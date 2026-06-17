import jax
from jax import random, numpy as jnp
import flax.linen as nn
from flax.core import pop
import optax
import numpy as np
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop5
from visturing.properties.config import default_prop5_config
import time

def main():
    print("Initializing model...")
    key = random.PRNGKey(42)
    x_init = jnp.ones((1, 128, 128, 3))
    model = Model()
    variables = model.init(key, x_init)
    state, params = pop(variables, "params")
    params, state = init_model(model, params, state)
    
    # Dummy forward pass to populate state
    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    
    # 1. Collect all stimuli first by running evaluate_gen with a mock function
    print("Collecting stimuli from prop5...")
    all_stimuli = []
    all_plains = []
    
    def collect_stimuli(a, b):
        all_stimuli.append(np.asarray(a))
        all_plains.append(np.asarray(b))
        return jnp.zeros(len(a))
        
    prop5.evaluate_gen(
        collect_stimuli,
        xp=jnp,
        batch_size=None,
        verbose=False,
        **default_prop5_config
    )
    
    stimuli_flat = np.concatenate(all_stimuli, axis=0)
    plain_flat = np.concatenate(all_plains, axis=0)
    slice_sizes = [len(s) for s in all_stimuli]
    total_stimuli = len(stimuli_flat)
    print(f"Total stimuli pairs collected: {total_stimuli}")
    print(f"Slice sizes (achrom, red-green, yellow-blue): {slice_sizes}")
    
    batch_size = 32
    
    # JIT-compiled function to calculate differences for a batch
    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        feat_a, _ = model.apply({"params": params_val, **state}, a, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    # 2. Define the loss function from a flat array of differences
    def loss_from_diffs(diffs_val):
        start = 0
        slices = []
        for size in slice_sizes:
            slices.append(diffs_val[start : start + size])
            start += size
            
        call_count = 0
        def mock_calculate_diffs(a, b):
            nonlocal call_count
            res = slices[call_count]
            call_count += 1
            return res
            
        res = prop5.evaluate_gen(
            mock_calculate_diffs,
            xp=jnp,
            batch_size=None, # No batching here to avoid JAX loops!
            verbose=False,
            **default_prop5_config
        )
        corr = res.correlations["non-weighted"]["global"]
        return -corr
        
    jit_loss_from_diffs_grad = jax.jit(jax.value_and_grad(loss_from_diffs))
    
    # 3. Define the custom gradient function
    def memory_efficient_grad_fn(params_val):
        t_start = time.time()
        # Step A: Forward pass (compute diffs batch-by-batch without gradient history)
        # We use stop_gradient on params to make sure JAX doesn't track activations
        params_sg = jax.lax.stop_gradient(params_val)
        diffs_list = []
        for idx in range(0, total_stimuli, batch_size):
            chunk_a = stimuli_flat[idx : idx + batch_size]
            chunk_b = plain_flat[idx : idx + batch_size]
            d = jit_calculate_diffs(params_sg, chunk_a, chunk_b)
            diffs_list.append(d)
            
        diffs = jnp.concatenate(diffs_list, axis=0)
        
        # Step B: Compute loss and gradient of loss w.r.t diffs
        loss_val, d_loss_d_diffs = jit_loss_from_diffs_grad(diffs)
        
        # Step C: Backward pass (compute VJPs batch-by-batch)
        # Initialize gradient accumulator with zeros
        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_val)
        
        # We loop and compute JAX VJP for each batch
        for idx in range(0, total_stimuli, batch_size):
            chunk_a = stimuli_flat[idx : idx + batch_size]
            chunk_b = plain_flat[idx : idx + batch_size]
            chunk_cotangent = d_loss_d_diffs[idx : idx + batch_size]
            
            # Helper function for VJP
            def batch_forward(p):
                return jit_calculate_diffs(p, chunk_a, chunk_b)
                
            # Compute VJP for this batch
            # We evaluate the forward pass and get the VJP function
            _, vjp_fn = jax.vjp(batch_forward, params_val)
            # Pull back the cotangent to parameters
            grads_chunk = vjp_fn(chunk_cotangent)[0]
            
            # Accumulate gradients
            grads = jax.tree_util.tree_map(lambda g, gc: g + gc, grads, grads_chunk)
            
        # correlation is -loss
        corr_val = -loss_val
        print(f"Memory efficient step took: {time.time() - t_start:.2f} seconds")
        return (loss_val, corr_val), grads

    print("\nEvaluating with memory efficient VJP grad_fn...")
    (loss_me, corr_me), grads_me = memory_efficient_grad_fn(params)
    jax.block_until_ready(grads_me)
    print("\nMemory-efficient step completed successfully!")
    print(f"  Loss: {loss_me:.4f}")
    print(f"  Correlation: {corr_me:.4f}")
    print(f"  Grad leaves count: {len(jax.tree_util.tree_leaves(grads_me))}")

if __name__ == "__main__":
    main()
