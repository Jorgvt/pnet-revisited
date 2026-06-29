from paramperceptnet.constraints import clip_layer, clip_param
import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from flax.core import pop
from visturing.properties import prop5
from visturing.properties.config import default_prop5_config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--weighted", action="store_true", help="Optimize weighted correlation instead of non-weighted")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Learning rate for optimization")
    parser.add_argument("--loss_type", type=str, default="correlation", choices=["correlation", "mse", "mse_z"], help="Loss function to optimize")
    args = parser.parse_args()

    print(f"Starting JAX PerceptNet (initialized from scratch) optimization experiment on Prop. 5 using {'WEIGHTED' if args.weighted else 'NON-WEIGHTED'} correlation...")

    key = jax.random.PRNGKey(42)
    x_init = jnp.ones((1, 128, 128, 3))
    model = Model()
    variables = model.init(key, x_init)
    state, params = pop(variables, "params")
    params, state = init_model(model, params, state)
    params = clip_layer(params, "GDN", a_min=0)
    params = clip_param(params, "A", a_min=0)
    params = clip_param(params, "K", a_min=1 + 1e-5)
    print("Model initialized from scratch with custom parameters!")

    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    from utils import collect_property_stimuli, make_loss_from_diffs, make_memory_efficient_grad_fn

    # 1. Collect all stimuli in a non-JIT context
    stimuli_flat, plain_flat, slice_sizes = collect_property_stimuli(
        prop5, default_prop5_config, args.batch_size
    )

    # 2. Define JITted batch difference function
    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    # 3. Create the flat loss function
    loss_from_diffs = make_loss_from_diffs(
        prop5, default_prop5_config, slice_sizes, args.weighted, loss_type=args.loss_type
    )

    # 4. Build the memory-efficient grad function
    grad_fn = make_memory_efficient_grad_fn(
        model, state, jit_calculate_diffs, loss_from_diffs,
        stimuli_flat, plain_flat, args.batch_size
    )

    # 5. Define loss_fn for evaluation steps
    def loss_fn(params_val):
        diffs = []
        for idx in range(0, len(stimuli_flat), args.batch_size):
            d = jit_calculate_diffs(params_val, stimuli_flat[idx : idx + args.batch_size], plain_flat[idx : idx + args.batch_size])
            diffs.append(d)
        return loss_from_diffs(jnp.concatenate(diffs, axis=0))

    import pickle
    suffix = "weighted" if args.weighted else "non_weighted"
    if args.loss_type != "correlation":
        save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_init_prop5_{suffix}_{args.loss_type}.pkl")
    else:
        save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_init_prop5_{suffix}.pkl")
    best_corr = -1.0

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    best_corr = float(init_corr)
    print(f"Initial global {'weighted' if args.weighted else 'non-weighted'} correlation: {init_corr:.4f}")

    print(f"\nRunning optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        (loss_val, corr_val), grads = grad_fn(params)
        
        # NaN protection
        if np.isnan(corr_val) or np.isnan(loss_val):
            print(f"NaN value detected at step {i+1}. Stopping training immediately to protect model parameters.")
            break
        
        # Update params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = clip_layer(params, "GDN", a_min=0)
        params = clip_param(params, "A", a_min=0)
        params = clip_param(params, "K", a_min=1 + 1e-5)
        
        # Save checkpoint if correlation improves
        if corr_val > best_corr:
            best_corr = float(corr_val)
            variables_to_save = {"params": params, "state": state}
            with open(save_path, "wb") as f_save:
                pickle.dump(variables_to_save, f_save)
            print(f"Step {i+1:02d} | New best correlation: {best_corr:.4f} | Checkpoint saved!")
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final global {'weighted' if args.weighted else 'non-weighted'} correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

    print(f"\nTraining complete! Best correlation achieved: {best_corr:.4f}")
    print(f"Trained variables saved to {save_path}")

if __name__ == "__main__":
    main()
