import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from flax.core import pop
from visturing.properties import prop6_7
from visturing.properties.config import default_prop6_7_config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--weighted", action="store_true", help="Optimize weighted correlation instead of non-weighted")
    args = parser.parse_args()

    print(f"Starting JAX PerceptNet (initialized from scratch) optimization experiment on Prop. 6 & 7 using {'WEIGHTED' if args.weighted else 'NON-WEIGHTED'} correlation...")

    key = jax.random.PRNGKey(42)
    x_init = jnp.ones((1, 128, 128, 3))
    model = Model()
    variables = model.init(key, x_init)
    state, params = pop(variables, "params")
    params, state = init_model(model, params, state)
    print("Model initialized from scratch with custom parameters!")

    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    tx = optax.adam(learning_rate=1e-4)
    opt_state = tx.init(params)

    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    def loss_fn(params_val):
        calculate_diffs = lambda a, b: jit_calculate_diffs(params_val, a, b)

        res = prop6_7.evaluate_gen(
            calculate_diffs,
            xp=jnp,
            batch_size=args.batch_size,
            verbose=False,
            **default_prop6_7_config
        )
        
        corr_key = "weighted" if args.weighted else "non-weighted"
        corr = res.correlations[corr_key]["global"]
        loss = -corr
        return loss, corr

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    print(f"Initial global {'weighted' if args.weighted else 'non-weighted'} correlation: {init_corr:.4f}")

    print(f"\nRunning optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        (loss_val, corr_val), grads = grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final global {'weighted' if args.weighted else 'non-weighted'} correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

    print("\nSaving trained model variables...")
    import pickle
    suffix = "weighted" if args.weighted else "non_weighted"
    save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_init_prop6_7_{suffix}.pkl")
    variables_to_save = {"params": params, "state": state}
    with open(save_path, "wb") as f_save:
        pickle.dump(variables_to_save, f_save)
    print(f"Saved variables to {save_path}")

if __name__ == "__main__":
    main()
