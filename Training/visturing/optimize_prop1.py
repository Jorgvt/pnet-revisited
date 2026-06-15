import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from flax.core import pop
from visturing.properties import prop1

def jax_pearson_correlation(x, y):
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    dev_x = x - mean_x
    dev_y = y - mean_y
    cov_xy = jnp.sum(dev_x * dev_y)
    var_x = jnp.sum(dev_x ** 2)
    var_y = jnp.sum(dev_y ** 2)
    return cov_xy / jnp.sqrt(var_x * var_y + 1e-8)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--weighted", action="store_true", help="Optimize weighted correlation instead of non-weighted")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Learning rate for optimization")
    args = parser.parse_args()

    print("Starting JAX PerceptNet (initialized from scratch) optimization experiment on Prop. 1 (Spectral Sensitivity)...")
    print("Note: Prop. 1 only has standard (non-weighted) human spectral sensitivities.")

    # Paths (relative to the repo root)
    data_path = "./Experiment_1"
    gt_path = "./ground_truth"

    if not os.path.exists(data_path):
        data_path = prop1.download_data(".")

    # Load data
    imgs, ref_img, lambdas = prop1.load_data(data_path)
    x, a, _, _ = prop1.load_ground_truth(gt_path)
    a_interp = np.interp(lambdas, x, a)
    a_interp_j = jnp.asarray(a_interp)

    print(f"Loaded {len(imgs)} stimuli images of shape {imgs.shape[1:]}")

    key = jax.random.PRNGKey(42)
    x_init = jnp.ones((1, 128, 128, 3))
    model = Model()
    variables = model.init(key, x_init)
    state, params = pop(variables, "params")
    params, state = init_model(model, params, state)
    print("Model initialized from scratch with custom parameters!")

    # Run a dummy forward pass to populate precalc_filter variables in state
    dummy_x = jnp.zeros((1, *imgs.shape[1:]))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    # Initialize Optax optimizer (Adam)
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    # Define loss function
    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    def loss_fn(params_val):
        from visturing.properties.utils import run_batched
        calculate_diffs = lambda a, b: jit_calculate_diffs(params_val, a, b)
            
        ref_img_expanded = np.repeat(ref_img[None, ...], len(imgs), axis=0)
        diffs = run_batched(
            calculate_diffs,
            imgs,
            ref_img_expanded,
            batch_size=args.batch_size
        )
        
        # Pearson correlation
        corr = jax_pearson_correlation(diffs, a_interp_j)
        loss = -corr
        return loss, corr

    # Compute value and grads
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    print(f"Initial Pearson correlation: {init_corr:.4f}")

    print(f"\nRunning optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        (loss_val, corr_val), grads = grad_fn(params)
        
        # Update params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Correlation: {corr_val:.4f}")

    print("\nOptimization finished!")
    final_loss, final_corr = loss_fn(params)
    print(f"Final Pearson correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

    print("\nSaving trained model variables...")
    import pickle
    save_path = os.path.join(os.path.dirname(__file__), "model_pnet_init_prop1.pkl")
    variables_to_save = {"params": params, "state": state}
    with open(save_path, "wb") as f_save:
        pickle.dump(variables_to_save, f_save)
    print(f"Saved variables to {save_path}")

if __name__ == "__main__":
    main()
