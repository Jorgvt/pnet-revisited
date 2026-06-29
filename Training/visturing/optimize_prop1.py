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
    parser.add_argument("--loss_type", type=str, default="correlation", choices=["correlation", "mse", "mse_z"], help="Loss function to optimize")
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
    params = clip_layer(params, "GDN", a_min=0)
    params = clip_param(params, "A", a_min=0)
    params = clip_param(params, "K", a_min=1 + 1e-5)
    print("Model initialized from scratch with custom parameters!")

    # Run a dummy forward pass to populate precalc_filter variables in state
    dummy_x = jnp.zeros((1, *imgs.shape[1:]))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    # Initialize Optax optimizer (Adam)
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    from utils import make_memory_efficient_grad_fn

    # 1. Prepare flat inputs
    ref_img_expanded = np.repeat(ref_img[None, ...], len(imgs), axis=0)
    stimuli_flat = imgs
    plain_flat = ref_img_expanded

    # 2. Define JITted batch difference function
    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    # 3. Create the flat loss function
    def loss_from_diffs(diffs_val):
        corr = jax_pearson_correlation(diffs_val, a_interp_j)
        if args.loss_type == "mse":
            loss = jnp.mean((diffs_val - a_interp_j) ** 2)
        elif args.loss_type == "mse_z":
            diffs_z = (diffs_val - jnp.mean(diffs_val)) / (jnp.std(diffs_val) + 1e-8)
            gt_z = (a_interp_j - jnp.mean(a_interp_j)) / (jnp.std(a_interp_j) + 1e-8)
            loss = jnp.mean((diffs_z - gt_z) ** 2)
        else:
            loss = -corr
        return loss, corr

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
    if args.loss_type != "correlation":
        save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_init_prop1_{args.loss_type}.pkl")
    else:
        save_path = os.path.join(os.path.dirname(__file__), "model_pnet_init_prop1.pkl")
    best_corr = -1.0

    print("Initial evaluation...")
    loss, init_corr = loss_fn(params)
    best_corr = float(init_corr)
    print(f"Initial Pearson correlation: {init_corr:.4f}")

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
    print(f"Final Pearson correlation: {final_corr:.4f}")
    print(f"Total improvement: {final_corr - init_corr:.4f}")

    print(f"\nTraining complete! Best correlation achieved: {best_corr:.4f}")
    print(f"Trained variables saved to {save_path}")

if __name__ == "__main__":
    main()
