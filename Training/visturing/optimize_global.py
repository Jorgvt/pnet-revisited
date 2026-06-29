import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
import gc
from flax.core import pop

from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from paramperceptnet.constraints import clip_layer, clip_param

# Visturing properties and configs
from visturing.properties import prop1, prop2, prop3_4, prop5, prop6_7, prop8, prop9, prop10
from visturing.properties.config import (
    default_prop2_config, default_prop3_4_config, default_prop5_config,
    default_prop6_7_config, default_prop8_config, default_prop9_config, default_prop10_config
)
from utils import collect_property_stimuli, make_loss_from_diffs

# Downsampled configurations to prevent RAM OOM and speed up global training
training_prop2_config = default_prop2_config

training_prop3_4_config = {
    **default_prop3_4_config,
    "freqs": np.arange(1, 17, step=2),  # 8 frequencies
}

training_prop5_config = {
    **default_prop5_config,
    "freqs": np.arange(1, 17, step=2),  # 8 frequencies
    "freqs_mask": np.array([2, 6, 10]),  # 3 mask frequencies
}

training_prop6_7_config = {
    **default_prop6_7_config,
    "freqs": np.arange(1, 17, step=2),  # 8 frequencies
    "Cs": np.linspace(0.01, 0.2, num=5),  # 5 contrast levels
}

training_prop8_config = {
    **default_prop8_config,
    "freqs": np.arange(1, 17, step=2),  # 8 frequencies
    "freqs_mask": np.array([2, 6, 10]),  # 3 mask frequencies
    "Cs": np.linspace(0.01, 0.2, num=5),  # 5 contrasts
    "Cs_mask": np.linspace(0.05, 0.2, num=2) / 0.3,  # 2 mask contrasts
}

training_prop9_config = {
    **default_prop9_config,
    "freqs_mask": np.arange(1, 17, step=2),  # 8 mask frequencies
}

training_prop10_config = {
    **default_prop10_config,
    "freqs": np.arange(2, 17, step=2),  # 8 frequencies
    "Cs": np.linspace(0.01, 0.2, num=6)[1:],  # 5 contrast levels
    "Cs_mask": np.linspace(0.05, 0.2, num=2) / 0.3,  # 2 mask contrasts
    "thetas_mask": np.linspace(0, 180, num=5)[:-1],  # 4 orientation mask angles
}

def load_prop1_stimuli():
    data_path = os.path.join(".", "Experiment_1")
    if not os.path.exists(data_path):
        print("Experiment_1 data not found. Downloading...")
        prop1.download_data(".")
    imgs, ref_img, _ = prop1.load_data(data_path)
    ref_img_expanded = np.repeat(ref_img[None, ...], len(imgs), axis=0)
    slice_sizes = [len(imgs)]
    return imgs, ref_img_expanded, slice_sizes

def jax_pearson_correlation(x, y):
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = jnp.sum(xm * ym)
    r_den = jnp.sqrt(jnp.sum(xm ** 2) * jnp.sum(ym ** 2) + 1e-8)
    return r_num / r_den

def make_prop1_loss(a_interp_jax, loss_type="correlation"):
    def loss_from_diffs(diffs_val):
        corr = jax_pearson_correlation(diffs_val, a_interp_jax)
        if loss_type == "mse":
            loss = jnp.mean((diffs_val - a_interp_jax) ** 2)
        elif loss_type == "mse_z":
            diffs_z = (diffs_val - jnp.mean(diffs_val)) / (jnp.std(diffs_val) + 1e-8)
            gt_z = (a_interp_jax - jnp.mean(a_interp_jax)) / (jnp.std(a_interp_jax) + 1e-8)
            loss = jnp.mean((diffs_z - gt_z) ** 2)
        else:
            loss = -corr
        return loss, corr
    return loss_from_diffs

def make_memory_efficient_grad_fn(model, state, jit_calculate_diffs, loss_from_diffs, batch_size):
    """Creates a VJP-based gradient function that computes backward pass batch-by-batch to save memory."""
    jit_loss_grad = jax.jit(jax.value_and_grad(loss_from_diffs, has_aux=True))
    
    def grad_fn(params_val, stimuli_flat, plain_flat):
        total_stimuli = len(stimuli_flat)
        
        # Step A: Forward pass (no gradient history/activations stored)
        params_sg = jax.lax.stop_gradient(params_val)
        diffs_list = []
        for idx in range(0, total_stimuli, batch_size):
            chunk_a = stimuli_flat[idx : idx + batch_size]
            chunk_b = plain_flat[idx : idx + batch_size]
            d = jit_calculate_diffs(params_sg, chunk_a, chunk_b)
            diffs_list.append(d)
            
        diffs = jnp.concatenate(diffs_list, axis=0)
        
        # Step B: Compute loss and loss gradient w.r.t differences
        (loss_val, aux_val), d_loss_d_diffs = jit_loss_grad(diffs)
        
        # Step C: Backward pass (compute VJPs batch-by-batch)
        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_val)
        
        for idx in range(0, total_stimuli, batch_size):
            chunk_a = stimuli_flat[idx : idx + batch_size]
            chunk_b = plain_flat[idx : idx + batch_size]
            chunk_cotangent = d_loss_d_diffs[idx : idx + batch_size]
            
            def batch_forward(p):
                return jit_calculate_diffs(p, chunk_a, chunk_b)
                
            _, vjp_fn = jax.vjp(batch_forward, params_val)
            grads_chunk = vjp_fn(chunk_cotangent)[0]
            grads = jax.tree_util.tree_map(lambda g, gc: g + gc, grads, grads_chunk)
            
        return (loss_val, aux_val), grads
        
    return grad_fn

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Global Visturing Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--weighted", action="store_true", help="Optimize weighted correlation instead of non-weighted")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Learning rate for optimization")
    parser.add_argument("--loss_type", type=str, default="correlation", choices=["correlation", "mse", "mse_z"], help="Loss function to optimize")
    args = parser.parse_args()

    print(f"Starting Global Visturing Optimization using loss type '{args.loss_type}' and {'WEIGHTED' if args.weighted else 'NON-WEIGHTED'} correlation...")

    # 1. Initialize the model
    key = jax.random.PRNGKey(42)
    x_init = jnp.ones((1, 128, 128, 3))
    model = Model()
    variables = model.init(key, x_init)
    state, params = pop(variables, "params")
    params, state = init_model(model, params, state)
    params = clip_layer(params, "GDN", a_min=0)
    params = clip_param(params, "A", a_min=0)
    params = clip_param(params, "K", a_min=1 + 1e-5)
    print("Model initialized and constrained!")

    # Populate precalc filters in state
    dummy_x = jnp.zeros((1, 128, 128, 3))
    _, state = model.apply({"params": params, **state}, dummy_x, train=True, mutable=list(state.keys()))
    print("Precalculated filters populated in state!")

    # 2. Setup optimizer
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    # 3. Define JITted batch difference function
    @jax.jit
    def jit_calculate_diffs(params_val, a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        feat_a, _ = model.apply({"params": params_val, **state}, a_j, train=True, mutable=list(state.keys()))
        feat_b, _ = model.apply({"params": params_val, **state}, b_j, train=True, mutable=list(state.keys()))
        return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

    # 4. Initialize properties and their gradient/loss functions
    properties_info = [
        {"name": "prop1", "module": prop1, "config": {}},
        {"name": "prop2", "module": prop2, "config": training_prop2_config},
        {"name": "prop3_4", "module": prop3_4, "config": training_prop3_4_config},
        {"name": "prop5", "module": prop5, "config": training_prop5_config},
        {"name": "prop6_7", "module": prop6_7, "config": training_prop6_7_config},
        {"name": "prop8", "module": prop8, "config": training_prop8_config},
        {"name": "prop9", "module": prop9, "config": training_prop9_config},
        {"name": "prop10", "module": prop10, "config": training_prop10_config},
    ]

    # Precompute a_interp for prop1
    gt_path = os.path.join(".", "ground_truth")
    if not os.path.exists(gt_path):
        print("ground_truth directory not found. Downloading...")
        from visturing.properties.utils import download_ground_truth
        download_ground_truth(".")
    data_path = os.path.join(".", "Experiment_1")
    if not os.path.exists(data_path):
        print("Experiment_1 data not found. Downloading...")
        prop1.download_data(".")
    x_gt, a_gt, _, _ = prop1.load_ground_truth(gt_path)
    _, _, lambdas = prop1.load_data(data_path)
    a_interp = np.interp(lambdas, x_gt, a_gt)
    a_interp_jax = jnp.asarray(a_interp)

    loss_fns = {}
    grad_fns = {}

    print("Initializing property loss and gradient functions sequentially...")
    for info in properties_info:
        name = info["name"]
        print(f"  Setting up {name}...")
        
        # Load stimuli briefly to get slice sizes
        if name == "prop1":
            imgs, ref_img_expanded, slice_sizes = load_prop1_stimuli()
        else:
            imgs, ref_img_expanded, slice_sizes = collect_property_stimuli(
                info["module"], info["config"], args.batch_size
            )
            
        # Create loss function
        if name == "prop1":
            loss_fns[name] = make_prop1_loss(a_interp_jax, loss_type=args.loss_type)
        else:
            loss_fns[name] = make_loss_from_diffs(
                info["module"], info["config"], slice_sizes, args.weighted, loss_type=args.loss_type
            )
            
        # Create gradient function
        grad_fns[name] = make_memory_efficient_grad_fn(
            model, state, jit_calculate_diffs, loss_fns[name], args.batch_size
        )
        
        # Free memory immediately
        del imgs, ref_img_expanded
        gc.collect()

    # 5. Define evaluation function
    def evaluate_global(params_val):
        total_loss = 0.0
        corrs_val = {}
        for info in properties_info:
            name = info["name"]
            if name == "prop1":
                stimuli_flat, plain_flat, _ = load_prop1_stimuli()
            else:
                stimuli_flat, plain_flat, _ = collect_property_stimuli(
                    info["module"], info["config"], args.batch_size
                )
                
            diffs = []
            for idx in range(0, len(stimuli_flat), args.batch_size):
                d = jit_calculate_diffs(params_val, stimuli_flat[idx : idx + args.batch_size], plain_flat[idx : idx + args.batch_size])
                diffs.append(d)
            
            loss_val, corr_val = loss_fns[name](jnp.concatenate(diffs, axis=0))
            total_loss += float(loss_val) / len(properties_info)
            corrs_val[name] = float(corr_val)
            
            del stimuli_flat, plain_flat
            gc.collect()
            
        return total_loss, corrs_val

    suffix = "weighted" if args.weighted else "non_weighted"
    if args.loss_type != "correlation":
        save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_global_{suffix}_{args.loss_type}.pkl")
    else:
        save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_global_{suffix}.pkl")
    best_loss = 999.0

    print("Initial evaluation...")
    init_loss, init_corrs = evaluate_global(params)
    best_loss = float(init_loss)
    print(f"Initial global loss: {init_loss:.4f}")
    for name, c in init_corrs.items():
        print(f"  {name:<8}: {c:.4f}")

    print(f"\nRunning global optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        accumulated_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        total_loss = 0.0
        corrs_val = {}
        
        for info in properties_info:
            name = info["name"]
            
            # Load stimuli for this property
            if name == "prop1":
                stimuli_flat, plain_flat, _ = load_prop1_stimuli()
            else:
                stimuli_flat, plain_flat, _ = collect_property_stimuli(
                    info["module"], info["config"], args.batch_size
                )
                
            # Compute loss and gradients
            (loss_val, corr_val), grads = grad_fns[name](params, stimuli_flat, plain_flat)
            
            # Accumulate loss and gradients
            total_loss += float(loss_val) / len(properties_info)
            corrs_val[name] = float(corr_val)
            accumulated_grads = jax.tree_util.tree_map(
                lambda g, gc: g + gc / len(properties_info),
                accumulated_grads,
                grads
            )
            
            # Free memory immediately
            del stimuli_flat, plain_flat
            gc.collect()
            
        # NaN protection
        if np.isnan(total_loss):
            print(f"NaN value detected at step {i+1}. Stopping training immediately to protect parameters.")
            break
            
        # Update params
        updates, opt_state = tx.update(accumulated_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = clip_layer(params, "GDN", a_min=0)
        params = clip_param(params, "A", a_min=0)
        params = clip_param(params, "K", a_min=1 + 1e-5)
        
        # Checkpoint if joint loss improves
        if total_loss < best_loss:
            best_loss = float(total_loss)
            variables_to_save = {"params": params, "state": state}
            with open(save_path, "wb") as f_save:
                pickle.dump(variables_to_save, f_save)
            print(f"Step {i+1:02d} | New best joint loss: {best_loss:.4f} | Checkpoint saved!")
            
        print(f"Step {i+1:02d} | Joint Loss: {total_loss:.4f}")
        for name, c in corrs_val.items():
            print(f"  {name:<8}: {c:.4f}")

    print(f"\nGlobal training complete! Best joint loss achieved: {best_loss:.4f}")
    print(f"Trained variables saved to {save_path}")

if __name__ == "__main__":
    main()
