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

def jax_z_score(x):
    mean = jnp.mean(x)
    std = jnp.std(x) + 1e-8
    return (x - mean) / std

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Global Concatenated Visturing Optimization Experiment")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--weighted", action="store_true", help="Optimize weighted correlation instead of non-weighted")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Learning rate for optimization")
    args = parser.parse_args()

    print(f"Starting Global Concatenated Visturing Optimization using {'WEIGHTED' if args.weighted else 'NON-WEIGHTED'} correlation...")

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

    # 4. Initialize properties
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
    x_gt, a_gt, _, _ = prop1.load_ground_truth(os.path.join(".", "ground_truth"))
    _, _, lambdas = prop1.load_data(os.path.join(".", "Experiment_1"))
    a_interp = np.interp(lambdas, x_gt, a_gt)
    prop1_gt = a_interp

    # Load all ground truths and slice information
    gts_list = []
    slices_info = []
    slice_sizes = {}
    current_idx = 0

    print("Loading and preparing ground truths sequentially...")
    for info in properties_info:
        name = info["name"]
        print(f"  Preparing {name}...")
        
        if name == "prop1":
            gt_p = prop1_gt
            sz = len(gt_p)
            slice_sizes[name] = [sz]
        else:
            # We call evaluate_gen with dummy prediction to load the ground truths
            res = info["module"].evaluate_gen(
                lambda a, b: np.zeros(len(a)),
                verbose=False,
                return_gt=True,
                xp=np,
                **info["config"]
            )
            gt_p = np.concatenate([res.gt[k].ravel() for k in res.results.keys()])
            sz = len(gt_p)
            slice_sizes[name] = [len(res.results[k].ravel()) for k in res.results.keys()]
            
        # Z-score the ground truth for this property to put them on a comparable scale
        gt_mean = gt_p.mean()
        gt_std = gt_p.std() + 1e-8
        gt_p_normalized = (gt_p - gt_mean) / gt_std
        
        gts_list.append(gt_p_normalized)
        slices_info.append((current_idx, current_idx + sz))
        current_idx += sz
        gc.collect()

    gts_global_jax = jnp.asarray(np.concatenate(gts_list, axis=0))
    print(f"Total concatenated dataset size: {len(gts_global_jax)}")

    # 5. Define JITted Global Concatenated Loss Function
    @jax.jit
    def global_loss_fn(diffs_val):
        z_diffs_list = []
        for start, end in slices_info:
            z_diffs_list.append(jax_z_score(diffs_val[start:end]))
        z_diffs = jnp.concatenate(z_diffs_list, axis=0)
        
        corr = jax_pearson_correlation(z_diffs, gts_global_jax)
        return -corr, corr

    jit_global_loss_grad = jax.jit(jax.value_and_grad(global_loss_fn, has_aux=True))

    # 6. Define Individual Property Loss Functions (for logging only)
    individual_loss_fns = {}
    for info in properties_info:
        name = info["name"]
        if name == "prop1":
            individual_loss_fns[name] = lambda diffs: (-jax_pearson_correlation(diffs, jnp.asarray(prop1_gt)), jax_pearson_correlation(diffs, jnp.asarray(prop1_gt)))
        else:
            individual_loss_fns[name] = make_loss_from_diffs(
                info["module"], info["config"], slice_sizes[name], args.weighted
            )

    # 7. Define Evaluation Function
    def evaluate_global(params_val):
        diffs_list = []
        for info in properties_info:
            name = info["name"]
            if name == "prop1":
                stimuli_flat, plain_flat, _ = load_prop1_stimuli()
            else:
                stimuli_flat, plain_flat, _ = collect_property_stimuli(
                    info["module"], info["config"], args.batch_size
                )
                
            diffs_p = []
            for idx in range(0, len(stimuli_flat), args.batch_size):
                d = jit_calculate_diffs(params_val, stimuli_flat[idx : idx + args.batch_size], plain_flat[idx : idx + args.batch_size])
                diffs_p.append(d)
            
            diffs_list.append(jnp.concatenate(diffs_p, axis=0))
            del stimuli_flat, plain_flat
            gc.collect()
            
        all_diffs = jnp.concatenate(diffs_list, axis=0)
        loss_val, corr_val = global_loss_fn(all_diffs)
        
        # Calculate individual property correlations for logging
        indiv_corrs = {}
        for idx, info in enumerate(properties_info):
            name = info["name"]
            start, end = slices_info[idx]
            _, c_val = individual_loss_fns[name](all_diffs[start:end])
            indiv_corrs[name] = float(c_val)
            
        return float(loss_val), float(corr_val), indiv_corrs

    # 8. Define Training Step Function
    def training_step(params_val):
        # Step A: Forward pass (collect differences, no parameter gradients tracked)
        params_sg = jax.lax.stop_gradient(params_val)
        diffs_list = []
        for info in properties_info:
            name = info["name"]
            if name == "prop1":
                stimuli_flat, plain_flat, _ = load_prop1_stimuli()
            else:
                stimuli_flat, plain_flat, _ = collect_property_stimuli(
                    info["module"], info["config"], args.batch_size
                )
                
            diffs_p = []
            for idx in range(0, len(stimuli_flat), args.batch_size):
                d = jit_calculate_diffs(params_sg, stimuli_flat[idx : idx + args.batch_size], plain_flat[idx : idx + args.batch_size])
                diffs_p.append(d)
            
            diffs_list.append(jnp.concatenate(diffs_p, axis=0))
            del stimuli_flat, plain_flat
            gc.collect()
            
        all_diffs = jnp.concatenate(diffs_list, axis=0)
        
        # Step B: Compute loss and loss gradient w.r.t differences
        (loss_val, corr_val), d_loss_d_diffs = jit_global_loss_grad(all_diffs)
        
        # Step C: Backward pass (VJPs property-by-property, batch-by-batch)
        accumulated_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_val)
        
        for idx, info in enumerate(properties_info):
            name = info["name"]
            start, end = slices_info[idx]
            prop_cotangent = d_loss_d_diffs[start:end]
            
            if name == "prop1":
                stimuli_flat, plain_flat, _ = load_prop1_stimuli()
            else:
                stimuli_flat, plain_flat, _ = collect_property_stimuli(
                    info["module"], info["config"], args.batch_size
                )
                
            total_stimuli = len(stimuli_flat)
            for j in range(0, total_stimuli, args.batch_size):
                chunk_a = stimuli_flat[j : j + args.batch_size]
                chunk_b = plain_flat[j : j + args.batch_size]
                chunk_cotangent = prop_cotangent[j : j + args.batch_size]
                
                def batch_forward(p):
                    return jit_calculate_diffs(p, chunk_a, chunk_b)
                    
                _, vjp_fn = jax.vjp(batch_forward, params_val)
                grads_chunk = vjp_fn(chunk_cotangent)[0]
                accumulated_grads = jax.tree_util.tree_map(lambda g, gc: g + gc, accumulated_grads, grads_chunk)
                
            del stimuli_flat, plain_flat
            gc.collect()
            
        return loss_val, corr_val, accumulated_grads

    suffix = "weighted" if args.weighted else "non_weighted"
    save_path = os.path.join(os.path.dirname(__file__), f"model_pnet_global_concat_{suffix}.pkl")
    best_loss = 999.0

    print("Initial evaluation...")
    init_loss, init_corr, init_corrs = evaluate_global(params)
    best_loss = float(init_loss)
    print(f"Initial global loss: {init_loss:.4f} | Global Concatenated Correlation: {init_corr:.4f}")
    for name, c in init_corrs.items():
        print(f"  {name:<8}: {c:.4f}")

    print(f"\nRunning global optimization loop ({args.iterations} steps)...")
    for i in range(args.iterations):
        loss_val, corr_val, grads = training_step(params)
        
        # NaN protection
        if np.isnan(loss_val):
            print(f"NaN value detected at step {i+1}. Stopping training immediately to protect parameters.")
            break
            
        # Update params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = clip_layer(params, "GDN", a_min=0)
        params = clip_param(params, "A", a_min=0)
        params = clip_param(params, "K", a_min=1 + 1e-5)
        
        # Checkpoint if joint loss improves
        if loss_val < best_loss:
            best_loss = float(loss_val)
            variables_to_save = {"params": params, "state": state}
            with open(save_path, "wb") as f_save:
                pickle.dump(variables_to_save, f_save)
            print(f"Step {i+1:02d} | New best joint loss: {best_loss:.4f} | Checkpoint saved!")
            
        # Calculate individual property correlations for logging
        _, _, indiv_corrs = evaluate_global(params)
        print(f"Step {i+1:02d} | Loss: {loss_val:.4f} | Global Concatenated Correlation: {corr_val:.4f}")
        for name, c in indiv_corrs.items():
            print(f"  {name:<8}: {c:.4f}")

    print(f"\nGlobal training complete! Best joint loss achieved: {best_loss:.4f}")
    print(f"Trained variables saved to {save_path}")

if __name__ == "__main__":
    main()
