#!/usr/bin/env python
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import pop
import pnet_revisited
from pnet_revisited.model import Model
from pnet_revisited.initialization import (
    init_dn_gamma,
    init_cs,
    init_dn_cs,
    init_v1,
)
from pnet_revisited.utils import download_imagenet_subset

def main():
    # Determine default paths in package directory
    pnet_revisited_dir = os.path.dirname(pnet_revisited.__file__)
    default_cs = os.path.join(pnet_revisited_dir, "a_star_gdn_cs.npy")
    default_v1 = os.path.join(pnet_revisited_dir, "a_star_gdn_v1.npy")

    parser = argparse.ArgumentParser(description="Recalculate a_star GDN parameters for Center-Surround and Gabor layers.")
    parser.add_argument("--num_images", type=int, default=64, help="Number of images to stream from mini-imagenet.")
    parser.add_argument("--quantile", type=float, default=0.90, help="Quantile value to calculate (default 0.90).")
    parser.add_argument("--seed", type=int, default=42, help="PRNG Key seed.")
    parser.add_argument("--output_cs", type=str, default=default_cs, help="Path to save recalculated a_star_gdn_cs.npy")
    parser.add_argument("--output_v1", type=str, default=default_v1, help="Path to save recalculated a_star_gdn_v1.npy")
    args = parser.parse_args()

    # Normalize paths to ensure they have the .npy extension
    if not args.output_cs.endswith(".npy"):
        args.output_cs += ".npy"
    if not args.output_v1.endswith(".npy"):
        args.output_v1 += ".npy"

    print("--- STEP 0: Downloading natural images subset ---")
    imgs = download_imagenet_subset(args.num_images)
    print(f"Loaded image batch with shape: {imgs.shape}")

    # Set up JAX model
    model = Model()
    key = jax.random.PRNGKey(args.seed)
    # Use dummy input of size 128 to initialize shapes
    x_dummy = jnp.ones((1, 128, 128, 3))

    # --- STEP 1: Recalculate a_star_gdn_cs ---
    print("\n--- STEP 1: Recalculating a_star_gdn_cs ---")
    variables = model.init(key, x_dummy)
    state, params = pop(variables, "params")

    # Apply Center-Surround initializations
    params = init_dn_gamma(params)
    params["CenterSurroundLogSigmaK_0"] = init_cs(params["CenterSurroundLogSigmaK_0"])

    # Run dummy pass to compile precalc filters
    _, state = model.apply(
        {"params": params, **state},
        x_dummy,
        train=True,
        mutable=list(state.keys())
    )

    # Run actual images and capture intermediates
    print("Evaluating model up to Center-Surround layer...")
    _, intermediates = model.apply(
        {"params": params, **state},
        imgs,
        train=False,
        capture_intermediates=True
    )

    cs_output = intermediates["intermediates"]["CenterSurroundLogSigmaK_0"]["__call__"][0]
    print(f"Center-Surround output shape: {cs_output.shape}")

    # Calculate quantile
    a_star_cs = jnp.quantile(jnp.abs(cs_output), q=args.quantile, axis=(0, 1, 2))
    a_star_cs = a_star_cs[None, None, None, :]

    # Save to file
    np.save(args.output_cs, a_star_cs)
    print(f"Successfully saved a_star_gdn_cs to {args.output_cs} with shape {a_star_cs.shape}")

    # --- STEP 2: Recalculate a_star_gdn_v1 ---
    print("\n--- STEP 2: Recalculating a_star_gdn_v1 ---")
    
    # Re-initialize to load the newly computed a_star_gdn_cs npy file
    variables = model.init(key, x_dummy)
    state, params = pop(variables, "params")

    # Apply parameters up to V1
    params = init_dn_gamma(params)
    params["CenterSurroundLogSigmaK_0"] = init_cs(params["CenterSurroundLogSigmaK_0"])
    params, state = init_dn_cs(params, state, a_star_path=args.output_cs) # Reads the newly updated npy file
    params = init_v1(params)

    # Run dummy pass to compile precalc filters
    _, state = model.apply(
        {"params": params, **state},
        x_dummy,
        train=True,
        mutable=list(state.keys())
    )

    # Run actual images and capture intermediates
    print("Evaluating model up to Gabor layer...")
    _, intermediates = model.apply(
        {"params": params, **state},
        imgs,
        train=False,
        capture_intermediates=True
    )

    gabor_output_tuple = intermediates["intermediates"]["GaborLayerGammaHumanLike__0"]["__call__"][0]
    gabor_output = gabor_output_tuple[0]
    print(f"Gabor output shape: {gabor_output.shape}")

    # Calculate quantile
    a_star_v1 = jnp.quantile(jnp.abs(gabor_output), q=args.quantile, axis=(0, 1, 2))
    a_star_v1 = a_star_v1[None, None, None, :]

    # Save to file
    np.save(args.output_v1, a_star_v1)
    print(f"Successfully saved a_star_gdn_v1 to {args.output_v1} with shape {a_star_v1.shape}")
    print("\nGDN initialization parameters recalculation completed successfully!")

if __name__ == "__main__":
    main()
