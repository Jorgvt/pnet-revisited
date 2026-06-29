import os
from datasets import load_dataset
from PIL import Image
import numpy as np

from perceptualtests.color_matrices import Mng2xyz, Mxyz2atd, gamma
from flax.serialization import to_state_dict
from orbax.checkpoint.msgpack_utils import msgpack_serialize, msgpack_restore

def rgb2atd(img):
    return img**(1/gamma) @ Mng2xyz.T @ Mxyz2atd.T

def download_imagenet_subset(num_images=64, output_dir="imagenet_samples"):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Streaming {num_images} images from Hugging Face (timm/mini-imagenet)...")
    
    # "streaming=True" allows us to load data without downloading the whole dataset
    dataset = load_dataset("timm/mini-imagenet", split="train", streaming=True)
    
    count = 0
    # Iterate through the stream
    imgs = []
    for sample in dataset:
        if count >= num_images:
            break
            
        image = sample['image']
        label = sample['label']
        imgs.append(image.resize((256,256)))
        count += 1
    return imgs

def get_imagenet_ready(num_images):
    """Downloads num_images imagenet images and returns them together
    with their atd transformation."""

    imgs = download_imagenet_subset(num_images=64)
    imgs = np.stack([np.array(img) for img in imgs])/255.
    atd = np.stack([rgb2atd(i) for i in imgs])

    return imgs, atd


def save_state(state, path):
    """Saves the state as .msgpack"""
    if path.split(".")[-1] != "msgpack":
        path = path + ".msgpack"
    with open(path, "wb") as f:
        f.write(msgpack_serialize(
            to_state_dict(state)
            ))

def load_state(path):
    """Loads the state in .msgpack"""
    
    with open(path, "rb") as f:
        state = msgpack_restore(f.read())

    return state

# -----------------------------------------------------------------------------
# Memory-Efficient Batch-by-Batch VJP Gradient Utilities
# -----------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from typing import Sequence

def collect_property_stimuli(prop_module, config, batch_size):
    """Collects all stimuli and plain/reference images by running evaluate_gen once with a mock function."""
    all_stimuli = []
    all_plains = []
    
    def collect_fn(a, b):
        all_stimuli.append(np.asarray(a))
        all_plains.append(np.asarray(b))
        return jnp.zeros(len(a))
        
    prop_module.evaluate_gen(
        collect_fn,
        xp=jnp,
        batch_size=None,
        verbose=False,
        **config
    )
    
    stimuli_flat = np.concatenate(all_stimuli, axis=0)
    plain_flat = np.concatenate(all_plains, axis=0)
    slice_sizes = [len(s) for s in all_stimuli]
    return stimuli_flat, plain_flat, slice_sizes

def make_loss_from_diffs(prop_module, config, slice_sizes, weighted, loss_type="correlation"):
    """Constructs a JAX-traceable loss function that computes correlation or MSE from a flat differences array."""
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
            
        current_config = {**config}
        if loss_type in ["mse", "mse_z"]:
            current_config["return_gt"] = True

        res = prop_module.evaluate_gen(
            mock_calculate_diffs,
            xp=jnp,
            batch_size=None,
            verbose=False,
            **current_config
        )
        corr_key = "weighted" if weighted else "non-weighted"
        corr = res.correlations[corr_key]["global"]
        
        if loss_type == "mse":
            preds = jnp.concatenate([jnp.ravel(res.results[k]) for k in res.results.keys()])
            gts_flat = jnp.concatenate([jnp.ravel(res.gt[k]) for k in res.results.keys()])
            loss = jnp.mean((preds - gts_flat) ** 2)
        elif loss_type == "mse_z":
            preds = jnp.concatenate([jnp.ravel(res.results[k]) for k in res.results.keys()])
            gts_flat = jnp.concatenate([jnp.ravel(res.gt[k]) for k in res.results.keys()])
            preds_z = (preds - jnp.mean(preds)) / (jnp.std(preds) + 1e-8)
            gts_z = (gts_flat - jnp.mean(gts_flat)) / (jnp.std(gts_flat) + 1e-8)
            loss = jnp.mean((preds_z - gts_z) ** 2)
        else:
            loss = -corr
            
        return loss, corr
    return loss_from_diffs

def make_memory_efficient_grad_fn(model, state, jit_calculate_diffs, loss_from_diffs, stimuli_flat, plain_flat, batch_size):
    """Creates a VJP-based gradient function that computes backward pass batch-by-batch to save memory."""
    jit_loss_grad = jax.jit(jax.value_and_grad(loss_from_diffs, has_aux=True))
    total_stimuli = len(stimuli_flat)
    
    def grad_fn(params_val):
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


