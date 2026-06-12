import os
import sys
import jax
from jax import random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax.training import orbax_utils
from flax import struct
import optax
import orbax.checkpoint
import numpy as np
from ml_collections import config_flags, ConfigDict
from clu import metrics

# Local imports
from pnet_revisited.model import Model, ModelDenoising
from pnet_revisited.initialization import init_model
from paramperceptnet.constraints import clip_layer, clip_param
from paramperceptnet.training import TrainState

# -----------------------------------------------------------------------------
# Configuration Setup
# -----------------------------------------------------------------------------
# Config flags
_CONFIG = config_flags.DEFINE_config_file("config", default="Training/Denoising/config.py")
from absl import flags
flags.FLAGS(sys.argv)
config = _CONFIG.value
print("Using configuration:")
print(config)

# -----------------------------------------------------------------------------
# Dataset Loading and Preprocessing using HF Datasets
# -----------------------------------------------------------------------------
from datasets import load_dataset

print("Loading dataset jonathanli/imagenette2-320-split from Hugging Face...")
train_ds = load_dataset("jonathanli/imagenette2-320-split", split="train")
val_ds = load_dataset("jonathanli/imagenette2-320-split", split="test")

# Preprocessing transform (unsupervised denoising, only need images)
def preprocess(batch):
    target_size = (224, 224)
    images = np.array([
        np.array(img.convert("RGB").resize(target_size), dtype=np.float32) / 255.0 
        for img in batch['image']
    ])
    return {
        'image': images
    }

train_ds = train_ds.with_transform(preprocess)
val_ds = val_ds.with_transform(preprocess)

def get_epoch_iterator(dataset, batch_size, shuffle=False, seed=None):
    ds = dataset
    if shuffle:
        ds = ds.shuffle(seed=seed)
    for batch in ds.iter(batch_size=batch_size, drop_last_batch=True):
        yield batch['image']

# -----------------------------------------------------------------------------
# Metrics and Custom Train State definition
# -----------------------------------------------------------------------------
@struct.dataclass
class DenoiseMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")

class DenoiseTrainState(TrainState):
    metrics: DenoiseMetrics

def create_denoise_train_state(module, key, tx, input_shape):
    variables = module.init(key, jnp.ones(input_shape))
    state, params = flax.core.pop(variables, "params")
    return DenoiseTrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=DenoiseMetrics.empty(),
    )

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
tx = optax.adam(config.LEARNING_RATE)

# Create Outer ModelDenoising train state
state = create_denoise_train_state(
    ModelDenoising(), random.PRNGKey(config.SEED), tx, input_shape=(1, 224, 224, 3)
)

# Extract and initialize inner feature extractor Model
params = unfreeze(state.params)
batch_stats = unfreeze(state.state)

perceptnet_params = params["perceptnet"]
# Construct the state dictionary for the inner perceptnet containing all collections
perceptnet_state = {
    "batch_stats": batch_stats["batch_stats"]["perceptnet"],
    "precalc_filter": batch_stats["precalc_filter"]["perceptnet"],
}

# Call original human-like initialization on inner perceptnet
perceptnet_params, perceptnet_state = init_model(Model(), perceptnet_params, perceptnet_state)

params["perceptnet"] = perceptnet_params
batch_stats["batch_stats"]["perceptnet"] = perceptnet_state["batch_stats"]
batch_stats["precalc_filter"]["perceptnet"] = perceptnet_state["precalc_filter"]

# Put initialized parameters and state back into train state
state = state.replace(params=freeze(params), state=freeze(batch_stats))

# Initial parameter clipping for inner model
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# -----------------------------------------------------------------------------
# Trainable Tree and Optimizer Setup
# -----------------------------------------------------------------------------
def is_param_trainable(path):
    path_str = "/".join(path)
    if hasattr(config, "FREEZE_PATTERNS") and config.FREEZE_PATTERNS:
        return not any(pattern in path_str for pattern in config.FREEZE_PATTERNS)
    return True

trainable_tree = freeze(
    flax.traverse_util.path_aware_map(
        lambda path, v: "trainable" if is_param_trainable(path) else "non_trainable",
        state.params,
    )
)

print("Trainable parameters tree:")
for k, v in flax.traverse_util.flatten_dict(trainable_tree).items():
    print(f"  {'/'.join(k)}: {v}")

optimizers = {
    "trainable": optax.adam(learning_rate=config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}
tx_multi = optax.multi_transform(optimizers, trainable_tree)

# Update state with multi-transform optimizer and its initialized state
state = state.replace(tx=tx_multi, opt_state=tx_multi.init(state.params))

# Calculate parameter counts
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum(
    [
        w.size if t == "trainable" else 0
        for w, t in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(trainable_tree),
        )
    ]
)
print(f"Total parameters: {param_count}, Trainable parameters: {trainable_param_count}")

# -----------------------------------------------------------------------------
# Denoising Training and Evaluation steps
# -----------------------------------------------------------------------------
@jax.jit
def train_step_denoise(state, clean_images, key):
    # Generate noisy inputs dynamically on GPU
    noise_key, next_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, shape=clean_images.shape) * config.NOISE_STD
    noisy_images = jnp.clip(clean_images + noise, 0.0, 1.0)
    
    def loss_fn(params):
        reconstructed, updated_state = state.apply_fn(
            {"params": params, **state.state},
            noisy_images,
            mutable=list(state.state.keys()),
            train=True,
        )
        loss = jnp.mean((reconstructed - clean_images) ** 2)
        return loss, updated_state

    (loss, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics, state=updated_state)
    return state, next_key

@jax.jit
def eval_step_denoise(state, clean_images, key):
    # Generate noisy inputs dynamically
    noise = jax.random.normal(key, shape=clean_images.shape) * config.NOISE_STD
    noisy_images = jnp.clip(clean_images + noise, 0.0, 1.0)
    
    reconstructed = state.apply_fn(
        {"params": state.params, **state.state},
        noisy_images,
        train=False,
    )
    loss = jnp.mean((reconstructed - clean_images) ** 2)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

# -----------------------------------------------------------------------------
# Checkpoint setup
# -----------------------------------------------------------------------------
checkpoint_dir = os.path.abspath("./checkpoints_denoise")
os.makedirs(checkpoint_dir, exist_ok=True)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-0"), state, save_args=orbax_utils.save_args_from_target(state), force=True
)

metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
rng_key = jax.random.PRNGKey(config.SEED)
print("Starting denoising training loop...")
for epoch in range(config.EPOCHS):
    # Split key for epoch-specific random noise generation
    epoch_key, rng_key = jax.random.split(rng_key)
    
    ## Training
    for batch in get_epoch_iterator(train_ds, config.BATCH_SIZE, shuffle=True, seed=config.SEED + epoch):
        # We need a different key for each step noise generation
        step_key, epoch_key = jax.random.split(epoch_key)
        state, epoch_key = train_step_denoise(state, batch, step_key)
        
        # Apply parameter clipping to feature extractor if it is trainable
        if not hasattr(config, "FREEZE_PATTERNS") or "perceptnet" not in config.FREEZE_PATTERNS:
            state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
            state = state.replace(params=clip_param(state.params, "A", a_min=0))
            state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))

    ## Compute train metrics
    computed_train_metrics = state.metrics.compute()
    metrics_history["train_loss"].append(computed_train_metrics["loss"])
    state = state.replace(metrics=DenoiseMetrics.empty())

    ## Evaluation
    for batch in get_epoch_iterator(val_ds, config.BATCH_SIZE, shuffle=False):
        val_key, epoch_key = jax.random.split(epoch_key)
        state = eval_step_denoise(state, batch, val_key)

    ## Compute validation metrics
    computed_val_metrics = state.metrics.compute()
    metrics_history["val_loss"].append(computed_val_metrics["loss"])
    state = state.replace(metrics=DenoiseMetrics.empty())

    ## Checkpointing (best validation loss)
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        save_args = orbax_utils.save_args_from_target(state)
        orbax_checkpointer.save(
            os.path.join(checkpoint_dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )

    print(
        f'Epoch {epoch + 1}/{config.EPOCHS} -> '
        f'[Train] MSE Loss: {metrics_history["train_loss"][-1]:.6f} | '
        f'[Val] MSE Loss: {metrics_history["val_loss"][-1]:.6f}'
    )

# Save final model
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-final"), state, save_args=save_args, force=True
)
print("Denoising training completed successfully! Checkpoints saved to:", checkpoint_dir)
