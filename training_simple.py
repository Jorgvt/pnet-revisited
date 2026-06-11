import os
import sys
import jax
from jax import random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax.training import orbax_utils
import optax
import orbax.checkpoint
import numpy as np
from ml_collections import config_flags

# Local imports
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from paramperceptnet.constraints import clip_layer, clip_param
from paramperceptnet.training import create_train_state, train_step, compute_metrics
from paramperceptnet.configs import param_config as default_config

# Config flags
_CONFIG = config_flags.DEFINE_config_file("config", default=None)
from absl import flags
flags.FLAGS(sys.argv)
config = _CONFIG.value if _CONFIG.value is not None else default_config
print("Using configuration:")
print(config)

# -----------------------------------------------------------------------------
# Dataset Loading and Preprocessing using HF Datasets
# -----------------------------------------------------------------------------
from datasets import load_dataset

print("Loading dataset Jorgvt/TID2008 from Hugging Face...")
train_ds = load_dataset("Jorgvt/TID2008", split="train")
print("Loading dataset Jorgvt/TID2013 from Hugging Face...")
val_ds = load_dataset("Jorgvt/TID2013", split="train")

# Filter out excluded reference images once
train_ds = train_ds.filter(lambda x: x['reference_id'] != 25)
val_ds = val_ds.filter(lambda x: x['reference_id'] != 25)

# Preprocessing transform
def preprocess(batch):
    ref = np.array([np.array(img, dtype=np.float32) / 255.0 for img in batch['reference']])
    dist = np.array([np.array(img, dtype=np.float32) / 255.0 for img in batch['distorted']])
    mos = np.array(batch['mos'], dtype=np.float32)
    return {
        'reference': ref,
        'distorted': dist,
        'mos': mos
    }

train_ds = train_ds.with_transform(preprocess)
val_ds = val_ds.with_transform(preprocess)

def get_epoch_iterator(dataset, batch_size, shuffle=False, seed=None):
    ds = dataset
    if shuffle:
        ds = ds.shuffle(seed=seed)
    for batch in ds.iter(batch_size=batch_size, drop_last_batch=True):
        yield batch['reference'], batch['distorted'], batch['mos']

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
# Define learning rate schedule
if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(config.LEARNING_RATE)
else:
    steps_per_epoch = len(train_ds) // config.BATCH_SIZE
    schedule_lr = optax.warmup_cosine_decay_schedule(
        init_value=config.INITIAL_LR,
        peak_value=config.PEAK_LR,
        end_value=config.END_LR,
        warmup_steps=steps_per_epoch * config.WARMUP_EPOCHS,
        decay_steps=steps_per_epoch * config.EPOCHS,
    )
    tx = optax.adam(learning_rate=schedule_lr)

# Create train state
state = create_train_state(
    Model(), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)

# Apply custom human-like initialization
params = unfreeze(state.params)
batch_stats = unfreeze(state.state)
params, batch_stats = init_model(Model(), params, batch_stats)
state = state.replace(params=freeze(params), state=freeze(batch_stats))

# Initial parameter clipping
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# -----------------------------------------------------------------------------
# Trainable Tree and Optimizer Setup
# -----------------------------------------------------------------------------
def check_trainable(path):
    if hasattr(config, "TRAIN_ONLY_B") and config.TRAIN_ONLY_B:
        if "B" in path:
            return False
        else:
            return True
    if "GDNGamma_0" in path:
        if not config.TRAIN_GDNGAMMA:
            return True
    if "Color" in path:
        if not config.TRAIN_JH:
            return True
    if "GDN_0" in path:
        if not config.TRAIN_GDNCOLOR:
            return True
    if "CenterSurroundLogSigmaK_0" in path:
        if not config.TRAIN_CS:
            return True
    if "GDNGaussian_0" in path:
        if not config.TRAIN_GDNGAUSSIAN:
            return True
    if "Gabor" in "".join(path):
        if not config.TRAIN_GABOR:
            return True
    if not config.A_GDNSPATIOFREQORIENT:
        if ("GDNSpatioChromaFreqOrient_0" in path) and ("A" in path):
            return True
    if "GDNSpatioChromaFreqOrient_0" not in path and config.TRAIN_ONLY_LAST_GDN:
        return True
    return False

trainable_tree = freeze(
    flax.traverse_util.path_aware_map(
        lambda path, v: "non_trainable" if check_trainable(path) else "trainable",
        state.params,
    )
)

# Define optimizers for trainable/non-trainable parts
if hasattr(config, "LEARNING_RATE"):
    tx_trainable = optax.adam(learning_rate=config.LEARNING_RATE)
else:
    tx_trainable = optax.adam(learning_rate=schedule_lr)

optimizers = {
    "trainable": tx_trainable,
    "non_trainable": optax.set_to_zero(),
}
tx_multi = optax.multi_transform(optimizers, trainable_tree)

# Update state with multi-transform optimizer and its initialized state
state = state.replace(tx=tx_multi, opt_state=tx_multi.init(state.params))

# Calculate parameters count
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
# Checkpoint setup
# -----------------------------------------------------------------------------
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup checkpointer
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-0"), state, save_args=save_args, force=True
)

metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
step = 0
for epoch in range(config.EPOCHS):
    ## Training
    for batch in get_epoch_iterator(train_ds, config.BATCH_SIZE, shuffle=True, seed=config.SEED + epoch):
        state, grads = train_step(state, batch, return_grads=True)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        state = state.replace(params=clip_param(state.params, "A", a_min=0))
        state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))
        step += 1

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in get_epoch_iterator(val_ds, config.BATCH_SIZE, shuffle=False):
        state = compute_metrics(state=state, batch=batch)

    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(
            os.path.join(checkpoint_dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )

    print(
        f'Epoch {epoch + 1}/{config.EPOCHS} -> [Train] Loss: {metrics_history["train_loss"][-1]:.4f} [Val] Loss: {metrics_history["val_loss"][-1]:.4f}'
    )

# Save final model
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-final"), state, save_args=save_args, force=True
)
print("Training completed successfully! Checkpoints saved to:", checkpoint_dir)
