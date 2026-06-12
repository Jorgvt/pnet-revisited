import os
import sys
import jax
from jax import random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax.training import orbax_utils
import optax
import orbax.checkpoint
import wandb
import numpy as np
from ml_collections import config_flags, ConfigDict

# Local imports
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from paramperceptnet.constraints import clip_layer, clip_param
from paramperceptnet.training import create_train_state, train_step, compute_metrics

# Config flags
config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.py"))
_CONFIG = config_flags.DEFINE_config_file("config", default=config_file_path)
from absl import flags
flags.FLAGS(sys.argv)
config = _CONFIG.value
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
print(trainable_tree)

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
# Weights & Biases Initialization
# -----------------------------------------------------------------------------
wandb.init(
    project="PerceptNet_v15",
    name="FinalModel_Freeze_CS_GoodInit",
    job_type="training",
    config=config,
    mode="online",
)
wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

# -----------------------------------------------------------------------------
# Training and Evaluation functions
# -----------------------------------------------------------------------------
@jax.jit
def forward_intermediates(state, inputs):
    return state.apply_fn(
        {"params": state.params, **state.state},
        inputs,
        train=False,
        capture_intermediates=True,
    )

def flatten_params(tree):
    return flax.traverse_util.flatten_dict(tree, sep="/")

def filter_extra(extra):
    def filter_intermediates(path, x):
        path = "/".join(path)
        if "Gabor" in path:
            return (x[0][0],)
        else:
            return x

    extra = unfreeze(extra)
    extra["intermediates"] = flax.traverse_util.path_aware_map(
        filter_intermediates, extra["intermediates"]
    )
    return freeze(extra)

# Setup checkpointer
checkpoint_dir = os.path.abspath(wandb.run.dir)
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
        
        wandb.log(
            {f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()},
            commit=False,
        )
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

    ## Obtain activations of last validation batch
    _, extra = forward_intermediates(state, batch[0])
    extra = filter_extra(extra)

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        save_args = orbax_utils.save_args_from_target(state)
        orbax_checkpointer.save(
            os.path.join(checkpoint_dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )

    # Log histograms and metrics
    wandb.log(
        {f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()},
        commit=False,
    )
    wandb.log(
        {
            f"{k}": wandb.Histogram(v)
            for k, v in flatten_params(extra["intermediates"]).items()
        },
        commit=False,
    )
    
    current_lr = config.LEARNING_RATE if hasattr(config, "LEARNING_RATE") else schedule_lr(step)
    wandb.log(
        {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            **{name: values[-1] for name, values in metrics_history.items()},
        }
    )
    print(
        f'Epoch {epoch + 1} -> [Train] Loss: {metrics_history["train_loss"][-1]:.4f} [Val] Loss: {metrics_history["val_loss"][-1]:.4f}'
    )

# Save final model
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-final"), state, save_args=save_args, force=True
)
wandb.finish()
