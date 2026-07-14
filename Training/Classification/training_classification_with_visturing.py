import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import jax
from jax import random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax.training import orbax_utils
from flax import struct
import optax
import orbax.checkpoint
import wandb
import numpy as np
from ml_collections import config_flags, ConfigDict
from clu import metrics
from functools import partial

# Local imports
from pnet_revisited.model import Model, ModelCls
from pnet_revisited.initialization import init_model
from paramperceptnet.constraints import clip_layer, clip_param
from paramperceptnet.training import TrainState

# Visturing imports
from visturing.properties.utils import evaluate_all_gen, build_evaluation_table_gen

# -----------------------------------------------------------------------------
# Configuration Setup
# -----------------------------------------------------------------------------
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

print("Loading dataset jonathanli/imagenette2-320-split from Hugging Face...")
train_ds = load_dataset("jonathanli/imagenette2-320-split", split="train")
val_ds = load_dataset("jonathanli/imagenette2-320-split", split="test")

# Preprocessing transform
def preprocess(batch):
    target_size = (224, 224)
    # Convert all images to RGB, resize, and normalize to [0.0, 1.0]
    images = np.array([
        np.array(img.convert("RGB").resize(target_size), dtype=np.float32) / 255.0 
        for img in batch['image']
    ])
    labels = np.array(batch['label'], dtype=np.int32)
    return {
        'image': images,
        'label': labels
    }

train_ds = train_ds.with_transform(preprocess)
val_ds = val_ds.with_transform(preprocess)

def get_epoch_iterator(dataset, batch_size, shuffle=False, seed=None):
    ds = dataset
    if shuffle:
        ds = ds.shuffle(seed=seed)
    for batch in ds.iter(batch_size=batch_size, drop_last_batch=True):
        yield batch['image'], batch['label']

# -----------------------------------------------------------------------------
# Metrics and Custom Train State definition
# -----------------------------------------------------------------------------
@struct.dataclass
class ClassificationMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")

class ClsTrainState(TrainState):
    metrics: ClassificationMetrics

def create_cls_train_state(module, key, tx, input_shape):
    variables = module.init(key, jnp.ones(input_shape))
    state, params = flax.core.pop(variables, "params")
    return ClsTrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=ClassificationMetrics.empty(),
    )

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
tx = optax.adam(config.LEARNING_RATE)

# Create Outer ModelCls train state
state = create_cls_train_state(
    ModelCls(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 224, 224, 3)
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
# Weights & Biases Initialization
# -----------------------------------------------------------------------------
wandb.init(
    project="PerceptNet_v2",
    name="Classification",
    job_type="training-cls",
    config=config,
    mode="online",
)
wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

# -----------------------------------------------------------------------------
# Classification Training and Evaluation steps
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnums=2)
def train_step_cls(state, batch, return_grads=False):
    images, labels = batch
    
    def loss_fn(params):
        logits, updated_state = state.apply_fn(
            {"params": params, **state.state},
            images,
            mutable=list(state.state.keys()),
            train=True,
            update_stats=True,
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        return loss, (updated_state, acc)

    (loss, (updated_state, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss, accuracy=acc)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics, state=updated_state)
    if return_grads:
        return state, grads
    else:
        return state

@jax.jit
def eval_step_cls(state, batch):
    images, labels = batch
    
    logits = state.apply_fn(
        {"params": state.params, **state.state},
        images,
        train=False,
    )
    one_hot = jax.nn.one_hot(labels, 10)
    loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == labels)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss, accuracy=acc)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

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

# -----------------------------------------------------------------------------
# Checkpoint setup
# -----------------------------------------------------------------------------
checkpoint_dir = os.path.abspath(wandb.run.dir)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-0"), state, save_args=save_args, force=True
)

metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

# -----------------------------------------------------------------------------
# JIT-compiled Visturing evaluation helper
# -----------------------------------------------------------------------------
@jax.jit
def calculate_diffs_jit(params_val, state_val, a, b):
    perceptnet_variables = {
        "params": params_val["perceptnet"],
        "batch_stats": state_val["batch_stats"]["perceptnet"],
        "precalc_filter": state_val["precalc_filter"]["perceptnet"],
    }
    feat_a = Model().apply(perceptnet_variables, a, train=False)
    feat_b = Model().apply(perceptnet_variables, b, train=False)
    return jnp.sqrt(jnp.mean((feat_a - feat_b) ** 2, axis=(-3, -2, -1)) + 1e-8)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
print("Starting classification training loop with Visturing...")
step = 0
for epoch in range(config.EPOCHS):
    ## Training
    for batch in get_epoch_iterator(train_ds, config.BATCH_SIZE, shuffle=True, seed=config.SEED + epoch):
        state, grads = train_step_cls(state, batch, return_grads=True)
        # Apply parameter clipping to feature extractor if it is trainable
        if not hasattr(config, "FREEZE_PATTERNS") or "perceptnet" not in config.FREEZE_PATTERNS:
            state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
            state = state.replace(params=clip_param(state.params, "A", a_min=0))
            state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))

        wandb.log(
            {f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()},
            commit=False,
        )
        step += 1

    ## Compute train metrics
    computed_train_metrics = state.metrics.compute()
    metrics_history["train_loss"].append(computed_train_metrics["loss"])
    metrics_history["train_accuracy"].append(computed_train_metrics["accuracy"])
    state = state.replace(metrics=ClassificationMetrics.empty())

    ## Evaluation
    for batch in get_epoch_iterator(val_ds, config.BATCH_SIZE, shuffle=False):
        state = eval_step_cls(state, batch)

    ## Compute validation metrics
    computed_val_metrics = state.metrics.compute()
    metrics_history["val_loss"].append(computed_val_metrics["loss"])
    metrics_history["val_accuracy"].append(computed_val_metrics["accuracy"])
    state = state.replace(metrics=ClassificationMetrics.empty())

    ## Obtain activations of last validation batch
    _, extra = forward_intermediates(state, batch[0])
    extra = filter_extra(extra)

    ## Run Visturing benchmark evaluation
    print(f"\nEvaluating on Visturing benchmark after epoch {epoch + 1}...")
    def calculate_diffs(a, b):
        a_j = jnp.asarray(a)
        b_j = jnp.asarray(b)
        return np.asarray(calculate_diffs_jit(state.params, state.state, a_j, b_j))

    # Evaluate on all visturing properties using batch_size=128 to prevent OOM
    results_visturing = evaluate_all_gen(
        calculate_diffs, 
        batch_size=128, 
        data_path=".", 
        gt_path=".", 
        verbose=False
    )
    table_visturing = build_evaluation_table_gen(results_visturing)
    print("Visturing Evaluation Table:")
    print(table_visturing.to_string(index=False))

    # Prepare Visturing metrics logs for W&B
    visturing_logs = {}
    for _, row in table_visturing.iterrows():
        prop_label = row["Property"].replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        # Log Global, Achromatic, Red-Green, and Yellow-Blue correlations
        for col in ["Global (r)", "Achromatic (r)", "Red-Green (r)", "Yellow-Blue (r)"]:
            val = row[col]
            if val != "-":
                try:
                    visturing_logs[f"visturing/{prop_label}/{col.split(' ')[0]}"] = float(val)
                except ValueError:
                    pass

    # Compute mean of individual correlations for both weighted and non-weighted properties
    non_weighted_corrs = []
    weighted_corrs = []
    for p in ["prop1", "prop2", "prop3_4", "prop5", "prop6_7", "prop8", "prop9", "prop10"]:
        prop_data = results_visturing.get(p)
        if prop_data is not None and hasattr(prop_data, "correlations"):
            corrs = prop_data.correlations
            # Extract non-weighted correlation
            if "non-weighted" in corrs:
                c_nw = corrs["non-weighted"].get("global")
                if c_nw is not None and not np.isnan(c_nw):
                    non_weighted_corrs.append(float(c_nw))
            elif "global" in corrs:
                c_nw = corrs.get("global")
                if c_nw is not None and not np.isnan(c_nw):
                    non_weighted_corrs.append(float(c_nw))
            
            # Extract weighted correlation
            if "weighted" in corrs:
                c_w = corrs["weighted"].get("global")
                if c_w is not None and not np.isnan(c_w):
                    weighted_corrs.append(float(c_w))
            elif "global" in corrs:
                c_w = corrs.get("global")
                if c_w is not None and not np.isnan(c_w):
                    weighted_corrs.append(float(c_w))

    if non_weighted_corrs:
        visturing_logs["visturing/mean_correlation/non_weighted"] = np.mean(non_weighted_corrs)
    if weighted_corrs:
        visturing_logs["visturing/mean_correlation/weighted"] = np.mean(weighted_corrs)

    ## Checkpointing (best validation accuracy)
    if metrics_history["val_accuracy"][-1] >= max(metrics_history["val_accuracy"]):
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
    
    current_lr = config.LEARNING_RATE
    wandb.log(
        {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            **{name: values[-1] for name, values in metrics_history.items()},
            **visturing_logs,
        }
    )
    print(
        f'Epoch {epoch + 1}/{config.EPOCHS} -> '
        f'[Train] Loss: {metrics_history["train_loss"][-1]:.4f}, Acc: {metrics_history["train_accuracy"][-1]*100:.2f}% | '
        f'[Val] Loss: {metrics_history["val_loss"][-1]:.4f}, Acc: {metrics_history["val_accuracy"][-1]*100:.2f}%\n'
    )

# Save final model
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(checkpoint_dir, "model-final"), state, save_args=save_args, force=True
)
wandb.finish()
print("Classification training with Visturing completed successfully! Checkpoints saved to:", checkpoint_dir)
