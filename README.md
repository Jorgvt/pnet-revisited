# PerceptNet Revisited (pnet-revisited)

This project contains an adapted, simplified training and evaluation pipeline for the visual system model defined in `src/pnet_revisited/model.py` and initialized in `src/pnet_revisited/initialization.py`.

## Core Features

- **Hugging Face Datasets Integration:** The pipeline loads the TID2008 and TID2013 datasets directly from Hugging Face (`Jorgvt/TID2008` and `Jorgvt/TID2013`).
- **Efficient Data Streaming:** Rather than loading entire datasets into memory, the pipeline leverages the Hugging Face `datasets` API:
  - Uses `with_transform` for efficient, on-the-fly PIL image decoding and normalization (scaling pixels to `[0.0, 1.0]`).
  - Employs `Dataset.iter(batch_size=...)` to stream batches dynamically with minimal memory footprint.
  - Automatically filters out excluded images (such as reference image `25`).
- **JAX/Flax Optimization:** Built on top of `JAX` and `Flax`, supporting custom multi-optimizer transformations to freeze specific parameter groups (e.g. Center-Surround layers, Gabor filters, GDN parameters) according to config requirements.
- **W&B Integration:** Integrates with Weights & Biases for real-time tracking of training/validation loss, parameter distributions, gradients, and model checkpoints.

## Installation

This project is managed with [uv](https://github.com/astral-sh/uv). To install the dependencies, simply run:

```bash
uv sync
```

This will automatically configure a virtual environment with the necessary JAX, Flax, Optax, HF Datasets, and custom dependencies (such as the `perceptualtests` package).

## Usage

### 1. Training with Default Configuration
To start training using the default `param_config` values from `paramperceptnet`:

```bash
uv run training.py
```

### 2. Training with Custom Configurations
To define custom settings (e.g. override hyperparameters or freeze/unfreeze specific layers), you can use the local `config.py` file:

```bash
uv run training.py --config=config.py
```

You can also override parameters directly from the command line:

```bash
uv run training.py --config=config.py --config.EPOCHS=100 --config.BATCH_SIZE=32
```

### 3. Training Locally Without Weights & Biases (Simple Mode)
If you prefer a lightweight, self-contained training execution that runs locally and does not require `wandb` logging, you can run the simplified training script instead:

```bash
uv run training_simple.py
```

Or with custom config overrides:

```bash
uv run training_simple.py --config=config.py --config.EPOCHS=5
```

All checkpoints (`model-0`, `model-best`, and `model-final`) will be saved locally inside a `./checkpoints/` directory.

### 4. Training on Image Classification (Imagenette Task)
To test training the model on a classification task using `ModelCls` (attaching a dense layer after our model acting as a classifier) on the Imagenette dataset:

```bash
uv run training_classification.py
```

By default, the script freezes the feature extractor and only trains the dense classifier head using Global Average Pooling. You can customize these behaviors (such as fine-tuning the full model or changing configuration) by passing custom flags or overrides:

```bash
uv run training_classification.py --config=config.py --config.FREEZE_FEATURE_EXTRACTOR=False --config.LEARNING_RATE=1e-4
```

Checkpoints for the classification task (`model-0`, `model-best`, and `model-final`) will be saved locally inside a `./checkpoints_cls/` directory.
