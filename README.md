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
To start training using the default configuration (which automatically loads from its corresponding `config.py` file):

```bash
uv run Training/IQA/training.py
```

### 2. Training with Custom Configurations and CLI Overrides
You can override specific parameters directly from the command line:

```bash
uv run Training/IQA/training.py --config.EPOCHS=100 --config.BATCH_SIZE=32
```

Or pass a custom configuration file path to load completely different settings:

```bash
uv run Training/IQA/training.py --config=/path/to/custom_config.py
```

### 3. Training Locally Without Weights & Biases (Simple Mode)
If you prefer a lightweight, self-contained training execution that runs locally and does not require `wandb` logging, you can run the simplified training script instead:

```bash
uv run Training/IQA/training_simple.py
```

Or with custom config overrides:

```bash
uv run Training/IQA/training_simple.py --config.EPOCHS=5
```

All checkpoints (`model-0`, `model-best`, and `model-final`) will be saved locally inside a `./checkpoints/` directory.

### 4. Training on Image Classification (Imagenette Task)
To test training the model on a classification task using `ModelCls` (attaching a dense layer after our model acting as a classifier) on the Imagenette dataset:

```bash
uv run Training/Classification/training_classification.py
```

By default, the script freezes the feature extractor and only trains the dense classifier head using Global Average Pooling. You can customize these behaviors (such as fine-tuning the full model or changing configuration) by passing overrides directly:

```bash
uv run Training/Classification/training_classification.py --config.FREEZE_PATTERNS="[]" --config.LEARNING_RATE=1e-4
```

Checkpoints for the classification task (`model-0`, `model-best`, and `model-final`) will be saved locally inside a `./checkpoints_cls/` directory.

### 5. Evaluating Visturing Properties
The project includes a suite for evaluating model predictions against human visual psychophysical properties (e.g., spectral sensitivities, noise masking, contrast thresholds). These scripts are located in `Evaluate/visturing/`.

To evaluate all properties together and generate an evaluation summary table:

```bash
uv run Evaluate/visturing/all.py
```

To run individual property evaluations (e.g., spectral sensitivity):

```bash
uv run Evaluate/visturing/prop1.py
```

### 6. Optimizing Model parameters on Visturing Properties
You can also optimize/fine-tune the model parameters directly to align model representations with ground truth human visual behavior on specific properties. These scripts are located in `Training/visturing/`.

To optimize parameters on a specific property (e.g., Spectral Sensitivity):

```bash
uv run Training/visturing/optimize_prop1.py --iterations=10
```

All trained parameter checkpoints from these optimization tasks are saved locally within the `Training/visturing/` folder (e.g., `model_pnet_init_prop1.pkl`).

### 7. Training on Image Denoising (Unsupervised Task)
To test training the model on an unsupervised denoising task using `ModelDenoising` (an encoder-decoder architecture wrapping the perceptnet model as the encoder and a custom convolutional decoder) on the Imagenette dataset:

```bash
uv run Training/Denoising/training_denoising.py
```

By default, the script freezes the encoder submodule and only trains the simple decoder. You can customize the training behaviour (such as the noise level or training the full model) by providing configuration overrides directly:

```bash
uv run Training/Denoising/training_denoising.py --config.NOISE_STD=0.15 --config.FREEZE_PATTERNS="[]"
```

Checkpoints for the denoising task (`model-0`, `model-best`, and `model-final`) will be saved locally inside a `./checkpoints_denoise/` directory. Note that all checkpoints are saved using resolved absolute paths to satisfy Orbax requirement constraints.

### 8. Analyzing Optimized Models
The project includes tools for analyzing model parameters before and after optimization to visualize how visual filters adapt to fit visual properties. These scripts are located in the `Analysis/` folder.

To compare parameters and plot Center-Surround and Gabor filters for a model optimized on Property 3 & 4:

```bash
uv run Analysis/analyze_prop3_4.py
```

This script generates comparative parameter tables in the terminal and saves 2D receptive field comparison plots (`cs_comparison.png` and `gabor_comparison.png`), as well as Contrast Sensitivity Function curve comparison plots (`csf_comparison.png`) directly inside the `Analysis/` directory.

### 9. Recalculating GDN Initialization Parameters (`a_star_gdn` constants)
When changing the model initialization, the `a_star_gdn_cs` and `a_star_gdn_v1` constants (stored in `src/pnet_revisited/`) must be recalculated. 

To run the recalculation utility, you can use the registered command-line script entrypoint:

```bash
uv run recalculate-astar
```

Options:
- `--num_images`: Number of images to stream from mini-imagenet (default: 64)
- `--img_size`: Resolution of processed images (default: 256)
- `--quantile`: Quantile of activation magnitudes to calculate (default: 0.90)

The script automatically handles the sequential dependency:
1. Recalculates `a_star_gdn_cs` first and saves it to `src/pnet_revisited/a_star_gdn_cs.npy`.
2. Re-initializes the model, applying the newly calculated `a_star_gdn_cs` to the Center-Surround Divisive Normalization layer.
3. Evaluates the model up to the Gabor layer to calculate and save the updated `a_star_gdn_v1` to `src/pnet_revisited/a_star_gdn_v1.npy`.
