from ml_collections import ConfigDict

def get_config():
    config = ConfigDict({
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
        "LEARNING_RATE": 1e-3,
        "SEED": 42,
        
        # Noise settings
        "NOISE_STD": 0.1, # Standard deviation of synthetic Gaussian noise
        
        # Freezing configuration: list of path substrings to freeze
        # By default, freeze the entire feature extractor ("perceptnet") and train only the decoder
        "FREEZE_PATTERNS": ["perceptnet"],
    })
    return config
