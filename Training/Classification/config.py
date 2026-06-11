from ml_collections import ConfigDict

def get_config():
    config = ConfigDict({
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
        "LEARNING_RATE": 1e-3,
        "SEED": 42,
        "GAP": True, # Global Average Pooling before the classification head
        
        # Freezing configuration: list of path substrings to freeze
        # By default, freeze the entire feature extractor ("perceptnet")
        "FREEZE_PATTERNS": ["perceptnet"],
    })
    return config
