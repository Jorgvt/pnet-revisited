from ml_collections import ConfigDict

def get_config():
    config = ConfigDict({
        "BATCH_SIZE": 64,
        "EPOCHS": 500,
        "SEED": 42,
        "LEARNING_RATE": 0.03,
        # "INITIAL_LR": 0.01,
        # "PEAK_LR": 0.04,
        # "END_LR": 0.005,
        # "WARMUP_EPOCHS": 15,
        # Default freeze patterns for IQA: freeze everything except the final GDN Control layer
        "FREEZE_PATTERNS": [],
    })
    return config
