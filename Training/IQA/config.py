from ml_collections import ConfigDict

def get_config():
    config = ConfigDict({
        "BATCH_SIZE": 64,
        "EPOCHS": 500,
        "SEED": 42,
        "LEARNING_RATE": 0.003,
        "INITIAL_LR": 0.01,
        "PEAK_LR": 0.04,
        "END_LR": 0.005,
        "WARMUP_EPOCHS": 15,
        # Default freeze patterns for IQA: freeze everything except the final GDN Control layer
        "FREEZE_PATTERNS": [
            "GDNGamma_0",
            "CenterSurroundLogSigmaK_0",
            "DN_0",
            "GaborLayerGammaHumanLike__0",
        ],
    })
    return config
