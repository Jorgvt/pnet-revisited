from paramperceptnet.configs import param_config

def get_config():
    config = param_config
    # Default freeze patterns for IQA: freeze everything except the final GDN Control layer
    config.FREEZE_PATTERNS = [
        "GDNGamma_0",
        "CenterSurroundLogSigmaK_0",
        "DN_0",
        "GaborLayerGammaHumanLike__0",
    ]
    return config
