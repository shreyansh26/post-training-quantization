from ptq.compatibility import validate_supported_config
from ptq.config import PTQRunConfig, load_run_config


def quantize_from_config(*args, **kwargs):
    from ptq.pipeline import quantize_from_config as _quantize_from_config

    return _quantize_from_config(*args, **kwargs)


def evaluate_model_ref(*args, **kwargs):
    from ptq.pipeline import evaluate_model_ref as _evaluate_model_ref

    return _evaluate_model_ref(*args, **kwargs)

__all__ = [
    "PTQRunConfig",
    "load_run_config",
    "quantize_from_config",
    "evaluate_model_ref",
    "validate_supported_config",
]
