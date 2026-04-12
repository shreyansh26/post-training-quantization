from ptq.compatibility import validate_supported_config
from ptq.config import PTQRunConfig, load_run_config


def run_pipeline(*args, **kwargs):
    from ptq.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)

__all__ = [
    "PTQRunConfig",
    "load_run_config",
    "run_pipeline",
    "validate_supported_config",
]
