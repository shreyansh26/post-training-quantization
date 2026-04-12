import pytest

from ptq.compatibility import validate_supported_config
from ptq.config import PTQRunConfig


def _base_config() -> dict:
    return {
        "model": {
            "model_id": "Qwen/Qwen3-4B",
            "trust_remote_code": False,
            "chat_template_source": "huggingface",
        },
        "runtime": {
            "backend": "vllm",
            "gpu_id": 0,
            "enforce_eager": True,
        },
        "method": {"name": "dynamic"},
        "artifacts": {
            "weights": {
                "enabled": True,
                "dtype": "int8",
                "granularity": "channel",
                "symmetric": True,
            },
            "activations": {
                "enabled": True,
                "dtype": "int8",
                "granularity": "token",
                "symmetric": True,
            },
            "attention": {
                "enabled": False,
                "dtype": "none",
                "granularity": "none",
                "symmetric": True,
            },
            "kv_cache": {
                "enabled": False,
                "dtype": "none",
                "granularity": "none",
                "symmetric": True,
            },
        },
        "calibration": {
            "dataset": "HuggingFaceH4/ultrachat_200k",
            "split": "train_sft",
            "num_samples": 256,
            "max_sequence_length": 512,
            "seed": 42,
            "shuffle": False,
        },
        "export": {
            "output_root": "/mnt/ssd2/shreyansh/ptq_experiments/artifacts",
            "format": "compressed-tensors",
            "require_vllm_compatibility": True,
            "prefer_hf_compatibility": True,
        },
        "evaluation": {
            "mode": "dev",
            "tasks": ["gsm8k", "ifeval", "mmlu"],
            "use_vllm": True,
            "cache_baseline": True,
        },
        "logging": {
            "metrics_dir": ".",
            "sanity_prompts_file": "prompts/sanity_prompts.txt",
        },
    }


def test_dynamic_w8a8_config_is_supported() -> None:
    config = PTQRunConfig.model_validate(_base_config())
    validate_supported_config(config)


def test_attention_quantization_is_supported_with_valid_granularity() -> None:
    raw = _base_config()
    raw["artifacts"]["attention"] = {
        "enabled": True,
        "dtype": "fp8",
        "granularity": "token",
        "symmetric": True,
    }
    config = PTQRunConfig.model_validate(raw)
    validate_supported_config(config)


def test_kv_cache_requires_attention_enabled() -> None:
    raw = _base_config()
    raw["artifacts"]["kv_cache"] = {
        "enabled": True,
        "dtype": "fp8",
        "granularity": "token",
        "symmetric": True,
    }
    config = PTQRunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="kv_cache quantization requires attention"):
        validate_supported_config(config)


def test_static_activation_quantization_requires_calibration() -> None:
    raw = _base_config()
    raw["method"]["name"] = "static"
    raw["artifacts"]["activations"]["granularity"] = "tensor"
    raw["calibration"]["num_samples"] = 0
    with pytest.raises(ValueError, match="requires calibration samples"):
        PTQRunConfig.model_validate(raw)


def test_calibration_metadata_is_none_for_dynamic() -> None:
    config = PTQRunConfig.model_validate(_base_config())
    dataset, samples = config.calibration_metadata()
    assert dataset == "none"
    assert samples == 0


def test_calibration_metadata_is_used_for_static_activation() -> None:
    raw = _base_config()
    raw["method"]["name"] = "static"
    raw["artifacts"]["activations"]["granularity"] = "tensor"
    config = PTQRunConfig.model_validate(raw)
    dataset, samples = config.calibration_metadata()
    assert dataset == "HuggingFaceH4/ultrachat_200k"
    assert samples == 256


def test_fp8_block_w8a8_config_is_supported() -> None:
    raw = _base_config()
    raw["artifacts"]["weights"]["dtype"] = "fp8"
    raw["artifacts"]["weights"]["granularity"] = "block"
    raw["artifacts"]["weights"]["block_size"] = 128
    raw["artifacts"]["activations"]["dtype"] = "fp8"
    raw["artifacts"]["activations"]["granularity"] = "group"
    raw["artifacts"]["activations"]["group_size"] = 128
    config = PTQRunConfig.model_validate(raw)
    validate_supported_config(config)


def test_block_w8a8_requires_grouped_dynamic_activations() -> None:
    raw = _base_config()
    raw["artifacts"]["weights"]["dtype"] = "fp8"
    raw["artifacts"]["weights"]["granularity"] = "block"
    raw["artifacts"]["weights"]["block_size"] = 128
    raw["artifacts"]["activations"]["dtype"] = "fp8"
    raw["artifacts"]["activations"]["granularity"] = "token"
    config = PTQRunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="requires grouped dynamic activations"):
        validate_supported_config(config)


def test_static_activation_group_granularity_is_rejected() -> None:
    raw = _base_config()
    raw["method"]["name"] = "static"
    raw["artifacts"]["activations"]["granularity"] = "group"
    raw["artifacts"]["activations"]["group_size"] = 128
    config = PTQRunConfig.model_validate(raw)
    with pytest.raises(
        ValueError,
        match="static activation quantization does not support group granularity",
    ):
        validate_supported_config(config)


def test_smoothquant_requires_activation_quantization() -> None:
    raw = _base_config()
    raw["method"]["name"] = "smoothquant"
    raw["artifacts"]["activations"] = {
        "enabled": False,
        "dtype": "none",
        "granularity": "none",
        "symmetric": True,
    }
    config = PTQRunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="smoothquant requires activation"):
        validate_supported_config(config)
