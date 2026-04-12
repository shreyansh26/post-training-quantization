from __future__ import annotations

from ptq.config import (
    ArtifactQuantizationSettings,
    PTQRunConfig,
    QuantizationDType,
    QuantizationGranularity,
    QuantizationMethod,
    RuntimeBackend,
)


def _validate_enabled_artifact(
    name: str,
    artifact: ArtifactQuantizationSettings,
) -> None:
    if not artifact.enabled:
        return

    if name == "weights":
        if artifact.granularity not in {
            QuantizationGranularity.TENSOR,
            QuantizationGranularity.CHANNEL,
            QuantizationGranularity.GROUP,
            QuantizationGranularity.BLOCK,
        }:
            raise ValueError(
                "weight quantization only supports "
                "tensor/channel/group/block granularity"
            )
        if (
            artifact.granularity is QuantizationGranularity.BLOCK
            and artifact.dtype is not QuantizationDType.FP8
        ):
            raise ValueError("block-level weight quantization is restricted to fp8")

    if name == "activations":
        if artifact.granularity not in {
            QuantizationGranularity.TENSOR,
            QuantizationGranularity.GROUP,
            QuantizationGranularity.TOKEN,
        }:
            raise ValueError(
                "activation quantization only supports tensor/group/token granularity"
            )
        if (
            artifact.granularity is QuantizationGranularity.TOKEN
            and artifact.block_size is not None
        ):
            raise ValueError("token granularity cannot set block_size")

    if name in {"attention", "kv_cache"}:
        if artifact.granularity not in {
            QuantizationGranularity.TENSOR,
            QuantizationGranularity.TOKEN,
        }:
            raise ValueError(
                f"{name} quantization currently supports tensor/token granularity"
            )


def validate_supported_config(config: PTQRunConfig) -> None:
    if not config.runtime.single_gpu_only:
        raise ValueError("only single-GPU execution is supported")
    if config.runtime.backend is not RuntimeBackend.VLLM:
        raise ValueError("vLLM is the only supported runtime backend")

    for name in ("weights", "activations", "attention", "kv_cache"):
        _validate_enabled_artifact(name, getattr(config.artifacts, name))

    if config.method.name is QuantizationMethod.BASELINE:
        return

    if config.artifacts.kv_cache.enabled and not config.artifacts.attention.enabled:
        raise ValueError(
            "kv_cache quantization requires attention quantization enabled"
        )

    if config.method.name is QuantizationMethod.RTN:
        if not config.artifacts.weights.enabled:
            raise ValueError("rtn requires weight quantization")
        if config.artifacts.activations.enabled:
            raise ValueError("rtn is weight-only; use dynamic/static for activations")
        if config.calibration.num_samples <= 0:
            return

    if config.method.name in {QuantizationMethod.DYNAMIC, QuantizationMethod.STATIC}:
        if not config.artifacts.weights.enabled:
            raise ValueError("dynamic/static quantization requires weight quantization")

    if config.method.name in {
        QuantizationMethod.SMOOTHQUANT,
        QuantizationMethod.AWQ,
        QuantizationMethod.GPTQ,
    }:
        if config.calibration.num_samples <= 0:
            raise ValueError(f"{config.method.name} requires calibration data")
        if not config.artifacts.weights.enabled:
            raise ValueError(f"{config.method.name} requires weights enabled")

    if (
        config.method.name is QuantizationMethod.STATIC
        and config.artifacts.activations.enabled
        and config.calibration.num_samples <= 0
    ):
        raise ValueError("static activation quantization requires calibration samples")

    if config.artifacts.activations.enabled:
        dynamic_mode = config.method.name is QuantizationMethod.DYNAMIC
        static_mode = config.method.name in {
            QuantizationMethod.STATIC,
            QuantizationMethod.SMOOTHQUANT,
            QuantizationMethod.AWQ,
            QuantizationMethod.GPTQ,
        }
        if (
            dynamic_mode
            and config.artifacts.activations.granularity
            is QuantizationGranularity.TENSOR
        ):
            raise ValueError(
                "dynamic activation quantization should use token granularity "
                "for vLLM-aligned behavior"
            )
        if (
            static_mode
            and config.artifacts.activations.granularity
            is QuantizationGranularity.TOKEN
        ):
            raise ValueError(
                "static activation quantization does not support token granularity"
            )
        if (
            static_mode
            and config.artifacts.activations.granularity
            is QuantizationGranularity.GROUP
        ):
            raise ValueError(
                "static activation quantization does not support group granularity"
            )

    if config.artifacts.weights.granularity is QuantizationGranularity.BLOCK:
        if not config.artifacts.activations.enabled:
            return
        if config.method.name is not QuantizationMethod.DYNAMIC:
            raise ValueError(
                "block-weight w8a8 is only supported with dynamic activations"
            )
        if (
            config.artifacts.activations.granularity
            is not QuantizationGranularity.GROUP
        ):
            raise ValueError(
                "block-weight w8a8 requires grouped dynamic activations"
            )
        if (
            config.artifacts.activations.group_size
            != config.artifacts.weights.block_size
        ):
            raise ValueError(
                "block-weight w8a8 requires activation group_size to match "
                "weight block_size"
            )
