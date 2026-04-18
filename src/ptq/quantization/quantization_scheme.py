"""Quantization scheme builders.

This module translates repo-level artifact settings into the compressed-tensors
objects that drive export. The intent is to keep artifact composition separate
from method-specific preprocessing such as SmoothQuant, AWQ, and GPTQ.
"""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.initialize import is_attention_module

from ptq.config import (
    ArtifactQuantizationSettings,
    PTQRunConfig,
    QuantizationDType,
    QuantizationGranularity,
    QuantizationMethod,
)


def dtype_to_quant_type(dtype: QuantizationDType) -> QuantizationType:
    """Map repo dtypes to compressed-tensors quantization types."""
    if dtype is QuantizationDType.INT8:
        return QuantizationType.INT
    if dtype is QuantizationDType.FP8:
        return QuantizationType.FLOAT
    raise ValueError(f"unsupported quantization dtype: {dtype}")


def weight_strategy(
    settings: ArtifactQuantizationSettings,
) -> tuple[QuantizationStrategy, list[int] | None]:
    """Translate weight granularity into compressed-tensors strategy metadata."""
    if settings.granularity is QuantizationGranularity.TENSOR:
        return QuantizationStrategy.TENSOR, None
    if settings.granularity is QuantizationGranularity.CHANNEL:
        return QuantizationStrategy.CHANNEL, None
    if settings.granularity is QuantizationGranularity.GROUP:
        return QuantizationStrategy.GROUP, None
    if settings.granularity is QuantizationGranularity.BLOCK:
        block_size = settings.block_size if settings.block_size is not None else 128
        return QuantizationStrategy.BLOCK, [block_size, block_size]
    raise ValueError(
        f"unsupported weight granularity for compressed export: {settings.granularity}"
    )


def activation_strategy(
    settings: ArtifactQuantizationSettings,
) -> QuantizationStrategy:
    """Translate activation-like granularities into export strategy enums."""
    if settings.granularity is QuantizationGranularity.TOKEN:
        return QuantizationStrategy.TOKEN
    if settings.granularity is QuantizationGranularity.GROUP:
        return QuantizationStrategy.GROUP
    if settings.granularity is QuantizationGranularity.TENSOR:
        return QuantizationStrategy.TENSOR
    raise ValueError(
        "activation/attention/kv_cache quantization only supports token/group/tensor"
    )


def build_artifact_quant_args(
    settings: ArtifactQuantizationSettings,
    *,
    dynamic: bool,
    for_weights: bool,
) -> QuantizationArgs | None:
    """Build the compressed-tensors quantization args for one artifact family."""
    if not settings.enabled:
        return None

    strategy, block_structure = (
        weight_strategy(settings)
        if for_weights
        else (activation_strategy(settings), None)
    )

    # Symmetry is intentionally config-driven for PTQ INT8 activations. FP8 also
    # flows through this flag, but in compressed-tensors "symmetric=True" means
    # a zero-centered scale for the floating-point lattice, not an affine INT8-
    # style learned nonzero zero-point.
    return QuantizationArgs(
        num_bits=8,
        type=dtype_to_quant_type(settings.dtype),
        symmetric=settings.symmetric,
        strategy=strategy,
        group_size=settings.group_size,
        block_structure=block_structure,
        dynamic=dynamic if not for_weights else False,
    )


def quantization_format_for_config(config: PTQRunConfig) -> str:
    """Infer the export format label expected by compressed-tensors."""
    if (
        config.artifacts.weights.enabled
        and not config.artifacts.activations.enabled
        and not config.artifacts.attention.enabled
        and not config.artifacts.kv_cache.enabled
        and config.artifacts.weights.dtype is QuantizationDType.INT8
    ):
        return "pack-quantized"

    enabled_dtypes = []
    for name in ("weights", "activations", "attention", "kv_cache"):
        artifact = getattr(config.artifacts, name)
        if artifact.enabled:
            enabled_dtypes.append(artifact.dtype)

    if not enabled_dtypes:
        return "dense"

    if all(dtype is QuantizationDType.FP8 for dtype in enabled_dtypes):
        return "float-quantized"
    if all(dtype is QuantizationDType.INT8 for dtype in enabled_dtypes):
        return "int-quantized"
    return "mixed-precision"


def discover_attention_targets(model: torch.nn.Module) -> list[str]:
    """Return unique attention class names to use as compressed-tensors targets."""
    targets: list[str] = []
    for module in model.modules():
        if not is_attention_module(module):
            continue
        target = module.__class__.__name__
        if target not in targets:
            targets.append(target)
    return targets


def build_quantization_config(
    config: PTQRunConfig,
    model: torch.nn.Module | None = None,
) -> QuantizationConfig:
    """Build the compressed-tensors config for the enabled artifact combination."""
    dynamic_activations = config.method.name is QuantizationMethod.DYNAMIC

    weights = build_artifact_quant_args(
        config.artifacts.weights,
        dynamic=False,
        for_weights=True,
    )
    input_activations = build_artifact_quant_args(
        config.artifacts.activations,
        dynamic=dynamic_activations,
        for_weights=False,
    )

    groups: dict[str, QuantizationScheme] = {}
    if weights is not None or input_activations is not None:
        groups["linear"] = QuantizationScheme(
            targets=["Linear"],
            weights=weights,
            input_activations=input_activations,
        )

    if config.artifacts.attention.enabled:
        if model is None:
            raise ValueError("attention quantization requires the loaded model")
        attention_args = build_artifact_quant_args(
            config.artifacts.attention,
            dynamic=dynamic_activations,
            for_weights=False,
        )
        attention_targets = discover_attention_targets(model)
        if not attention_targets:
            raise ValueError(
                "could not discover any attention modules for quantization"
            )
        for index, target in enumerate(attention_targets):
            groups[f"attention_{index}"] = QuantizationScheme(
                targets=[target],
                input_activations=attention_args,
            )

    kv_args = None
    if config.artifacts.kv_cache.enabled:
        kv_args = build_artifact_quant_args(
            config.artifacts.kv_cache,
            dynamic=dynamic_activations,
            for_weights=False,
        )

    return QuantizationConfig(
        config_groups=groups,
        kv_cache_scheme=kv_args,
        format=quantization_format_for_config(config),
        quantization_status=QuantizationStatus.FROZEN,
        ignore=[
            "lm_head",
            "re:.*norm.*",
            "re:.*embed_tokens.*",
        ],
    )
