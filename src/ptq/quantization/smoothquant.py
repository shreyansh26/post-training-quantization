"""SmoothQuant transforms for Qwen/LLaMA-style transformer blocks."""

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig
from ptq.data import batched_texts

MIN_SMOOTH_SCALE = 1e-5

DEFAULT_SMOOTHQUANT_MAPPINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "re:.*input_layernorm$",
        ("re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"),
    ),
    (
        "re:.*post_attention_layernorm$",
        ("re:.*gate_proj$", "re:.*up_proj$"),
    ),
)

SMOOTHQUANT_MAPPING_REGISTRY: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "LlamaForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MistralForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Qwen2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Qwen3ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
}


@dataclass(frozen=True)
class ResolvedBalanceMapping:
    """A normalization layer and the linear layers that consume its activations."""

    smooth_name: str
    smooth_layer: nn.Module
    balance_names: list[str]
    balance_layers: list[nn.Linear]


def _pattern_matches(name: str, pattern: str) -> bool:
    """Match either an exact module name or a ``re:``-prefixed regex."""
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return name == pattern


def _mapping_spec_for_model(
    model: nn.Module,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the SmoothQuant mapping template for the model architecture."""
    return SMOOTHQUANT_MAPPING_REGISTRY.get(
        model.__class__.__name__,
        DEFAULT_SMOOTHQUANT_MAPPINGS,
    )


def resolve_balance_mappings(model: nn.Module) -> list[ResolvedBalanceMapping]:
    """Resolve architecture mappings into concrete module references."""
    named_modules = dict(model.named_modules())
    resolved: list[ResolvedBalanceMapping] = []

    for smooth_pattern, balance_patterns in _mapping_spec_for_model(model):
        for smooth_name, smooth_layer in named_modules.items():
            if not _pattern_matches(smooth_name, smooth_pattern):
                continue

            block_prefix = smooth_name.rsplit(".", maxsplit=1)[0]
            balance_names: list[str] = []
            balance_layers: list[nn.Linear] = []
            for balance_pattern in balance_patterns:
                for balance_name, balance_layer in named_modules.items():
                    if not balance_name.startswith(f"{block_prefix}."):
                        continue
                    if not isinstance(balance_layer, nn.Linear):
                        continue
                    if not _pattern_matches(balance_name, balance_pattern):
                        continue
                    balance_names.append(balance_name)
                    balance_layers.append(balance_layer)

            if not balance_layers:
                raise ValueError(
                    "failed to resolve SmoothQuant balance layers for "
                    f"{smooth_name}"
                )

            resolved.append(
                ResolvedBalanceMapping(
                    smooth_name=smooth_name,
                    smooth_layer=smooth_layer,
                    balance_names=balance_names,
                    balance_layers=balance_layers,
                )
            )

    if not resolved:
        raise ValueError(
            "failed to resolve any SmoothQuant mappings for "
            f"{model.__class__.__name__}"
        )

    return resolved


def collect_smoothquant_activation_scales(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
    mappings: list[ResolvedBalanceMapping],
) -> dict[str, torch.Tensor]:
    """Collect channel-wise activation maxima on the smooth-layer outputs."""
    scales: dict[str, torch.Tensor] = {}
    hooks = []

    for mapping in mappings:
        def hook_fn(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            smooth_name: str = mapping.smooth_name,
        ) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            tensor = output.detach().to(torch.float32)
            flat = tensor.reshape(-1, tensor.shape[-1])
            observed = flat.abs().amax(dim=0).cpu()
            if smooth_name in scales:
                scales[smooth_name] = torch.maximum(scales[smooth_name], observed)
            else:
                scales[smooth_name] = observed

        hooks.append(mapping.smooth_layer.register_forward_hook(hook_fn))

    if calibration_texts:
        with torch.no_grad():
            for batch in batched_texts(
                calibration_texts,
                config.calibration.batch_size,
            ):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=config.calibration.max_sequence_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded)

    for hook in hooks:
        hook.remove()

    return scales


def _weight_channel_absmax(layer: nn.Linear) -> torch.Tensor:
    """Return the largest absolute weight per input channel."""
    return layer.weight.detach().to(torch.float32).abs().amax(dim=0)


def resolve_smoothquant_mappings(model: nn.Module) -> list[ResolvedBalanceMapping]:
    """Backward-compatible alias for the shared balance-layer mapping resolver."""
    return resolve_balance_mappings(model)


def _apply_scale_to_smooth_layer(layer: nn.Module, scale: torch.Tensor) -> None:
    """Fold the inverse SmoothQuant scale into the smooth layer parameters."""
    weight = layer.weight if hasattr(layer, "weight") else None
    if weight is not None:
        if weight.ndim == 1 and weight.shape[0] == scale.numel():
            weight.data.div_(scale.to(device=weight.device, dtype=weight.dtype))
    bias = layer.bias if hasattr(layer, "bias") else None
    if bias is not None:
        if bias.ndim == 1 and bias.shape[0] == scale.numel():
            bias.data.div_(scale.to(device=bias.device, dtype=bias.dtype))


def _apply_scale_to_balance_layers(
    balance_layers: list[nn.Linear],
    scale: torch.Tensor,
) -> None:
    """Fold the SmoothQuant scale into each downstream linear weight."""
    for layer in balance_layers:
        scale_view = scale.to(device=layer.weight.device, dtype=layer.weight.dtype)
        layer.weight.data.mul_(scale_view.reshape(1, -1))


def apply_smoothquant(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> None:
    """Apply SmoothQuant rebalancing before quantization parameters are computed."""
    mappings = resolve_balance_mappings(model)
    activation_scales = collect_smoothquant_activation_scales(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
        mappings=mappings,
    )

    alpha = config.method.smoothquant_alpha
    with torch.no_grad():
        for mapping in mappings:
            device = mapping.balance_layers[0].weight.device
            act_scale = activation_scales[mapping.smooth_name].to(
                device=device,
                dtype=torch.float32,
            )
            weight_scale = torch.stack(
                [_weight_channel_absmax(layer) for layer in mapping.balance_layers],
                dim=0,
            ).amax(dim=0)
            smooth_scale = torch.clamp(
                act_scale.clamp_min(MIN_SMOOTH_SCALE).pow(alpha)
                / weight_scale.clamp_min(MIN_SMOOTH_SCALE).pow(1.0 - alpha),
                min=MIN_SMOOTH_SCALE,
            )

            _apply_scale_to_smooth_layer(mapping.smooth_layer, smooth_scale)
            _apply_scale_to_balance_layers(mapping.balance_layers, smooth_scale)
