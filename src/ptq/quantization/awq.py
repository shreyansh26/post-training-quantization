"""Activation-aware weight quantization preprocessing."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig
from ptq.data import batched_texts
from ptq.quantization.smoothquant import (
    MIN_SMOOTH_SCALE,
    ResolvedBalanceMapping,
    resolve_balance_mappings,
)
from ptq.quantization.weight_only import quantize_dequantize_weight

_AWQ_GRID_SIZE = 20
_AWQ_MAX_SAMPLE_TOKENS = 1024


@dataclass
class MappingCalibrationData:
    """Cached activation summaries for one AWQ mapping."""

    mean_abs_activation: torch.Tensor
    sample_inputs: torch.Tensor


def collect_awq_calibration_data(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
    mappings: list[ResolvedBalanceMapping],
) -> dict[str, MappingCalibrationData]:
    """Collect activation statistics and representative samples for AWQ search."""
    activation_sums: dict[str, torch.Tensor] = {}
    activation_counts: dict[str, int] = {}
    sample_inputs: dict[str, list[torch.Tensor]] = {m.smooth_name: [] for m in mappings}
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
            flat = tensor.reshape(-1, tensor.shape[-1]).cpu()
            if flat.numel() == 0:
                return

            current_sum = flat.abs().sum(dim=0)
            activation_sums[smooth_name] = (
                current_sum
                if smooth_name not in activation_sums
                else activation_sums[smooth_name] + current_sum
            )
            activation_counts[smooth_name] = (
                activation_counts.get(smooth_name, 0) + flat.shape[0]
            )

            cached_rows = sum(chunk.shape[0] for chunk in sample_inputs[smooth_name])
            remaining = _AWQ_MAX_SAMPLE_TOKENS - cached_rows
            if remaining > 0:
                sample_inputs[smooth_name].append(flat[:remaining])

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

    results: dict[str, MappingCalibrationData] = {}
    for mapping in mappings:
        smooth_name = mapping.smooth_name
        count = max(activation_counts.get(smooth_name, 0), 1)
        mean_abs = activation_sums[smooth_name] / count
        samples = torch.cat(sample_inputs[smooth_name], dim=0)
        results[smooth_name] = MappingCalibrationData(
            mean_abs_activation=mean_abs,
            sample_inputs=samples,
        )
    return results


def _weight_channel_mean(balance_layers: list[nn.Linear]) -> torch.Tensor:
    """Aggregate per-input-channel weight magnitudes across balance layers."""
    stats = [
        layer.weight.detach().to(torch.float32).abs().mean(dim=0)
        for layer in balance_layers
    ]
    return torch.stack(stats, dim=0).mean(dim=0)


def _normalize_awq_scale(scale: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """Keep AWQ search scales numerically stable before applying them."""
    scale = scale / torch.sqrt(scale.max() * scale.min())
    if clip_ratio != 1.0:
        scale = torch.clamp(scale, min=1.0 / clip_ratio, max=clip_ratio)
    return torch.clamp(scale, min=MIN_SMOOTH_SCALE)


def _awq_loss_for_mapping(
    mapping: ResolvedBalanceMapping,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    config: PTQRunConfig,
) -> float:
    """Measure reconstruction error for one candidate AWQ scaling vector."""
    total_loss = 0.0
    scaled_inputs = inputs.to(torch.float32) / scale.reshape(1, -1)

    for layer in mapping.balance_layers:
        weight = layer.weight.detach().to(torch.float32)
        scaled_weight = weight * scale.reshape(1, -1)
        quantized_weight, _params = quantize_dequantize_weight(
            scaled_weight,
            config.artifacts.weights,
        )

        bias = None if layer.bias is None else layer.bias.detach().to(torch.float32)
        reference = F.linear(inputs.to(torch.float32), weight, bias)
        candidate = F.linear(scaled_inputs, quantized_weight, bias)
        total_loss += torch.mean((reference - candidate).pow(2)).item()

    return total_loss


def _find_best_awq_scale(
    mapping: ResolvedBalanceMapping,
    calibration: MappingCalibrationData,
    config: PTQRunConfig,
) -> torch.Tensor:
    """Search for the AWQ scaling vector that minimizes reconstruction error."""
    device = mapping.balance_layers[0].weight.device
    act_mean = calibration.mean_abs_activation.to(device=device).clamp_min(
        MIN_SMOOTH_SCALE
    )
    weight_mean = (
        _weight_channel_mean(mapping.balance_layers)
        .to(device=device)
        .clamp_min(MIN_SMOOTH_SCALE)
    )
    inputs = calibration.sample_inputs.to(device=device)

    best_loss = float("inf")
    best_scale = torch.ones_like(act_mean)

    for grid_index in range(_AWQ_GRID_SIZE):
        ratio = grid_index / max(_AWQ_GRID_SIZE - 1, 1)
        scale = act_mean.pow(ratio) / weight_mean.pow(1.0 - ratio)
        scale = _normalize_awq_scale(scale, config.method.awq_clip_ratio)
        loss = _awq_loss_for_mapping(mapping, inputs, scale, config)
        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    return best_scale


def _apply_awq_scale(mapping: ResolvedBalanceMapping, scale: torch.Tensor) -> None:
    """Fold the selected AWQ scale into the smooth and balance layers."""
    smooth_layer = mapping.smooth_layer
    if hasattr(smooth_layer, "weight") and smooth_layer.weight is not None:
        weight = smooth_layer.weight
        if weight.ndim == 1 and weight.shape[0] == scale.numel():
            weight.data.div_(scale.to(device=weight.device, dtype=weight.dtype))
    if hasattr(smooth_layer, "bias") and smooth_layer.bias is not None:
        bias = smooth_layer.bias
        if bias.ndim == 1 and bias.shape[0] == scale.numel():
            bias.data.div_(scale.to(device=bias.device, dtype=bias.dtype))

    for layer in mapping.balance_layers:
        scale_view = scale.to(device=layer.weight.device, dtype=layer.weight.dtype)
        layer.weight.data.mul_(scale_view.reshape(1, -1))


def apply_awq(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> None:
    """Apply AWQ-style layer rebalancing before weight quantization."""
    mappings = resolve_balance_mappings(model)
    calibration_data = collect_awq_calibration_data(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
        mappings=mappings,
    )

    with torch.no_grad():
        for mapping in mappings:
            scale = _find_best_awq_scale(
                mapping,
                calibration_data[mapping.smooth_name],
                config,
            )
            _apply_awq_scale(mapping, scale)
