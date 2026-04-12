"""Artifact parameter collection and serialization helpers.

This module owns the calibration stats and qparam population steps used by the
main quantization pipeline. It deliberately stays method-agnostic except for
accepting GPTQ-produced weight qparams when a method has already computed them.
"""

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import DynamicType, QuantizationArgs
from compressed_tensors.quantization.utils import calculate_range
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig
from ptq.data import batched_texts
from ptq.quantization.weight_only import WeightQuantizationParameters


@dataclass
class ModuleActivationStats:
    """Per-linear activation statistics collected during calibration."""

    global_absmax: float
    channel_absmax: torch.Tensor
    channel_sq_mean: torch.Tensor


def strategy_name(args: QuantizationArgs) -> str:
    """Return a stable strategy name across enum- and string-backed configs."""
    strategy = args.strategy
    if isinstance(strategy, str):
        return strategy.upper()
    return str(strategy).split(".")[-1].upper()


def calculate_scale_zero_point(
    values: torch.Tensor,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert observed ranges into scale and zero-point tensors."""
    q_min, q_max = calculate_range(args, values.device)
    q_min = float(q_min.item())
    q_max = float(q_max.item())
    eps = torch.tensor(1e-8, dtype=torch.float32, device=values.device)

    if args.symmetric:
        max_val = values.abs()
        denom = max(abs(q_min), abs(q_max))
        scale = torch.maximum(max_val / denom, eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point

    min_val = -values.abs()
    max_val = values.abs()
    scale = torch.maximum((max_val - min_val) / max(q_max - q_min, 1.0), eps)
    zero_point = torch.clamp(torch.round(q_min - min_val / scale), q_min, q_max)
    return scale, zero_point


def reshape_weight_for_scale(
    weight: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    """Reduce a weight matrix into the serialized qparam shape."""
    strategy = strategy_name(args)

    if strategy == "TENSOR":
        return weight.reshape(1)

    if strategy == "CHANNEL":
        return weight.abs().amax(dim=1, keepdim=True)

    if strategy in {"GROUP", "TENSOR_GROUP"}:
        if args.group_size is None:
            raise ValueError("group quantization requires group_size")
        cols = weight.shape[1]
        padded_cols = ceil(cols / args.group_size) * args.group_size
        if padded_cols != cols:
            weight = F.pad(weight, (0, padded_cols - cols), value=0.0)
        grouped = weight.reshape(weight.shape[0], -1, args.group_size)
        return grouped.abs().amax(dim=-1)

    if strategy == "BLOCK":
        if not args.block_structure:
            raise ValueError("block quantization requires block_structure")
        block_h, block_w = args.block_structure
        rows, cols = weight.shape
        padded_rows = ceil(rows / block_h) * block_h
        padded_cols = ceil(cols / block_w) * block_w
        if padded_rows != rows or padded_cols != cols:
            weight = F.pad(weight, (0, padded_cols - cols, 0, padded_rows - rows))
        view = weight.reshape(
            padded_rows // block_h,
            block_h,
            padded_cols // block_w,
            block_w,
        )
        blocks = view.transpose(1, 2)
        return blocks.abs().amax(dim=(-1, -2))

    raise ValueError(f"unsupported weight strategy: {args.strategy}")


def set_module_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """Copy a serialized qparam tensor into an instrumented module parameter."""
    if not hasattr(module, name):
        return
    parameter = getattr(module, name)
    if parameter is None:
        return
    if parameter.shape != value.shape:
        raise ValueError(
            f"shape mismatch for {name}: expected {tuple(parameter.shape)}, "
            f"got {tuple(value.shape)}"
        )
    parameter.data.copy_(value.to(parameter.device, dtype=parameter.dtype))


def collect_activation_statistics(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> dict[str, ModuleActivationStats]:
    """Collect calibration statistics from dense linear inputs before quantization."""
    stats: dict[str, dict[str, torch.Tensor | int | float]] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def pre_hook(mod, args, _name=name):
            del mod
            inputs = args[0]
            if not isinstance(inputs, torch.Tensor):
                return
            tensor = inputs.detach().to(torch.float32)
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"non-finite activations observed during calibration for "
                    f"module {_name}"
                )
            flat = tensor.reshape(-1, tensor.shape[-1])
            channel_absmax = flat.abs().amax(dim=0).cpu()
            channel_sq_sum = flat.pow(2).sum(dim=0).cpu()
            token_count = flat.shape[0]
            global_absmax = float(channel_absmax.max().item())

            current = stats.get(_name)
            if current is None:
                stats[_name] = {
                    "global_absmax": global_absmax,
                    "channel_absmax": channel_absmax,
                    "channel_sq_sum": channel_sq_sum,
                    "token_count": token_count,
                }
            else:
                current["global_absmax"] = max(
                    float(current["global_absmax"]),
                    global_absmax,
                )
                current["channel_absmax"] = torch.maximum(
                    current["channel_absmax"],
                    channel_absmax,
                )
                current["channel_sq_sum"] = current["channel_sq_sum"] + channel_sq_sum
                current["token_count"] = int(current["token_count"]) + token_count

        hooks.append(module.register_forward_pre_hook(pre_hook))

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
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model(**encoded)

    for hook in hooks:
        hook.remove()

    results: dict[str, ModuleActivationStats] = {}
    for name, value in stats.items():
        token_count = max(int(value["token_count"]), 1)
        channel_sq_mean = value["channel_sq_sum"] / token_count
        results[name] = ModuleActivationStats(
            global_absmax=float(value["global_absmax"]),
            channel_absmax=value["channel_absmax"],
            channel_sq_mean=channel_sq_mean,
        )
    return results


def populate_weight_quantization_parameters(
    model: torch.nn.Module,
    gptq_parameters: dict[str, WeightQuantizationParameters] | None = None,
) -> None:
    """Populate serialized weight scales and zero-points after instrumentation."""
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        if not hasattr(module, "weight"):
            continue

        if gptq_parameters is not None and name in gptq_parameters:
            scale = gptq_parameters[name].scale
            zero_point = gptq_parameters[name].zero_point
        else:
            weight = module.weight.detach().to(torch.float32)
            reduced = reshape_weight_for_scale(weight, scheme.weights)
            scale, zero_point = calculate_scale_zero_point(reduced, scheme.weights)
        set_module_param(module, "weight_scale", scale)
        set_module_param(module, "weight_zero_point", zero_point)


def activation_tensor_for_strategy(
    stat: ModuleActivationStats,
    args: QuantizationArgs,
) -> torch.Tensor:
    """Project activation stats into the serialized qparam shape."""
    strategy = strategy_name(args)
    channel_absmax = stat.channel_absmax.to(torch.float32)
    if strategy == "TENSOR":
        return torch.tensor([stat.global_absmax], dtype=torch.float32)
    if strategy == "CHANNEL":
        return channel_absmax.reshape(-1, 1)
    if strategy in {"GROUP", "TENSOR_GROUP"}:
        if args.group_size is None:
            raise ValueError("group activation quantization requires group_size")
        width = channel_absmax.numel()
        padded = ceil(width / args.group_size) * args.group_size
        if padded != width:
            channel_absmax = F.pad(channel_absmax, (0, padded - width), value=0.0)
        return channel_absmax.reshape(-1, args.group_size).amax(dim=-1).reshape(1, -1)
    raise ValueError(f"unsupported static activation strategy: {args.strategy}")


def populate_static_activation_parameters(
    model: torch.nn.Module,
    activation_stats: dict[str, ModuleActivationStats],
) -> None:
    """Populate frozen activation scales for static quantization exports."""
    for name, module in model.named_modules():
        if name not in activation_stats:
            continue
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.input_activations is None:
            continue
        args = scheme.input_activations
        if args.dynamic in {True, DynamicType.LOCAL}:
            continue
        stat_tensor = activation_tensor_for_strategy(activation_stats[name], args)
        scale, zero_point = calculate_scale_zero_point(stat_tensor, args)
        set_module_param(module, "input_scale", scale)
        set_module_param(module, "input_zero_point", zero_point)
