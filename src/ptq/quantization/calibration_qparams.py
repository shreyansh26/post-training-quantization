"""Calibration-statistics and qparam population helpers.

This module owns the calibration stats and qparam population steps used by the
main quantization pipeline. It deliberately stays method-agnostic except for
accepting GPTQ-produced weight qparams when a method has already computed them.
"""

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from compressed_tensors.modeling.attention import (
    initialize_hooked_attention,
    register_query_hook,
)
from compressed_tensors.modeling.kvcache import (
    initialize_hooked_kv_cache,
    register_key_hook,
    register_value_hook,
)
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.initialize import is_attention_module
from compressed_tensors.quantization.utils import calculate_qparams
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig, QuantizationMethod
from ptq.data import batched_texts
from ptq.quantization.weight_only import WeightQuantizationParameters


@dataclass
class ModuleActivationStats:
    """Per-linear activation statistics collected during calibration."""

    global_absmax: float
    global_min: float
    global_max: float
    channel_absmax: torch.Tensor
    channel_min: torch.Tensor
    channel_max: torch.Tensor
    channel_sq_mean: torch.Tensor


@dataclass
class AttentionQKVStats:
    """Per-attention-module calibration statistics for query, key, and value."""

    q_absmax: float
    k_absmax: float
    v_absmax: float


def strategy_name(args: QuantizationArgs) -> str:
    """Return a stable strategy name across enum- and string-backed configs."""
    strategy = args.strategy
    if isinstance(strategy, str):
        return strategy.upper()
    return str(strategy).split(".")[-1].upper()


def calculate_bounds_qparams(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert observed min/max tensors into serialized scale and zero-point."""
    return calculate_qparams(
        min_vals=min_vals,
        max_vals=max_vals,
        quantization_args=args,
        global_scale=None,
    )


def weight_bounds_for_strategy(
    weight: torch.Tensor,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a weight matrix into min/max tensors matching serialized qparams."""
    strategy = strategy_name(args)

    if strategy == "TENSOR":
        return weight.reshape(1).amin().reshape(1), weight.reshape(1).amax().reshape(1)

    if strategy == "CHANNEL":
        return weight.amin(dim=1, keepdim=True), weight.amax(dim=1, keepdim=True)

    if strategy in {"GROUP", "TENSOR_GROUP"}:
        if args.group_size is None:
            raise ValueError("group quantization requires group_size")
        cols = weight.shape[1]
        padded_cols = ceil(cols / args.group_size) * args.group_size
        if padded_cols != cols:
            weight = F.pad(weight, (0, padded_cols - cols), value=0.0)
        grouped = weight.reshape(weight.shape[0], -1, args.group_size)
        return grouped.amin(dim=-1), grouped.amax(dim=-1)

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
        return blocks.amin(dim=(-1, -2)), blocks.amax(dim=(-1, -2))

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


def clear_module_param(module: torch.nn.Module, name: str) -> None:
    """Drop a registered qparam when the runtime/export contract does not use it."""
    if hasattr(module, name):
        delattr(module, name)


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
            channel_min = flat.amin(dim=0).cpu()
            channel_max = flat.amax(dim=0).cpu()
            channel_sq_sum = flat.pow(2).sum(dim=0).cpu()
            token_count = flat.shape[0]
            global_absmax = float(channel_absmax.max().item())
            global_min = float(channel_min.min().item())
            global_max = float(channel_max.max().item())

            current = stats.get(_name)
            if current is None:
                stats[_name] = {
                    "global_absmax": global_absmax,
                    "global_min": global_min,
                    "global_max": global_max,
                    "channel_absmax": channel_absmax,
                    "channel_min": channel_min,
                    "channel_max": channel_max,
                    "channel_sq_sum": channel_sq_sum,
                    "token_count": token_count,
                }
            else:
                current["global_absmax"] = max(
                    float(current["global_absmax"]),
                    global_absmax,
                )
                current["global_min"] = min(float(current["global_min"]), global_min)
                current["global_max"] = max(float(current["global_max"]), global_max)
                current["channel_absmax"] = torch.maximum(
                    current["channel_absmax"],
                    channel_absmax,
                )
                current["channel_min"] = torch.minimum(
                    current["channel_min"],
                    channel_min,
                )
                current["channel_max"] = torch.maximum(
                    current["channel_max"],
                    channel_max,
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
            global_min=float(value["global_min"]),
            global_max=float(value["global_max"]),
            channel_absmax=value["channel_absmax"],
            channel_min=value["channel_min"],
            channel_max=value["channel_max"],
            channel_sq_mean=channel_sq_mean,
        )
    return results


def _update_module_activation_stats(
    stats: dict[str, dict[str, torch.Tensor | int | float]],
    module_name: str,
    inputs: torch.Tensor,
) -> ModuleActivationStats:
    """Update running activation stats for one module and return the aggregate."""
    tensor = inputs.detach().to(torch.float32)
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            "non-finite activations observed during calibration for module "
            f"{module_name}"
        )

    flat = tensor.reshape(-1, tensor.shape[-1])
    channel_absmax = flat.abs().amax(dim=0).cpu()
    channel_min = flat.amin(dim=0).cpu()
    channel_max = flat.amax(dim=0).cpu()
    channel_sq_sum = flat.pow(2).sum(dim=0).cpu()
    token_count = flat.shape[0]
    global_absmax = float(channel_absmax.max().item())
    global_min = float(channel_min.min().item())
    global_max = float(channel_max.max().item())

    current = stats.get(module_name)
    if current is None:
        current = {
            "global_absmax": global_absmax,
            "global_min": global_min,
            "global_max": global_max,
            "channel_absmax": channel_absmax,
            "channel_min": channel_min,
            "channel_max": channel_max,
            "channel_sq_sum": channel_sq_sum,
            "token_count": token_count,
        }
        stats[module_name] = current
    else:
        current["global_absmax"] = max(float(current["global_absmax"]), global_absmax)
        current["global_min"] = min(float(current["global_min"]), global_min)
        current["global_max"] = max(float(current["global_max"]), global_max)
        current["channel_absmax"] = torch.maximum(
            current["channel_absmax"],
            channel_absmax,
        )
        current["channel_min"] = torch.minimum(
            current["channel_min"],
            channel_min,
        )
        current["channel_max"] = torch.maximum(
            current["channel_max"],
            channel_max,
        )
        current["channel_sq_sum"] = current["channel_sq_sum"] + channel_sq_sum
        current["token_count"] = int(current["token_count"]) + token_count

    total_tokens = max(int(current["token_count"]), 1)
    return ModuleActivationStats(
        global_absmax=float(current["global_absmax"]),
        global_min=float(current["global_min"]),
        global_max=float(current["global_max"]),
        channel_absmax=current["channel_absmax"],
        channel_min=current["channel_min"],
        channel_max=current["channel_max"],
        channel_sq_mean=current["channel_sq_sum"] / total_tokens,
    )


def _max_abs_from_attention_tensor(tensor: torch.Tensor, name: str) -> float:
    """Reduce one observed attention tensor to a scalar static calibration range."""
    value = tensor.detach().to(torch.float32)
    if not torch.isfinite(value).all():
        raise RuntimeError(
            f"non-finite attention states observed during calibration for module {name}"
        )
    return float(value.abs().amax().item())


def collect_attention_statistics(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> dict[str, AttentionQKVStats]:
    """Collect static q/k/v calibration ranges from attention modules.

    The installed compressed-tensors attention and kv-cache hooks both read
    `module.quantization_scheme.input_activations` at runtime. Because of that
    runtime contract, we calibrate q, k, and v together whenever either
    attention or kv-cache quantization is enabled in a non-dynamic mode.
    """
    static_attention_like = (
        config.method.name is not QuantizationMethod.DYNAMIC
        and (config.artifacts.attention.enabled or config.artifacts.kv_cache.enabled)
    )
    if not static_attention_like or not calibration_texts:
        return {}

    stats: dict[str, dict[str, float]] = {}
    hooks = []

    for name, module in model.named_modules():
        if not is_attention_module(module):
            continue

        if config.artifacts.attention.enabled:
            initialize_hooked_attention(model, module)
        else:
            initialize_hooked_kv_cache(model, module)

        def query_hook(attn_module, query_states, module_name=name):
            del attn_module
            observed = _max_abs_from_attention_tensor(query_states, module_name)
            current = stats.setdefault(
                module_name,
                {"q_absmax": 0.0, "k_absmax": 0.0, "v_absmax": 0.0},
            )
            current["q_absmax"] = max(current["q_absmax"], observed)
            return None

        def key_hook(attn_module, key_states, module_name=name):
            del attn_module
            observed = _max_abs_from_attention_tensor(key_states, module_name)
            current = stats.setdefault(
                module_name,
                {"q_absmax": 0.0, "k_absmax": 0.0, "v_absmax": 0.0},
            )
            current["k_absmax"] = max(current["k_absmax"], observed)
            return None

        def value_hook(attn_module, value_states, module_name=name):
            del attn_module
            observed = _max_abs_from_attention_tensor(value_states, module_name)
            current = stats.setdefault(
                module_name,
                {"q_absmax": 0.0, "k_absmax": 0.0, "v_absmax": 0.0},
            )
            current["v_absmax"] = max(current["v_absmax"], observed)
            return None

        if config.artifacts.attention.enabled:
            hooks.append(register_query_hook(module, query_hook))
        hooks.append(register_key_hook(module, key_hook))
        hooks.append(register_value_hook(module, value_hook))

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

    return {
        name: AttentionQKVStats(
            q_absmax=value["q_absmax"],
            k_absmax=value["k_absmax"],
            v_absmax=value["v_absmax"],
        )
        for name, value in stats.items()
    }


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
            min_vals, max_vals = weight_bounds_for_strategy(weight, scheme.weights)
            scale, zero_point = calculate_bounds_qparams(
                min_vals,
                max_vals,
                scheme.weights,
            )
        set_module_param(module, "weight_scale", scale)
        set_module_param(module, "weight_zero_point", zero_point)


def activation_bounds_for_strategy(
    stat: ModuleActivationStats,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project activation stats into min/max tensors for asymmetric qparams."""
    strategy = strategy_name(args)
    channel_min = stat.channel_min.to(torch.float32)
    channel_max = stat.channel_max.to(torch.float32)

    if strategy == "TENSOR":
        return (
            torch.tensor([stat.global_min], dtype=torch.float32),
            torch.tensor([stat.global_max], dtype=torch.float32),
        )

    if strategy == "CHANNEL":
        return channel_min.reshape(-1, 1), channel_max.reshape(-1, 1)

    if strategy in {"GROUP", "TENSOR_GROUP"}:
        if args.group_size is None:
            raise ValueError("group activation quantization requires group_size")
        width = channel_min.numel()
        padded = ceil(width / args.group_size) * args.group_size
        if padded != width:
            channel_min = F.pad(channel_min, (0, padded - width), value=0.0)
            channel_max = F.pad(channel_max, (0, padded - width), value=0.0)
        reduced_min = channel_min.reshape(-1, args.group_size).amin(dim=-1)
        reduced_max = channel_max.reshape(-1, args.group_size).amax(dim=-1)
        return reduced_min.reshape(1, -1), reduced_max.reshape(1, -1)

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
        min_tensor, max_tensor = activation_bounds_for_strategy(
            activation_stats[name],
            args,
        )
        scale, zero_point = calculate_bounds_qparams(
            min_tensor,
            max_tensor,
            args,
        )
        set_module_param(module, "input_scale", scale)
        set_module_param(module, "input_zero_point", zero_point)


def calibrate_static_activation_parameters(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> dict[str, ModuleActivationStats]:
    """Calibrate static activation qparams through instrumented fake-quant forwards.

    This mirrors llm-compressor more closely than the previous dense-only stats
    pass: each linear layer updates its running activation observer state in a
    forward pre-hook, then immediately executes with fake-quantized inputs and
    weights. Downstream layers therefore see upstream quantization effects
    during calibration.
    """
    static_modules: list[tuple[str, torch.nn.Linear]] = []
    running_stats: dict[str, dict[str, torch.Tensor | int | float]] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.input_activations is None:
            continue
        args = scheme.input_activations
        if args.dynamic in {True, DynamicType.LOCAL}:
            continue

        static_modules.append((name, module))
        module.quantization_status = QuantizationStatus.CALIBRATION

        def pre_hook(
            calibrated_module: torch.nn.Linear,
            hook_args: tuple[object, ...],
            *,
            module_name: str = name,
        ) -> None:
            inputs = hook_args[0]
            if not isinstance(inputs, torch.Tensor):
                return
            stats = _update_module_activation_stats(running_stats, module_name, inputs)
            min_tensor, max_tensor = activation_bounds_for_strategy(
                stats,
                calibrated_module.quantization_scheme.input_activations,
            )
            scale, zero_point = calculate_bounds_qparams(
                min_tensor,
                max_tensor,
                calibrated_module.quantization_scheme.input_activations,
            )
            set_module_param(calibrated_module, "input_scale", scale)
            set_module_param(calibrated_module, "input_zero_point", zero_point)

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
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded)

    for hook in hooks:
        hook.remove()

    results: dict[str, ModuleActivationStats] = {}
    for name, module in static_modules:
        module.quantization_status = QuantizationStatus.FROZEN
        current = running_stats.get(name)
        if current is None:
            continue
        total_tokens = max(int(current["token_count"]), 1)
        results[name] = ModuleActivationStats(
            global_absmax=float(current["global_absmax"]),
            global_min=float(current["global_min"]),
            global_max=float(current["global_max"]),
            channel_absmax=current["channel_absmax"],
            channel_min=current["channel_min"],
            channel_max=current["channel_max"],
            channel_sq_mean=current["channel_sq_sum"] / total_tokens,
        )
    return results


def populate_static_attention_parameters(
    model: torch.nn.Module,
    attention_stats: dict[str, AttentionQKVStats],
) -> None:
    """Populate frozen q/k/v scales for static attention and kv-cache exports."""
    for name, module in model.named_modules():
        if name not in attention_stats:
            continue
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.input_activations is None:
            continue
        args = scheme.input_activations
        if args.dynamic in {True, DynamicType.LOCAL}:
            continue

        stats = attention_stats[name]
        for base_name, absmax in (
            ("q", stats.q_absmax),
            ("k", stats.k_absmax),
            ("v", stats.v_absmax),
        ):
            min_tensor = torch.tensor([-absmax], dtype=torch.float32)
            max_tensor = torch.tensor([absmax], dtype=torch.float32)
            scale, zero_point = calculate_bounds_qparams(
                min_tensor,
                max_tensor,
                args,
            )
            set_module_param(module, f"{base_name}_scale", scale)
            if args.symmetric:
                clear_module_param(module, f"{base_name}_zero_point")
            else:
                set_module_param(module, f"{base_name}_zero_point", zero_point)
