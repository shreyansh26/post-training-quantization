from dataclasses import dataclass
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from compressed_tensors import ModelCompressor
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
    apply_quantization_config,
)
from compressed_tensors.quantization.utils import calculate_range
from transformers import PreTrainedTokenizerBase

from ptq.config import (
    ArtifactQuantizationSettings,
    PTQRunConfig,
    QuantizationDType,
    QuantizationGranularity,
    QuantizationMethod,
)
from ptq.data import batched_texts


@dataclass
class ModuleActivationStats:
    """Per-linear activation statistics collected during calibration."""

    global_absmax: float
    channel_absmax: torch.Tensor
    channel_sq_mean: torch.Tensor


def _dtype_to_quant_type(dtype: QuantizationDType) -> QuantizationType:
    """Map the repo's dtype enum to compressed-tensors' quantization type."""
    if dtype is QuantizationDType.INT8:
        return QuantizationType.INT
    if dtype is QuantizationDType.FP8:
        return QuantizationType.FLOAT
    raise ValueError(f"unsupported quantization dtype: {dtype}")


def _weight_strategy(
    settings: ArtifactQuantizationSettings,
) -> tuple[QuantizationStrategy, list[int] | None]:
    """Translate weight granularity settings into compressed-tensors strategy."""
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


def _activation_strategy(
    settings: ArtifactQuantizationSettings,
) -> QuantizationStrategy:
    """Translate activation granularity settings into export strategy enums."""
    if settings.granularity is QuantizationGranularity.TOKEN:
        return QuantizationStrategy.TOKEN
    if settings.granularity is QuantizationGranularity.GROUP:
        return QuantizationStrategy.GROUP
    if settings.granularity is QuantizationGranularity.TENSOR:
        return QuantizationStrategy.TENSOR
    raise ValueError(
        "activation/attention/kv_cache quantization only supports token/group/tensor"
    )


def _build_quant_args(
    settings: ArtifactQuantizationSettings,
    *,
    dynamic: bool,
    for_weights: bool,
) -> QuantizationArgs | None:
    """Build a compressed-tensors quantization spec for one artifact family."""
    if not settings.enabled:
        return None

    strategy, block_structure = (
        _weight_strategy(settings)
        if for_weights
        else (_activation_strategy(settings), None)
    )

    return QuantizationArgs(
        num_bits=8,
        type=_dtype_to_quant_type(settings.dtype),
        symmetric=settings.symmetric,
        strategy=strategy,
        group_size=settings.group_size,
        block_structure=block_structure,
        dynamic=dynamic if not for_weights else False,
    )


def _quantization_format_for_config(config: PTQRunConfig) -> str:
    """Infer the export format label expected by compressed-tensors."""
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


def build_quantization_config(config: PTQRunConfig) -> QuantizationConfig:
    """Build the compressed-tensors config applied before parameter export."""
    dynamic_activations = config.method.name is QuantizationMethod.DYNAMIC

    weights = _build_quant_args(
        config.artifacts.weights,
        dynamic=False,
        for_weights=True,
    )
    input_activations = _build_quant_args(
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
        attention_args = _build_quant_args(
            config.artifacts.attention,
            dynamic=dynamic_activations,
            for_weights=False,
        )
        groups["attention"] = QuantizationScheme(
            targets=["re:.*self_attn$"],
            input_activations=attention_args,
        )

    kv_args = None
    if config.artifacts.kv_cache.enabled:
        kv_args = _build_quant_args(
            config.artifacts.kv_cache,
            dynamic=dynamic_activations,
            for_weights=False,
        )

    return QuantizationConfig(
        config_groups=groups,
        kv_cache_scheme=kv_args,
        format=_quantization_format_for_config(config),
        quantization_status=QuantizationStatus.FROZEN,
        ignore=[
            "lm_head",
            "re:.*norm.*",
            "re:.*embed_tokens.*",
        ],
    )


def _calculate_scale_zp(
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


def _reshape_for_weight_scale(
    weight: torch.Tensor,
    strategy: QuantizationStrategy,
    group_size: int | None,
    block_structure: list[int] | None,
) -> torch.Tensor:
    """Reduce a weight matrix into the shape expected for serialized scales."""
    if strategy == QuantizationStrategy.TENSOR:
        return weight.reshape(1)

    if strategy == QuantizationStrategy.CHANNEL:
        return weight.abs().amax(dim=1, keepdim=True)

    if strategy in {QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP}:
        if group_size is None:
            raise ValueError("group quantization requires group_size")
        cols = weight.shape[1]
        padded_cols = ceil(cols / group_size) * group_size
        if padded_cols != cols:
            pad = padded_cols - cols
            weight = F.pad(weight, (0, pad), value=0.0)
        grouped = weight.reshape(weight.shape[0], -1, group_size)
        return grouped.abs().amax(dim=-1)

    if strategy == QuantizationStrategy.BLOCK:
        if not block_structure:
            raise ValueError("block quantization requires block_structure")
        block_h, block_w = block_structure
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

    raise ValueError(f"unsupported weight strategy: {strategy}")


def _set_module_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """Safely copy exported quantization tensors into an instrumented module."""
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


def _weight_importance(
    method: QuantizationMethod,
    module_name: str,
    stats: dict[str, ModuleActivationStats],
    config: PTQRunConfig,
) -> torch.Tensor | None:
    """Derive method-specific weighting signals from calibration statistics."""
    if module_name not in stats:
        return None
    module_stats = stats[module_name]
    if method is QuantizationMethod.AWQ:
        importance = module_stats.channel_absmax
        denom = torch.clamp(importance.mean(), min=1e-8)
        return torch.clamp(importance / denom, min=1e-4)
    if method is QuantizationMethod.GPTQ:
        damp = config.method.gptq_damp_percent
        return torch.sqrt(torch.clamp(module_stats.channel_sq_mean + damp, min=1e-8))
    if method is QuantizationMethod.SMOOTHQUANT:
        alpha = config.method.smoothquant_alpha
        return torch.clamp(module_stats.channel_absmax.pow(alpha), min=1e-4)
    return None


def populate_weight_quantization_parameters(
    model: torch.nn.Module,
    method: QuantizationMethod,
    config: PTQRunConfig,
    activation_stats: dict[str, ModuleActivationStats],
) -> None:
    """Populate serialized weight scales and zero-points after instrumentation."""
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        if not hasattr(module, "weight"):
            continue

        weight = module.weight.detach().to(torch.float32)
        importance = _weight_importance(method, name, activation_stats, config)
        if importance is not None and importance.numel() == weight.shape[1]:
            adjusted_weight = weight * importance.to(weight.device).reshape(1, -1)
        else:
            adjusted_weight = weight

        args = scheme.weights
        reduced = _reshape_for_weight_scale(
            adjusted_weight,
            args.strategy,
            args.group_size,
            args.block_structure,
        )
        scale, zero_point = _calculate_scale_zp(reduced, args)
        _set_module_param(module, "weight_scale", scale)
        _set_module_param(module, "weight_zero_point", zero_point)


def _activation_tensor_for_strategy(
    stat: ModuleActivationStats,
    strategy: QuantizationStrategy,
    group_size: int | None,
) -> torch.Tensor:
    """Project activation stats into the serialized shape for each strategy."""
    channel_absmax = stat.channel_absmax.to(torch.float32)
    if strategy == QuantizationStrategy.TENSOR:
        return torch.tensor([stat.global_absmax], dtype=torch.float32)
    if strategy == QuantizationStrategy.CHANNEL:
        return channel_absmax.reshape(-1, 1)
    if strategy in {QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP}:
        if group_size is None:
            raise ValueError("group activation quantization requires group_size")
        width = channel_absmax.numel()
        padded = ceil(width / group_size) * group_size
        if padded != width:
            channel_absmax = F.pad(channel_absmax, (0, padded - width), value=0.0)
        return channel_absmax.reshape(-1, group_size).amax(dim=-1).reshape(1, -1)
    raise ValueError(f"unsupported static activation strategy: {strategy}")


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
        stat_tensor = _activation_tensor_for_strategy(
            activation_stats[name],
            args.strategy,
            args.group_size,
        )
        scale, zero_point = _calculate_scale_zp(stat_tensor, args)
        _set_module_param(module, "input_scale", scale)
        _set_module_param(module, "input_zero_point", zero_point)


def prepare_model_for_quantization(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: PTQRunConfig,
    calibration_texts: list[str],
    device: str,
) -> QuantizationConfig:
    """Collect stats on the dense model, then instrument and populate scales."""
    activation_stats = collect_activation_statistics(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
    )
    quant_config = build_quantization_config(config)
    apply_quantization_config(model, quant_config)
    populate_weight_quantization_parameters(
        model=model,
        method=config.method.name,
        config=config,
        activation_stats=activation_stats,
    )
    populate_static_activation_parameters(
        model=model,
        activation_stats=activation_stats,
    )
    return quant_config


def export_quantized_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    quantization_format: str,
) -> Path:
    """Write a model artifact that both HF and vLLM can load."""
    save_dir.mkdir(parents=True, exist_ok=True)
    compressor = ModelCompressor.from_pretrained_model(
        model,
        quantization_format=quantization_format,
    )
    if compressor is None:
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        return save_dir

    state_dict = compressor.compress(model, show_progress=True)
    model.save_pretrained(
        save_dir,
        state_dict=state_dict,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(save_dir)
    compressor.update_config(str(save_dir))
    return save_dir
