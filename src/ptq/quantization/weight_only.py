"""Shared helpers for weight-only quantization algorithms.

This module owns the reusable weight quantization math that is shared by:

- weight-only fake quantization helpers
- AWQ
- GPTQ
"""

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_range

from ptq.config import (
    ArtifactQuantizationSettings,
    QuantizationDType,
    QuantizationGranularity,
)

_EPS = 1e-8


@dataclass(frozen=True)
class WeightQuantizationParameters:
    """Precomputed quantization parameters for a weight tensor."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    args: QuantizationArgs


def _dtype_to_quant_type(dtype: QuantizationDType) -> QuantizationType:
    """Map repo dtypes to compressed-tensors quantization types."""
    if dtype is QuantizationDType.INT8:
        return QuantizationType.INT
    if dtype is QuantizationDType.FP8:
        return QuantizationType.FLOAT
    raise ValueError(f"unsupported quantization dtype: {dtype}")


def build_weight_quant_args(
    settings: ArtifactQuantizationSettings,
) -> QuantizationArgs:
    """Translate repo weight settings into compressed-tensors quant args."""
    if settings.granularity is QuantizationGranularity.TENSOR:
        strategy = QuantizationStrategy.TENSOR
        block_structure = None
    elif settings.granularity is QuantizationGranularity.CHANNEL:
        strategy = QuantizationStrategy.CHANNEL
        block_structure = None
    elif settings.granularity is QuantizationGranularity.GROUP:
        strategy = QuantizationStrategy.GROUP
        block_structure = None
    elif settings.granularity is QuantizationGranularity.BLOCK:
        block_size = settings.block_size if settings.block_size is not None else 128
        strategy = QuantizationStrategy.BLOCK
        block_structure = [block_size, block_size]
    else:
        raise ValueError(
            "weight-only quantization supports tensor/channel/group/block granularity"
        )

    return QuantizationArgs(
        num_bits=8,
        type=_dtype_to_quant_type(settings.dtype),
        symmetric=settings.symmetric,
        strategy=strategy,
        group_size=settings.group_size,
        block_structure=block_structure,
        dynamic=False,
    )


def _reduce_weight_for_qparams(
    weight: torch.Tensor,
    args: QuantizationArgs,
) -> torch.Tensor:
    """Reduce a weight matrix into the serialized qparam shape."""
    if args.strategy == QuantizationStrategy.TENSOR:
        return weight.reshape(1)

    if args.strategy == QuantizationStrategy.CHANNEL:
        return weight.abs().amax(dim=1, keepdim=True)

    if args.strategy in {QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP}:
        if args.group_size is None:
            raise ValueError("group quantization requires group_size")
        cols = weight.shape[1]
        padded_cols = ceil(cols / args.group_size) * args.group_size
        if padded_cols != cols:
            weight = F.pad(weight, (0, padded_cols - cols), value=0.0)
        return weight.reshape(weight.shape[0], -1, args.group_size).abs().amax(dim=-1)

    if args.strategy == QuantizationStrategy.BLOCK:
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
        return view.transpose(1, 2).abs().amax(dim=(-1, -2))

    raise ValueError(f"unsupported weight strategy: {args.strategy}")


def _calculate_scale_zero_point(
    values: torch.Tensor,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert observed values into scale and zero-point tensors."""
    q_min, q_max = calculate_range(args, values.device)
    q_min = float(q_min.item())
    q_max = float(q_max.item())
    eps = torch.tensor(_EPS, dtype=torch.float32, device=values.device)

    if args.symmetric:
        max_val = values.abs()
        denom = max(abs(q_min), abs(q_max))
        scale = torch.maximum(max_val / denom, eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point

    min_val = values.min()
    max_val = values.max()
    scale = torch.maximum((max_val - min_val) / max(q_max - q_min, 1.0), eps)
    zero_point = torch.clamp(torch.round(q_min - min_val / scale), q_min, q_max)
    return scale, zero_point


def compute_weight_quantization_parameters(
    weight: torch.Tensor,
    settings: ArtifactQuantizationSettings,
) -> WeightQuantizationParameters:
    """Build the fixed qparams used to quantize a weight matrix."""
    args = build_weight_quant_args(settings)
    reduced = _reduce_weight_for_qparams(weight.to(torch.float32), args)
    scale, zero_point = _calculate_scale_zero_point(reduced, args)
    return WeightQuantizationParameters(scale=scale, zero_point=zero_point, args=args)


def _expand_qparams_for_weight(
    weight: torch.Tensor,
    params: WeightQuantizationParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast serialized qparams over a full weight matrix."""
    rows, cols = weight.shape
    args = params.args

    if args.strategy == QuantizationStrategy.TENSOR:
        scale = params.scale.reshape(1, 1).expand(rows, cols)
        zero_point = params.zero_point.reshape(1, 1).expand(rows, cols)
        return scale, zero_point

    if args.strategy == QuantizationStrategy.CHANNEL:
        scale = params.scale.reshape(rows, 1).expand(rows, cols)
        zero_point = params.zero_point.reshape(rows, 1).expand(rows, cols)
        return scale, zero_point

    if args.strategy in {QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP}:
        if args.group_size is None:
            raise ValueError("group quantization requires group_size")
        scale = params.scale.repeat_interleave(args.group_size, dim=1)[:, :cols]
        zero_point = params.zero_point.repeat_interleave(args.group_size, dim=1)[
            :, :cols
        ]
        return scale, zero_point

    if args.strategy == QuantizationStrategy.BLOCK:
        if not args.block_structure:
            raise ValueError("block quantization requires block_structure")
        block_h, block_w = args.block_structure
        row_repeat = params.scale.repeat_interleave(block_h, dim=0)
        scale = row_repeat.repeat_interleave(block_w, dim=1)[:rows, :cols]
        row_repeat_zp = params.zero_point.repeat_interleave(block_h, dim=0)
        zero_point = row_repeat_zp.repeat_interleave(block_w, dim=1)[:rows, :cols]
        return scale, zero_point

    raise ValueError(f"unsupported weight strategy: {args.strategy}")


def quantize_dequantize_weight(
    weight: torch.Tensor,
    settings: ArtifactQuantizationSettings,
    params: WeightQuantizationParameters | None = None,
) -> tuple[torch.Tensor, WeightQuantizationParameters]:
    """Round-trip a weight matrix through the configured quantizer."""
    params = params or compute_weight_quantization_parameters(weight, settings)

    if settings.dtype is QuantizationDType.FP8:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")
        return weight.to(torch.float8_e4m3fn).to(weight.dtype), params

    scale, zero_point = _expand_qparams_for_weight(weight, params)
    q_min, q_max = calculate_range(params.args, weight.device)
    q = torch.round(weight / scale + zero_point)
    q = torch.clamp(q, float(q_min.item()), float(q_max.item()))
    return (q - zero_point) * scale, params


def quantize_dequantize_weight_column(
    column: torch.Tensor,
    column_index: int,
    settings: ArtifactQuantizationSettings,
    params: WeightQuantizationParameters,
) -> torch.Tensor:
    """Round-trip one weight column with the qparams used by GPTQ."""
    if settings.dtype is QuantizationDType.FP8:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")
        return column.to(torch.float8_e4m3fn).to(column.dtype)

    args = params.args
    q_min, q_max = calculate_range(args, column.device)

    if args.strategy == QuantizationStrategy.TENSOR:
        scale = params.scale.reshape(1).expand_as(column)
        zero_point = params.zero_point.reshape(1).expand_as(column)
    elif args.strategy == QuantizationStrategy.CHANNEL:
        scale = params.scale[:, 0]
        zero_point = params.zero_point[:, 0]
    elif args.strategy in {
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    }:
        if args.group_size is None:
            raise ValueError("group quantization requires group_size")
        group_index = column_index // args.group_size
        scale = params.scale[:, group_index]
        zero_point = params.zero_point[:, group_index]
    elif args.strategy == QuantizationStrategy.BLOCK:
        if not args.block_structure:
            raise ValueError("block quantization requires block_structure")
        block_h, block_w = args.block_structure
        col_block = column_index // block_w
        row_index = torch.arange(column.numel(), device=column.device)
        row_block = torch.div(row_index, block_h, rounding_mode="floor")
        scale = params.scale[row_block, col_block]
        zero_point = params.zero_point[row_block, col_block]
    else:
        raise ValueError(f"unsupported weight strategy: {args.strategy}")

    q = torch.round(column / scale + zero_point)
    q = torch.clamp(q, float(q_min.item()), float(q_max.item()))
    return (q - zero_point) * scale


def quantize_linear_weight_rtn(
    weight: torch.Tensor,
    settings: ArtifactQuantizationSettings,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply RTN-style fake quantization to a linear weight matrix."""
    quantized, params = quantize_dequantize_weight(weight, settings)
    scale = None
    if settings.dtype is QuantizationDType.INT8:
        scale = params.scale
    return quantized, scale


def apply_rtn_qdq(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    settings: ArtifactQuantizationSettings,
) -> torch.Tensor:
    """Run a linear projection with RTN-quantized weights."""
    qweight, _scale = quantize_linear_weight_rtn(weight, settings)
    return F.linear(inputs, qweight, bias)
