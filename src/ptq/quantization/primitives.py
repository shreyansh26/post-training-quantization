from __future__ import annotations

import torch

from ptq.config import QuantizationDType, QuantizationGranularity

_EPS = 1e-12


def _normalize_axes(axes: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(sorted(axis if axis >= 0 else ndim + axis for axis in axes))


def _keepdim_amax(tensor: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    resolved = _normalize_axes(axes, tensor.ndim)
    result = tensor.abs()
    for axis in reversed(resolved):
        result = result.amax(dim=axis, keepdim=True)
    return result


def int8_scale(
    tensor: torch.Tensor,
    granularity: QuantizationGranularity,
    axis: int | tuple[int, ...],
) -> torch.Tensor:
    if granularity is QuantizationGranularity.NONE:
        raise ValueError("granularity cannot be none for enabled quantization")
    return (_keepdim_amax(tensor, axis) / 127.0).clamp_min(_EPS)


def qdq_int8(
    tensor: torch.Tensor,
    granularity: QuantizationGranularity,
    axis: int | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = int8_scale(tensor, granularity, axis)
    q = torch.clamp(torch.round(tensor / scale), -127, 127)
    return q * scale, scale


def qdq_fp8(tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build does not expose float8_e4m3fn")
    return tensor.to(torch.float8_e4m3fn).to(tensor.dtype)


def activation_axes(
    tensor: torch.Tensor,
    granularity: QuantizationGranularity,
) -> int | tuple[int, ...]:
    if granularity is QuantizationGranularity.TENSOR:
        return tuple(range(tensor.ndim))
    if granularity is QuantizationGranularity.TOKEN:
        return (-1,)
    raise ValueError(f"unsupported activation granularity: {granularity}")


def weight_axes(
    weight: torch.Tensor,
    granularity: QuantizationGranularity,
) -> int | tuple[int, ...]:
    if weight.ndim != 2:
        raise ValueError("only 2D linear weights are currently supported")
    if granularity is QuantizationGranularity.TENSOR:
        return (0, 1)
    if granularity is QuantizationGranularity.CHANNEL:
        return (1,)
    raise ValueError(f"unsupported weight granularity: {granularity}")


def simulate_quantization(
    tensor: torch.Tensor,
    dtype: QuantizationDType,
    granularity: QuantizationGranularity,
    axis: int | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if dtype is QuantizationDType.INT8:
        if axis is None:
            raise ValueError("int8 simulation requires an axis")
        qdq, scale = qdq_int8(tensor, granularity, axis)
        return qdq, scale
    if dtype is QuantizationDType.FP8:
        return qdq_fp8(tensor), None
    raise ValueError(f"unsupported quantization dtype: {dtype}")

