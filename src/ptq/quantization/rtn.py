from __future__ import annotations

import torch
import torch.nn.functional as F

from ptq.config import ArtifactQuantizationSettings, QuantizationDType
from ptq.quantization.primitives import simulate_quantization, weight_axes


def quantize_linear_weight_rtn(
    weight: torch.Tensor,
    settings: ArtifactQuantizationSettings,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    axis = None
    if settings.dtype is QuantizationDType.INT8:
        axis = weight_axes(weight, settings.granularity)
    return simulate_quantization(weight, settings.dtype, settings.granularity, axis)


def apply_rtn_qdq(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    settings: ArtifactQuantizationSettings,
) -> torch.Tensor:
    qweight, _scale = quantize_linear_weight_rtn(weight, settings)
    return F.linear(inputs, qweight, bias)
