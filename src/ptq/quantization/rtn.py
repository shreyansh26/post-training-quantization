import torch
import torch.nn.functional as F

from ptq.config import ArtifactQuantizationSettings, QuantizationDType
from ptq.quantization.primitives import (
    simulate_quantization,
    weight_reduction_axes,
)


def quantize_linear_weight_rtn(
    weight: torch.Tensor,
    settings: ArtifactQuantizationSettings,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply RTN-style fake quantization to a linear weight matrix."""
    axis = None
    if settings.dtype is QuantizationDType.INT8:
        axis = weight_reduction_axes(weight, settings.granularity)
    return simulate_quantization(weight, settings.dtype, settings.granularity, axis)


def apply_rtn_qdq(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    settings: ArtifactQuantizationSettings,
) -> torch.Tensor:
    """Run a linear projection with RTN-quantized weights."""
    qweight, _scale = quantize_linear_weight_rtn(weight, settings)
    return F.linear(inputs, qweight, bias)
