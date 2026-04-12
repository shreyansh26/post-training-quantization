"""GPTQ weight quantization for linear layers."""

import math

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig
from ptq.data import batched_texts
from ptq.quantization.weight_only import (
    WeightQuantizationParameters,
    compute_weight_quantization_parameters,
    quantize_dequantize_weight_column,
)

GPTQ_PRECISION = torch.float32


def _target_linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return the linear layers quantized by the repo's compressed-tensors path."""
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head":
            continue
        if "norm" in name or "embed_tokens" in name:
            continue
        targets.append((name, module))
    return targets


def make_empty_hessian(module: nn.Linear, device: torch.device) -> torch.Tensor:
    """Allocate the Hessian accumulator for one linear layer."""
    columns = module.weight.shape[1]
    return torch.zeros((columns, columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inputs: torch.Tensor,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Accumulate the GPTQ Hessian approximation from a calibration batch."""
    tensor = inputs.to(device=hessian.device, dtype=GPTQ_PRECISION)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    tensor = math.sqrt(2.0) * tensor.t()
    hessian += tensor.matmul(tensor.t())
    return hessian


def collect_gptq_hessians(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> dict[str, torch.Tensor]:
    """Collect per-layer Hessian approximations from calibration inputs."""
    hessians: dict[str, torch.Tensor] = {}
    hooks = []

    for name, module in _target_linear_modules(model):
        hessians[name] = make_empty_hessian(module, device=torch.device(device))

        def pre_hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            *,
            module_name: str = name,
        ) -> None:
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                return
            hessians[module_name] = accumulate_hessian(
                tensor.detach(),
                hessians[module_name],
            )

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

    return hessians


def quantize_linear_weight_gptq(
    module: nn.Linear,
    settings,
    hessian: torch.Tensor,
    damp_percent: float,
    block_size: int,
) -> tuple[torch.Tensor, WeightQuantizationParameters]:
    """Quantize a linear weight matrix with Hessian-aware GPTQ updates."""
    params = compute_weight_quantization_parameters(module.weight.detach(), settings)
    weight = module.weight.detach().to(GPTQ_PRECISION).clone()
    inverse_hessian = hessian.to(weight.device, dtype=GPTQ_PRECISION).clone()

    dead = torch.diag(inverse_hessian) == 0
    inverse_hessian[dead, dead] = 1
    weight[:, dead] = 0

    damp = damp_percent * torch.mean(torch.diag(inverse_hessian))
    diagonal = torch.arange(inverse_hessian.shape[0], device=weight.device)
    inverse_hessian[diagonal, diagonal] += damp

    try:
        inverse_hessian = torch.linalg.cholesky(inverse_hessian)
        inverse_hessian = torch.cholesky_inverse(inverse_hessian)
        inverse_hessian = torch.linalg.cholesky(inverse_hessian, upper=True)
    except torch._C._LinAlgError:
        inverse_hessian = torch.eye(
            weight.shape[1],
            device=weight.device,
            dtype=GPTQ_PRECISION,
        )

    columns = weight.shape[1]
    for block_start in range(0, columns, block_size):
        block_end = min(block_start + block_size, columns)
        count = block_end - block_start

        weight_block = weight[:, block_start:block_end].clone()
        quantized_block = torch.zeros_like(weight_block)
        error_block = torch.zeros_like(weight_block)
        inverse_block = inverse_hessian[block_start:block_end, block_start:block_end]

        for column_offset in range(count):
            column = weight_block[:, column_offset]
            diag_entry = inverse_block[column_offset, column_offset]
            quantized = quantize_dequantize_weight_column(
                column,
                block_start + column_offset,
                settings,
                params,
            )
            quantized_block[:, column_offset] = quantized

            error = (column - quantized) / diag_entry
            weight_block[:, column_offset:] -= error.unsqueeze(1).matmul(
                inverse_block[column_offset, column_offset:].unsqueeze(0)
            )
            error_block[:, column_offset] = error

        weight[:, block_start:block_end] = quantized_block
        if block_end < columns:
            weight[:, block_end:] -= error_block.matmul(
                inverse_hessian[block_start:block_end, block_end:]
            )

    return weight.to(module.weight.dtype), params


def apply_gptq(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    config: PTQRunConfig,
    device: str,
) -> dict[str, tuple[torch.Tensor, WeightQuantizationParameters]]:
    """Run GPTQ over each targeted linear layer and return serialized qparams."""
    hessians = collect_gptq_hessians(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
    )

    quantized: dict[str, tuple[torch.Tensor, WeightQuantizationParameters]] = {}
    sample_count = max(config.calibration.num_samples, 1)
    for name, module in _target_linear_modules(model):
        quantized_weight, params = quantize_linear_weight_gptq(
            module=module,
            settings=config.artifacts.weights,
            hessian=hessians[name] / sample_count,
            damp_percent=config.method.gptq_damp_percent,
            block_size=config.method.gptq_block_size,
        )
        module.weight.data.copy_(quantized_weight.to(module.weight.device))
        quantized[name] = (
            quantized_weight,
            WeightQuantizationParameters(
                scale=params.scale.to(module.weight.device, dtype=module.weight.dtype),
                zero_point=params.zero_point.to(
                    module.weight.device,
                    dtype=module.weight.dtype,
                ),
                args=params.args,
            ),
        )

    return quantized
