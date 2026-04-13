"""Method-specific preparation before general quantization/export.

Artifact serialization is intentionally handled elsewhere. This module only
decides which method-specific preprocessing step should run before artifact
quantization/export.
"""

from dataclasses import dataclass, field

from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig, QuantizationMethod
from ptq.quantization.awq import apply_awq
from ptq.quantization.gptq import apply_gptq
from ptq.quantization.smoothquant import apply_smoothquant
from ptq.quantization.weight_only import WeightQuantizationParameters


@dataclass
class MethodPreparationResult:
    """Method-specific artifacts produced before general export instrumentation."""

    gptq_parameters: dict[str, WeightQuantizationParameters] = field(
        default_factory=dict
    )


def prepare_method_for_quantization(
    model,
    tokenizer: PreTrainedTokenizerBase,
    config: PTQRunConfig,
    calibration_texts: list[str],
    device: str,
) -> MethodPreparationResult:
    """Run the method-specific preprocessing step selected by the config."""
    if config.smoothquant_enabled():
        apply_smoothquant(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            config=config,
            device=device,
        )

    if config.method.name is QuantizationMethod.AWQ:
        apply_awq(
            model=model,
            tokenizer=tokenizer,
            calibration_texts=calibration_texts,
            config=config,
            device=device,
        )
        return MethodPreparationResult()

    if config.method.name is QuantizationMethod.GPTQ:
        gptq_parameters = {
            name: params
            for name, (_quantized_weight, params) in apply_gptq(
                model=model,
                tokenizer=tokenizer,
                calibration_texts=calibration_texts,
                config=config,
                device=device,
            ).items()
        }
        return MethodPreparationResult(gptq_parameters=gptq_parameters)

    return MethodPreparationResult()
