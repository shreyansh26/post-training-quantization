"""Production quantization pipeline and artifact export entrypoints."""

from pathlib import Path

import torch
from compressed_tensors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from transformers import PreTrainedTokenizerBase

from ptq.config import PTQRunConfig
from ptq.quantization.calibration_qparams import (
    calibrate_static_activation_parameters,
    collect_attention_statistics,
    populate_static_attention_parameters,
    populate_weight_quantization_parameters,
)
from ptq.quantization.method_preparation import prepare_method_for_quantization
from ptq.quantization.quantization_scheme import build_quantization_config


def prepare_model_for_quantization(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: PTQRunConfig,
    calibration_texts: list[str],
    device: str,
) -> QuantizationConfig:
    """Run method preprocessing, then instrument and populate artifact qparams."""
    method_result = prepare_method_for_quantization(
        model=model,
        tokenizer=tokenizer,
        config=config,
        calibration_texts=calibration_texts,
        device=device,
    )
    attention_stats = collect_attention_statistics(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
    )
    quant_config = build_quantization_config(config, model=model)
    # This is the main in-memory mutation step for plain PTQ W8A8. After this,
    # compressed-tensors has attached the quantization scheme and the required
    # qparam buffers to the matching modules.
    apply_quantization_config(model, quant_config)
    populate_weight_quantization_parameters(
        model=model,
        gptq_parameters=method_result.gptq_parameters or None,
    )
    calibrate_static_activation_parameters(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        config=config,
        device=device,
    )
    populate_static_attention_parameters(
        model=model,
        attention_stats=attention_stats,
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
