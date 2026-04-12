from types import SimpleNamespace

import torch

from ptq.config import (
    ArtifactQuantizationSettings,
    PTQRunConfig,
    QuantizationDType,
    QuantizationGranularity,
)
from ptq.quantization.compressed import prepare_model_for_quantization
from ptq.quantization.rtn import quantize_linear_weight_rtn
from ptq.quantization.w8a8 import SimulatedW8A8Linear


def test_rtn_int8_channel_quantization_preserves_shape() -> None:
    weight = torch.randn(8, 16)
    settings = ArtifactQuantizationSettings(
        enabled=True,
        dtype=QuantizationDType.INT8,
        granularity=QuantizationGranularity.CHANNEL,
    )
    quantized, scale = quantize_linear_weight_rtn(weight, settings)

    assert quantized.shape == weight.shape
    assert scale is not None
    assert scale.shape == (8, 1)


def test_simulated_dynamic_w8a8_linear_runs_forward() -> None:
    layer = torch.nn.Linear(16, 8, bias=True)
    wrapper = SimulatedW8A8Linear(
        linear=layer,
        weight_settings=ArtifactQuantizationSettings(
            enabled=True,
            dtype=QuantizationDType.INT8,
            granularity=QuantizationGranularity.CHANNEL,
        ),
        activation_settings=ArtifactQuantizationSettings(
            enabled=True,
            dtype=QuantizationDType.INT8,
            granularity=QuantizationGranularity.TOKEN,
        ),
        method="dynamic",
    )

    inputs = torch.randn(2, 4, 16)
    outputs = wrapper(inputs)

    assert outputs.shape == (2, 4, 8)
    assert torch.isfinite(outputs).all()


class _TinyTokenizer:
    def __call__(
        self,
        batch: list[str],
        return_tensors: str,
        truncation: bool,
        padding: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        encoded = []
        for text in batch:
            ids = [((ord(ch) % 31) + 1) for ch in text][:max_length]
            encoded.append(ids or [1])

        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, 16)
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.lm_head = torch.nn.Linear(16, 32)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        x = self.embed(input_ids)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits, attention_mask=attention_mask)


def _static_fp8_config() -> PTQRunConfig:
    return PTQRunConfig.model_validate(
        {
            "model": {
                "model_id": "dummy",
                "trust_remote_code": False,
                "chat_template_source": "huggingface",
            },
            "runtime": {
                "backend": "vllm",
                "gpu_id": 0,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.2,
                "max_model_len": 128,
            },
            "method": {
                "name": "static",
                "smoothquant_alpha": 0.5,
                "awq_clip_ratio": 1.0,
                "gptq_damp_percent": 0.01,
                "gptq_block_size": 128,
                "activation_ordering": "static",
            },
            "artifacts": {
                "weights": {
                    "enabled": True,
                    "dtype": "fp8",
                    "granularity": "channel",
                    "symmetric": True,
                },
                "activations": {
                    "enabled": True,
                    "dtype": "fp8",
                    "granularity": "tensor",
                    "symmetric": True,
                },
                "attention": {
                    "enabled": False,
                    "dtype": "none",
                    "granularity": "none",
                    "symmetric": True,
                },
                "kv_cache": {
                    "enabled": False,
                    "dtype": "none",
                    "granularity": "none",
                    "symmetric": True,
                },
            },
            "calibration": {
                "dataset": "dummy",
                "split": "train",
                "num_samples": 2,
                "max_sequence_length": 32,
                "batch_size": 1,
                "seed": 42,
                "shuffle": False,
            },
            "export": {
                "output_root": "/tmp",
                "format": "compressed-tensors",
                "require_vllm_compatibility": True,
                "prefer_hf_compatibility": True,
            },
            "evaluation": {
                "mode": "dev",
                "tasks": ["gsm8k"],
                "use_vllm": True,
                "cache_baseline": False,
                "dev_limit": 1,
                "lm_eval_enable_thinking": False,
                "lm_eval_max_gen_toks": 32,
            },
            "logging": {
                "metrics_dir": ".",
                "sanity_prompts_file": "prompts/sanity_prompts.txt",
                "save_lm_eval_raw_json": True,
            },
        }
    )


def test_prepare_static_fp8_quantization_keeps_input_scales_finite() -> None:
    model = _TinyLM().eval()
    tokenizer = _TinyTokenizer()
    config = _static_fp8_config()

    prepare_model_for_quantization(
        model=model,
        tokenizer=tokenizer,
        config=config,
        calibration_texts=["hello", "quantization"],
        device="cpu",
    )

    input_scales = [
        module.input_scale.detach()
        for module in model.modules()
        if hasattr(module, "input_scale") and module.input_scale is not None
    ]
    assert input_scales
    assert all(torch.isfinite(scale).all() for scale in input_scales)
