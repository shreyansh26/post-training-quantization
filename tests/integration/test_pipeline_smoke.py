import json
from pathlib import Path

import pytest

from ptq.config import PTQRunConfig
from ptq.eval import VLLMEvaluationSettings
from ptq.pipeline import evaluate_model_ref, quantize_from_config


def _base_config(tmp_path: Path, method: str = "dynamic") -> PTQRunConfig:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello\n", encoding="utf-8")
    quantized = method != "baseline"

    return PTQRunConfig.model_validate(
        {
            "model": {
                "model_id": "Qwen/Qwen3-4B",
                "trust_remote_code": False,
                "chat_template_source": "huggingface",
            },
            "runtime": {
                "backend": "vllm",
                "gpu_id": 0,
                "enforce_eager": True,
            },
            "method": {"name": method},
            "artifacts": {
                "weights": {
                    "enabled": quantized,
                    "dtype": "int8" if quantized else "none",
                    "granularity": "channel" if quantized else "none",
                    "symmetric": True,
                },
                "activations": {
                    "enabled": False,
                    "dtype": "none",
                    "granularity": "none",
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
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "split": "train_sft",
                "num_samples": 1,
                "max_sequence_length": 32,
                "batch_size": 1,
                "seed": 42,
                "shuffle": False,
            },
            "export": {
                "output_root": str(tmp_path / "artifacts"),
                "format": "compressed-tensors",
                "require_vllm_compatibility": True,
                "prefer_hf_compatibility": True,
            },
            "evaluation": {
                "mode": "dev",
                "tasks": ["gsm8k"],
                "use_vllm": True,
                "cache_baseline": False,
                "dev_limit": 2,
            },
            "logging": {
                "metrics_dir": str(tmp_path / "metrics"),
                "sanity_prompts_file": str(prompts),
                "save_lm_eval_raw_json": True,
            },
        }
    )


def test_quantize_from_config_smoke(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    def fake_export(config, artifact_dir, device):
        del config, device
        model_dir = artifact_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def fake_sanity(model_ref, prompts, settings, output_path):
        del prompts, settings
        output_path.write_text(json.dumps([{"model_ref": model_ref}]), encoding="utf-8")

    monkeypatch.setattr("ptq.pipeline._export_quantized_artifact", fake_export)
    monkeypatch.setattr("ptq.pipeline.run_vllm_sanity_generation", fake_sanity)

    outcome = quantize_from_config(config)
    assert Path(outcome.model_ref).exists()
    assert (outcome.artifact_dir / "run_metadata.json").exists()
    assert (outcome.artifact_dir / "sanity_outputs.json").exists()


def test_quantize_rejects_baseline(monkeypatch, tmp_path: Path) -> None:
    del monkeypatch
    config = _base_config(tmp_path, method="baseline")
    with pytest.raises(ValueError, match="does not support baseline"):
        quantize_from_config(config)


def test_evaluate_model_ref_smoke(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ptq.pipeline.DEFAULT_METRICS_DIR", tmp_path / "metrics")

    def fake_eval(model_ref, tasks, settings, num_samples, output_dir):
        del model_ref, tasks, settings, num_samples
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "results.json"
        raw.write_text("{}", encoding="utf-8")
        return {"gsm8k": {"exact_match,strict-match": 0.5}}, raw

    monkeypatch.setattr("ptq.pipeline.run_lm_eval_vllm", fake_eval)

    outcome = evaluate_model_ref(
        model_ref=str(model_dir),
        tasks=["gsm8k"],
        num_samples=2,
        output_dir=tmp_path / "eval",
        settings=VLLMEvaluationSettings(),
    )
    assert (outcome.output_dir / "lm_eval_results.json").exists()
    assert (outcome.output_dir / "eval_metadata.json").exists()
    assert ((tmp_path / "metrics") / "metrics_gsm8k.csv").exists()
