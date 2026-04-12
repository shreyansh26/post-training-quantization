import json
from pathlib import Path

from ptq.config import PTQRunConfig
from ptq.pipeline import run_pipeline


def _base_config(tmp_path: Path, method: str = "baseline") -> PTQRunConfig:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello\n", encoding="utf-8")

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
                    "enabled": method != "baseline",
                    "dtype": "int8" if method != "baseline" else "none",
                    "granularity": "channel" if method != "baseline" else "none",
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


def test_run_pipeline_baseline_smoke(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path, method="baseline")

    def fake_sanity(model_ref, prompts, config, output_path):
        output_path.write_text(json.dumps([{"model_ref": model_ref}]), encoding="utf-8")

    def fake_eval(model_ref, config, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "results.json"
        raw.write_text("{}", encoding="utf-8")
        return {"gsm8k": {"exact_match": 0.5}}, raw

    monkeypatch.setattr("ptq.pipeline.run_vllm_sanity_generation", fake_sanity)
    monkeypatch.setattr("ptq.pipeline.run_lm_eval_vllm", fake_eval)

    outcome = run_pipeline(config, ensure_baseline=False)
    metadata_file = outcome.artifact_dir / "run_metadata.json"
    assert metadata_file.exists()
    assert (config.logging.metrics_dir / "metrics_gsm8k.csv").exists()


def test_run_pipeline_quantized_smoke(monkeypatch, tmp_path: Path) -> None:
    config = _base_config(tmp_path, method="rtn")

    def fake_export(config, artifact_dir, device):
        model_dir = artifact_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def fake_sanity(model_ref, prompts, config, output_path):
        output_path.write_text(json.dumps([{"model_ref": model_ref}]), encoding="utf-8")

    def fake_eval(model_ref, config, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "results.json"
        raw.write_text("{}", encoding="utf-8")
        return {"gsm8k": {"exact_match": 0.4}}, raw

    monkeypatch.setattr("ptq.pipeline._export_quantized_artifact", fake_export)
    monkeypatch.setattr("ptq.pipeline.run_vllm_sanity_generation", fake_sanity)
    monkeypatch.setattr("ptq.pipeline.run_lm_eval_vllm", fake_eval)

    outcome = run_pipeline(config, ensure_baseline=False)
    assert Path(outcome.model_ref).exists()
    assert (outcome.artifact_dir / "run_metadata.json").exists()
