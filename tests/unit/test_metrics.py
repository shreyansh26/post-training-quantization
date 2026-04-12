import csv
from pathlib import Path

from ptq.config import PTQRunConfig
from ptq.metrics import MetricRow, upsert_metric_rows


def _config() -> PTQRunConfig:
    return PTQRunConfig.model_validate(
        {
            "model": {
                "model_id": "Qwen/Qwen3-4B",
                "trust_remote_code": False,
                "chat_template_source": "huggingface",
            },
            "runtime": {
                "backend": "vllm",
                "enforce_eager": True,
                "single_gpu_only": True,
                "excluded_gpus": [6, 7],
            },
            "method": {"name": "dynamic"},
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
                    "granularity": "token",
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
                "num_samples": 256,
                "max_sequence_length": 512,
                "seed": 42,
                "shuffle": False,
            },
            "export": {
                "output_root": "/mnt/ssd2/shreyansh/ptq_experiments/artifacts",
                "format": "compressed-tensors",
                "require_vllm_compatibility": True,
                "prefer_hf_compatibility": True,
            },
            "evaluation": {
                "mode": "dev",
                "tasks": ["gsm8k", "ifeval", "mmlu"],
                "use_vllm": True,
                "cache_baseline": True,
            },
            "logging": {
                "metrics_dir": ".",
                "sanity_prompts_file": "prompts/sanity_prompts.txt",
            },
        }
    )


def test_upsert_metric_rows_is_idempotent(tmp_path: Path) -> None:
    config = _config()
    csv_path = tmp_path / "metrics_gsm8k.csv"

    first = MetricRow.from_config(
        config=config,
        run_id="run-1",
        metric_name="exact_match",
        metric_value=0.5,
        artifact_path="/tmp/artifact",
    )
    another_metric = MetricRow.from_config(
        config=config,
        run_id="run-1",
        metric_name="f1",
        metric_value=0.6,
        artifact_path="/tmp/artifact",
    )
    replacement = MetricRow.from_config(
        config=config,
        run_id="run-1",
        metric_name="exact_match",
        metric_value=0.75,
        artifact_path="/tmp/artifact",
    )

    upsert_metric_rows(csv_path, [first, another_metric])
    upsert_metric_rows(csv_path, [replacement])

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["metric_value"] == "0.75"
    assert rows[0]["calibration_dataset"] == "none"
    assert rows[0]["num_calibration_samples"] == "0"
