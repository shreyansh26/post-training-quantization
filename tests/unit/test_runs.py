from pathlib import Path

from ptq.config import PTQRunConfig, RunStatus
from ptq.runs import (
    artifact_dir_for_run,
    build_run_metadata,
    generate_run_id,
    resolve_run_metadata,
    write_run_metadata,
)


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
            "method": {"name": "rtn"},
            "artifacts": {
                "weights": {
                    "enabled": True,
                    "dtype": "int8",
                    "granularity": "tensor",
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


def test_run_id_is_short_hex_digest() -> None:
    config = _config()

    run_id = generate_run_id(config)

    assert len(run_id) == 10
    int(run_id, 16)
    assert run_id == generate_run_id(config)


def test_artifact_directory_uses_model_and_method() -> None:
    config = _config()
    run_id = "deadbeefaa"
    artifact_dir = artifact_dir_for_run(config, run_id)

    assert artifact_dir.as_posix().endswith(
        "/Qwen3-4B/rtn/deadbeefaa"
    )


def test_resolve_run_metadata_by_run_id(tmp_path: Path) -> None:
    config = _config()
    config = config.model_copy(
        update={
            "export": config.export.model_copy(update={"output_root": tmp_path}),
        }
    )
    metadata = build_run_metadata(
        config=config,
        run_id="abcde12345",
        status=RunStatus.PENDING,
    )
    write_run_metadata(metadata)

    resolved = resolve_run_metadata(tmp_path, "abcde12345")

    assert resolved.run_id == "abcde12345"
