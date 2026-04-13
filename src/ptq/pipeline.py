import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ptq.compatibility import validate_supported_config
from ptq.config import (
    PTQRunConfig,
    QuantizationDType,
    QuantizationGranularity,
    QuantizationMethod,
    RunStatus,
)
from ptq.data import load_calibration_texts
from ptq.eval import (
    load_sanity_prompts,
    run_lm_eval_vllm,
    run_vllm_sanity_generation,
)
from ptq.metrics import MetricRow, has_metrics_for_method, upsert_metric_rows
from ptq.modeling import load_model_and_tokenizer
from ptq.quantization.quantization_pipeline import (
    export_quantized_model,
    prepare_model_for_quantization,
)
from ptq.runs import (
    artifact_dir_for_run,
    build_run_metadata,
    generate_run_id,
    transition_run_metadata,
    write_run_metadata,
)


@dataclass(frozen=True)
class RunOutcome:
    """Minimal summary returned after a pipeline run completes."""

    run_id: str
    artifact_dir: Path
    model_ref: str


def _configure_gpu_environment(gpu_id: int) -> str:
    """Restrict the process to one explicit physical GPU for torch and vLLM."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return "cuda:0"


def _baseline_config(config: PTQRunConfig) -> PTQRunConfig:
    """Clone a config into the dense baseline variant used for cached comparisons."""
    disabled = {
        "enabled": False,
        "dtype": QuantizationDType.NONE,
        "granularity": QuantizationGranularity.NONE,
        "symmetric": True,
    }
    return config.model_copy(
        update={
            "method": config.method.model_copy(
                update={"name": QuantizationMethod.BASELINE}
            ),
            "artifacts": config.artifacts.model_copy(
                update={
                    "weights": config.artifacts.weights.model_copy(update=disabled),
                    "activations": config.artifacts.activations.model_copy(
                        update=disabled
                    ),
                    "attention": config.artifacts.attention.model_copy(update=disabled),
                    "kv_cache": config.artifacts.kv_cache.model_copy(update=disabled),
                }
            ),
        }
    )


def _write_metrics(
    config: PTQRunConfig,
    run_id: str,
    model_ref: str,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Append or update per-task metric CSV rows for a completed run."""
    for task, task_metrics in metrics.items():
        csv_path = config.logging.metrics_dir / f"metrics_{task}.csv"
        rows = [
            MetricRow.from_config(
                config=config,
                run_id=run_id,
                metric_name=metric_name,
                metric_value=metric_value,
                artifact_path=model_ref,
            )
            for metric_name, metric_value in task_metrics.items()
        ]
        upsert_metric_rows(csv_path, rows)


def _run_eval_and_sanity(
    config: PTQRunConfig,
    run_id: str,
    artifact_dir: Path,
    model_ref: str,
) -> None:
    """Run the qualitative sanity pass and the lm-eval task suite."""
    prompts = load_sanity_prompts(config.logging.sanity_prompts_file)
    sanity_out = artifact_dir / "sanity_outputs.json"
    run_vllm_sanity_generation(
        model_ref=model_ref,
        prompts=prompts,
        config=config,
        output_path=sanity_out,
    )
    lm_eval_out = artifact_dir / "lm_eval"
    metrics, results_json = run_lm_eval_vllm(
        model_ref=model_ref,
        config=config,
        output_dir=lm_eval_out,
    )
    if config.logging.save_lm_eval_raw_json:
        shutil.copy2(results_json, artifact_dir / "lm_eval_results.json")
    _write_metrics(config=config, run_id=run_id, model_ref=model_ref, metrics=metrics)


def _ensure_baseline_if_needed(config: PTQRunConfig) -> None:
    """Materialize a cached baseline once per model/task combination."""
    if (
        config.method.name is QuantizationMethod.BASELINE
        or not config.evaluation.cache_baseline
    ):
        return

    missing = False
    for task in config.evaluation.tasks:
        csv_path = config.logging.metrics_dir / f"metrics_{task}.csv"
        if not has_metrics_for_method(
            csv_path,
            model_name=config.model.model_id,
            method=QuantizationMethod.BASELINE.value,
        ):
            missing = True
            break

    if not missing:
        return

    baseline_config = _baseline_config(config)
    run_pipeline(baseline_config, ensure_baseline=False)


def _export_quantized_artifact(
    config: PTQRunConfig,
    artifact_dir: Path,
    device: str,
) -> str:
    """Load, calibrate, quantize, and export the requested model artifact."""
    model, tokenizer = load_model_and_tokenizer(
        config.model,
        device=device,
    )
    calibration_texts = []
    if config.requires_calibration_data() and config.calibration.num_samples > 0:
        calibration_texts = load_calibration_texts(config.calibration, tokenizer)

    quant_config = prepare_model_for_quantization(
        model=model,
        tokenizer=tokenizer,
        config=config,
        calibration_texts=calibration_texts,
        device=device,
    )
    model_dir = artifact_dir / "model"
    export_quantized_model(
        model=model,
        tokenizer=tokenizer,
        save_dir=model_dir,
        quantization_format=quant_config.format,
    )
    return str(model_dir)


def run_pipeline(config: PTQRunConfig, ensure_baseline: bool = True) -> RunOutcome:
    """Execute the full PTQ pipeline for one config on a single selected device."""
    validate_supported_config(config)

    selected_device = _configure_gpu_environment(config.runtime.gpu_id)

    if ensure_baseline:
        _ensure_baseline_if_needed(config)

    run_id = generate_run_id(config)
    artifact_dir = artifact_dir_for_run(config, run_id)
    metadata = build_run_metadata(
        config=config,
        run_id=run_id,
        status=RunStatus.PENDING,
    )
    write_run_metadata(metadata)
    metadata = transition_run_metadata(metadata, status=RunStatus.RUNNING)
    write_run_metadata(metadata)

    try:
        # Baseline runs evaluate the original HF model directly, while all
        # quantized methods materialize an exported artifact first.
        if config.method.name is QuantizationMethod.BASELINE:
            model_ref = config.model.model_id
        else:
            model_ref = _export_quantized_artifact(
                config,
                artifact_dir,
                device=selected_device,
            )

        _run_eval_and_sanity(
            config=config,
            run_id=run_id,
            artifact_dir=artifact_dir,
            model_ref=model_ref,
        )

        metadata = transition_run_metadata(
            metadata,
            status=RunStatus.SUCCEEDED,
            model_ref=model_ref,
        )
        write_run_metadata(metadata)
        return RunOutcome(
            run_id=run_id,
            artifact_dir=artifact_dir,
            model_ref=model_ref,
        )
    except Exception as exc:
        metadata = transition_run_metadata(
            metadata,
            status=RunStatus.FAILED,
            error=str(exc),
        )
        write_run_metadata(metadata)
        raise
