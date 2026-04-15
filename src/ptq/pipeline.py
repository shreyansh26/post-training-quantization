import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ptq.compatibility import validate_supported_config
from ptq.config import PTQRunConfig, QuantizationMethod, RunStatus
from ptq.data import load_calibration_texts
from ptq.eval import (
    VLLMEvaluationSettings,
    load_sanity_prompts,
    run_lm_eval_vllm,
    run_vllm_sanity_generation,
)
from ptq.metrics import MetricRow, external_metric_rows, upsert_metric_rows
from ptq.modeling import load_model_and_tokenizer
from ptq.quantization.quantization_pipeline import (
    export_quantized_model,
    prepare_model_for_quantization,
)
from ptq.runs import (
    RunMetadata,
    artifact_dir_for_run,
    build_run_metadata,
    generate_run_id,
    load_run_metadata,
    transition_run_metadata,
    write_run_metadata,
)

DEFAULT_EVAL_OUTPUT_ROOT = Path("/mnt/ssd2/shreyansh/ptq_experiments/evals")
DEFAULT_METRICS_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QuantizeOutcome:
    """Minimal summary returned after artifact export completes."""

    run_id: str
    artifact_dir: Path
    model_ref: str


@dataclass(frozen=True)
class EvaluationOutcome:
    """Minimal summary returned after an evaluation run completes."""

    output_dir: Path
    model_ref: str


@dataclass(frozen=True)
class EvaluationMetadata:
    """Persistent metadata stored alongside each evaluation output."""

    created_at_utc: str
    model_ref: str
    tasks: list[str]
    num_samples: int | None
    output_dir: str
    source_run_id: str | None
    quantization_method: str | None
    quantization_artifact: str | None
    quantization_dtype: str | None
    quantization_granularity: str | None
    vllm: dict[str, str | int | float | bool]


def _configure_gpu_environment(gpu_id: int) -> str:
    """Restrict quantization runs to one explicit physical GPU."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return "cuda:0"


def _apply_output_root_override(
    config: PTQRunConfig,
    output_root: Path | None,
) -> PTQRunConfig:
    """Return a config with an overridden export root when requested."""
    if output_root is None:
        return config
    return config.model_copy(
        update={
            "export": config.export.model_copy(update={"output_root": output_root}),
        }
    )


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
    calibration_texts: list[str] = []
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


def quantize_from_config(
    config: PTQRunConfig,
    output_root: Path | None = None,
) -> QuantizeOutcome:
    """Export a quantized artifact and run the post-export sanity prompts."""
    config = _apply_output_root_override(config, output_root)
    validate_supported_config(config)

    if config.method.name is QuantizationMethod.BASELINE:
        raise ValueError("quantize does not support baseline configs")

    selected_device = _configure_gpu_environment(config.runtime.gpu_id)
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
        model_ref = _export_quantized_artifact(
            config,
            artifact_dir,
            device=selected_device,
        )
        prompts = load_sanity_prompts(config.logging.sanity_prompts_file)
        run_vllm_sanity_generation(
            model_ref=model_ref,
            prompts=prompts,
            settings=VLLMEvaluationSettings(
                trust_remote_code=config.model.trust_remote_code,
                enforce_eager=config.runtime.enforce_eager,
                gpu_memory_utilization=config.runtime.gpu_memory_utilization,
                max_model_len=config.runtime.max_model_len,
                enable_thinking=config.evaluation.lm_eval_enable_thinking,
                max_gen_toks=config.evaluation.lm_eval_max_gen_toks,
            ),
            output_path=artifact_dir / "sanity_outputs.json",
        )
        metadata = transition_run_metadata(
            metadata,
            status=RunStatus.SUCCEEDED,
            model_ref=model_ref,
        )
        write_run_metadata(metadata)
        return QuantizeOutcome(
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


def _resolved_output_dir(model_ref: str, output_dir: Path | None) -> Path:
    """Return the exact evaluation output directory for this invocation."""
    if output_dir is not None:
        return output_dir
    sanitized = model_ref.replace("/", "__").replace(":", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EVAL_OUTPUT_ROOT / sanitized / timestamp


def _resolve_local_metadata(model_ref: str) -> RunMetadata | None:
    """Best-effort recovery of run metadata for exported local artifacts."""
    candidate = Path(model_ref).expanduser()
    if not candidate.exists():
        return None

    search_roots = [candidate]
    search_roots.extend(candidate.parents[:3])
    for root in search_roots:
        metadata_path = root / "run_metadata.json"
        if metadata_path.exists():
            return load_run_metadata(metadata_path)
    return None


def _write_metrics_from_results(
    model_ref: str,
    tasks: list[str],
    num_samples: int | None,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Append per-task metrics using rich metadata when it is recoverable."""
    metadata = _resolve_local_metadata(model_ref)
    if metadata is not None:
        config = PTQRunConfig.model_validate(metadata.config)
        run_id = metadata.run_id
        for task, task_metrics in metrics.items():
            csv_path = DEFAULT_METRICS_DIR / f"metrics_{task}.csv"
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
        return

    payload = json.dumps(
        {
            "model_ref": model_ref,
            "tasks": tasks,
            "num_samples": num_samples,
        },
        sort_keys=True,
    )
    external_run_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    for task, task_metrics in metrics.items():
        csv_path = DEFAULT_METRICS_DIR / f"metrics_{task}.csv"
        rows = external_metric_rows(
            run_id=external_run_id,
            model_ref=model_ref,
            task_metrics=task_metrics,
        )
        upsert_metric_rows(csv_path, rows)


def _write_evaluation_metadata(
    output_dir: Path,
    model_ref: str,
    tasks: list[str],
    num_samples: int | None,
    settings: VLLMEvaluationSettings,
) -> None:
    """Persist the exact evaluation request alongside lm-eval outputs."""
    metadata = _resolve_local_metadata(model_ref)
    config = (
        PTQRunConfig.model_validate(metadata.config)
        if metadata is not None
        else None
    )
    payload = EvaluationMetadata(
        created_at_utc=datetime.now(UTC).isoformat(),
        model_ref=model_ref,
        tasks=tasks,
        num_samples=num_samples,
        output_dir=str(output_dir),
        source_run_id=metadata.run_id if metadata is not None else None,
        quantization_method=(
            config.method_key() if config is not None else "external"
        ),
        quantization_artifact=(
            config.artifact_key() if config is not None else "unknown"
        ),
        quantization_dtype=config.dtype_key() if config is not None else "unknown",
        quantization_granularity=(
            config.granularity_key() if config is not None else "unknown"
        ),
        vllm=asdict(settings),
    )
    (output_dir / "eval_metadata.json").write_text(
        json.dumps(asdict(payload), indent=2),
        encoding="utf-8",
    )


def evaluate_model_ref(
    model_ref: str,
    tasks: list[str],
    num_samples: int | None,
    output_dir: Path | None,
    settings: VLLMEvaluationSettings,
) -> EvaluationOutcome:
    """Run lm-eval for a local artifact or HF model ID and record the results."""
    resolved_output_dir = _resolved_output_dir(model_ref, output_dir)
    metrics, results_json = run_lm_eval_vllm(
        model_ref=model_ref,
        tasks=tasks,
        settings=settings,
        num_samples=num_samples,
        output_dir=resolved_output_dir / "lm_eval",
    )
    (resolved_output_dir / "lm_eval_results.json").write_text(
        results_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_evaluation_metadata(
        output_dir=resolved_output_dir,
        model_ref=model_ref,
        tasks=tasks,
        num_samples=num_samples,
        settings=settings,
    )
    _write_metrics_from_results(
        model_ref=model_ref,
        tasks=tasks,
        num_samples=num_samples,
        metrics=metrics,
    )
    return EvaluationOutcome(
        output_dir=resolved_output_dir,
        model_ref=model_ref,
    )
