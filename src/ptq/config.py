from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantizationMethod(StrEnum):
    BASELINE = "baseline"
    RTN = "rtn"
    DYNAMIC = "dynamic"
    STATIC = "static"
    SMOOTHQUANT = "smoothquant"
    AWQ = "awq"
    GPTQ = "gptq"


class QuantizationDType(StrEnum):
    NONE = "none"
    INT8 = "int8"
    FP8 = "fp8"


class QuantizationGranularity(StrEnum):
    NONE = "none"
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    TOKEN = "token"
    BLOCK = "block"


class EvaluationMode(StrEnum):
    DEV = "dev"
    FULL = "full"


class RuntimeBackend(StrEnum):
    VLLM = "vllm"


class ExportFormat(StrEnum):
    COMPRESSED_TENSORS = "compressed-tensors"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-4B"
    trust_remote_code: bool = False
    chat_template_source: str = "huggingface"


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: RuntimeBackend = RuntimeBackend.VLLM
    enforce_eager: bool = True
    single_gpu_only: bool = True
    excluded_gpus: list[int] = Field(default_factory=lambda: [6, 7])
    gpu_memory_utilization: float = 0.2
    max_model_len: int = 4096

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeSettings:
        if self.gpu_memory_utilization <= 0.0 or self.gpu_memory_utilization > 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        return self


class MethodSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: QuantizationMethod
    smoothquant_alpha: float = 0.5
    awq_clip_ratio: float = 1.0
    gptq_damp_percent: float = 0.01
    gptq_block_size: int = 128
    activation_ordering: str = "static"

    @model_validator(mode="after")
    def validate_method_params(self) -> MethodSettings:
        if not (0.0 <= self.smoothquant_alpha <= 1.0):
            raise ValueError("smoothquant_alpha must be in [0, 1]")
        if self.awq_clip_ratio <= 0.0:
            raise ValueError("awq_clip_ratio must be positive")
        if self.gptq_damp_percent < 0.0:
            raise ValueError("gptq_damp_percent must be non-negative")
        if self.gptq_block_size <= 0:
            raise ValueError("gptq_block_size must be positive")
        if self.activation_ordering not in {"static", "dynamic"}:
            raise ValueError("activation_ordering must be one of: static, dynamic")
        return self


class ArtifactQuantizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    dtype: QuantizationDType = QuantizationDType.NONE
    granularity: QuantizationGranularity = QuantizationGranularity.NONE
    symmetric: bool = True
    group_size: int | None = None
    block_size: int | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> ArtifactQuantizationSettings:
        if self.enabled:
            if self.dtype is QuantizationDType.NONE:
                raise ValueError("enabled artifacts must declare a non-none dtype")
            if self.granularity is QuantizationGranularity.NONE:
                raise ValueError(
                    "enabled artifacts must declare a non-none granularity"
                )
        else:
            if self.dtype is not QuantizationDType.NONE:
                raise ValueError("disabled artifacts must use dtype=none")
            if self.granularity is not QuantizationGranularity.NONE:
                raise ValueError("disabled artifacts must use granularity=none")
            if self.group_size is not None or self.block_size is not None:
                raise ValueError("disabled artifacts cannot declare group/block sizes")

        if (
            self.granularity is QuantizationGranularity.BLOCK
            and self.block_size is None
        ):
            raise ValueError("block granularity requires block_size")
        if self.granularity is not QuantizationGranularity.BLOCK and self.block_size:
            raise ValueError("block_size is only valid for block granularity")
        if (
            self.granularity is QuantizationGranularity.GROUP
            and self.group_size is None
        ):
            raise ValueError("group granularity requires group_size")
        if self.granularity is not QuantizationGranularity.GROUP and self.group_size:
            raise ValueError("group_size is only valid for group granularity")

        if self.granularity in {
            QuantizationGranularity.CHANNEL,
            QuantizationGranularity.TENSOR,
        } and self.group_size:
            raise ValueError(
                "group_size is only valid for group granularity"
            )

        return self


class ArtifactSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: ArtifactQuantizationSettings = Field(
        default_factory=ArtifactQuantizationSettings
    )
    activations: ArtifactQuantizationSettings = Field(
        default_factory=ArtifactQuantizationSettings
    )
    attention: ArtifactQuantizationSettings = Field(
        default_factory=ArtifactQuantizationSettings
    )
    kv_cache: ArtifactQuantizationSettings = Field(
        default_factory=ArtifactQuantizationSettings
    )


class CalibrationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = "HuggingFaceH4/ultrachat_200k"
    split: str = "train_sft"
    num_samples: int = 256
    max_sequence_length: int = 512
    batch_size: int = 1
    seed: int = 42
    shuffle: bool = False


class ExportSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("/mnt/ssd2/shreyansh/ptq_experiments/artifacts")
    format: ExportFormat = ExportFormat.COMPRESSED_TENSORS
    require_vllm_compatibility: bool = True
    prefer_hf_compatibility: bool = True


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: EvaluationMode = EvaluationMode.DEV
    tasks: list[str] = Field(default_factory=lambda: ["gsm8k", "ifeval", "mmlu"])
    use_vllm: bool = True
    cache_baseline: bool = True
    dev_limit: int = 10
    lm_eval_enable_thinking: bool = False
    lm_eval_max_gen_toks: int = 512

    @model_validator(mode="after")
    def validate_eval(self) -> EvaluationSettings:
        if self.dev_limit <= 0:
            raise ValueError("dev_limit must be positive")
        if not self.tasks:
            raise ValueError("evaluation.tasks cannot be empty")
        if self.lm_eval_max_gen_toks <= 0:
            raise ValueError("lm_eval_max_gen_toks must be positive")
        return self


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics_dir: Path = Path(".")
    sanity_prompts_file: Path = Path("prompts/sanity_prompts.txt")
    save_lm_eval_raw_json: bool = True


class PTQRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings
    runtime: RuntimeSettings
    method: MethodSettings
    artifacts: ArtifactSettings
    calibration: CalibrationSettings
    export: ExportSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings

    @model_validator(mode="after")
    def validate_basic_constraints(self) -> PTQRunConfig:
        enabled_artifacts = [
            name
            for name, artifact in self.artifacts
            if artifact.enabled
        ]

        if self.method.name is QuantizationMethod.BASELINE and enabled_artifacts:
            raise ValueError("baseline runs cannot enable quantization artifacts")

        if (
            self.method.name is not QuantizationMethod.BASELINE
            and not enabled_artifacts
        ):
            raise ValueError("non-baseline runs must enable at least one artifact")

        if self.method.name in {
            QuantizationMethod.AWQ,
            QuantizationMethod.GPTQ,
        } and not self.artifacts.weights.enabled:
            raise ValueError(f"{self.method.name} requires weight quantization")

        if self.method.name in {
            QuantizationMethod.SMOOTHQUANT,
            QuantizationMethod.AWQ,
            QuantizationMethod.GPTQ,
        } and self.calibration.num_samples <= 0:
            raise ValueError(f"{self.method.name} requires calibration samples")

        if (
            self.method.name is QuantizationMethod.STATIC
            and self.artifacts.activations.enabled
            and self.calibration.num_samples <= 0
        ):
            raise ValueError(
                "static activation quantization requires calibration samples"
            )

        return self

    def artifact_key(self) -> str:
        names = [
            name
            for name, artifact in self.artifacts
            if artifact.enabled
        ]
        return "+".join(names) if names else "none"

    def dtype_key(self) -> str:
        values = [
            f"{name}:{artifact.dtype}"
            for name, artifact in self.artifacts
            if artifact.enabled
        ]
        return "+".join(values) if values else "none"

    def requires_calibration_data(self) -> bool:
        if self.method.name in {
            QuantizationMethod.SMOOTHQUANT,
            QuantizationMethod.AWQ,
            QuantizationMethod.GPTQ,
        }:
            return True
        if (
            self.method.name is QuantizationMethod.STATIC
            and self.artifacts.activations.enabled
        ):
            return True
        return False

    def calibration_metadata(self) -> tuple[str, int]:
        if self.requires_calibration_data() and self.calibration.num_samples > 0:
            return self.calibration.dataset, self.calibration.num_samples
        return "none", 0

    def granularity_key(self) -> str:
        values = [
            f"{name}:{artifact.granularity}"
            for name, artifact in self.artifacts
            if artifact.enabled
        ]
        return "+".join(values) if values else "none"


def load_run_config(path: str | Path) -> PTQRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    config = PTQRunConfig.model_validate(raw)

    from ptq.compatibility import validate_supported_config

    validate_supported_config(config)
    return config
