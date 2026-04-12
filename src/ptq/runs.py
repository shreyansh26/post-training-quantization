from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ptq.config import PTQRunConfig, RunStatus


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at_utc: str
    config_digest: str
    status: RunStatus
    artifact_dir: Path
    config: dict[str, Any]
    error: str | None = None
    model_ref: str | None = None


def normalized_config_dict(config: PTQRunConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def config_digest(config: PTQRunConfig) -> str:
    payload = json.dumps(
        normalized_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def generate_run_id(config: PTQRunConfig, now: datetime | None = None) -> str:
    del now
    return config_digest(config)[:10]


def artifact_dir_for_run(config: PTQRunConfig, run_id: str) -> Path:
    model_name = config.model.model_id.rsplit("/", maxsplit=1)[-1]
    return config.export.output_root / model_name / config.method.name / run_id


def build_run_metadata(
    config: PTQRunConfig,
    run_id: str,
    status: RunStatus,
    error: str | None = None,
    model_ref: str | None = None,
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        config_digest=config_digest(config),
        status=status,
        artifact_dir=artifact_dir_for_run(config, run_id),
        config=normalized_config_dict(config),
        error=error,
        model_ref=model_ref,
    )


def write_run_metadata(metadata: RunMetadata) -> Path:
    metadata.artifact_dir.mkdir(parents=True, exist_ok=True)
    target = metadata.artifact_dir / "run_metadata.json"
    target.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return target


def load_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_run_metadata(output_root: Path, run_id: str) -> RunMetadata:
    pattern = f"*/*/{run_id}/run_metadata.json"
    matches = sorted(output_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"could not resolve run metadata for run_id={run_id} under {output_root}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"run_id={run_id} is ambiguous under {output_root}: "
            f"{[str(path) for path in matches]}"
        )
    return load_run_metadata(matches[0])


def run_id_from_metadata(metadata: RunMetadata) -> str:
    return metadata.run_id


def transition_run_metadata(
    metadata: RunMetadata,
    status: RunStatus,
    *,
    error: str | None = None,
    model_ref: str | None = None,
) -> RunMetadata:
    return metadata.model_copy(
        update={
            "status": status,
            "error": error,
            "model_ref": model_ref if model_ref is not None else metadata.model_ref,
        }
    )
