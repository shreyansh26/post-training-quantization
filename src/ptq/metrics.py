import csv
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from ptq.config import PTQRunConfig


class MetricRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_name: str
    quantization_artifact: str
    quantization_dtype: str
    quantization_granularity: str
    quantization_method: str
    metric_name: str
    metric_value: float
    artifact_path: str
    calibration_dataset: str
    num_calibration_samples: int

    @classmethod
    def from_config(
        cls,
        config: PTQRunConfig,
        run_id: str,
        metric_name: str,
        metric_value: float,
        artifact_path: str,
    ) -> Self:
        calibration_dataset, num_calibration_samples = config.calibration_metadata()
        return cls(
            run_id=run_id,
            model_name=config.model.model_id,
            quantization_artifact=config.artifact_key(),
            quantization_dtype=config.dtype_key(),
            quantization_granularity=config.granularity_key(),
            quantization_method=config.method_key(),
            metric_name=metric_name,
            metric_value=metric_value,
            artifact_path=artifact_path,
            calibration_dataset=calibration_dataset,
            num_calibration_samples=num_calibration_samples,
        )


def upsert_metric_rows(csv_path: Path, rows: list[MetricRow]) -> None:
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))

    replacement_run_ids = {row.run_id for row in rows}
    retained = [
        row
        for row in existing
        if row["run_id"] not in replacement_run_ids
    ]
    merged = retained + [row.model_dump(mode="json") for row in rows]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MetricRow.model_fields))
        writer.writeheader()
        writer.writerows(merged)


def has_metrics_for_method(
    csv_path: Path,
    *,
    model_name: str,
    method: str,
) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_name") == model_name and row.get(
                "quantization_method"
            ) == method:
                return True
    return False
