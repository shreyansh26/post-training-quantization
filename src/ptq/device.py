from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceChoice:
    torch_device: str
    cuda_index: int | None


def select_single_device(excluded_gpus: list[int]) -> DeviceChoice:
    """
    Pick one CUDA device with the lowest memory usage, excluding configured IDs.
    Falls back to CPU when no visible GPU can be queried.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DeviceChoice(torch_device="cpu", cuda_index=None)

    rows: list[tuple[int, int, int]] = []
    for raw in csv.reader(result.stdout.strip().splitlines()):
        index = int(raw[0].strip())
        used = int(raw[1].strip())
        total = int(raw[2].strip())
        if index in excluded_gpus:
            continue
        rows.append((index, used, total))

    if not rows:
        return DeviceChoice(torch_device="cpu", cuda_index=None)

    selected = min(rows, key=lambda row: row[1])
    return DeviceChoice(torch_device=f"cuda:{selected[0]}", cuda_index=selected[0])


def configure_single_gpu_environment(choice: DeviceChoice) -> str:
    """
    Restrict the process to the selected physical GPU for vLLM and torch.
    Returns the logical torch device to use after the environment update.
    """
    if choice.cuda_index is None:
        return "cpu"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choice.cuda_index)
    return "cuda:0"
