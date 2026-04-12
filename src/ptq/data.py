from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ptq.config import CalibrationSettings


def _to_chat_text(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    if "messages" in sample and sample["messages"] is not None:
        messages = sample["messages"]
        if isinstance(messages, list):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    for key in ("text", "prompt", "instruction", "question"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return str(sample)


def load_calibration_texts(
    calibration: CalibrationSettings,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    split_expr = f"{calibration.split}[:{calibration.num_samples}]"
    dataset = load_dataset(calibration.dataset, split=split_expr)
    if calibration.shuffle:
        dataset = dataset.shuffle(seed=calibration.seed)

    return [_to_chat_text(sample, tokenizer) for sample in dataset]


def batched_texts(texts: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]
