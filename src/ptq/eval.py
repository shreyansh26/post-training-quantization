import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TASK_METRIC_FILTERS: dict[str, set[str]] = {
    "gsm8k": {
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    },
    "ifeval": {
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
    },
    "mmlu": {
        "acc,none",
    },
}


@dataclass(frozen=True)
class VLLMEvaluationSettings:
    """Runtime knobs shared by sanity generation and lm-eval invocations."""

    trust_remote_code: bool = False
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 4096
    enable_thinking: bool = False
    max_gen_toks: int = 512


def load_sanity_prompts(path: Path) -> list[str]:
    """Load the non-empty sanity prompts used for quick qualitative checks."""
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            prompts.append(stripped)
    return prompts


def run_vllm_sanity_generation(
    model_ref: str,
    prompts: list[str],
    settings: VLLMEvaluationSettings,
    output_path: Path,
) -> None:
    """Render chat prompts and persist short deterministic vLLM generations."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=settings.trust_remote_code,
    )
    rendered_prompts: list[str] = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=settings.enable_thinking,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered_prompts.append(rendered)

    llm = LLM(
        model=model_ref,
        trust_remote_code=settings.trust_remote_code,
        dtype="auto",
        enforce_eager=settings.enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        max_model_len=settings.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )
    responses = llm.generate(rendered_prompts, sampling_params=sampling_params)
    data = []
    for prompt, response in zip(prompts, responses, strict=False):
        generated = response.outputs[0].text if response.outputs else ""
        data.append(
            {
                "prompt": prompt,
                "completion": generated,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _find_lm_eval_results_json(path: Path) -> Path:
    """Resolve the final ``lm-eval`` JSON artifact from a file or directory."""
    if path.is_file():
        return path
    candidates = sorted(path.glob("**/*.json"))
    if not candidates:
        raise FileNotFoundError(f"no lm-eval json outputs found under {path}")
    return candidates[-1]


def _collect_numeric_metrics(
    task_name: str,
    task_metrics: dict[str, Any],
) -> dict[str, float]:
    """Keep only the task-level scalar metrics that are useful for reporting."""
    selected: dict[str, float] = {}
    allowed_metrics = TASK_METRIC_FILTERS.get(task_name)
    for key, value in task_metrics.items():
        if not isinstance(value, (float, int)):
            continue
        if allowed_metrics is not None and key not in allowed_metrics:
            continue
        if key.endswith("_stderr"):
            continue
        selected[key] = float(value)
    return selected


def run_lm_eval_vllm(
    model_ref: str,
    tasks: list[str],
    settings: VLLMEvaluationSettings,
    num_samples: int | None,
    output_dir: Path,
) -> tuple[dict[str, dict[str, float]], Path]:
    """Run the configured lm-eval task set against vLLM and collect scalars."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = ",".join(
        [
            f"pretrained={model_ref}",
            f"trust_remote_code={str(settings.trust_remote_code)}",
            "dtype=auto",
            "tensor_parallel_size=1",
            f"enforce_eager={str(settings.enforce_eager)}",
            f"gpu_memory_utilization={settings.gpu_memory_utilization}",
            f"max_model_len={settings.max_model_len}",
            f"enable_thinking={str(settings.enable_thinking)}",
            f"max_gen_toks={settings.max_gen_toks}",
        ]
    )
    cmd = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "auto",
        "--output_path",
        str(output_dir),
        "--apply_chat_template",
    ]
    if num_samples is not None:
        cmd.extend(["--limit", str(num_samples)])

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)

    results_json = _find_lm_eval_results_json(output_dir)
    raw = json.loads(results_json.read_text(encoding="utf-8"))
    task_results = raw.get("results", {})
    metrics: dict[str, dict[str, float]] = {}
    for task in tasks:
        if task in task_results:
            metrics[task] = _collect_numeric_metrics(task, task_results[task])
    return metrics, results_json
