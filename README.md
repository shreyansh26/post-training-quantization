# Post-Training Quantization

Lightweight, educational post-training quantization experiments for `Qwen/Qwen3-4B`, aligned with `llm-compressor` and `vLLM` behavior where practical, but implemented locally in a smaller codebase.

The project focuses on:

- explicit YAML configs
- clean, readable PTQ passes
- `compressed-tensors` export
- `vLLM` inference compatibility
- evaluation with `lm-eval`

## Status

Implemented and validated in this repo:

- baseline inference
- dynamic W8A8 (`fp8`, `int8`)
- static W8A8 (`fp8`, `int8`)
- SmoothQuant (`fp8`, `int8`)
- AWQ (`int8`)
- GPTQ (`int8`)
- FP8 block dynamic W8A8 (`weights:block`, `activations:group`)

Current scope:

- target modules: `Linear` layers only
- runtime: `vLLM`
- model: `Qwen/Qwen3-4B`

## Repo Layout

- [src/ptq](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq) core library
- [configs](/mnt/ssd1/shreyansh/home_dir/ptq/configs) explicit YAML configs
- [tests](/mnt/ssd1/shreyansh/home_dir/ptq/tests) unit and integration tests
- [prompts](/mnt/ssd1/shreyansh/home_dir/ptq/prompts) sanity prompts
- [metrics_gsm8k.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_gsm8k.csv), [metrics_ifeval.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_ifeval.csv), [metrics_mmlu.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_mmlu.csv) evaluation logs
- `/mnt/ssd2/shreyansh/ptq_experiments/artifacts` exported artifacts

## Setup

```bash
uv sync
```

Useful checks:

```bash
uv run ruff check src tests
uv run pytest tests -q
```

## How Runs Work

Every run is driven by one explicit YAML config.

Standard workflow:

```bash
uv run ptq validate-config <config.yaml>
uv run ptq run <config.yaml>
```

The config controls:

- quantization method
- enabled artifacts
- dtype and granularity
- calibration dataset and sample count
- export settings
- evaluation tasks
- GPU selection

Before running, edit `runtime.gpu_id` in the YAML to a free GPU on your machine.

## Best Validated Configs

These are the canonical configs for the strongest validated run per method family in this repo.

### Baseline

```bash
uv run ptq validate-config configs/example_baseline_qwen3.yaml
uv run ptq run configs/example_baseline_qwen3.yaml
```

### Dynamic W8A8 FP8

This is the best complete dynamic W8A8 config currently validated across `gsm8k`, `ifeval`, and `mmlu`.

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_fp8_qwen3.yaml
uv run ptq run configs/generated/eval_dynamic_w8a8_fp8_qwen3.yaml
```

Reference dev run:

- `run_id: a4583c303f`
- `gsm8k flexible-extract: 0.8`
- `ifeval prompt_level_strict_acc: 0.8`
- `mmlu acc: 0.5825`

### Dynamic W8A8 FP8 Block

This is the `llm-compressor`-style block path:

- weights: `fp8`, `block`
- activations: `fp8`, `group`

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_fp8_block_qwen3.yaml
uv run ptq run configs/generated/eval_dynamic_w8a8_fp8_block_qwen3.yaml
```

Reference dev run:

- `run_id: e8fd400e60`
- `gsm8k flexible-extract: 0.6`
- `ifeval prompt_level_strict_acc: 0.8`
- `mmlu acc: 0.5351`

### Dynamic W8A8 INT8

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_int8_qwen3.yaml
uv run ptq run configs/generated/eval_dynamic_w8a8_int8_qwen3.yaml
```

Reference dev run:

- `run_id: a84ae91456`
- `gsm8k flexible-extract: 0.9`
- `ifeval prompt_level_strict_acc: 0.8`
- `mmlu acc: 0.5632`

### Static W8A8 FP8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_fp8_qwen3.yaml
uv run ptq run configs/generated/eval_static_w8a8_fp8_qwen3.yaml
```

Reference dev run:

- `run_id: 0aadee2984`
- `gsm8k flexible-extract: 1.0`
- `ifeval prompt_level_strict_acc: 0.7`
- `mmlu acc: 0.5667`

### Static W8A8 INT8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_int8_qwen3.yaml
uv run ptq run configs/generated/eval_static_w8a8_int8_qwen3.yaml
```

Important note:

- this path is implemented correctly and exports a valid model
- outputs are coherent English, not gibberish
- quality is materially worse than FP8 static in current form

Reference dev run:

- `run_id: c9a784c317`
- `gsm8k flexible-extract: 0.0`
- `ifeval prompt_level_strict_acc: 0.1`
- `mmlu acc: 0.2649`

### SmoothQuant FP8

```bash
uv run ptq validate-config configs/example_smoothquant_w8a8_fp8.yaml
uv run ptq run configs/example_smoothquant_w8a8_fp8.yaml
```

Reference smoke run:

- `run_id: d7e2456df1`
- coherent sanity outputs
- `gsm8k flexible-extract: 1.0` on `limit=2`

### SmoothQuant INT8

```bash
uv run ptq validate-config configs/example_smoothquant_w8a8_int8.yaml
uv run ptq run configs/example_smoothquant_w8a8_int8.yaml
```

Reference smoke run:

- `run_id: c6fb9c3fcf`
- coherent sanity outputs
- `gsm8k flexible-extract: 0.5` on `limit=2`

### AWQ INT8

AWQ is currently validated as a weight-only method in this repo.

```bash
uv run ptq validate-config configs/example_awq_w8a8_int8.yaml
uv run ptq run configs/example_awq_w8a8_int8.yaml
```

Reference smoke run:

- `run_id: 665350b60d`
- coherent sanity outputs
- `gsm8k flexible-extract: 1.0` on `limit=2`

### GPTQ INT8

GPTQ is currently validated as a weight-only method in this repo.

```bash
uv run ptq validate-config configs/example_gptq_w8a8_int8.yaml
uv run ptq run configs/example_gptq_w8a8_int8.yaml
```

Reference smoke run:

- `run_id: 6d4cbf3646`
- coherent sanity outputs
- `gsm8k flexible-extract: 0.5` on `limit=2`

## FP8 AWQ and GPTQ Issue

The repo contains FP8 configs for AWQ and GPTQ:

- `configs/example_awq_w8a8_fp8.yaml`
- `configs/example_gptq_w8a8_fp8.yaml`

You can run them with:

```bash
uv run ptq validate-config configs/example_awq_w8a8_fp8.yaml
uv run ptq run configs/example_awq_w8a8_fp8.yaml

uv run ptq validate-config configs/example_gptq_w8a8_fp8.yaml
uv run ptq run configs/example_gptq_w8a8_fp8.yaml
```

Current state:

- the repo-side quantization passes complete
- the artifacts export successfully
- `vLLM` reaches weight-only FP8 startup
- inference then fails inside the shared Marlin FP8 path with:

```text
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

This failure reproduces for both AWQ FP8 and GPTQ FP8, which strongly suggests an environment or kernel compatibility issue in the current `vLLM` FP8 weight-only path rather than a method-specific bug in the AWQ or GPTQ logic in this repo.

Reference failing runs:

- AWQ FP8: `run_id: f1fb8caca0`
- GPTQ FP8: `run_id: e96ebba238`

## Quick Smoke Runs

For fast validation, use the `smoke_*.yaml` configs in [configs](/mnt/ssd1/shreyansh/home_dir/ptq/configs:1). They keep the same quantization structures but shorten evaluation.

Examples:

```bash
uv run ptq run configs/smoke_dynamic_w8a8_fp8_qwen3.yaml
uv run ptq run configs/smoke_dynamic_w8a8_int8_qwen3.yaml
uv run ptq run configs/smoke_static_w8a8_fp8_qwen3.yaml
uv run ptq run configs/smoke_static_w8a8_int8_qwen3.yaml
```

## Metrics and Artifacts

Artifacts are written under:

```text
/mnt/ssd2/shreyansh/ptq_experiments/artifacts/<model>/<method>/<run_id>/
```

Each run directory contains:

- exported model
- tokenizer artifacts
- `run_metadata.json`
- `sanity_outputs.json`
- evaluation outputs

Metrics are appended or updated in:

- [metrics_gsm8k.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_gsm8k.csv:1)
- [metrics_ifeval.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_ifeval.csv:1)
- [metrics_mmlu.csv](/mnt/ssd1/shreyansh/home_dir/ptq/metrics_mmlu.csv:1)

## Notes

- calibration uses `HuggingFaceH4/ultrachat_200k`
- dynamic methods do not use calibration samples
- static activation quantization requires calibration
- chat templating follows the official Hugging Face Qwen template path used by the repo
- the hard inference requirement is `vLLM`
