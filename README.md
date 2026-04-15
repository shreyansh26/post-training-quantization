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
- dynamic KV-cache quantization (`fp8`)
- static attention + KV-cache quantization (`fp8`)
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

## Quantization Module Guide

The quantization code is intentionally split by responsibility rather than by
"one file per algorithm family."

- [src/ptq/quantization/quantization_pipeline.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/quantization_pipeline.py:1)
  Thin orchestration layer for the production quantization/export path. It runs
  method preparation, collects calibration stats, applies the
  `compressed-tensors` instrumentation, populates frozen qparams, and exports an
  artifact that HF and `vLLM` can load.
- [src/ptq/quantization/quantization_scheme.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/quantization_scheme.py:1)
  Builds artifact-level quantization configs for weights, activations,
  attention, and KV cache. This is where the orthogonal artifact combinations
  are assembled from YAML config.
- [src/ptq/quantization/calibration_qparams.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/calibration_qparams.py:1)
  Collects calibration statistics and populates serialized scales / zero-points
  for the enabled artifacts after instrumentation.
- [src/ptq/quantization/method_preparation.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/method_preparation.py:1)
  Dispatches method-specific preprocessing before the general artifact export
  path. This is where `smoothquant`, `awq`, and `gptq` hook into the pipeline.
- [src/ptq/quantization/weight_only.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/weight_only.py:1)
  Shared weight-quantization math used by RTN-style helpers and weight-only
  methods like AWQ and GPTQ.
- [src/ptq/quantization/smoothquant.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/smoothquant.py:1)
  SmoothQuant-specific balancing logic that folds activation scaling into the
  upstream normalization / downstream linear weights.
- [src/ptq/quantization/awq.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/awq.py:1)
  AWQ-specific calibration and scaling search for weight-only quantization.
- [src/ptq/quantization/gptq.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/gptq.py:1)
  GPTQ-specific Hessian accumulation and blockwise weight updates.
- [src/ptq/quantization/simulated_w8a8_linear.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/simulated_w8a8_linear.py:1)
  Simulation-only fake-quant linear wrappers used by tests and local validation.
  This is not the production export path.
- [src/ptq/quantization/primitives.py](/mnt/ssd1/shreyansh/home_dir/ptq/src/ptq/quantization/primitives.py:1)
  Small tensor-level fake-quant helpers used by the simulation layer.

## Quantization Flow

For a production quantized run, the main flow is:

1. `pipeline.py`
   Loads the dense HF model, determines whether calibration data is needed, and
   calls into the quantization pipeline.
2. `method_preparation.py`
   Runs any method-specific preprocessing before general quantization. For
   `dynamic` and `static` W8A8 this is a no-op. For `smoothquant`, `awq`, and
   `gptq`, this is where those method-specific transformations happen.
3. `quantization_scheme.py`
   Converts the YAML artifact settings into the `compressed-tensors`
   `QuantizationConfig`. This is where weights, activations, attention, and KV
   cache are combined orthogonally.
4. `quantization_pipeline.py`
   Applies the quantization scheme to the in-memory model with
   `apply_quantization_config(...)`.
5. `calibration_qparams.py`
   Computes and writes the frozen qparams needed by the exported artifact:
   weight scales / zero-points, and for static activation quantization,
   `input_scale` / `input_zero_point`. For static attention and KV-cache
   quantization, this is also where `q_scale`, `k_scale`, and `v_scale` are
   calibrated and serialized.
6. `quantization_pipeline.py`
   Exports the instrumented model through `ModelCompressor.compress(...)` into a
   HF- and `vLLM`-loadable artifact.

For dynamic W8A8 specifically:

- weights are quantized offline during export
- activations are marked as dynamically quantized in the exported scheme
- `vLLM` performs the actual dynamic activation quantization at inference time

For static W8A8 specifically:

- weights are quantized offline during export
- activation qparams are computed from calibration data and exported per layer
- `vLLM` reads those frozen activation qparams at inference time

For attention and KV cache in the current runtime stack:

- dynamic KV-cache quantization is supported and composes cleanly with weight and
  activation quantization
- static attention quantization is calibrated offline and exported through
  `q_scale`, `k_scale`, and `v_scale`
- `vLLM` currently couples attention quantization with KV-cache quantization, so
  the repo requires matching `attention` and `kv_cache` settings when both are
  enabled
- dynamic attention quantization is not exposed because the installed
  `compressed-tensors` + `vLLM` path does not support it cleanly

## Compatibility Matrix

The YAML surface treats quantization artifacts separately, but the actual
runtime contract is slightly stricter than a fully orthogonal matrix.

| Artifact | Dynamic | Static-like (`static`, `smoothquant`, `awq`, `gptq`) | Supported granularity in this repo | Notes |
|---|---|---|---|---|
| weights | yes | yes | `tensor`, `channel`, `group`, `block` | `block` is FP8-only and requires grouped dynamic activations for W8A8 |
| activations | yes | yes | dynamic: `token`, `group`; static-like: `tensor` | static `token` and static `group` are rejected |
| attention | no | yes | `tensor` | must be enabled together with `kv_cache` using identical settings |
| kv_cache | yes | yes | `tensor` | vLLM also has an internal `attn_head` mode, but this repo does not expose it yet |

Practical combinations:

- `weights + activations` is the standard W8A8 path
- `weights + activations + kv_cache` is supported for dynamic FP8 and validated
- `weights + activations + attention + kv_cache` is supported for static FP8 and validated
- `attention` by itself is intentionally rejected because `vLLM` couples it with KV-cache quantization

Valid combinations by method family:

- dynamic:
  - `weights`
  - `weights + activations`
  - `weights + activations + kv_cache`
- static-like (`static`, `smoothquant`, `awq`, `gptq`):
  - `weights`
  - `weights + activations`
  - `weights + activations + kv_cache`
  - `weights + activations + attention + kv_cache`

Important limitation:

- `weights + activations + attention + kv_cache` is not supported in `dynamic`
  mode in the current `compressed-tensors` + `vLLM` stack, because attention
  quantization does not have a clean dynamic runtime path there.

Reference validated runs:

- dynamic FP8 `weights+activations+kv_cache`: `run_id: d2d17f88a6`
- static FP8 `weights+activations+attention+kv_cache`: `run_id: e50f60b86a`

## Setup

```bash
uv sync
```

Useful checks:

```bash
uv run ruff check src tests
uv run pytest tests -q
```

## CLI

The public CLI is now split into two explicit flows:

```bash
uv run ptq validate-config <config.yaml>
uv run ptq quantize --config <config.yaml> [--output-path <artifact-root>]
uv run ptq evaluate --model-ref <path-or-hf-model> --tasks <task1,task2,...> [--num-samples N] [--output-path <exact-eval-dir>]
```

`quantize`:

- accepts one YAML config
- exports the quantized artifact
- writes `run_metadata.json`
- runs the 3-prompt sanity check and writes `sanity_outputs.json`
- does not run `lm-eval` and does not write `metrics_*.csv`

`evaluate`:

- accepts a local artifact path, local model directory, or HF model ID
- does not require the YAML
- writes `lm_eval_results.json` and `eval_metadata.json`
- appends `metrics_*.csv`
- uses `--num-samples` as the optional eval limit; omit it for full evaluation
- exposes optional runtime overrides with code defaults:
  `--gpu-memory-utilization` (`0.5`), `--max-model-len` (`4096`), and
  `--max-gen-toks` (`512`)

Runtime behavior:

- `quantize` still uses `runtime.gpu_id` from the YAML
- `evaluate` uses the current `CUDA_VISIBLE_DEVICES` environment instead of a CLI GPU flag
- `evaluate --output-path` is the exact directory where eval results are written
- `quantize --output-path` overrides the parent artifact root, while preserving the normal nested run layout

For quantized configs, the normal sequence is:

```bash
uv run ptq validate-config <config.yaml>
uv run ptq quantize --config <config.yaml>
# then use the printed model_ref
uv run ptq evaluate --model-ref <printed-model-ref> --tasks gsm8k,ifeval,mmlu
```

## Benchmark Snapshot

Full-eval metrics below are reported by quantization description rather than run
ID. The task metrics tracked in this repo are:

- `gsm8k`: `exact_match,strict-match` / `exact_match,flexible-extract`
- `ifeval`: `prompt_level_strict_acc,none` / `inst_level_strict_acc,none`
- `mmlu`: `acc,none`

| Quantization | GSM8K strict / flexible | IFEval prompt / inst strict | MMLU acc |
|---|---:|---:|---:|
| Dynamic W8A8 FP8 | `0.6626 / 0.8400` | `0.7985 / 0.8597` | `0.5446` |
| Dynamic W8A8 INT8 | `0.6027 / 0.8438` | `0.7930 / 0.8549` | `0.5638` |
| Static W8A8 FP8 | `0.6611 / 0.8287` | `0.8096 / 0.8645` | `0.5474` |
| Static W8A8 INT8 | `0.0061 / 0.0705` | `0.2458 / 0.3717` | `0.2374` |
| Dynamic W8A8 FP8 + KV cache FP8 | `0.6664 / 0.8408` | `0.7930 / 0.8561` | `0.5713` |
| Dynamic W8A8 INT8 + KV cache INT8 | `0.5565 / 0.8309` | `0.7930 / 0.8597` | `0.5611` |
| Static W8A8 FP8 + KV cache FP8 | `0.7005 / 0.8461` | `0.8004 / 0.8561` | `0.5345` |
| Static W8A8 INT8 + KV cache INT8 | `0.0000 / 0.0546` | `0.2015 / 0.3237` | `0.2314` |
| Static W8A8 FP8 + attention FP8 + KV cache FP8 | `0.7066 / 0.8491` | `0.7893 / 0.8525` | `0.5246` |
| Static W8A8 INT8 + attention INT8 + KV cache INT8 | `0.0008 / 0.0758` | `0.1756 / 0.3141` | `0.2321` |
| SmoothQuant W8A8 FP8 | `0.6535 / 0.8461` | `0.8078 / 0.8633` | `0.5397` |
| SmoothQuant W8A8 INT8 | `0.0053 / 0.1221` | `0.2366 / 0.3741` | `0.2317` |
| AWQ W8A8 INT8 | `0.0045 / 0.1198` | `0.2514 / 0.3657` | `0.2342` |
| SmoothQuant + GPTQ W8A8 INT8 | `0.0159 / 0.1994` | `0.2957 / 0.4317` | `0.2336` |

Current INT8 takeaway:

- dynamic W8A8 INT8 is strong and competitive with dynamic FP8
- dynamic W8A8 INT8 + KV cache INT8 is also healthy
- the current static-like INT8 family is valid but underperforms badly in this repo:
  `static`, `static + kv`, `static + attention + kv`, `smoothquant`, and `awq`
- a follow-up investigation fixed one real asymmetric activation calibration bug,
  but smoke reruns still show low-quality outputs for static-like INT8
- that means the remaining gap is most likely recipe / algorithm fidelity rather
  than artifact corruption or a broken `vLLM` load path

## Best Validated Configs

These are the canonical configs for the strongest validated run per method family in this repo.

For every quantized config below, use the same two-step pattern:

```bash
uv run ptq validate-config <config.yaml>
uv run ptq quantize --config <config.yaml>
uv run ptq evaluate --model-ref <printed-model-ref> --tasks gsm8k,ifeval,mmlu
```

### Baseline

```bash
uv run ptq evaluate --model-ref Qwen/Qwen3-4B --tasks gsm8k,ifeval,mmlu
```

### Dynamic W8A8 FP8

This is the best complete dynamic W8A8 config currently validated across `gsm8k`, `ifeval`, and `mmlu`.

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_dynamic_w8a8_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.6626 strict` / `0.8400 flexible`
- `ifeval`: `0.7985 prompt strict` / `0.8597 inst strict`
- `mmlu`: `0.5446`

### Dynamic W8A8 FP8 Block

This is the `llm-compressor`-style block path:

- weights: `fp8`, `block`
- activations: `fp8`, `group`

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_fp8_block_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_dynamic_w8a8_fp8_block_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.2 strict` / `0.6 flexible` on the validated dev slice
- `ifeval`: `0.8 prompt strict` on the validated dev slice
- `mmlu`: `0.5351` on the validated dev slice

### Dynamic W8A8 INT8

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_dynamic_w8a8_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.6027 strict` / `0.8438 flexible`
- `ifeval`: `0.7930 prompt strict` / `0.8549 inst strict`
- `mmlu`: `0.5638`

### Static W8A8 FP8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.6611 strict` / `0.8287 flexible`
- `ifeval`: `0.8096 prompt strict` / `0.8645 inst strict`
- `mmlu`: `0.5474`

### Dynamic W8A8 FP8 + KV Cache FP8

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_kv_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_dynamic_w8a8_kv_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.6664 strict` / `0.8408 flexible`
- `ifeval`: `0.7930 prompt strict` / `0.8561 inst strict`
- `mmlu`: `0.5713`

### Dynamic W8A8 INT8 + KV Cache INT8

```bash
uv run ptq validate-config configs/generated/eval_dynamic_w8a8_kv_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_dynamic_w8a8_kv_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.5565 strict` / `0.8309 flexible`
- `ifeval`: `0.7930 prompt strict` / `0.8597 inst strict`
- `mmlu`: `0.5611`

### Static W8A8 FP8 + Attention FP8 + KV Cache FP8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_attention_kv_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_attention_kv_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.7066 strict` / `0.8491 flexible`
- `ifeval`: `0.7893 prompt strict` / `0.8525 inst strict`
- `mmlu`: `0.5246`

### Static W8A8 FP8 + KV Cache FP8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_kv_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_kv_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.7005 strict` / `0.8461 flexible`
- `ifeval`: `0.8004 prompt strict` / `0.8561 inst strict`
- `mmlu`: `0.5345`

### Static W8A8 INT8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0061 strict` / `0.0705 flexible`
- `ifeval`: `0.2458 prompt strict` / `0.3717 inst strict`
- `mmlu`: `0.2374`

Important note:

- the exported model is loadable by `vLLM`
- outputs are coherent English, not gibberish
- a follow-up asymmetric activation calibration fix did not materially change the
  qualitative result in smoke reruns

### Static W8A8 INT8 + KV Cache INT8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_kv_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_kv_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0000 strict` / `0.0546 flexible`
- `ifeval`: `0.2015 prompt strict` / `0.3237 inst strict`
- `mmlu`: `0.2314`

### Static W8A8 INT8 + Attention INT8 + KV Cache INT8

```bash
uv run ptq validate-config configs/generated/eval_static_w8a8_attention_kv_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_static_w8a8_attention_kv_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0008 strict` / `0.0758 flexible`
- `ifeval`: `0.1756 prompt strict` / `0.3141 inst strict`
- `mmlu`: `0.2321`

### SmoothQuant FP8

```bash
uv run ptq validate-config configs/generated/eval_smoothquant_w8a8_fp8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_smoothquant_w8a8_fp8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.6535 strict` / `0.8461 flexible`
- `ifeval`: `0.8078 prompt strict` / `0.8633 inst strict`
- `mmlu`: `0.5397`

### SmoothQuant INT8

```bash
uv run ptq validate-config configs/generated/eval_smoothquant_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_smoothquant_w8a8_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0053 strict` / `0.1221 flexible`
- `ifeval`: `0.2366 prompt strict` / `0.3741 inst strict`
- `mmlu`: `0.2317`

Follow-up smoke after the asymmetric activation calibration fix:

- `gsm8k`: `0.0 strict` / `0.2 flexible` on `limit=10`
- `ifeval`: `0.1 prompt strict` / `0.2222 inst strict` on `limit=10`
- `mmlu`: `0.2772` on `limit=10`

### AWQ INT8

```bash
uv run ptq validate-config configs/generated/eval_awq_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_awq_w8a8_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0045 strict` / `0.1198 flexible`
- `ifeval`: `0.2514 prompt strict` / `0.3657 inst strict`
- `mmlu`: `0.2342`

### SmoothQuant + GPTQ INT8

This is the closest repo-local equivalent to the stronger llm-compressor INT8
recipe shape: SmoothQuant preprocessing followed by GPTQ weight quantization,
with static tensor activations.

```bash
uv run ptq validate-config configs/generated/eval_smoothquant_gptq_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_smoothquant_gptq_w8a8_int8_qwen3.yaml
```

Metrics:

- `gsm8k`: `0.0159 strict` / `0.1994 flexible`
- `ifeval`: `0.2957 prompt strict` / `0.4317 inst strict`
- `mmlu`: `0.2336`

### GPTQ INT8

```bash
uv run ptq validate-config configs/generated/eval_gptq_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/generated/eval_gptq_w8a8_int8_qwen3.yaml
```

Current state:

- export succeeds
- smoke validation produced coherent outputs
- the plain GPTQ-only INT8 path does not yet have a completed full three-task
  eval recorded in this README
- the composed `smoothquant+gptq` INT8 path above does have a completed full
  eval

## FP8 AWQ and GPTQ Issue

The repo contains FP8 configs for AWQ and GPTQ:

- `configs/example_awq_w8a8_fp8.yaml`
- `configs/example_gptq_w8a8_fp8.yaml`

You can run them with:

```bash
uv run ptq validate-config configs/example_awq_w8a8_fp8.yaml
uv run ptq quantize --config configs/example_awq_w8a8_fp8.yaml

uv run ptq validate-config configs/example_gptq_w8a8_fp8.yaml
uv run ptq quantize --config configs/example_gptq_w8a8_fp8.yaml
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
uv run ptq quantize --config configs/smoke_dynamic_w8a8_fp8_qwen3.yaml
uv run ptq quantize --config configs/smoke_dynamic_w8a8_int8_qwen3.yaml
uv run ptq quantize --config configs/smoke_static_w8a8_fp8_qwen3.yaml
uv run ptq quantize --config configs/smoke_static_w8a8_int8_qwen3.yaml
# then evaluate the printed model_ref if you want task metrics
uv run ptq evaluate --model-ref <printed-model-ref> --tasks gsm8k --num-samples 10
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
- static attention and static KV-cache quantization require calibration
- the current `vLLM` runtime couples attention quantization with KV-cache quantization
- dynamic attention quantization is intentionally disabled in the validator
- chat templating follows the official Hugging Face Qwen template path used by the repo
- the hard inference requirement is `vLLM`
