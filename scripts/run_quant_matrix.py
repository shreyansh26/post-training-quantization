#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

CONFIGS = [
    "configs/example_baseline_qwen3.yaml",
    "configs/example_rtn_w8_int8_datafree.yaml",
    "configs/example_dynamic_w8a8_int8.yaml",
    "configs/example_dynamic_w8a8_fp8.yaml",
    "configs/example_dynamic_w8a8_fp8_block.yaml",
    "configs/example_static_w8a8_int8.yaml",
    "configs/example_static_w8a8_fp8.yaml",
    "configs/example_static_w8a8_fp8_block.yaml",
    "configs/example_smoothquant_w8a8_int8.yaml",
    "configs/example_smoothquant_w8a8_fp8.yaml",
    "configs/example_smoothquant_w8a8_fp8_block.yaml",
    "configs/example_awq_w8a8_int8.yaml",
    "configs/example_awq_w8a8_fp8.yaml",
    "configs/example_awq_w8_fp8_block.yaml",
    "configs/example_gptq_w8a8_int8.yaml",
    "configs/example_gptq_w8a8_fp8.yaml",
    "configs/example_gptq_w8_fp8_block.yaml",
    "configs/example_attention_kv_fp8.yaml",
    "configs/example_attention_kv_int8.yaml",
]


def run_configs(stop_on_error: bool) -> int:
    exit_code = 0
    for config in CONFIGS:
        print(f"\n=== running {config} ===")
        result = subprocess.run(["ptq", "run", config], check=False)
        if result.returncode != 0:
            exit_code = result.returncode
            if stop_on_error:
                return exit_code
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort the matrix on the first failed config",
    )
    args = parser.parse_args()
    return run_configs(stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
