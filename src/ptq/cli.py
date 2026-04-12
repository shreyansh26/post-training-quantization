from __future__ import annotations

import argparse
from pathlib import Path

from ptq.config import load_run_config
from ptq.runs import artifact_dir_for_run, generate_run_id


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    run_id = generate_run_id(config)
    artifact_dir = artifact_dir_for_run(config, run_id)
    print(f"config: {Path(args.config).resolve()}")
    print(f"run_id: {run_id}")
    print(f"artifact_dir: {artifact_dir}")
    print("status: valid")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from ptq.pipeline import run_pipeline

    config = load_run_config(args.config)
    outcome = run_pipeline(config)
    print(f"run_id: {outcome.run_id}")
    print(f"artifact_dir: {outcome.artifact_dir}")
    print(f"model_ref: {outcome.model_ref}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    validate.set_defaults(func=_cmd_validate_config)

    run = subparsers.add_parser("run")
    run.add_argument("config")
    run.set_defaults(func=_cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
