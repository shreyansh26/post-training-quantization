import argparse
from pathlib import Path

from ptq.config import load_run_config
from ptq.eval import VLLMEvaluationSettings
from ptq.pipeline import evaluate_model_ref, quantize_from_config
from ptq.runs import artifact_dir_for_run, generate_run_id


def _csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    run_id = generate_run_id(config)
    artifact_dir = artifact_dir_for_run(config, run_id)
    print(f"config: {Path(args.config).resolve()}")
    print(f"run_id: {run_id}")
    print(f"artifact_dir: {artifact_dir}")
    print("status: valid")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    outcome = quantize_from_config(
        config=config,
        output_root=Path(args.output_path).resolve() if args.output_path else None,
    )
    print(f"run_id: {outcome.run_id}")
    print(f"artifact_dir: {outcome.artifact_dir}")
    print(f"model_ref: {outcome.model_ref}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    outcome = evaluate_model_ref(
        model_ref=args.model_ref,
        tasks=_csv_list(args.tasks),
        num_samples=args.num_samples,
        output_dir=Path(args.output_path).resolve() if args.output_path else None,
        settings=VLLMEvaluationSettings(
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enable_thinking=args.enable_thinking,
            max_gen_toks=args.max_gen_toks,
        ),
    )
    print(f"output_dir: {outcome.output_dir}")
    print(f"model_ref: {outcome.model_ref}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    validate.set_defaults(func=_cmd_validate_config)

    quantize = subparsers.add_parser("quantize")
    quantize.add_argument("--config", required=True)
    quantize.add_argument("--output-path")
    quantize.set_defaults(func=_cmd_quantize)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--model-ref", required=True)
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--num-samples", type=int)
    evaluate.add_argument("--output-path")
    evaluate.add_argument("--trust-remote-code", action="store_true")
    evaluate.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    evaluate.add_argument("--max-model-len", type=int, default=4096)
    evaluate.add_argument("--enable-thinking", action="store_true")
    evaluate.add_argument("--max-gen-toks", type=int, default=512)
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
