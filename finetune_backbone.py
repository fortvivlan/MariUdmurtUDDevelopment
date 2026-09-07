"""Train, audit, resume, or evaluate a reproducible Glot500 MLM experiment."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for TrainConfig defaults")
    parser.add_argument("--output", type=Path, help="New run directory; defaults to a unique path under runs/")
    parser.add_argument("--resume", type=Path, help="Existing run directory (uses its saved configuration)")
    parser.add_argument("--evaluate-run", type=Path, help="Evaluate a run's selected checkpoint without training")
    parser.add_argument("--parent-run", type=Path, help="Provenance link for a new experiment")
    parser.add_argument("--audit-only", action="store_true", help="Prepare tokenizer audit without loading model weights")
    parser.add_argument("--stop-after-steps", type=int, help="Pause after this many additional optimizer steps; resumable")
    args = parser.parse_args()
    if sum(bool(x) for x in (args.resume, args.evaluate_run, args.output)) > 1:
        parser.error("--resume, --evaluate-run and --output are mutually exclusive")
    if args.resume and (args.config or args.audit_only or args.parent_run):
        parser.error("Resume uses its immutable saved configuration; do not provide config/audit/parent overrides")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        parser.error("--stop-after-steps must be positive")
    from uralic_lm.common import read_json
    from uralic_lm.training import TrainConfig, evaluate_run, run_experiment
    if args.evaluate_run:
        print(evaluate_run(args.evaluate_run))
        return
    settings = read_json(args.resume / "config.json") if args.resume else read_json(args.config) if args.config else {}
    config = TrainConfig(**settings)
    print(run_experiment(config, args.output, args.resume, args.stop_after_steps, args.audit_only, args.parent_run))


if __name__ == "__main__":
    main()
