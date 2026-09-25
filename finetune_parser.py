"""Train or resume a joint UD transfer parser."""

import argparse
from pathlib import Path

from uralic_parser.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resume:
        if args.output:
            parser.error("--output is incompatible with --resume")
        config = TrainConfig.from_json(args.resume / "config.json")
        output = args.resume
    else:
        if not args.output:
            parser.error("--output is required with --config")
        config = TrainConfig.from_json(args.config)
        output = args.output
    train(config, output, resume=bool(args.resume))


if __name__ == "__main__":
    main()
