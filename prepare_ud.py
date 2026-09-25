"""Fetch and prepare the pinned UD source treebanks."""

import argparse
from pathlib import Path

from uralic_parser.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/ud/r2.18/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/ud/r2.18/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Require already-downloaded raw files")
    args = parser.parse_args()
    manifest = prepare(args.source_root, args.output, args.seed, fetch=not args.offline)
    print(manifest["counts"])


if __name__ == "__main__":
    main()
