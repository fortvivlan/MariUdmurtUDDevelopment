"""Prepare immutable, documented corpus splits for backbone adaptation."""

import argparse
from pathlib import Path
from uralic_lm.data import prepare, FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("rawtexts"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/original-v1"))
    parser.add_argument("--nganasan-archive", type=Path, default=Path("rawtexts/sources/nganasan-1.0-lite.zip"))
    parser.add_argument("--languages", nargs="+", choices=list(FILES), default=list(FILES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--test-fraction", type=float, default=.05)
    parser.add_argument("--overlap-words", type=int, default=12)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="External evaluation .txt or .conllu files")
    parser.add_argument("--source-metadata", type=Path, help="JSON metadata overrides keyed by language")
    args = parser.parse_args()
    result = prepare(args.raw_dir, args.output, args.nganasan_archive, args.seed, args.languages,
                     args.exclude, args.validation_fraction, args.test_fraction, args.overlap_words,
                     args.source_metadata)
    print("Dataset:", result["fingerprint"])
    print("Documents:", result["counts"])
    print("Nganasan recovery:", result["nganasan_recovery"])


if __name__ == "__main__":
    main()
