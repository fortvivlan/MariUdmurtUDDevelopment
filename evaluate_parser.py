"""Score a saved parser with official UD metrics on gold CoNLL-U."""

import argparse
from pathlib import Path

from uralic_parser.evaluation import evaluate_run
from uralic_lm.common import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-mhr", type=Path)
    parser.add_argument("--gold-udm", type=Path)
    parser.add_argument("--source-tests", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    files = {}
    if args.gold_mhr:
        files["mhr"] = args.gold_mhr
    if args.gold_udm:
        files["udm"] = args.gold_udm
    if args.source_tests:
        manifest = read_json(args.run / "dataset_manifest.json")
        for language in read_json(args.run / "config.json")["source_languages"]:
            files[language] = Path(manifest["outputs"][f"{language}/test"]["path"])
    if not files:
        parser.error("Supply target gold files and/or --source-tests")
    report = evaluate_run(args.run, files, args.output, args.device, args.batch_size)
    for language, result in report["languages"].items():
        print(language, {metric: round(result["metrics"][metric]["f1"], 4)
                         for metric in ("UAS", "LAS", "UPOS", "UFeats", "Lemmas")})


if __name__ == "__main__":
    main()
