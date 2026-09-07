"""Compare complete or partial experiment records without loading models."""

import argparse
from pathlib import Path
from uralic_lm.common import read_json, write_json


def summarize(run):
    config = read_json(run / "config.json")
    experiment = read_json(run / "experiment.json")
    summary = read_json(run / "summary.json") if (run / "summary.json").exists() else {}
    return {"run": str(run), "status": read_json(run / "status.json")["status"],
            "data_fingerprint": experiment["data_fingerprint"], "model": experiment["model"],
            "config": config, "validation": summary.get("adapted", {}).get("validation"),
            "test": summary.get("adapted", {}).get("test"),
            "baseline_test": summary.get("baseline", {}).get("test"),
            "best_checkpoint": summary.get("best_checkpoint"),
            "comparison_note": "Compare like-for-like data, tokenizer, representation, masks, and scored tokens."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON comparison report")
    args = parser.parse_args()
    rows = [summarize(run) for run in args.runs]
    for row in rows:
        print(f"{row['run']}: {row['status']}, data={row['data_fingerprint'][:12]}")
        for language, metrics in (row["validation"] or {}).items():
            print(f"  {language} validation loss={metrics['mlm_loss']:.4f}, top1={metrics['top1']:.4f}")
    if args.output:
        if args.output.exists():
            parser.error("Comparison output already exists; choose a new filename")
        write_json(args.output, rows)


if __name__ == "__main__":
    main()
