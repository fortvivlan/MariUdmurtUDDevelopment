"""Compare the three preplanned parser conditions on identical target gold."""

import argparse
from pathlib import Path

from uralic_lm.common import read_json, utc_now, write_json
from uralic_parser.evaluation import _article_counts, paired_difference


def compare(primary: Path, no_nenets: Path, original: Path, output: Path) -> dict:
    reports = {name: read_json(path / "report.json") for name, path in (
        ("all_sources", primary), ("without_nenets", no_nenets),
        ("original_glot500", original))}
    result = {"created_at": utc_now(), "reports": {name: str(path) for name, path in (
        ("all_sources", primary), ("without_nenets", no_nenets),
        ("original_glot500", original))}, "targets": {}}
    for language in ("mhr", "udm"):
        rows = {name: report["languages"][language] for name, report in reports.items()}
        if len({row["gold_sha256"] for row in rows.values()}) != 1:
            raise ValueError(f"{language}: conditions used different gold files")
        groups = {name: _article_counts(Path(row["gold"]), Path(row["predicted"]))
                  for name, row in rows.items()}
        result["targets"][language] = {
            "scores": {name: {metric: row["metrics"][metric]["f1"]
                              for metric in ("UAS", "LAS")}
                       for name, row in rows.items()},
            "nenets_effect": paired_difference(groups["all_sources"],
                                                groups["without_nenets"]),
            "backbone_effect": paired_difference(groups["all_sources"],
                                                  groups["original_glot500"]),
        }
    if output.exists():
        raise ValueError("Comparison output must be new")
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--no-nenets", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.primary, args.no_nenets, args.original, args.output)
    print(report["targets"])


if __name__ == "__main__":
    main()
