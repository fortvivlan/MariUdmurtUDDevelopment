"""Select article-disjoint Mari/Udmurt sentences for human UD annotation."""

import argparse
import csv
from pathlib import Path

from uralic_lm.common import file_hash, fingerprint, read_json, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu


INPUTS = {"mhr": "mari.conllu", "udm": "udmurt.conllu"}


def prepare_drafts(split_file: Path, input_dir: Path, output: Path,
                   per_language: int = 200, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Draft output must be a new/empty directory")
    source_splits = read_json(split_file)
    heldout = {sid for row in source_splits if row["split"] == "test"
               for sid in row["source_ids"]}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utc_now(), "seed": seed, "per_language": per_language,
                "split_file": str(split_file), "split_sha256": file_hash(split_file),
                "languages": {}}
    for language, filename in INPUTS.items():
        source = input_dir / filename
        by_article: dict[str, tuple[str, Sentence]] = {}
        eligible = 0
        for sentence in read_conllu(source):
            article = sentence.metadata.get("article", "")
            source_id = language + ":wiki:" + fingerprint(article)[:20]
            if not article or source_id not in heldout or len(sentence.words) < 3:
                continue
            eligible += 1
            rank = fingerprint([seed, source_id, sentence.metadata.get("sent_id"),
                                sentence.metadata.get("text")])
            if source_id not in by_article or rank < by_article[source_id][0]:
                by_article[source_id] = (rank, sentence)
        if len(by_article) < per_language:
            raise ValueError(f"{language}: only {len(by_article)} eligible held-out articles")
        selected = sorted(by_article.items(), key=lambda item: fingerprint([seed, item[0]]))[:per_language]
        destination = output / f"{language}-draft.conllu"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for source_id, (_, sentence) in selected:
                lines = [f"# source_id = {source_id}",
                         "# annotation_status = DRAFT_AUTOMATIC",
                         *sentence.lines]
                stream.write(Sentence(lines).render())
        manifest["languages"][language] = {
            "source": str(source), "source_sha256": file_hash(source),
            "draft": str(destination), "draft_sha256": file_hash(destination),
            "eligible_sentences": eligible, "eligible_articles": len(by_article),
            "selected_source_ids": [source_id for source_id, _ in selected],
        }
    manifest["fingerprint"] = fingerprint({key: value for key, value in manifest.items()
                                           if key != "created_at"})
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-splits", type=Path,
                        default=Path("data/prepared/original-v1/source_splits.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/ud/targets/drafts-v1"))
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_drafts(args.source_splits, args.input_dir, args.output,
                              args.per_language, args.seed)
    print({language: len(data["selected_source_ids"])
           for language, data in manifest["languages"].items()})


if __name__ == "__main__":
    main()
