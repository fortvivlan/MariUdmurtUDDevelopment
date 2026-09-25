"""Validate human-adjudicated Mari/Udmurt gold and its held-out article IDs."""

import argparse
from pathlib import Path

from uralic_lm.common import read_json
from uralic_parser.conllu import read_conllu, validate_sentence


def validate_gold(path: Path, language: str, split_file: Path,
                  expected_sentences: int = 200) -> dict:
    heldout = {sid for row in read_json(split_file) if row["split"] == "test"
               for sid in row["source_ids"]}
    sentences = list(read_conllu(path))
    if len(sentences) != expected_sentences:
        raise ValueError(f"Expected {expected_sentences} gold sentences; found {len(sentences)}")
    articles = set()
    words = 0
    for sentence in sentences:
        validate_sentence(sentence)
        metadata = sentence.metadata
        source_id = metadata.get("source_id", "")
        if metadata.get("annotation_status") != "GOLD_ADJUDICATED":
            raise ValueError("Gold sentences need # annotation_status = GOLD_ADJUDICATED")
        if not source_id.startswith(language + ":") or source_id not in heldout:
            raise ValueError(f"Target source is outside backbone test split: {source_id}")
        if source_id in articles:
            raise ValueError(f"Repeated article in target gold: {source_id}")
        articles.add(source_id)
        words += len(sentence.words)
    return {"language": language, "sentences": len(sentences),
            "articles": len(articles), "syntactic_words": words}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--language", choices=("mhr", "udm"), required=True)
    parser.add_argument("--source-splits", type=Path,
                        default=Path("data/prepared/original-v1/source_splits.json"))
    args = parser.parse_args()
    print(validate_gold(args.gold, args.language, args.source_splits))


if __name__ == "__main__":
    main()
