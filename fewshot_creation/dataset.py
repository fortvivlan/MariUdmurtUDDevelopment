"""Validate corrected batches and make cumulative target train/dev files."""

from pathlib import Path

from uralic_lm.common import file_hash, fingerprint, read_json, utc_now, write_json
from uralic_parser.conllu import Sentence, parse_feats, read_conllu, validate_sentence


LANGUAGES = ("mhr", "udm")


def source_splits(path: Path) -> dict[str, str]:
    result = {}
    for row in read_json(path):
        for source_id in row["source_ids"]:
            if source_id in result and result[source_id] != row["split"]:
                raise ValueError(f"Conflicting source split: {source_id}")
            result[source_id] = row["split"]
    return result


def _validated_batch(corrected_dir: Path, draft_dir: Path, language: str,
                     splits: dict[str, str]) -> list[Sentence]:
    corrected = list(read_conllu(corrected_dir / f"{language}-corrected.conllu"))
    draft = list(read_conllu(draft_dir / f"{language}-predicted.conllu"))
    expected = {row.metadata.get("sent_id"): row for row in draft}
    if len(expected) != len(draft) or not corrected:
        raise ValueError(f"{language}: empty or duplicate draft IDs")
    if len(corrected) != len(draft):
        raise ValueError(f"{language}: corrected sentence count differs from draft")
    found = set()
    for row in corrected:
        validate_sentence(row)
        meta = row.metadata
        sent_id = meta.get("sent_id")
        if sent_id in found or sent_id not in expected:
            raise ValueError(f"{language}: duplicate or unknown corrected ID {sent_id}")
        found.add(sent_id)
        original = expected[sent_id].metadata
        if any(meta.get(key) != original.get(key) for key in ("source_id", "source_sent_id")):
            raise ValueError(f"{language}:{sent_id}: source identity changed")
        if meta.get("text") != original.get("text"):
            raise ValueError(f"{language}:{sent_id}: original text changed")
        if meta.get("annotation_status") != "GOLD_ADJUDICATED":
            raise ValueError(f"{language}:{sent_id}: expected GOLD_ADJUDICATED")
        if splits.get(meta["source_id"]) != "train":
            raise ValueError(f"{language}:{sent_id}: source article is not in train split")
        if any(word.upos == "_" or (word.lemma == "_" or "?" in word.lemma)
               and any(char.isalpha() or char.isdigit() for char in word.form)
               for word in row.words):
            raise ValueError(f"{language}:{sent_id}: unresolved LEMMA/UPOS")
        for word in row.words:
            parse_feats(word.feats)
        if not meta.get("text"):
            raise ValueError(f"{language}:{sent_id}: missing original text")
    return corrected


def prepare_targets(corrected_dirs: list[Path], draft_dirs: list[Path],
                    split_file: Path, output: Path, seed: int = 42,
                    dev_cap: int = 500) -> dict:
    if not corrected_dirs or len(corrected_dirs) != len(draft_dirs):
        raise ValueError("Supply matching corrected and draft directories in batch order")
    if dev_cap < 1 or output.exists() and any(output.iterdir()):
        raise ValueError("Development cap must be positive and output new/empty")
    splits = source_splits(split_file)
    prepared = {}
    for language in LANGUAGES:
        train, dev, used = [], [], set()
        article_split = {}
        sources = []
        for corrected_dir, draft_dir in zip(corrected_dirs, draft_dirs, strict=True):
            rows = _validated_batch(corrected_dir, draft_dir, language, splits)
            source = corrected_dir / f"{language}-corrected.conllu"
            sources.append({"corrected": str(source), "corrected_sha256": file_hash(source),
                            "draft": str(draft_dir / f"{language}-predicted.conllu"),
                            "draft_sha256": file_hash(draft_dir / f"{language}-predicted.conllu")})
            for row in rows:
                key = (row.metadata["source_id"], row.metadata["source_sent_id"])
                if key in used:
                    raise ValueError(f"Repeated corrected source sentence: {key}")
                used.add(key)
            quota = min(round(len(rows) * .1), dev_cap - len(dev))
            candidates = {row.metadata["source_id"] for row in rows
                          if row.metadata["source_id"] not in article_split}
            chosen = set(sorted(candidates, key=lambda article: fingerprint(
                [seed, language, article]))[:max(0, quota)])
            for row in rows:
                article = row.metadata["source_id"]
                group = article_split.setdefault(article, "dev" if article in chosen else "train")
                (dev if group == "dev" else train).append(row)
            if len(dev) > dev_cap:
                raise ValueError(f"{language}: development cap exceeded by repeated article")
        if not train or not dev:
            raise ValueError(f"{language}: corrected data must yield train and dev rows")
        prepared[language] = {"train_rows": train, "dev_rows": dev,
                              "dev_articles": sorted(article for article, group in
                                                     article_split.items() if group == "dev"),
                              "sources": sources}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utc_now(), "seed": seed, "dev_cap_per_language": dev_cap,
                "script_sha256": file_hash(Path(__file__)),
                "dev_fraction_per_batch": .1, "source_splits": str(split_file),
                "source_splits_sha256": file_hash(split_file), "languages": {}}
    for language, info in prepared.items():
        paths = {}
        for split in ("train", "dev"):
            path = output / f"{language}-{split}.conllu"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in info[f"{split}_rows"]:
                    stream.write(row.render())
            paths[split] = str(path)
            paths[split + "_sha256"] = file_hash(path)
            paths[split + "_sentences"] = len(info[f"{split}_rows"])
        manifest["languages"][language] = {
            **paths, "dev_articles": info["dev_articles"], "sources": info["sources"]}
    write_json(output / "manifest.json", manifest)
    return manifest
