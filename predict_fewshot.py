"""Create form-only target inputs and zero-shot CoNLL-U drafts for annotation."""

import argparse
from pathlib import Path
import string

import torch

from uralic_lm.common import file_hash, fingerprint, read_json, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence, write_predictions
from uralic_parser.evaluation import load_best


INPUTS = {"mhr": "mari.conllu", "udm": "udmurt.conllu"}


def usable_forms(forms: list[str]) -> bool:
    """Avoid image markup, mixed-script debris and punctuation-heavy fragments."""
    if any("|" in form or any(char in string.ascii_letters for char in form)
           for form in forms):
        return False
    cyrillic_words = sum(any("\u0400" <= char <= "\u052f" for char in form)
                         for form in forms)
    return cyrillic_words >= 3 and 4 * cyrillic_words >= 3 * len(forms)


def select_form_only(source: Path, language: str, splits: dict[str, str],
                     count: int, seed: int, min_words: int,
                     max_words: int) -> tuple[list[Sentence], dict]:
    """Select one sentence per training article, discarding all source annotations."""
    by_article: dict[str, tuple[str, str, list[str]]] = {}
    eligible_sentences = 0
    for sentence in read_conllu(source):
        article = sentence.metadata.get("article", "")
        source_id = language + ":wiki:" + fingerprint(article)[:20]
        if not article or splits.get(source_id) != "train":
            continue
        words = sentence.words
        if not min_words <= len(words) <= max_words:
            continue
        ids = [word.id for word in words]
        if ids != list(range(1, len(words) + 1)) or any(
                not word.form or word.form == "_" for word in words):
            continue
        forms = [word.form for word in words]
        if not usable_forms(forms):
            continue
        sent_id = sentence.metadata.get("sent_id", "")
        if not sent_id:
            continue
        eligible_sentences += 1
        rank = fingerprint([seed, source_id, sent_id, forms])
        if source_id not in by_article or rank < by_article[source_id][0]:
            by_article[source_id] = (rank, sent_id, forms)
    if len(by_article) < count:
        raise ValueError(f"{language}: only {len(by_article)} eligible training articles")
    selected = sorted(by_article.items(), key=lambda item: fingerprint([seed, item[0]]))[:count]
    sentences = []
    selection = []
    for number, (source_id, (_, source_sent_id, forms)) in enumerate(selected, 1):
        sent_id = f"{language}-fewshot-{number:04d}"
        lines = [f"# sent_id = {sent_id}", f"# source_id = {source_id}",
                 f"# source_sent_id = {source_sent_id}",
                 "# annotation_status = FORM_ONLY"]
        lines.extend(f"{index}\t{form}\t_\t_\t_\t_\t_\t_\t_\t_"
                     for index, form in enumerate(forms, 1))
        sentence = Sentence(lines)
        validate_sentence(sentence, require_gold=False)
        sentences.append(sentence)
        selection.append({"sent_id": sent_id, "source_id": source_id,
                          "source_sent_id": source_sent_id, "words": len(forms)})
    return sentences, {"eligible_sentences": eligible_sentences,
                       "eligible_articles": len(by_article), "selection": selection}


def run(args: argparse.Namespace) -> dict:
    if args.per_language < 1 or args.batch_size < 1 or not 1 <= args.min_words <= args.max_words:
        raise ValueError("Counts and batch size must be positive; min_words <= max_words")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    split_rows = read_json(args.source_splits)
    splits = {}
    for row in split_rows:
        for source_id in row["source_ids"]:
            if source_id in splits and splits[source_id] != row["split"]:
                raise ValueError(f"Conflicting source split: {source_id}")
            splits[source_id] = row["split"]
    selected = {}
    for language, filename in INPUTS.items():
        source = args.input_dir / filename
        selected[language] = select_form_only(
            source, language, splits, args.per_language, args.seed,
            args.min_words, args.max_words)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(), "purpose": "zero-shot drafts for few-shot training annotation",
        "parent_run": str(args.run), "checkpoint": "best_model",
        "checkpoint_sha256": file_hash(args.run / "best_model" / "model.safetensors"),
        "vocab_sha256": file_hash(args.run / "vocab.json"),
        "run_config_sha256": file_hash(args.run / "config.json"),
        "source_splits": str(args.source_splits),
        "source_splits_sha256": file_hash(args.source_splits),
        "split": "train", "representation": "identity; syntactic word FORM only",
        "selection_filter": "5–50 words; no ASCII Latin or |; >=75% Cyrillic word forms",
        "seed": args.seed, "per_language": args.per_language,
        "min_words": args.min_words, "max_words": args.max_words,
        "batch_size": args.batch_size, "device": "cuda",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "languages": {},
    }
    model = load_best(args.run, "cuda")
    for language, filename in INPUTS.items():
        sentences, selection = selected[language]
        source = args.input_dir / filename
        forms_path = args.output / f"{language}-forms.conllu"
        predicted_path = args.output / f"{language}-predicted.conllu"
        with forms_path.open("w", encoding="utf-8", newline="\n") as stream:
            for sentence in sentences:
                stream.write(sentence.render())
        with predicted_path.open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(sentences), args.batch_size):
                part = sentences[start:start + args.batch_size]
                for sentence, rows in zip(part, model.predict(part), strict=True):
                    lines = write_predictions(sentence, rows).strip().splitlines()
                    predicted = Sentence([
                        line.replace("# annotation_status = FORM_ONLY",
                                     "# annotation_status = ZERO_SHOT_DRAFT")
                        for line in lines
                    ])
                    validate_sentence(predicted)
                    stream.write(predicted.render())
            print(f"{language}: predicted {len(sentences)} sentences", flush=True)
        manifest["languages"][language] = {
            "source": str(source), "source_sha256": file_hash(source),
            "forms": str(forms_path), "forms_sha256": file_hash(forms_path),
            "predicted": str(predicted_path),
            "predicted_sha256": file_hash(predicted_path), **selection,
        }
        write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/parser-all-001"))
    parser.add_argument("--source-splits", type=Path,
                        default=Path("data/prepared/original-v1/source_splits.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/parser-all-001/fewshot-001"))
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
