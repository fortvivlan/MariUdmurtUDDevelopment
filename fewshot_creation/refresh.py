"""Refill a cleaned annotation batch, preserving existing human edits.

Run prepare with the cleaning output, predict in the parser environment, then
the regular fst/add-translations commands. Finally assemble merges unchanged
review entries back verbatim. Publication is staged; this command never
overwrites the annotation directory or the full corpora.
"""

import argparse
from pathlib import Path
import shutil

from enrich_fewshot import spaces_after
from uralic_lm.common import file_hash, read_json, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence, write_predictions
from .dataset import source_splits
from .prediction import excluded_sentences, select_new


def form_only(row: Sentence) -> Sentence:
    forms = [word.form for word in row.words]
    spacing = spaces_after(forms, row.metadata["text"])
    lines = [line for line in row.lines if line.startswith("#")
             and not line.startswith(("# annotation_status", "# lemma_source"))]
    lines.append("# annotation_status = FORM_ONLY")
    for index, (form, adjacent) in enumerate(zip(forms, spacing, strict=True), 1):
        lines.append(f"{index}\t{form}\t_\t_\t_\t_\t_\t_\t_\t"
                     + ("SpaceAfter=No" if adjacent else "_"))
    return Sentence(lines)


def prepare(args):
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    cleaning = read_json(args.cleaned / "manifest.json")
    splits = source_splits(args.source_splits)
    excluded = excluded_sentences(args.history)
    manifest = {"created_at": utc_now(), "cleaning": str(args.cleaned),
                "cleaning_manifest_sha256": file_hash(args.cleaned / "manifest.json"),
                "source_splits": str(args.source_splits),
                "source_splits_sha256": file_hash(args.source_splits),
                "seed": args.seed, "count": args.count, "min_words": args.min_words,
                "max_words": args.max_words, "languages": {},
                "original_batch": str(args.batch), "original_hashes": {
                    path.name: file_hash(path) for path in args.batch.glob("*.conllu")}}
    args.output.mkdir(parents=True)
    (args.output / "forms").mkdir()
    request = {}
    for language, filename in (("mhr", "mari.conllu"), ("udm", "udmurt.conllu")):
        original = args.batch / f"{language}-model-review.conllu"
        if file_hash(original) != cleaning["languages"][language]["batch_input_sha256"]:
            raise ValueError(f"{original}: changed since cleaning")
        retained = list(read_conllu(args.cleaned / f"{language}-retained-review.conllu"))
        original_rows = {row.metadata["sent_id"]: row for row in read_conllu(original)}
        count = args.count - len(retained)
        if count < 0:
            raise ValueError("Retained batch exceeds desired count")
        texts = {row.metadata["text"] for row in retained}
        # Exclude retained articles from additions, and all 200 original source
        # identities, including foreign sentences manually deleted earlier.
        articles = {row.metadata["source_id"] for row in retained}
        if count:
            additions, selection = select_new(
                args.cleaned / filename, language, splits, excluded[language],
                articles, "refresh", count, args.seed, args.min_words, args.max_words)
        else:
            additions, selection = [], {}
        for index, row in enumerate(additions, 201):
            row.lines[0] = f"# sent_id = {language}-fewshot-{index:04d}"
            if row.metadata["text"] in texts:
                raise ValueError("Addition duplicates retained batch text")
            texts.add(row.metadata["text"])
        preserved_ids = [row.metadata["sent_id"] for row in retained
                         if row.metadata["text"] == original_rows[
                             row.metadata["sent_id"]].metadata["text"]]
        rows = [form_only(row) for row in retained + additions]
        for row in rows:
            validate_sentence(row, require_gold=False)
            if "text_en" not in row.metadata:
                request[row.metadata["sent_id"]] = row.metadata["text"]
        path = args.output / "forms" / f"{language}-forms.conllu"
        path.write_text("".join(row.render() for row in rows), encoding="utf-8")
        manifest["languages"][language] = {
            "retained": len(retained), "added": len(additions),
            "reparsed_after_boilerplate_removal": len(retained) - len(preserved_ids),
            "preserved_ids": preserved_ids, "forms_sha256": file_hash(path),
            "selection": selection,
        }
    write_json(args.output / "translation-request.json", request)
    write_json(args.output / "manifest.json", manifest)
    print({key: {k: v for k, v in value.items() if k not in ("selection", "preserved_ids")}
           for key, value in manifest["languages"].items()})


def predict(args):
    from uralic_parser.evaluation import load_best

    output = args.work / "predictions"
    if output.exists():
        raise ValueError("Prediction directory already exists")
    manifest = read_json(args.work / "manifest.json")
    model = load_best(args.parser_run, args.device)
    output.mkdir()
    record = {"created_at": utc_now(), "parser_run": str(args.parser_run),
              "checkpoint_sha256": file_hash(args.parser_run / "best_model/model.safetensors"),
              "device": args.device, "batch_size": args.batch_size,
              "preparation_manifest_sha256": file_hash(args.work / "manifest.json"),
              "languages": {}}
    for language in ("mhr", "udm"):
        path = args.work / "forms" / f"{language}-forms.conllu"
        if file_hash(path) != manifest["languages"][language]["forms_sha256"]:
            raise ValueError(f"Forms changed: {path}")
        rows = list(read_conllu(path))
        drafts = []
        for start in range(0, len(rows), args.batch_size):
            part = rows[start:start + args.batch_size]
            for row, prediction in zip(part, model.predict(part), strict=True):
                draft = Sentence(write_predictions(row, prediction).rstrip("\n").splitlines())
                draft.lines = [line.replace("# annotation_status = FORM_ONLY",
                                           "# annotation_status = PARSER_DRAFT") for line in draft.lines]
                validate_sentence(draft)
                drafts.append(draft)
        target = output / f"{language}-predicted.conllu"
        target.write_text("".join(row.render() for row in drafts), encoding="utf-8")
        record["languages"][language] = {"sentences": len(drafts), "sha256": file_hash(target)}
        print(language, len(drafts), flush=True)
    write_json(output / "manifest.json", record)


def assemble(args):
    output = args.work / "annotation"
    if output.exists():
        raise ValueError("Annotation directory already exists")
    manifest = read_json(args.work / "manifest.json")
    batch = Path(manifest["original_batch"])
    for name, digest in manifest["original_hashes"].items():
        if file_hash(batch / name) != digest:
            raise ValueError(f"Original batch changed: {name}")
    translations = read_json(args.translations)
    request = read_json(args.work / "translation-request.json")
    if translations.keys() != request.keys():
        raise ValueError("Translation IDs differ from request")
    output.mkdir()
    report = {"created_at": utc_now(), "preparation": str(args.work / "manifest.json"),
              "review": str(args.review), "files": {}, "preserved_tree_issues": []}
    for language in ("mhr", "udm"):
        originals = {row.metadata["sent_id"]: row for row in read_conllu(
            batch / f"{language}-model-review.conllu")}
        preserved = set(manifest["languages"][language]["preserved_ids"])
        pairs = []
        for kind in ("model-review", "fst"):
            rows = list(read_conllu(args.review / f"{language}-{kind}.conllu"))
            result = []
            for row in rows:
                key = row.metadata["sent_id"]
                if kind == "model-review" and key in preserved:
                    row = originals[key]
                elif key in translations:
                    text_en = translations[key]
                    if not isinstance(text_en, str) or not text_en.strip() or "\n" in text_en:
                        raise ValueError(f"Invalid translation: {key}")
                    row.lines.insert(2, f"# text_en = {text_en}")
                validate_sentence(row, require_gold=False)
                if not row.metadata.get("text_en"):
                    raise ValueError(f"Missing translation: {key}")
                if kind == "model-review":
                    try:
                        validate_sentence(row)
                    except ValueError as error:
                        if key not in preserved:
                            raise
                        report["preserved_tree_issues"].append({"sent_id": key, "issue": str(error)})
                result.append(row)
            if len(result) != manifest["count"]:
                raise ValueError("Wrong sentence count")
            if len({row.metadata["text"] for row in result}) != len(result):
                raise ValueError("Duplicate sentence text")
            pairs.append(result)
            target = output / f"{language}-{kind}.conllu"
            target.write_text("".join(row.render() for row in result), encoding="utf-8")
            report["files"][target.name] = {"sha256": file_hash(target), "sentences": len(result)}
        for model, fst in zip(*pairs, strict=True):
            if any(model.metadata[key] != fst.metadata[key] for key in
                   ("sent_id", "text", "text_en", "source_id", "source_sent_id")):
                raise ValueError("Model/FST metadata differs")
            if [word.form for word in model.words] != [word.form for word in fst.words]:
                raise ValueError("Model/FST forms differ")
    write_json(args.work / "annotation-manifest.json", report)
    print(report)


def apply(args):
    """Apply the validated staging files with byte-for-byte source backups."""
    manifest = read_json(args.work / "manifest.json")
    cleaned = Path(manifest["cleaning"])
    cleaning = read_json(cleaned / "manifest.json")
    annotation = read_json(args.work / "annotation-manifest.json")
    batch = Path(manifest["original_batch"])
    backup = args.work / "backup"
    if backup.exists():
        raise ValueError("Backup already exists; refusing to apply twice")
    pairs = []
    for language, name in (("mhr", "mari.conllu"), ("udm", "udmurt.conllu")):
        info = cleaning["languages"][language]
        source = Path(info["input"])
        if file_hash(source) != info["input_sha256"]:
            raise ValueError(f"Full corpus changed since cleaning: {source}")
        if file_hash(cleaned / name) != info["output_sha256"]:
            raise ValueError(f"Staged corpus changed: {name}")
        pairs.append((cleaned / name, source, Path(name)))
        for kind in ("model-review", "fst"):
            name = f"{language}-{kind}.conllu"
            target = batch / name
            staged = args.work / "annotation" / name
            if file_hash(target) != manifest["original_hashes"][name]:
                raise ValueError(f"Annotation input changed: {target}")
            if file_hash(staged) != annotation["files"][name]["sha256"]:
                raise ValueError(f"Staged annotation changed: {staged}")
            pairs.append((staged, target, Path("batch") / name))
    # Preflight every pair before the first source mutation. Backups retain
    # original bytes, including newlines, metadata, and manual annotations.
    record = {"created_at": utc_now(), "files": []}
    for staged, target, relative in pairs:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, saved)
        if file_hash(saved) != file_hash(target):
            raise ValueError(f"Backup verification failed: {target}")
        record["files"].append({"path": str(target), "backup": str(saved),
                                "before_sha256": file_hash(saved),
                                "after_sha256": file_hash(staged)})
    write_json(args.work / "application-plan.json", record)
    for staged, target, _ in pairs:
        temporary = target.with_suffix(target.suffix + ".cleaning-tmp")
        if temporary.exists():
            raise ValueError(f"Unexpected temporary file: {temporary}")
        shutil.copyfile(staged, temporary)
        temporary.replace(target)
    for entry in record["files"]:
        if file_hash(Path(entry["path"])) != entry["after_sha256"]:
            raise ValueError(f"Applied file checksum differs: {entry['path']}")
    write_json(args.work / "applied.json", record)
    print("Applied cleaned corpora and annotation files; originals backed up in", backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--cleaned", type=Path, required=True)
    p.add_argument("--batch", type=Path, default=Path("data/fewshot_batches/004"))
    p.add_argument("--history", type=Path, action="append", required=True)
    p.add_argument("--source-splits", type=Path,
                   default=Path("data/prepared/original-v1/source_splits.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=35)
    p.set_defaults(func=prepare)
    p = commands.add_parser("predict")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--parser-run", type=Path, default=Path("runs/parser-all-001"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.set_defaults(func=predict)
    p = commands.add_parser("assemble")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--translations", type=Path, required=True)
    p.set_defaults(func=assemble)
    p = commands.add_parser("apply")
    p.add_argument("--work", type=Path, required=True)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
