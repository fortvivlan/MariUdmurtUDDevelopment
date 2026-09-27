"""Sample source forms and create linked parser drafts."""

from pathlib import Path
import re
import string

from enrich_fewshot import spaces_after
from uralic_lm.common import file_hash, fingerprint, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence, write_predictions

from .dataset import source_splits


INPUTS = {"mhr": "mari.conllu", "udm": "udmurt.conllu"}


def usable_forms(forms: list[str]) -> bool:
    if any("|" in form or any(char in string.ascii_letters for char in form)
           for form in forms):
        return False
    cyrillic = sum(any("\u0400" <= char <= "\u052f" for char in form)
                   for form in forms)
    return cyrillic >= 3 and 4 * cyrillic >= 3 * len(forms)


def excluded_sentences(draft_dirs: list[Path]) -> dict[str, set[tuple[str, str]]]:
    excluded = {language: set() for language in INPUTS}
    for directory in draft_dirs:
        for language in INPUTS:
            path = directory / f"{language}-predicted.conllu"
            for row in read_conllu(path):
                meta = row.metadata
                excluded[language].add((meta["source_id"], meta["source_sent_id"]))
    return excluded


def select_new(source: Path, language: str, splits: dict[str, str],
               excluded: set[tuple[str, str]], dev_articles: set[str],
               round_id: str, count: int, seed: int, min_words: int,
               max_words: int) -> tuple[list[Sentence], dict]:
    if count < 1 or not 1 <= min_words <= max_words:
        raise ValueError("Invalid selection count or word range")
    by_article = {}
    eligible = 0
    for original in read_conllu(source):
        meta = original.metadata
        article = meta.get("article", "")
        if not article:
            continue
        source_id = language + ":wiki:" + fingerprint(article)[:20]
        source_sent_id = meta.get("sent_id", "")
        if (splits.get(source_id) != "train" or source_id in dev_articles
                or not source_sent_id or (source_id, source_sent_id) in excluded):
            continue
        words = original.words
        forms = [word.form for word in words]
        if (not min_words <= len(words) <= max_words or
                [word.id for word in words] != list(range(1, len(words) + 1)) or
                any(not form or form == "_" for form in forms) or
                not usable_forms(forms)):
            continue
        original_text = meta.get("text", "")
        try:
            no_space = spaces_after(forms, original_text)
        except ValueError:
            continue
        eligible += 1
        rank = fingerprint([seed, source_id, source_sent_id, forms])
        if source_id not in by_article or rank < by_article[source_id][0]:
            by_article[source_id] = (rank, source_sent_id, forms, original_text, no_space)
    if len(by_article) < count:
        raise ValueError(f"{language}: only {len(by_article)} eligible articles")
    chosen = sorted(by_article.items(), key=lambda pair: fingerprint(
        [seed, pair[0]]))[:count]
    sentences, selection = [], []
    for index, (source_id, (_, original_id, forms, original_text, no_space)) in enumerate(chosen, 1):
        sent_id = f"{language}-fewshot-{round_id}-{index:04d}"
        lines = [f"# sent_id = {sent_id}", f"# text = {original_text}",
                 f"# source_id = {source_id}", f"# source_sent_id = {original_id}",
                 "# annotation_status = FORM_ONLY"]
        for word_id, (form, adjacent) in enumerate(zip(forms, no_space, strict=True), 1):
            lines.append(f"{word_id}\t{form}\t_\t_\t_\t_\t_\t_\t_\t"
                         f"{'SpaceAfter=No' if adjacent else '_'}")
        sentence = Sentence(lines)
        validate_sentence(sentence, require_gold=False)
        sentences.append(sentence)
        selection.append({"sent_id": sent_id, "source_id": source_id,
                          "source_sent_id": original_id, "words": len(forms)})
    return sentences, {"eligible_sentences": eligible,
                       "eligible_articles": len(by_article), "selection": selection}


def predict_batch(run: Path, source_dir: Path, split_file: Path,
                  prior_drafts: list[Path], target_manifest: Path, output: Path,
                  round_id: str, count: int = 200, seed: int = 42,
                  min_words: int = 5, max_words: int = 50,
                  batch_size: int = 2, device: str = "cuda") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", round_id):
        raise ValueError("round_id must contain only letters, digits, underscore or hyphen")
    if batch_size < 1 or output.exists() and any(output.iterdir()):
        raise ValueError("Invalid batch size or output is not new/empty")
    splits = source_splits(split_file)
    excluded = excluded_sentences(prior_drafts)
    prior_ids = {row.metadata["sent_id"] for directory in prior_drafts
                 for language in INPUTS
                 for row in read_conllu(directory / f"{language}-predicted.conllu")}
    from uralic_lm.common import read_json
    targets = read_json(target_manifest)
    selected = {}
    for language, filename in INPUTS.items():
        selected[language] = select_new(
            source_dir / filename, language, splits, excluded[language],
            set(targets["languages"][language]["dev_articles"]),
            round_id, count, seed, min_words, max_words)
        if any(row.metadata["sent_id"] in prior_ids for row in selected[language][0]):
            raise ValueError(f"{language}: round_id reuses prior sentence IDs")
    from uralic_parser.evaluation import load_best

    model = load_best(run, device)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utc_now(), "round_id": round_id,
                "script_sha256": file_hash(Path(__file__)),
                "parser_run": str(run), "parser_checkpoint_sha256": file_hash(
                    run / "best_model" / "model.safetensors"),
                "source_splits": str(split_file), "source_splits_sha256": file_hash(split_file),
                "target_manifest": str(target_manifest),
                "target_manifest_sha256": file_hash(target_manifest),
                "prior_drafts": [{"path": str(path), "manifest_sha256": (
                    file_hash(path / "manifest.json") if (path / "manifest.json").exists()
                    else None)} for path in prior_drafts],
                "selection": {"count": count, "seed": seed, "min_words": min_words,
                              "max_words": max_words, "batch_size": batch_size},
                "languages": {}}
    translations = {}
    for language, filename in INPUTS.items():
        sentences, details = selected[language]
        forms_path = output / f"{language}-forms.conllu"
        predicted_path = output / f"{language}-predicted.conllu"
        with forms_path.open("w", encoding="utf-8", newline="\n") as stream:
            for sentence in sentences:
                stream.write(sentence.render())
        with predicted_path.open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(sentences), batch_size):
                part = sentences[start:start + batch_size]
                for sentence, predictions in zip(part, model.predict(part), strict=True):
                    draft = Sentence(write_predictions(sentence, predictions).rstrip("\n").split("\n"))
                    draft.lines = [line.replace("# annotation_status = FORM_ONLY",
                                                "# annotation_status = PARSER_DRAFT")
                                   for line in draft.lines]
                    validate_sentence(draft)
                    stream.write(draft.render())
                    translations[draft.metadata["sent_id"]] = draft.metadata["text"]
        manifest["languages"][language] = {
            "source": str(source_dir / filename), "source_sha256": file_hash(source_dir / filename),
            "forms": str(forms_path), "forms_sha256": file_hash(forms_path),
            "predicted": str(predicted_path), "predicted_sha256": file_hash(predicted_path),
            **details}
    write_json(output / "translation-request.json", translations)
    manifest["translation_request_sha256"] = file_hash(output / "translation-request.json")
    write_json(output / "manifest.json", manifest)
    return manifest
