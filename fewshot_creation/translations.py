"""Merge separately produced English translations into annotation drafts."""

from pathlib import Path

from uralic_lm.common import file_hash, read_json, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence


LANGUAGES = ("mhr", "udm")
KINDS = ("model-review", "fst")


def add_translation(sentence: Sentence, translation: str, gold_tree: bool) -> Sentence:
    if (not isinstance(translation, str) or not translation.strip() or
            "\n" in translation or "\r" in translation):
        raise ValueError("Translation must be one nonempty line")
    if "text" not in sentence.metadata or "text_en" in sentence.metadata:
        raise ValueError("Expected source text and no existing English translation")
    lines = []
    for line in sentence.lines:
        lines.append(line)
        if line.startswith("# text = "):
            lines.append(f"# text_en = {translation}")
    result = Sentence(lines)
    validate_sentence(result, require_gold=gold_tree)
    return result


def run(review_dir: Path, translations_file: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Translated output must be new/empty")
    translation_hash = file_hash(translations_file)
    translations = read_json(translations_file)
    if not isinstance(translations, dict):
        raise ValueError("Translations must be a JSON object keyed by sent_id")
    prepared = {}
    expected = set()
    for language in LANGUAGES:
        for kind in KINDS:
            path = review_dir / f"{language}-{kind}.conllu"
            rows = list(read_conllu(path))
            ids = [row.metadata.get("sent_id") for row in rows]
            if len(set(ids)) != len(rows) or not rows:
                raise ValueError(f"Duplicate or missing IDs in {path}")
            if kind == "model-review":
                expected.update(ids)
            elif set(ids) != {row.metadata["sent_id"] for row in
                              prepared[(language, "model-review")]}:
                raise ValueError(f"{language}: FST and model review IDs differ")
            prepared[(language, kind)] = rows
    if set(translations) != expected:
        raise ValueError(f"Translation IDs differ: missing={len(expected-set(translations))}, "
                         f"extra={len(set(translations)-expected)}")
    converted = {key: [add_translation(row, translations[row.metadata["sent_id"]],
                                       key[1] == "model-review") for row in rows]
                 for key, rows in prepared.items()}
    if file_hash(translations_file) != translation_hash:
        raise ValueError("Translation file changed while being read")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utc_now(), "review_dir": str(review_dir),
                "script_sha256": file_hash(Path(__file__)),
                "review_manifest_sha256": file_hash(review_dir / "manifest.json"),
                "translations": str(translations_file),
                "translations_sha256": translation_hash, "outputs": {}}
    for (language, kind), rows in converted.items():
        name = f"{language}-{kind}.conllu"
        path = output / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(row.render())
        manifest["outputs"][name] = {"path": str(path), "sha256": file_hash(path),
                                      "sentences": len(rows)}
    if file_hash(translations_file) != translation_hash:
        raise ValueError("Translation file changed while writing output")
    write_json(output / "manifest.json", manifest)
    return manifest
