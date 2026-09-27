"""Restore source text and spacing in zero-shot target CoNLL-U drafts."""

import argparse
import hashlib
import json
from pathlib import Path

from uralic_lm.common import file_hash, fingerprint, utc_now, write_json
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence


SOURCES = {"mhr": "mari.conllu", "udm": "udmurt.conllu"}


def spaces_after(forms: list[str], text: str) -> list[bool]:
    """Return whether each form is immediately followed by the next form."""
    if not forms or not text:
        raise ValueError("Cannot align empty forms or source text")
    spans = []
    cursor = 0
    for form in forms:
        start = text.find(form, cursor)
        if start < 0:
            raise ValueError(f"FORM {form!r} is absent after character {cursor}")
        spans.append((start, start + len(form)))
        cursor = start + len(form)
    gaps = [text[left[1]:right[0]] for left, right in zip(spans, spans[1:])]
    if (text[:spans[0][0]].strip() or text[spans[-1][1]:].strip()
            or any(gap and not gap.isspace() for gap in gaps)):
        raise ValueError("Source text contains non-whitespace outside aligned forms")
    return [gap == "" for gap in gaps] + [False]


def enrich(sentence: Sentence, text: str, translation: str) -> Sentence:
    if not translation or "\n" in translation or "\r" in translation:
        raise ValueError("English translation must be one nonempty line")
    if "text" in sentence.metadata or "text_en" in sentence.metadata:
        raise ValueError("Input already has text comments")
    no_space = spaces_after([word.form for word in sentence.words], text)
    lines = []
    word_index = 0
    for line in sentence.lines:
        lines.append(line)
        if line.startswith("# sent_id = "):
            lines.extend((f"# text = {text}", f"# text_en = {translation}"))
        elif not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) != 10 or not fields[0].isdigit():
                raise ValueError("Expected ten-column syntactic word row")
            misc = [item for item in fields[9].split("|")
                    if item not in {"_", "SpaceAfter=No"}]
            if no_space[word_index]:
                misc.append("SpaceAfter=No")
            fields[9] = "|".join(misc) or "_"
            lines[-1] = "\t".join(fields)
            word_index += 1
    if word_index != len(no_space):
        raise ValueError("Word count changed while enriching")
    result = Sentence(lines)
    validate_sentence(result)
    return result


def source_texts(path: Path, language: str,
                 wanted: dict[str, Sentence]) -> dict[str, str]:
    """Match original sentences by both local sentence ID and article source ID."""
    by_original_id = {}
    for sentence in wanted.values():
        original_id = sentence.metadata["source_sent_id"]
        if original_id in by_original_id:
            raise ValueError(f"Repeated source_sent_id: {original_id}")
        by_original_id[original_id] = sentence
    found = {}
    for original in read_conllu(path):
        original_id = original.metadata.get("sent_id")
        selected = by_original_id.get(original_id)
        if selected is None:
            continue
        source_id = language + ":wiki:" + fingerprint(original.metadata.get("article", ""))[:20]
        if source_id != selected.metadata["source_id"]:
            raise ValueError(f"Source article mismatch for {language}:{original_id}")
        if [word.form for word in original.words] != [word.form for word in selected.words]:
            raise ValueError(f"Source forms changed for {language}:{original_id}")
        if original_id in found:
            raise ValueError(f"Repeated original sentence: {language}:{original_id}")
        found[original_id] = original.metadata.get("text", "")
    missing = set(by_original_id) - set(found)
    if missing:
        raise ValueError(f"Missing {language} source sentences: {sorted(missing)[:5]}")
    return found


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    translation_bytes = args.translations.read_bytes()
    translation_hash = hashlib.sha256(translation_bytes).hexdigest()
    translations = json.loads(translation_bytes.decode("utf-8"))
    inputs = {}
    for language in SOURCES:
        rows = list(read_conllu(args.input / f"{language}-predicted.conllu"))
        if len(rows) != args.per_language:
            raise ValueError(f"{language}: expected {args.per_language} input sentences")
        wanted = {sentence.metadata["sent_id"]: sentence for sentence in rows}
        if len(wanted) != len(rows):
            raise ValueError(f"{language}: duplicate sent_id")
        originals = source_texts(args.source_dir / SOURCES[language], language, wanted)
        inputs[language] = (rows, originals)
    expected = {sentence.metadata["sent_id"] for rows, _ in inputs.values()
                for sentence in rows}
    if set(translations) != expected:
        raise ValueError(f"Translation IDs differ: missing={len(expected-set(translations))}, "
                         f"extra={len(set(translations)-expected)}")
    enriched = {}
    for language, (rows, originals) in inputs.items():
        enriched[language] = [enrich(sentence,
                                     originals[sentence.metadata["source_sent_id"]],
                                     translations[sentence.metadata["sent_id"]])
                              for sentence in rows]
    if file_hash(args.translations) != translation_hash:
        raise ValueError("Translation file changed during enrichment")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(), "parent_draft": str(args.input),
        "parent_manifest_sha256": file_hash(args.input / "manifest.json"),
        "translations": str(args.translations),
        "translations_sha256": translation_hash,
        "script_sha256": file_hash(Path(__file__)),
        "text_source": "original CoNLL-U # text matched by source article and sent_id",
        "space_after_rule": "adjacent FORM spans in original # text",
        "per_language": args.per_language, "languages": {},
    }
    for language, rows in enriched.items():
        output = args.output / f"{language}-predicted.conllu"
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for sentence in rows:
                stream.write(sentence.render())
        source = args.source_dir / SOURCES[language]
        predecessor = args.input / f"{language}-predicted.conllu"
        manifest["languages"][language] = {
            "source": str(source), "source_sha256": file_hash(source),
            "parent_predictions": str(predecessor),
            "parent_predictions_sha256": file_hash(predecessor),
            "output": str(output), "output_sha256": file_hash(output),
            "sentences": len(rows), "words": sum(len(row.words) for row in rows),
            "space_after_no": sum("SpaceAfter=No" in line
                                  for row in rows for line in row.lines),
        }
    if file_hash(args.translations) != translation_hash:
        raise ValueError("Translation file changed while writing output")
    write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs/parser-all-001/fewshot-002"))
    parser.add_argument("--translations", type=Path,
                        default=Path("runs/parser-all-001/fewshot-002/translations-en.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/parser-all-001/fewshot-004"))
    parser.add_argument("--per-language", type=int, default=200)
    args = parser.parse_args()
    result = run(args)
    print({language: (item["sentences"], item["space_after_no"])
           for language, item in result["languages"].items()})


if __name__ == "__main__":
    main()
