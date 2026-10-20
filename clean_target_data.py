"""Reproducible, conservative cleaning of local Mari/Udmurt CoNLL-U drafts.

Language detection is a heuristic, not a gold language annotation. Require
Russian dictionary coverage as well as LID evidence; retain mixed sentences.
Dependencies for this optional workflow: fasttext-community, pymorphy3.
"""

import argparse
from collections import Counter
from functools import lru_cache
import importlib.metadata
from pathlib import Path
import re

from uralic_lm.common import file_hash, utc_now, write_json, write_jsonl
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence


BOILERPLATE = re.compile(
    r"Торалык Яндекс\.Карт(?:е-влак(?:ын)?|ын) "
    r"ӱзгар(?:ын|ыштын|тын) полышышт дене висалтын\."
)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
TARGET_CHARS = set("ӱӧҥӝӟӥӵӹӓӫöä")
# Corpus-specific Russian bibliography compounds/abbreviations missing from
# OpenCorpora. These are lexical exceptions, not permissive suffix guessing.
RUSSIAN_EXTRA = set("""финно угроведов угроведение угроведения угроведению
финноугроведения финноугроведению финноугроведение священномученик бесермян
конф науч научн практ посвящ посвящён посвящ посвящен отв междунар всерос
филол докл тр авт респ пед межвуз техн студ практич межд кл
волжско марийско русско историко северо карело гражданско романо лугово
угристика ревитализации билингвальной контактологии деятельностного
удмуртоведение фольклорно лит литературоведческий сост ред изд сост""".split())
# Homographs and frequent target forms must not become evidence for Russian.
TARGET_WORDS = {
    "mhr": {"дене", "деч", "але", "уло", "огыл", "тудо", "тиде", "влак", "россий",
            "ий", "кече", "ял", "сола", "шке", "лийын", "шоген", "марте"},
    "udm": {"пыре", "гурт", "гуртын", "сярысь", "вал", "вань", "луэ",
            "быдэ", "со", "татын", "шур", "шуре", "арын", "но", "кун",
            "выль", "вуон", "сылон", "кошкон", "кыл", "уж", "крезьгур"},
}


class LanguageFilter:
    def __init__(self, model_path: Path):
        import fasttext
        import pymorphy3

        self.model = fasttext.load_model(str(model_path))
        self.morph = pymorphy3.MorphAnalyzer()

    @lru_cache(maxsize=None)
    def known(self, word: str) -> bool:
        return word in RUSSIAN_EXTRA or self.morph.word_is_known(word)

    def inspect(self, text: str, language: str) -> dict:
        labels, probabilities = self.model.predict(text.replace("\n", " "), k=3)
        lid = {label.removeprefix("__label__"): float(prob)
               for label, prob in zip(labels, probabilities, strict=True)}
        words = WORD.findall(text)
        lower = [word.lower() for word in words]
        evidence = {"lid": lid, "reason": None}
        if any(char in TARGET_CHARS for char in text.lower()):
            return evidence
        if set(lower) & TARGET_WORDS[language]:
            return evidence
        if language == "udm" and any(re.search(r"(?:ысь|лэн|лэсь|ъёс)$", word)
                                     for word in lower):
            return evidence
        # Ukrainian citations also contaminate the current Mari batch.
        if (language == "mhr" and lid.get("uk", 0) >= .8
                and any(char in text.lower() for char in "іїєґ")):
            return {**evidence, "reason": "ukrainian"}
        cyrillic = [word for word in words if re.fullmatch("[А-Яа-яЁё]+", word)]
        known = [word for word in cyrillic if self.known(word.lower())]
        unknown = [word for word in words if word not in known]
        evidence.update(russian_words=len(known), unknown_words=unknown)
        # Capitalized unknowns may be names, never accept unknown lowercase
        # words or strings from another script as Russian dictionary evidence.
        names_only = all(re.fullmatch("[А-ЯЁ][А-Яа-яЁё]*", word) for word in unknown)
        top = next(iter(lid))
        if (known and not unknown and top == "ru" and lid["ru"] >= .5
                or len(known) >= 4 and names_only and top == "ru" and lid["ru"] >= .8):
            evidence["reason"] = "russian_only"
        return evidence


def strip_boilerplate(sentence: Sentence) -> tuple[Sentence | None, bool]:
    """Remove matched character spans and their tokens, preserving source IDs.

    Input corpora have no dependency heads. Changed annotation drafts must be
    re-predicted before publication. Handle even a token glued to the span.
    """
    text = sentence.metadata["text"]
    spans = [match.span() for match in BOILERPLATE.finditer(text)]
    if not spans:
        return sentence, False
    new_text = BOILERPLATE.sub("", text).strip()
    if not any(char.isalpha() or char.isdigit() for char in new_text):
        return None, True
    lines = []
    cursor = 0
    for line in sentence.lines:
        if line.startswith("# text = "):
            lines.append("# text = " + new_text)
        elif line.startswith("# text_en = "):
            continue  # Translation must be refreshed for the edited sentence.
        elif line.startswith("#"):
            lines.append(line)
        else:
            fields = line.split("\t")
            if not fields[0].isdigit():
                raise ValueError("Boilerplate removal requires simple word tokenization")
            start = text.find(fields[1], cursor)
            if start < 0:
                raise ValueError(f"Cannot align token {fields[1]!r} to {text!r}")
            end = start + len(fields[1])
            cursor = end
            form = "".join(char for pos, char in enumerate(text[start:end], start)
                           if not any(left <= pos < right for left, right in spans))
            if form:
                fields[1] = form
                fields[2:9] = ["_"] * 7
                lines.append("\t".join(fields))
    word_id = 0
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            word_id += 1
            fields = line.split("\t")
            fields[0] = str(word_id)
            lines[index] = "\t".join(fields)
    result = Sentence(lines)
    validate_sentence(result, require_gold=False)
    return result, True


def clean_rows(rows, language, classifier, protected_ids=frozenset(), preferred=None):
    """Deduplicate exact final # text values, optionally preferring batch sources."""
    candidates, audit = [], []
    for row in rows:
        meta = row.metadata
        if meta["sent_id"] in protected_ids:
            candidates.append((row, False))
            continue
        cleaned, changed = strip_boilerplate(row) if language == "mhr" else (row, False)
        entry = {"sent_id": meta["sent_id"], "text": meta["text"]}
        if cleaned is None:
            audit.append({**entry, "action": "remove", "reason": "boilerplate_only"})
            continue
        decision = classifier.inspect(cleaned.metadata["text"], language)
        if decision["reason"]:
            audit.append({**entry, **decision, "action": "remove"})
            continue
        if changed:
            audit.append({**entry, "action": "strip_boilerplate",
                          "cleaned_text": cleaned.metadata["text"]})
        candidates.append((cleaned, changed))
    preferred = preferred or {}
    winners = {}
    for row, _ in candidates:
        text = row.metadata["text"]
        old = winners.get(text)
        if old is None or row.metadata["sent_id"] == preferred.get(text):
            winners[text] = row
    result = []
    for row, changed in candidates:
        meta = row.metadata
        if winners[meta["text"]] is row:
            result.append(row)
        else:
            audit.append({"sent_id": meta["sent_id"], "text": meta["text"],
                          "action": "remove", "reason": "duplicate_text",
                          "kept_sent_id": winners[meta["text"]].metadata["sent_id"]})
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--batch", type=Path, default=Path("data/fewshot_batches/004"))
    parser.add_argument("--lid-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    args.output.mkdir(parents=True)
    classifier = LanguageFilter(args.lid_model)
    manifest = {"created_at": utc_now(), "script_sha256": file_hash(Path(__file__)),
                "lid_model_sha256": file_hash(args.lid_model),
                "lid_model_source": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
                "lid_model_license": "CC-BY-SA-3.0", "languages": {},
                "dependencies": {name: importlib.metadata.version(name) for name in
                                 ("fasttext-community", "pymorphy3", "pymorphy3-dicts-ru")},
                "note": "Heuristic Russian-only/explicit Ukrainian filtering; mixed text retained."}
    for language, name in (("mhr", "mari.conllu"), ("udm", "udmurt.conllu")):
        batch_path = args.batch / f"{language}-model-review.conllu"
        batch = list(read_conllu(batch_path))
        protected = {row.metadata["sent_id"] for row in batch if language == "mhr"
                     and int(row.metadata["sent_id"].split("-")[2]) <= 36}
        cleaned_batch, batch_audit = clean_rows(batch, language, classifier, protected)
        # Keep the exact original source identity of the surviving batch member
        # when identical text occurs in several source articles.
        preferred = {row.metadata["text"]: row.metadata["source_sent_id"]
                     for row in cleaned_batch}
        source = args.input_dir / name
        rows = list(read_conllu(source))
        cleaned, audit = clean_rows(rows, language, classifier, preferred=preferred)
        for filename, values in ((name, cleaned), (f"{language}-retained-review.conllu", cleaned_batch)):
            path = args.output / filename
            path.write_text("".join(row.render() for row in values), encoding="utf-8")
        write_jsonl(args.output / f"{language}-audit.jsonl", audit)
        write_jsonl(args.output / f"{language}-batch-audit.jsonl", batch_audit)
        counts = Counter(entry.get("reason", entry["action"]) for entry in audit)
        batch_counts = Counter(entry.get("reason", entry["action"]) for entry in batch_audit)
        manifest["languages"][language] = {
            "input": str(source), "input_sha256": file_hash(source),
            "input_sentences": len(rows), "output_sentences": len(cleaned),
            "output_sha256": file_hash(args.output / name), "counts": dict(counts),
            "batch_input_sha256": file_hash(batch_path), "batch_input_sentences": len(batch),
            "batch_retained_sentences": len(cleaned_batch), "batch_counts": dict(batch_counts),
            "protected_ids": sorted(protected),
        }
        print(language, manifest["languages"][language], flush=True)
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
