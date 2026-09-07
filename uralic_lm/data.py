"""Corpus preparation without ML dependencies or in-place corpus changes."""

from collections import Counter, defaultdict
import csv
import html
import hashlib
from pathlib import Path
import re
import unicodedata

from .common import file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl, utc_now
from .inel import recover_nganasan, NGANASAN_SOURCE, NGANASAN_MD5
from .representations import Representation

TARGETS = ("mhr", "udm", "nio")
FILES = {"mhr": "mari.csv", "udm": "udmurt.csv", "nio": "nganasan.txt",
         "enets": "enets.txt", "nenets": "nenets.txt", "sel": "selkup.txt"}
SOURCES = {
    "mhr": {"name": "Meadow/Eastern Mari Wikipedia", "url": "https://mhr.wikipedia.org/",
            "release": "unconfirmed; legacy extractor names a 2026-02-01 dump",
            "license": "Wikipedia text terms; exact dump metadata not retained"},
    "udm": {"name": "Standard Udmurt Wikipedia", "url": "https://udm.wikipedia.org/",
            "release": "unconfirmed; legacy extractor names a 2026-02-01 dump",
            "license": "Wikipedia text terms; exact dump metadata not retained"},
    "nio": {"name": "INEL Nganasan", "url": "https://www.fdr.uni-hamburg.de/communities/inel/",
            "local_release": "unknown; matched to cited reference archive", "reference": NGANASAN_SOURCE},
    "enets": {"name": "INEL Enets", "url": "https://www.fdr.uni-hamburg.de/communities/inel/",
              "portal": "https://inel.corpora.uni-hamburg.de/portal/corpora/enets/",
              "release": "unknown", "license": "unknown for local extract",
              "variety": "not retained; source corpus includes Forest and Tundra Enets"},
    "nenets": {"name": "INEL Nenets", "url": "https://www.fdr.uni-hamburg.de/communities/inel/",
               "portal": "https://inel.corpora.uni-hamburg.de/portal/corpora/nenets/",
               "release": "unknown", "license": "unknown for local extract",
               "variety": "not retained; source corpus includes Forest and Tundra Nenets"},
    "sel": {"name": "INEL Selkup", "url": "https://inel.corpora.uni-hamburg.de/portal/corpora/selkup/",
            "release": "local version unconfirmed; portal cites 2.0 (2021-12-31)",
            "reference_license": "CC-BY-NC-SA-4.0", "variety": "not retained in local extract"},
}


def canonical(text):
    """Matching key only; never used as model input."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def clean_wikipedia(text):
    kept, removed = [], []
    for original in text.splitlines():
        line = original.strip()
        if not line:
            continue
        if re.match(r"^(?:\{\||\|[}\-]|\||!|__(?:NOTOC|NOEDITSECTION)__|"
                    r"(?:Категорий|Категория|Category|Файл|File|Image):|\d+\s*px\|)", line, re.I):
            removed.append({"reason": "wiki_metadata_line", "text": original})
            continue
        line = re.sub(r"<!--.*?-->", "", line)
        line = re.sub(r"\{\{[^{}]*\}\}", "", line)
        line = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = html.unescape(line)
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if line != original:
            removed.append({"reason": "wiki_inline_cleanup", "text": original, "result": line})
        if line and any(c.isalpha() for c in line):
            kept.append(line)
    return "\n".join(kept), removed


def assign_splits(records, seed=42, validation_fraction=.05, test_fraction=.05):
    """Group exact document duplicates before assigning language-stratified splits."""
    # Union both identical texts and repeated source IDs (e.g. article revisions).
    parents = list(range(len(records)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, record in enumerate(records):
        keys = [("text", record["language"], canonical(record["text"]))]
        keys += [("source", record["language"], sid) for sid in record["source_ids"]]
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    by_language = defaultdict(list)
    for members in groups.values():
        members.sort(key=lambda r: (r["id"], r["text"]))
        record = dict(members[0])
        record["source_ids"] = sorted({sid for r in members for sid in r["source_ids"]})
        texts = {canonical(r["text"]): r["text"] for r in members}
        record["text"] = "\n".join(texts[key] for key in sorted(texts))
        by_language[record["language"]].append(record)
    result = []
    for language, rows in sorted(by_language.items()):
        rows.sort(key=lambda r: fingerprint([seed, r["source_ids"]]))
        nval = ntest = 0
        if language in TARGETS:
            if len(rows) < 3:
                raise ValueError(f"{language}: at least three distinct source documents are required.")
            nval = max(1, round(len(rows) * validation_fraction))
            ntest = max(1, round(len(rows) * test_fraction))
            if nval + ntest >= len(rows):
                raise ValueError(f"{language}: split fractions leave no training documents.")
        for i, row in enumerate(rows):
            row["split"] = "test" if i < ntest else "validation" if i < ntest + nval else "train"
            result.append(row)
    return result


def ngrams(text, width):
    words = canonical(text).split()
    return {tuple(words[i:i + width]) for i in range(len(words) - width + 1)}


def exclusion_texts(paths):
    for path in paths:
        if path.suffix == ".conllu":
            # Reconstruct sentences if # text is absent; skip MWT/empty-node rows.
            text, forms = None, []
            for line in path.read_text(encoding="utf-8-sig").splitlines() + [""]:
                if line.startswith("# text = "):
                    text = line[len("# text = "):]
                elif line and not line.startswith("#"):
                    fields = line.split("\t")
                    if len(fields) == 10 and fields[0].isdigit():
                        forms.append(fields[1])
                elif not line:
                    if text or forms:
                        yield text or " ".join(forms)
                    text, forms = None, []
        else:
            yield from path.read_text(encoding="utf-8-sig").splitlines()


def remove_overlap(records, exclusions=(), width=12):
    """Drop whole paragraphs matching external/held-out passages, preserving audit.

    Matching is exact after NFC/case/whitespace normalization. This is not a
    semantic near-duplicate detector. Test takes precedence over validation.
    """
    protected_lines, protected_grams = set(), set()
    for text in exclusions:
        if canonical(text):
            protected_lines.add(canonical(text))
            protected_grams.update(ngrams(text, width))
    audit, result = [], []
    for split in ("test", "validation", "train"):
        accepted = []
        for row in records:
            if row["split"] != split:
                continue
            kept = []
            for line in row["text"].splitlines():
                if canonical(line) in protected_lines or ngrams(line, width) & protected_grams:
                    audit.append({"id": row["id"], "split": split, "reason": "protected_passage",
                                  "text": line})
                elif canonical(line):
                    kept.append(line)
            if kept:
                accepted.append({**row, "text": "\n".join(kept)})
        result.extend(accepted)
        if split != "train":
            for row in accepted:
                for line in row["text"].splitlines():
                    protected_lines.add(canonical(line))
                    protected_grams.update(ngrams(line, width))
    return result, audit


def prepare(raw_dir, output, archive, seed=42, languages=tuple(FILES), exclusions=(),
            validation_fraction=.05, test_fraction=.05, overlap_words=12, source_metadata=None):
    raw_dir, output, archive = Path(raw_dir), Path(output), Path(archive)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared output must be a new/empty directory; datasets are immutable.")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    if overlap_words < 2:
        raise ValueError("overlap_words must be >= 2")
    unknown = set(languages) - FILES.keys()
    if unknown:
        raise ValueError(f"Unknown languages: {unknown}")
    metadata = {**SOURCES, **(read_json(source_metadata) if source_metadata else {})}
    records, audit, source_files = [], [], {}
    mapping_audit, recovery = [], {}
    for language in languages:
        path = raw_dir / FILES[language]
        source_files[language] = {"file": path.name, "sha256": file_hash(path), "metadata": metadata[language]}
        if language == "nio":
            if not archive.is_file():
                raise ValueError("Nganasan requires source IDs. Run fetch_inel_nganasan.py or supply --nganasan-archive.")
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "md5").hexdigest() != NGANASAN_MD5:
                    raise ValueError("Reference archive is not the pinned INEL Nganasan 1.0 release.")
            rows, mapping_audit, recovery = recover_nganasan(path, archive)
            records.extend(rows)
            source_files[language]["reference_archive_sha256"] = file_hash(archive)
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"article", "text"} <= set(reader.fieldnames or []):
                    raise ValueError(f"{path}: expected article and text columns")
                for i, row in enumerate(reader, 2):
                    title, text = row.get("article") or "", row.get("text") or ""
                    identifier = language + ":wiki:" + fingerprint(title or [i, text])[:20]
                    if title.strip() in {"Кумшо лаштык", "Кутскон бам", "Тӱҥ лаштык", "Main Page"}:
                        audit.append({"id": identifier, "reason": "wiki_main_page", "text": text})
                        continue
                    text, edits = clean_wikipedia(text)
                    audit.extend({"id": identifier, **e} for e in edits)
                    if text:
                        records.append({"id": identifier, "source_ids": [identifier],
                                        "language": language, "source": path.name,
                                        "article": title, "text": text})
        else:
            # No recording boundaries survived in the auxiliary exports.
            text = "\n".join(s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip())
            if text:
                records.append({"id": language + ":local-file", "source_ids": [language + ":local-file"],
                                "language": language, "source": path.name, "text": text,
                                "document_boundaries": "unknown; training only"})
    records = assign_splits(records, seed, validation_fraction, test_fraction)
    source_splits = [{"source_ids": r["source_ids"], "language": r["language"], "split": r["split"]}
                     for r in records]
    records, overlap_audit = remove_overlap(records, exclusion_texts(exclusions), overlap_words)
    audit.extend(overlap_audit)
    records = [Representation().apply(r) for r in records]
    counts = Counter((r["language"], r["split"]) for r in records)
    for language in set(languages) & set(TARGETS):
        if any(counts[language, split] == 0 for split in ("train", "validation", "test")):
            raise ValueError(f"{language}: empty split after filtering; inspect source mapping or use more data.")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "documents.jsonl", records)
    write_jsonl(output / "cleaning_audit.jsonl", audit)
    write_jsonl(output / "nganasan_mapping.jsonl", mapping_audit)
    write_json(output / "source_splits.json", source_splits)
    recipe = {"seed": seed, "languages": list(languages), "validation_fraction": validation_fraction,
              "test_fraction": test_fraction, "overlap_words": overlap_words,
              "representation": Representation().__dict__, "sources": source_files,
              "exclusions": [{"file": p.name, "sha256": file_hash(p)} for p in exclusions],
              "implementation": {name: file_hash(Path(__file__).parent / name)
                                 for name in ("data.py", "inel.py", "representations.py")}}
    manifest = {"created_at": utc_now(), "recipe": recipe, "nganasan_recovery": recovery,
                "counts": {f"{lang}/{split}": n for (lang, split), n in sorted(counts.items())},
                "characters": {f"{lang}/{split}": sum(len(r["text"]) for r in records
                                                       if r["language"] == lang and r["split"] == split)
                               for lang, split in counts},
                "files": {name: file_hash(output / name) for name in
                          ("documents.jsonl", "source_splits.json", "nganasan_mapping.jsonl", "cleaning_audit.jsonl")}}
    manifest["fingerprint"] = fingerprint({"recipe": recipe, "files": manifest["files"]})
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Prepared data changed: {name}. Create a new dataset.")
    if fingerprint({"recipe": manifest["recipe"], "files": manifest["files"]}) != manifest["fingerprint"]:
        raise ValueError("Invalid dataset manifest fingerprint")
    return read_jsonl(path / "documents.jsonl"), manifest
