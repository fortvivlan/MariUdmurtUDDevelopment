"""Analyze Mari and Udmurt few-shot forms with pinned UniParser grammars."""

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from uralic_parser.conllu import read_conllu, validate_sentence


ANALYZERS = {
    "mhr": ("uniparser-meadow-mari", "MeadowMariAnalyzer"),
    "udm": ("uniparser-udmurt", "UdmurtAnalyzer"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_analyzer(language: str, mode: str):
    if language == "mhr":
        from uniparser_meadow_mari import MeadowMariAnalyzer

        return MeadowMariAnalyzer(mode=mode)
    from uniparser_udmurt import UdmurtAnalyzer

    return UdmurtAnalyzer(mode=mode)


def analyze_file(path: Path, analyzer) -> tuple[list[dict], dict]:
    sentences = list(read_conllu(path))
    for sentence in sentences:
        validate_sentence(sentence)
    forms = [[word.form for word in sentence.words] for sentence in sentences]
    # One batch amortizes parser setup and preserves the exact sentence/token order.
    analyzed = analyzer.analyze_words(forms, format="json", disambiguate=False)
    if len(analyzed) != len(sentences):
        raise ValueError(f"Sentence count changed while analyzing {path}")
    rows = []
    counts = {"sentences": len(sentences), "tokens": 0, "analyzed_tokens": 0,
              "candidate_analyses": 0}
    for sentence, token_analyses in zip(sentences, analyzed):
        if len(token_analyses) != len(sentence.words):
            raise ValueError(f"Token count changed in {sentence.metadata.get('sent_id')}")
        tokens = []
        for word, candidates in zip(sentence.words, token_analyses):
            if not isinstance(candidates, list):
                raise ValueError(f"Unexpected analyzer output for {word.form!r}")
            # UniParser emits one empty Wordform for an unanalyzed token.
            if len(candidates) == 1 and not candidates[0].get("lemma") and not candidates[0].get("gramm"):
                candidates = []
            if any(candidate.get("wf") != word.form for candidate in candidates):
                raise ValueError(f"Analyzer form mismatch for {word.form!r}")
            candidates.sort(key=lambda candidate: json.dumps(
                candidate, ensure_ascii=False, sort_keys=True))
            tokens.append({"id": word.id, "form": word.form, "analyses": candidates})
            counts["tokens"] += 1
            counts["analyzed_tokens"] += bool(candidates)
            counts["candidate_analyses"] += len(candidates)
        rows.append({"sent_id": sentence.metadata["sent_id"],
                     "source_id": sentence.metadata["source_id"],
                     "source_sent_id": sentence.metadata["source_sent_id"],
                     "text": sentence.metadata["text"], "tokens": tokens})
    return rows, counts


def run(input_dir: Path, output_dir: Path, languages: list[str], mode: str) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output_dir}")
    prepared = {}
    for language in languages:
        path = input_dir / f"{language}-predicted.conllu"
        # UniParser creates errors.log in the working directory at construction.
        # Keep that side effect out of the repository and retain any real errors.
        with TemporaryDirectory(prefix=f"uniparser-{language}-") as temporary:
            original_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                analyzer = make_analyzer(language, mode)
                rows, counts = analyze_file(original_cwd / path, analyzer)
                error_log = Path("errors.log").read_bytes()
            finally:
                os.chdir(original_cwd)
        prepared[language] = (path, rows, counts, error_log)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": str(input_dir), "mode": mode, "disambiguation": False,
        "format": "one JSON object per source sentence; all UniParser candidate analyses",
        "candidate_order": "canonical JSON ascending",
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "dependencies": {name: version(name) for name in
                         ("uniparser-meadow-mari", "uniparser-udmurt",
                          "uniparser-morph", "textdistance", "importlib-resources")},
        "languages": {},
    }
    for language, (path, rows, counts, error_log) in prepared.items():
        output = output_dir / f"{language}-analyses.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["languages"][language] = {
            "analyzer": ANALYZERS[language][1],
            "source": str(path), "source_sha256": sha256(path),
            "output": str(output), "output_sha256": sha256(output), **counts,
        }
        if error_log:
            log_path = output_dir / f"{language}-errors.log"
            log_path.write_bytes(error_log)
            manifest["languages"][language]["errors_sha256"] = sha256(log_path)
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs/parser-all-001/fewshot-004"))
    parser.add_argument("--output", type=Path, default=Path("data/FST_parse_001"))
    parser.add_argument("--language", choices=(*ANALYZERS, "all"), default="all")
    parser.add_argument("--mode", choices=("strict", "nodiacritics"), default="strict")
    args = parser.parse_args()
    languages = list(ANALYZERS) if args.language == "all" else [args.language]
    manifest = run(args.input, args.output, languages, args.mode)
    for language, item in manifest["languages"].items():
        print(f"{language}: {item['analyzed_tokens']}/{item['tokens']} tokens analyzed; "
              f"{item['candidate_analyses']} candidate analyses")


if __name__ == "__main__":
    main()
