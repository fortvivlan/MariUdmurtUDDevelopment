"""Run UniParser and GiellaLT analyzers in the isolated FST environment."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .uniparser import run as run_uniparser
from uralic_lm.common import file_hash, utc_now, write_json, write_jsonl
from uralic_parser.conllu import read_conllu


LANGUAGES = ("mhr", "udm")
ANALYZER_NAME = "analyser-gt-norm.hfstol"


def lookup_forms(forms: list[str], fst: Path, lookup: str) -> dict[str, list[dict]]:
    unique = list(dict.fromkeys(forms))
    if any("\n" in form or "\r" in form for form in unique):
        raise ValueError("Forms cannot contain line breaks")
    result = subprocess.run([lookup, "--pipe-mode=input", str(fst)],
                            input="\n".join(unique) + "\n", capture_output=True,
                            text=True, encoding="utf-8", check=True)
    found = {form: set() for form in unique}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] not in found:
            raise ValueError(f"Unexpected HFST output: {line[:120]}")
        analysis = fields[1]
        if analysis not in {"+?", "?", ""} and not analysis.endswith("+?"):
            found[fields[0]].add(analysis)
    return {form: [{"analysis": analysis, "lemma": analysis.split("+", 1)[0]}
                   for analysis in sorted(analyses)]
            for form, analyses in found.items()}


def run(input_dir: Path, output_dir: Path, model_dirs: dict[str, Path],
        lookup: str = "hfst-optimized-lookup") -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("FST output must be new/empty")
    model_info = {}
    for language in LANGUAGES:
        directory = model_dirs[language]
        fst = directory / ANALYZER_NAME
        license_file = next((directory / name for name in
                             ("LICENCE.txt", "LICENSE.txt", "LICENSE")
                             if (directory / name).is_file()), None)
        metadata = directory / "metadata.json"
        source_record = directory / "source.json"
        if not fst.is_file() or license_file is None or not metadata.is_file():
            raise FileNotFoundError(f"{language}: need {ANALYZER_NAME}, metadata.json and license")
        snapshot = (json.loads(source_record.read_text(encoding="utf-8"))
                    if source_record.is_file() else
                    {"downloaded_at": "unknown", "release": "unknown nightly build"})
        model_info[language] = {"fst": str(fst), "fst_sha256": file_hash(fst),
                                "artifact_url": (snapshot.get("artifact_url") or
                                                 snapshot.get("source_url") or
                                                 "https://models.uralicnlp.com/nightly/"
                                                 f"{language}/{ANALYZER_NAME}"),
                                "metadata": str(metadata),
                                "metadata_sha256": file_hash(metadata),
                                "license": str(license_file),
                                "license_sha256": file_hash(license_file),
                                "model_metadata": json.loads(metadata.read_text(encoding="utf-8")),
                                "source_record": snapshot,
                                "source_record_sha256": (file_hash(source_record)
                                                         if source_record.is_file() else None)}
    lookup_path = Path(lookup) if Path(lookup).is_file() else Path(shutil.which(lookup) or "")
    if not lookup_path.is_file():
        raise FileNotFoundError(f"HFST lookup command not found: {lookup}")
    version = subprocess.run([lookup, "--version"], capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    uni_dir = output_dir / "uniparser"
    uni_manifest = run_uniparser(input_dir, uni_dir, list(LANGUAGES), "strict")
    manifest = {"created_at": utc_now(), "input": str(input_dir),
                "worker_sha256": file_hash(Path(__file__)),
                "model_catalog_url": "https://models.uralicnlp.com/nightly/",
                "uniparser_manifest": str(uni_dir / "manifest.json"),
                "uniparser_manifest_sha256": file_hash(uni_dir / "manifest.json"),
                "lookup_command": lookup, "lookup_version": version,
                "lookup_sha256": file_hash(lookup_path),
                "analysis_rule": "lemma is prefix before first + tag; exact Unicode strings",
                "languages": {}}
    for language in LANGUAGES:
        path = input_dir / f"{language}-predicted.conllu"
        sentences = list(read_conllu(path))
        forms = [word.form for sentence in sentences for word in sentence.words]
        candidates = lookup_forms(forms, Path(model_info[language]["fst"]), lookup)
        output = output_dir / f"{language}-giella.jsonl"
        rows = [{"sent_id": sentence.metadata["sent_id"],
                 "source_id": sentence.metadata["source_id"],
                 "source_sent_id": sentence.metadata["source_sent_id"],
                 "tokens": [{"id": word.id, "form": word.form,
                             "analyses": candidates[word.form]}
                            for word in sentence.words]}
                for sentence in sentences]
        write_jsonl(output, rows)
        manifest["languages"][language] = {
            **model_info[language], "source": str(path), "source_sha256": file_hash(path),
            "giella": str(output), "giella_sha256": file_hash(output),
            "uniparser": uni_manifest["languages"][language]["output"],
            "uniparser_sha256": uni_manifest["languages"][language]["output_sha256"],
            "tokens": len(forms),
            "analyzed_tokens": sum(bool(candidates[form]) for form in forms)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--giella-mhr-dir", type=Path, required=True)
    parser.add_argument("--giella-udm-dir", type=Path, required=True)
    parser.add_argument("--lookup", default="hfst-optimized-lookup")
    args = parser.parse_args()
    run(args.input, args.output, {"mhr": args.giella_mhr_dir,
                                  "udm": args.giella_udm_dir}, args.lookup)


if __name__ == "__main__":
    main()
