"""Fetch pinned UD treebanks and make source-only parser splits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

from uralic_lm.common import file_hash, fingerprint, utc_now, write_json
from .conllu import Sentence, read_conllu, validate_sentence


RELEASE = "r2.18"


@dataclass(frozen=True)
class Treebank:
    language: str
    repo: str
    prefix: str
    official_splits: tuple[str, ...]


TREEBANKS = (
    Treebank("fi", "UD_Finnish-TDT", "fi_tdt", ("train", "dev", "test")),
    Treebank("hu", "UD_Hungarian-Szeged", "hu_szeged", ("train", "dev", "test")),
    Treebank("myv", "UD_Erzya-JR", "myv_jr", ("train", "test")),
    Treebank("mdf", "UD_Moksha-JR", "mdf_jr", ("train", "test")),
    Treebank("et", "UD_Estonian-EDT", "et_edt", ("train", "dev", "test")),
    Treebank("koi", "UD_Komi_Permyak-UH", "koi_uh", ("train", "test")),
    Treebank("kpv", "UD_Komi_Zyrian-Lattice", "kpv_lattice", ("train", "test")),
    Treebank("yrk", "UD_Nenets-Tundra", "yrk_tundra", ("test",)),
)


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _group_key(sentence: Sentence, language: str) -> str:
    metadata = sentence.metadata
    if language == "yrk":
        # The Tundra treebank repeats doc_title for utterances from one recording.
        return metadata.get("doc_title") or metadata.get("sound_url") or ""
    sent_id = metadata.get("sent_id", "")
    if "brat_2018:" in sent_id:
        return "brat_2018"
    if ":" in sent_id:
        return sent_id.rsplit(":", 1)[0]
    return sent_id.rsplit("_", 1)[0]


def _select_groups(groups: dict[str, list[Sentence]], goal: int, seed: int,
                   forbidden: set[str] = frozenset()) -> set[str]:
    candidates = [key for key in groups if key not in forbidden]
    candidates.sort(key=lambda key: fingerprint([seed, key]))
    selected, count = set(), 0
    for key in candidates:
        n = len(groups[key])
        if count >= goal:
            break
        if count and count + n > max(goal * 1.2, goal + 2):
            continue
        selected.add(key)
        count += n
    if not selected and candidates:
        selected.add(min(candidates, key=lambda key: len(groups[key])))
    return selected


def _split_internal(sentences: list[Sentence], language: str, seed: int) -> dict[str, list[Sentence]]:
    groups: dict[str, list[Sentence]] = defaultdict(list)
    for sentence in sentences:
        key = _group_key(sentence, language)
        if not key:
            raise ValueError(f"{language}: missing source/recording ID for grouped split")
        groups[key].append(sentence)
    if language == "myv":
        dev = _select_groups(groups, round(len(sentences) * .10), seed, {"brat_2018"})
        return {"train": [s for s in sentences if _group_key(s, language) not in dev],
                "dev": [s for s in sentences if _group_key(s, language) in dev]}
    if language == "yrk":
        if len(groups) < 3:
            raise ValueError("Nenets needs at least three recording groups")
        test = _select_groups(groups, round(len(sentences) * .10), seed)
        remaining = {key: value for key, value in groups.items() if key not in test}
        dev = _select_groups(remaining, round(len(sentences) * .10), seed + 1)
        return {
            "train": [s for s in sentences if _group_key(s, language) not in test | dev],
            "dev": [s for s in sentences if _group_key(s, language) in dev],
            "test": [s for s in sentences if _group_key(s, language) in test],
        }
    raise ValueError(f"No internal split policy for {language}")


def _write_sentences(path: Path, sentences: list[Sentence]) -> None:
    if not sentences:
        raise ValueError(f"Empty output split: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for sentence in sentences:
            stream.write(sentence.render())


def prepare(source_root: Path, output: Path, seed: int = 42, fetch: bool = True) -> dict:
    source_root, output = Path(source_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Parser preparation output must be a new/empty directory")
    output.mkdir(parents=True, exist_ok=True)
    raw_inputs, written, counts = {}, {}, Counter()
    splits_by_language: dict[str, dict[str, list[Sentence]]] = {}
    for treebank in TREEBANKS:
        raw_dir = source_root / treebank.repo
        files = {}
        for split in treebank.official_splits:
            name = f"{treebank.prefix}-ud-{split}.conllu"
            path = raw_dir / name
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{treebank.repo}/{RELEASE}/{name}"
            if fetch:
                _download(url, path)
            if not path.is_file():
                raise FileNotFoundError(path)
            sentences = list(read_conllu(path))
            for sentence in sentences:
                validate_sentence(sentence)
            files[split] = sentences
            raw_inputs[f"{treebank.language}/{split}"] = {
                "path": str(path), "url": url, "sha256": file_hash(path),
                "sentences": len(sentences),
            }
        license_path = raw_dir / "LICENSE.txt"
        license_url = f"https://raw.githubusercontent.com/UniversalDependencies/{treebank.repo}/{RELEASE}/LICENSE.txt"
        if fetch:
            _download(license_url, license_path)
        if not license_path.is_file():
            raise FileNotFoundError(license_path)
        raw_inputs[f"{treebank.language}/license"] = {
            "path": str(license_path), "url": license_url,
            "sha256": file_hash(license_path),
            "text": license_path.read_text(encoding="utf-8"),
        }
        if treebank.language in {"myv", "yrk"}:
            split_source = files.pop("train" if treebank.language == "myv" else "test")
            files.update(_split_internal(split_source, treebank.language, seed))
        splits_by_language[treebank.language] = files

    # Parallel brat_2018 examples must not cross source train/evaluation splits.
    seen_parallel: dict[str, set[str]] = defaultdict(set)
    for language, splits in splits_by_language.items():
        for split, sentences in splits.items():
            for sentence in sentences:
                sid = sentence.metadata.get("sent_id", "")
                if "brat_2018:" in sid:
                    seen_parallel[sid.split("brat_2018:", 1)[1]].add(split)
    leaked = {key: value for key, value in seen_parallel.items() if len(value) > 1}
    if leaked:
        raise ValueError(f"Parallel source sentences cross splits: {list(leaked)[:8]}")

    for language, splits in splits_by_language.items():
        for split, sentences in splits.items():
            path = output / f"{language}-{split}.conllu"
            _write_sentences(path, sentences)
            written[f"{language}/{split}"] = {"path": str(path), "sha256": file_hash(path)}
            counts[f"{language}/{split}"] = len(sentences)
    manifest = {
        "created_at": utc_now(), "release": RELEASE, "seed": seed,
        "policy": {
            "official_tests_preserved_except": ["yrk"],
            "myv_dev": "10% of official train, grouped by sent_id source, brat_2018 in train",
            "yrk": "official test repurposed; 80/10/10 by doc_title/recording",
            "mdf_koi_kpv": "all official train retained; no development split",
        },
        "treebanks": [treebank.__dict__ for treebank in TREEBANKS],
        "raw_inputs": raw_inputs, "outputs": written, "counts": dict(counts),
    }
    manifest["fingerprint"] = fingerprint({key: value for key, value in manifest.items() if key != "created_at"})
    write_json(output / "manifest.json", manifest)
    return manifest
