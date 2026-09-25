"""Lossless CoNLL-U reading and safe basic-UD output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Word:
    id: int
    form: str
    lemma: str
    upos: str
    feats: str
    head: int | None
    deprel: str


@dataclass
class Sentence:
    lines: list[str]

    @property
    def metadata(self) -> dict[str, str]:
        result = {}
        for line in self.lines:
            if line.startswith("# ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                result[key] = value
        return result

    @property
    def words(self) -> list[Word]:
        result = []
        for line in self.lines:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"Expected ten CoNLL-U columns: {line[:100]}")
            if not fields[0].isdigit():
                continue  # Multiword-token range or enhanced empty node.
            result.append(Word(
                int(fields[0]), fields[1], fields[2], fields[3], fields[5],
                None if fields[6] == "_" else int(fields[6]), fields[7],
            ))
        return result

    def render(self) -> str:
        return "\n".join(self.lines) + "\n\n"


def read_conllu(path: str | Path) -> Iterator[Sentence]:
    lines: list[str] = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if line:
                lines.append(line)
            elif lines:
                yield Sentence(lines)
                lines = []
    if lines:
        yield Sentence(lines)


def validate_sentence(sentence: Sentence, require_gold: bool = True) -> None:
    words = sentence.words
    if not words:
        raise ValueError("Empty sentence")
    ids = [word.id for word in words]
    if ids != list(range(1, len(words) + 1)):
        raise ValueError(f"Nonconsecutive syntactic word IDs: {ids[:12]}")
    if not require_gold:
        return
    if any(word.head is None or word.deprel == "_" for word in words):
        raise ValueError("Gold HEAD/DEPREL missing")
    heads = {word.id: word.head for word in words}
    if sum(head == 0 for head in heads.values()) != 1:
        raise ValueError("A basic UD tree must have exactly one root")
    for word in words:
        if word.head not in range(len(words) + 1) or word.head == word.id:
            raise ValueError("Invalid head index")
        if (word.head == 0) != (word.deprel.split(":", 1)[0] == "root"):
            raise ValueError("Root head and relation disagree")
        seen = {word.id}
        parent = word.head
        while parent != 0:
            if parent in seen:
                raise ValueError("Dependency cycle")
            seen.add(parent)
            parent = heads[parent]


def parse_feats(value: str) -> dict[str, str]:
    if value == "_":
        return {}
    result = {}
    for item in value.split("|"):
        if "=" not in item:
            raise ValueError(f"Invalid FEATS item: {item}")
        key, feature_value = item.split("=", 1)
        if not key or not feature_value or key in result:
            raise ValueError(f"Invalid or repeated FEATS item: {item}")
        result[key] = feature_value
    return result


def format_feats(features: dict[str, str]) -> str:
    return "|".join(f"{key}={features[key]}" for key in sorted(features)) or "_"


def write_predictions(sentence: Sentence, predictions: list[dict]) -> str:
    """Replace labels without copying gold annotations into model output."""
    if len(predictions) != len(sentence.words):
        raise ValueError("Prediction count does not match syntactic word count")
    output = []
    word_index = 0
    for line in sentence.lines:
        if line.startswith("#"):
            output.append(line)
            continue
        fields = line.split("\t")
        if fields[0].isdigit():
            pred = predictions[word_index]
            word_index += 1
            fields[2] = pred["lemma"]
            fields[3] = pred["upos"]
            fields[4] = "_"  # XPOS has no shared target-language vocabulary.
            fields[5] = pred["feats"]
            fields[6] = str(pred["head"])
            fields[7] = pred["deprel"]
            fields[8] = "_"
            misc = [item for item in fields[9].split("|") if item == "SpaceAfter=No"]
            fields[9] = "|".join(misc) or "_"
        else:
            # Preserve tokenization rows but do not leak enhanced annotations.
            fields[2:9] = ["_"] * 7
            misc = [item for item in fields[9].split("|") if item == "SpaceAfter=No"]
            fields[9] = "|".join(misc) or "_"
        output.append("\t".join(fields))
    return "\n".join(output) + "\n\n"
