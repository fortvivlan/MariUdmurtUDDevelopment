"""Official CoNLL-2018 metrics and article-grouped target intervals."""

from collections import defaultdict
from pathlib import Path
import random

from uralic_lm.common import file_hash, read_json, utc_now, write_json
from . import conll18_ud_eval
from .conllu import read_conllu, validate_sentence, write_predictions


METRICS = ("UAS", "LAS", "UPOS", "UFeats", "Lemmas", "MLAS", "BLEX")


def load_best(run: Path, device: str = "cuda"):
    from safetensors.torch import load_file
    from .model import JointUDParser
    from .training import TrainConfig

    config = TrainConfig.from_json(run / "config.json")
    vocab = read_json(run / "vocab.json")
    model = JointUDParser(config.backbone, vocab, config.max_length,
                          config.window_overlap, config.dropout,
                          gradient_checkpointing=False,
                          revision=config.revision,
                          tokenizer_path=config.tokenizer).to(device)
    model.load_state_dict(load_file(str(run / "best_model" / "model.safetensors"),
                                    device=device))
    model.eval()
    return model


def _official_score(gold: Path, predicted: Path) -> dict:
    with gold.open(encoding="utf-8") as stream:
        gold_ud = conll18_ud_eval.load_conllu(stream)
    with predicted.open(encoding="utf-8") as stream:
        pred_ud = conll18_ud_eval.load_conllu(stream)
    scores = conll18_ud_eval.evaluate(gold_ud, pred_ud)
    return {metric: {"f1": scores[metric].f1,
                     "precision": scores[metric].precision,
                     "recall": scores[metric].recall}
            for metric in METRICS}


def _article_counts(gold_path: Path, predicted_path: Path) -> dict[str, tuple[int, int, int]]:
    groups = defaultdict(lambda: [0, 0, 0])
    for gold, predicted in zip(read_conllu(gold_path), read_conllu(predicted_path), strict=True):
        article = gold.metadata.get("source_id")
        if not article:
            raise ValueError("Target gold needs # source_id for article bootstrap")
        for gw, pw in zip(gold.words, predicted.words, strict=True):
            if gw.form != pw.form or gw.id != pw.id:
                raise ValueError("Predicted target words differ from gold input")
            groups[article][0] += 1
            correct_head = gw.head == pw.head
            groups[article][1] += correct_head
            groups[article][2] += correct_head and (
                gw.deprel.split(":", 1)[0] == pw.deprel.split(":", 1)[0]
            )
    return {key: tuple(value) for key, value in groups.items()}


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(.025 * (len(ordered) - 1))],
            ordered[int(.975 * (len(ordered) - 1))]]


def bootstrap_uas_las(counts: dict[str, tuple[int, int, int]],
                      repeats: int = 1000, seed: int = 1729) -> dict:
    if len(counts) < 2:
        raise ValueError("At least two source articles are needed for intervals")
    groups = list(counts.values())
    rng = random.Random(seed)
    uas, las = [], []
    for _ in range(repeats):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        total = sum(row[0] for row in chosen)
        uas.append(sum(row[1] for row in chosen) / total)
        las.append(sum(row[2] for row in chosen) / total)
    return {"uas_95": _interval(uas), "las_95": _interval(las),
            "articles": len(groups), "resamples": repeats, "seed": seed}


def paired_difference(first: dict[str, tuple[int, int, int]],
                      second: dict[str, tuple[int, int, int]],
                      repeats: int = 1000, seed: int = 1729) -> dict:
    if set(first) != set(second):
        raise ValueError("Paired comparisons require the same source articles")
    keys = sorted(first)
    if any(first[key][0] != second[key][0] for key in keys):
        raise ValueError("Paired comparisons require identical token counts")
    rng = random.Random(seed)
    samples = {"uas": [], "las": []}
    for _ in range(repeats):
        picked = [keys[rng.randrange(len(keys))] for _ in keys]
        total = sum(first[key][0] for key in picked)
        for name, offset in (("uas", 1), ("las", 2)):
            samples[name].append(sum(first[key][offset] - second[key][offset]
                                     for key in picked) / total)
    total = sum(first[key][0] for key in keys)
    return {name: {"difference": sum(first[key][offset] - second[key][offset]
                                     for key in keys) / total,
                   "ci95": _interval(samples[name])}
            for name, offset in (("uas", 1), ("las", 2))}


def evaluate_run(run: Path, gold_files: dict[str, Path], output: Path,
                 device: str = "cuda", batch_size: int = 2) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be new/empty; preserve test history")
    output.mkdir(parents=True, exist_ok=True)
    model = load_best(run, device)
    report = {"created_at": utc_now(), "run": str(run),
              "model_sha256": file_hash(run / "best_model" / "model.safetensors"),
              "official_evaluator_sha256": file_hash(Path(conll18_ud_eval.__file__)),
              "protocol": "CoNLL-2018 gold-tokenization UAS/LAS with punctuation; relation subtypes collapsed",
              "languages": {}}
    for language, gold_file in gold_files.items():
        gold_file = Path(gold_file)
        sentences = list(read_conllu(gold_file))
        if not sentences:
            raise ValueError(f"Empty gold input: {gold_file}")
        target = language in {"mhr", "udm"}
        for sentence in sentences:
            validate_sentence(sentence)
            if target and sentence.metadata.get("annotation_status") != "GOLD_ADJUDICATED":
                raise ValueError(f"{gold_file}: target sentences must be human-adjudicated gold")
        predicted_file = output / f"{language}-predicted.conllu"
        with predicted_file.open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(sentences), batch_size):
                part = sentences[start:start + batch_size]
                rows = model.predict(part)
                for sentence, row in zip(part, rows, strict=True):
                    stream.write(write_predictions(sentence, row))
        result = {"gold": str(gold_file), "gold_sha256": file_hash(gold_file),
                  "predicted": str(predicted_file), "predicted_sha256": file_hash(predicted_file),
                  "sentences": len(sentences),
                  "metrics": _official_score(gold_file, predicted_file)}
        if target:
            result["article_bootstrap"] = bootstrap_uas_las(
                _article_counts(gold_file, predicted_file)
            )
        else:
            result["source_split_note"] = (
                "locally repartitioned former official test" if language == "yrk"
                else "official UD r2.18 test"
            )
        report["languages"][language] = result
        write_json(output / "report.json", report)
    return report
