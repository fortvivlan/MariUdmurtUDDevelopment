"""Interactive helpers imported by the generated local notebook."""

import gc
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .common import fingerprint, read_json, read_jsonl
from .data import load_prepared


def sample_excerpts(run, language, count=5, split="validation", seed=42):
    config = read_json(Path(run) / "config.json")
    records, _ = load_prepared(config["data"])
    candidates = []
    for record in records:
        if record["language"] != language or record["split"] != split:
            continue
        for line in record["text"].splitlines():
            if 8 <= len(line.split()) <= 40:
                candidates.append({"text": line, "source_ids": record["source_ids"],
                                   "language": language, "split": split,
                                   "representation": record["representation"]})
    candidates.sort(key=lambda r: fingerprint([seed, r["source_ids"], r["text"]]))
    return candidates[:count]


class Inspector:
    def __init__(self, run, checkpoint="adapted", device=None):
        run = Path(run)
        self.config = read_json(run / "config.json")
        self.experiment = read_json(run / "experiment.json")
        self.tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer", use_fast=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if checkpoint not in ("baseline", "adapted"):
            raise ValueError("checkpoint must be baseline or adapted")
        identity = self.experiment["model"]
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(run / "best_model") if checkpoint == "adapted" else identity["model"],
            revision=identity["revision"] if checkpoint == "baseline" else None,
            trust_remote_code=False).to(self.device).eval()

    def tokens(self, text):
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        return [{"index": i, "id": token, "piece": self.tokenizer.convert_ids_to_tokens(token),
                 "span": span, "surface": text[span[0]:span[1]]}
                for i, (token, span) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"]))]

    @torch.inference_mode()
    def predict(self, text, token_index=None, top_k=5):
        encoded = self.tokenizer(text, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.config["max_length"]:
            raise ValueError("Choose a shorter excerpt; manual inspection never silently truncates input.")
        ids = encoded["input_ids"][0]
        positions = (ids == self.tokenizer.mask_token_id).nonzero().flatten().tolist()
        original = None
        if positions:
            if len(positions) != 1 or token_index is not None:
                raise ValueError("Use exactly one explicit mask OR a token_index on an unmasked sentence.")
            position = positions[0]
        else:
            eligible = [i for i, token in enumerate(ids.tolist()) if token not in self.tokenizer.all_special_ids]
            if not eligible:
                raise ValueError("The input contains no non-special tokens to mask")
            position = token_index if token_index is not None else eligible[len(eligible) // 2]
            if position not in eligible:
                raise ValueError("token_index must identify a non-special token")
            original = int(ids[position])
            ids[position] = self.tokenizer.mask_token_id
        logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits[0, position].float()
        probabilities = logits.softmax(-1)
        values, predictions = probabilities.topk(min(top_k, len(self.tokenizer)))
        return {"position": position, "original_token": self.tokenizer.convert_ids_to_tokens(original) if original is not None else None,
                "original_rank": int((logits > logits[original]).sum()) + 1 if original is not None else None,
                "masked_text": self.tokenizer.decode(ids),
                "predictions": [{"piece": self.tokenizer.convert_ids_to_tokens(int(token)),
                                 "decoded": self.tokenizer.decode([int(token)]), "probability": float(prob)}
                                for token, prob in zip(predictions, values)]}

    @torch.inference_mode()
    def candidates(self, prefix, suffix, candidates):
        """Score only candidate-overlapping subtokens; other pieces remain visible.

        Both sum and mean scores are reported. Neither is a calibrated candidate
        probability, and token-length/context-boundary effects require care.
        """
        results = []
        for candidate in candidates:
            if not candidate.strip():
                raise ValueError("Candidates must contain text")
            text = prefix + candidate + suffix
            encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")[0].tolist()
            if encoded["input_ids"].shape[1] > self.config["max_length"]:
                raise ValueError("Candidate context exceeds the experiment sequence length")
            start, end = len(prefix), len(prefix) + len(candidate)
            positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a
                         and int(encoded["input_ids"][0, i]) not in self.tokenizer.all_special_ids]
            if not positions:
                raise ValueError("Candidate has no scoreable subtokens")
            score = 0.
            for position in positions:
                batch = {k: v.clone().to(self.device) for k, v in encoded.items()}
                gold = int(batch["input_ids"][0, position])
                batch["input_ids"][0, position] = self.tokenizer.mask_token_id
                logits = self.model(**batch).logits[0, position].float()
                score += float(logits.log_softmax(-1)[gold])
            results.append({"candidate": candidate, "subtokens": len(positions),
                            "sum_pseudo_log_likelihood": score, "mean_pseudo_log_likelihood": score / len(positions)})
        return results

    def close(self):
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def compare_predictions(run, text, token_index=None, device=None):
    result = {}
    for checkpoint in ("baseline", "adapted"):
        inspector = Inspector(run, checkpoint, device)
        try:
            result[checkpoint] = inspector.predict(text, token_index)
        finally:
            inspector.close()
    return result


def training_events(run):
    return [r for r in read_jsonl(Path(run) / "events.jsonl") if r["event"] in ("train", "validation")]
