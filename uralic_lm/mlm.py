"""Token blocks, masking and token-weighted MLM evaluation."""

from collections import defaultdict
from contextlib import nullcontext
import math
import random

import torch
import torch.nn.functional as F

from .common import fingerprint


def build_blocks(records, tokenizer, max_length):
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("Sequence length must leave space for text tokens.")
    blocks, totals = [], defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    for row in records:
        ids = tokenizer(row["text"], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        lang = row["language"]
        totals[lang]["documents"] += 1
        totals[lang]["characters"] += len(row["text"])
        totals[lang]["whitespace_words"] += len(row["text"].split())
        totals[lang]["subtokens"] += len(ids)
        totals[lang]["unknown_tokens"] += ids.count(tokenizer.unk_token_id)
        totals[lang]["documents_over_block_capacity"] += int(len(ids) > capacity)
        totals[lang]["max_document_subtokens"] = max(totals[lang]["max_document_subtokens"], len(ids))
        if len(examples[lang]) < 3:
            examples[lang].append({"source_id": row["id"], "text": row["text"][:160],
                                   "tokens": tokenizer.tokenize(row["text"][:160])})
        for offset in range(0, len(ids), capacity):
            content = ids[offset:offset + capacity]
            sequence = tokenizer.build_inputs_with_special_tokens(content)
            special = tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True)
            # UNK and any other special tokens are never MLM targets.
            if all(special):
                continue
            blocks.append({"id": fingerprint([row["id"], offset, row["representation"]]),
                           "document_id": row["id"], "source_ids": row["source_ids"],
                           "language": lang, "split": row["split"], "offset": offset,
                           "input_ids": sequence, "special_tokens_mask": special})
    audit = {}
    for lang, values in totals.items():
        audit[lang] = {**values, "subtokens_per_whitespace_word": values["subtokens"] / max(1, values["whitespace_words"]),
                       "unknown_token_rate": values["unknown_tokens"] / max(1, values["subtokens"]),
                       "examples": examples[lang]}
    return blocks, audit


def fixed_mask(block, tokenizer, probability, seed):
    rng = random.Random(int(fingerprint([seed, block["id"]]), 16))
    return mask_block(block, tokenizer, probability, rng)


def mask_block(block, tokenizer, probability, rng):
    ids = list(block["input_ids"])
    eligible = [i for i, special in enumerate(block["special_tokens_mask"]) if not special]
    if not eligible:
        raise ValueError("Cannot mask a block containing only special tokens.")
    selected = [i for i in eligible if rng.random() < probability]
    if not selected:
        selected = [rng.choice(eligible)]
    labels = [-100] * len(ids)
    for i in selected:
        labels[i] = ids[i]
        choice = rng.random()
        if choice < .8:
            ids[i] = tokenizer.mask_token_id
        elif choice < .9:
            # Random replacement excludes special IDs, including padding.
            replacement = rng.randrange(len(tokenizer))
            while replacement in tokenizer.all_special_ids:
                replacement = rng.randrange(len(tokenizer))
            ids[i] = replacement
    return {"input_ids": ids, "labels": labels}


def pad_batch(rows, tokenizer, device):
    width = max(len(row["input_ids"]) for row in rows)
    ids, attention, labels = [], [], []
    for row in rows:
        n = len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * (width - n))
        attention.append([1] * n + [0] * (width - n))
        labels.append(row["labels"] + [-100] * (width - n))
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in (("input_ids", ids), ("attention_mask", attention), ("labels", labels))}


def autocast(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def safe_exp(value):
    return math.exp(value) if value < 709 else None


def selected_blocks(blocks, language, split, limit, seed):
    rows = [b for b in blocks if b["language"] == language and b["split"] == split]
    rows.sort(key=lambda b: fingerprint([seed, b["id"]]))
    return rows if limit is None else rows[:limit]


@torch.inference_mode()
def evaluate_mlm(model, tokenizer, blocks, device, precision="fp32", batch_size=2,
                 probability=.15, seed=1729):
    if not blocks:
        raise ValueError("Cannot evaluate an empty split.")
    model.eval()
    loss_sum, count, hits1, hits5 = 0., 0, 0, 0
    for start in range(0, len(blocks), batch_size):
        rows = [fixed_mask(b, tokenizer, probability, seed) for b in blocks[start:start + batch_size]]
        batch = pad_batch(rows, tokenizer, device)
        labels = batch.pop("labels")
        with autocast(device, precision):
            output = model(**batch)
        eligible = labels != -100
        logits = output.logits[eligible].float()
        gold = labels[eligible]
        loss_sum += F.cross_entropy(logits, gold, reduction="sum").item()
        k = min(5, logits.shape[-1])
        predictions = logits.topk(k, dim=-1).indices
        hits1 += (predictions[:, 0] == gold).sum().item()
        hits5 += (predictions == gold[:, None]).any(dim=-1).sum().item()
        count += gold.numel()
        del output, logits
    loss = loss_sum / count
    return {"mlm_loss": loss, "exp_mlm_loss": safe_exp(loss), "top1": hits1 / count,
            "top5": hits5 / count, "masked_tokens": count, "blocks": len(blocks),
            "loss_sum": loss_sum, "mask_seed": seed, "mask_probability": probability,
            "metric_note": "exp_mlm_loss is not ordinary perplexity"}


@torch.inference_mode()
def pseudo_perplexity(model, tokenizer, blocks, device, precision="fp32", max_tokens=512,
                      batch_size=2):
    """Single-subtoken leave-one-out score, with unchanged within-block context.

    This conventional PPPL can exploit other pieces of the same word. It is not
    whole-word PPPL and is not comparable across different tokenizers.
    """
    model.eval()
    pending, total, count = [], 0., 0

    def score(rows):
        batch = pad_batch(rows, tokenizer, device)
        labels = batch.pop("labels")
        with autocast(device, precision):
            output = model(**batch)
        chosen = labels != -100
        return F.cross_entropy(output.logits[chosen].float(), labels[chosen], reduction="sum").item()

    for block in blocks:
        for i, special in enumerate(block["special_tokens_mask"]):
            if special:
                continue
            ids = list(block["input_ids"])
            labels = [-100] * len(ids)
            labels[i], ids[i] = ids[i], tokenizer.mask_token_id
            pending.append({"input_ids": ids, "labels": labels})
            count += 1
            if len(pending) == batch_size:
                total += score(pending)
                pending = []
            if count >= max_tokens:
                break
        if count >= max_tokens:
            break
    if pending:
        total += score(pending)
    if not count:
        raise ValueError("No tokens available for pseudo-perplexity.")
    return {"pseudo_nll": total / count, "pseudo_perplexity": safe_exp(total / count),
            "scored_tokens": count, "protocol": "leave_one_subtoken_out_within_block"}
