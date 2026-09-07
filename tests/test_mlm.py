import math
import random

import pytest
import torch
from transformers import AutoModelForMaskedLM

from uralic_lm.mlm import build_blocks, evaluate_mlm, fixed_mask, mask_block, pseudo_perplexity


def test_blocks_keep_long_document_tail_and_boundaries(tiny_model, prepared):
    _, tokenizer = tiny_model
    _, rows = prepared
    blocks, audit = build_blocks(rows, tokenizer, 9)
    for record in rows:
        chunks = [b for b in blocks if b["document_id"] == record["id"]]
        recovered = [token for b in chunks for token, special in zip(b["input_ids"], b["special_tokens_mask"]) if not special]
        assert recovered == tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        assert all(len(b["input_ids"]) <= 9 for b in chunks)
        assert all(b["split"] == record["split"] for b in chunks)
        assert chunks[-1]["input_ids"][-1] == tokenizer.eos_token_id


def test_fixed_mask_reproducible_and_specials_never_targets(tiny_model, prepared):
    _, tokenizer = tiny_model
    blocks, _ = build_blocks(prepared[1], tokenizer, 16)
    block = blocks[-1]
    a = fixed_mask(block, tokenizer, .15, 42)
    assert a == fixed_mask(block, tokenizer, .15, 42)
    assert any(label != -100 for label in a["labels"])
    assert all(label == -100 for special, label in zip(block["special_tokens_mask"], a["labels"]) if special)
    changed = [mask_block(block, tokenizer, .5, random.Random(seed)) for seed in range(5)]
    assert len({str(item) for item in changed}) > 1


def test_metrics_independent_of_batch_grouping(tiny_model, prepared):
    path, tokenizer = tiny_model
    blocks, _ = build_blocks(prepared[1][:2], tokenizer, 12)
    model = AutoModelForMaskedLM.from_pretrained(path)
    torch.set_num_threads(1)
    a = evaluate_mlm(model, tokenizer, blocks, torch.device("cpu"), batch_size=1)
    b = evaluate_mlm(model, tokenizer, blocks, torch.device("cpu"), batch_size=3)
    assert a["masked_tokens"] == b["masked_tokens"]
    assert a["mlm_loss"] == pytest.approx(b["mlm_loss"], abs=1e-6)
    assert a["top1"] == b["top1"] and a["top5"] == b["top5"]
    score = pseudo_perplexity(model, tokenizer, blocks, torch.device("cpu"), max_tokens=7)
    assert score["scored_tokens"] == 7 and math.isfinite(score["pseudo_perplexity"])
