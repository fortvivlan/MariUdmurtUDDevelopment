from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizerFast
from uralic_lm.common import file_hash, fingerprint, write_json, write_jsonl


@pytest.fixture
def tiny_model(tmp_path):
    path = tmp_path / "tiny-model"
    vocabulary = ["<s>", "<pad>", "</s>", "<unk>", "<mask>",
                  "ӱ", "ӥ", "ŋuəɁ", "word", "different", "sample", "test", "other", "text", "end"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = XLMRobertaTokenizerFast(tokenizer_object=backend, bos_token="<s>", pad_token="<pad>",
                                      eos_token="</s>", unk_token="<unk>", mask_token="<mask>", model_max_length=64)
    tokenizer.save_pretrained(path)
    model = XLMRobertaForMaskedLM(XLMRobertaConfig(vocab_size=len(tokenizer), hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=66,
        bos_token_id=0, pad_token_id=1, eos_token_id=2))
    model.save_pretrained(path)
    return path, tokenizer


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "prepared"
    path.mkdir()
    rows = []
    for lang in ("mhr", "udm", "nio", "enets", "nenets", "sel"):
        for split in (("train", "validation", "test") if lang in ("mhr", "udm", "nio") else ("train",)):
            sid = f"{lang}:{split}"
            rows.append({"id": sid, "source_ids": [sid], "language": lang, "split": split,
                         "text": "ӱ ӥ ŋuəɁ word different sample test other text end " * 2,
                         "representation": {"name": "identity", "version": "1", "mapping_sha256": None}})
    write_jsonl(path / "documents.jsonl", rows)
    files = {"documents.jsonl": file_hash(path / "documents.jsonl")}
    manifest = {"recipe": {"test_fixture": True}, "files": files}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(path / "manifest.json", manifest)
    return path, rows
