"""Focused checks for repeatable few-shot annotation behavior."""

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fewshot_creation.dataset import prepare_targets
from fewshot_creation.fst_worker import lookup_forms
from fewshot_creation.prediction import select_new
from fewshot_creation.review import consensus, review_sentence
from fewshot_creation.translations import add_translation
from uralic_lm.common import file_hash, fingerprint, write_json, write_jsonl
from uralic_parser.conllu import Sentence, read_conllu
from uralic_parser.vocab import extend_vocab


def _draft(language: str, index: int) -> Sentence:
    source_id = f"{language}:wiki:{fingerprint('article' + str(index))[:20]}"
    return Sentence([
        f"# sent_id = {language}-fewshot-004-{index:04d}",
        "# text = Ӱдыр, йоча пӧрт.",
        f"# source_id = {source_id}", f"# source_sent_id = {index}",
        "# annotation_status = PARSER_DRAFT",
        "1\tӰдыр\tӱдыр\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No",
        "2\t,\t,\tPUNCT\t_\t_\t1\tpunct\t_\t_",
        "3\tйоча\tйоча\tNOUN\t_\t_\t1\tconj\t_\t_",
        "4\tпӧрт\tпӧрт\tNOUN\t_\t_\t1\tconj\t_\tSpaceAfter=No",
        "5\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_",
    ])


def _write(path: Path, rows: list[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.render() for row in rows), encoding="utf-8")


class FewshotCreationTests(unittest.TestCase):
    def test_unique_fst_intersection_and_review_marks(self):
        sentence = _draft("mhr", 1)
        words = sentence.words
        uni = {"tokens": [{"id": word.id, "form": word.form, "analyses": []}
                          for word in words]}
        giella = {"tokens": [{"id": word.id, "form": word.form, "analyses": []}
                             for word in words]}
        uni["tokens"][0]["analyses"] = [{"lemma": "ӱдыр"}, {"lemma": "ӱдыраш"}]
        giella["tokens"][0]["analyses"] = [{"lemma": "ӱдыр"}]
        uni["tokens"][2]["analyses"] = [{"lemma": "йоч"}]
        giella["tokens"][2]["analyses"] = [{"lemma": "йоч"}]
        uni["tokens"][3]["analyses"] = [{"lemma": "пӧрт"}]
        giella["tokens"][3]["analyses"] = [{"lemma": "пӧртыш"}]
        review, fst, audit = review_sentence(sentence, uni, giella)
        self.assertEqual([word.lemma for word in review.words],
                         ["ӱдыр", ",", "йоча?йоч", "пӧрт?", "."])
        self.assertEqual([word.lemma for word in fst.words],
                         ["ӱдыр", "_", "йоч", "_", "_"])
        self.assertEqual(audit[1]["status"], "nonlexical_exempt")
        self.assertEqual(audit[3]["status"], "disjoint")
        self.assertEqual(consensus([{"lemma": "a"}, {"lemma": "b"}],
                                   [{"lemma": "a"}, {"lemma": "b"}]),
                         (None, "ambiguous_common"))

    def test_sampling_excludes_sentences_but_can_reuse_train_article(self):
        article = "training article"
        source_id = "mhr:wiki:" + fingerprint(article)[:20]
        old = (source_id, "1")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mari.conllu"
            rows = []
            for index in (1, 2):
                rows.append(Sentence([
                    f"# sent_id = {index}", f"# article = {article}",
                    "# text = Ӱдыр йоча пӧрт.",
                    "1\tӰдыр\tWRONG\tX\t_\t_\t_\t_\t_\t_",
                    "2\tйоча\tWRONG\tX\t_\t_\t_\t_\t_\t_",
                    "3\tпӧрт\tWRONG\tX\t_\t_\t_\t_\t_\tSpaceAfter=No",
                    "4\t.\t.\tX\t_\t_\t_\t_\t_\t_",
                ]))
            _write(path, rows)
            selected, details = select_new(path, "mhr", {source_id: "train"},
                                           {old}, set(), "0002", 1, 42, 3, 50)
        self.assertEqual(details["selection"][0]["source_sent_id"], "2")
        self.assertEqual(selected[0].metadata["text"], "Ӱдыр йоча пӧрт.")
        self.assertEqual([word.lemma for word in selected[0].words], ["_"] * 4)
        self.assertIn("SpaceAfter=No", selected[0].lines[-2])

    def test_corrected_batches_are_split_by_article_and_reject_marks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft_dir, corrected_dir = root / "draft", root / "corrected"
            splits = []
            for language in ("mhr", "udm"):
                rows = [_draft(language, index) for index in range(1, 11)]
                _write(draft_dir / f"{language}-predicted.conllu", rows)
                corrected = [Sentence([line.replace("PARSER_DRAFT", "GOLD_ADJUDICATED")
                                       for line in row.lines]) for row in rows]
                _write(corrected_dir / f"{language}-corrected.conllu", corrected)
                splits.extend({"split": "train", "source_ids": [row.metadata["source_id"]]}
                              for row in rows)
            split_file = root / "splits.json"
            split_file.write_text(json.dumps(splits), encoding="utf-8")
            result = prepare_targets([corrected_dir], [draft_dir], split_file,
                                     root / "targets")
            self.assertEqual(result["languages"]["mhr"]["dev_sentences"], 1)
            self.assertEqual(result["languages"]["udm"]["train_sentences"], 9)
            self.assertEqual(len(list(read_conllu(root / "targets" / "mhr-dev.conllu"))), 1)
            path = corrected_dir / "mhr-corrected.conllu"
            path.write_text(path.read_text(encoding="utf-8").replace("ӱдыр\tNOUN", "ӱдыр?\tNOUN", 1),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unresolved"):
                prepare_targets([corrected_dir], [draft_dir], split_file, root / "again")

    def test_translation_and_vocab_extension_preserve_originals(self):
        sentence = _draft("mhr", 1)
        translated = add_translation(sentence, "A girl, a child house.", True)
        self.assertEqual(translated.metadata["text_en"], "A girl, a child house.")
        self.assertNotIn("text_en", sentence.metadata)
        parent = {"upos": ["NOUN"], "deprel": ["root"],
                  "lemma_rules": ["copy"], "feats": {"Case": ["_", "Nom"]}}
        observed = {"upos": ["ADJ", "NOUN"], "deprel": ["amod", "root"],
                    "lemma_rules": ["new", "copy"],
                    "feats": {"Number": ["_", "Plur"], "Case": ["_", "Acc"]}}
        merged = extend_vocab(parent, observed)
        self.assertEqual(merged["upos"], ["NOUN", "ADJ"])
        self.assertEqual(merged["feats"]["Case"], ["_", "Nom", "Acc"])
        self.assertEqual(list(merged["feats"]), ["Case", "Number"])

    def test_review_and_translation_file_handoff(self):
        from fewshot_creation.review import run as make_review
        from fewshot_creation.translations import run as merge_translations
        from fewshot_creation.publish import run as publish

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions, fst_dir = root / "predictions", root / "fst"
            entries = {}
            for language in ("mhr", "udm"):
                sentence = _draft(language, 1)
                predicted = predictions / f"{language}-predicted.conllu"
                _write(predicted, [sentence])
                tokens = [{"id": word.id, "form": word.form,
                           "analyses": [{"lemma": word.lemma}] if word.id == 1 else []}
                          for word in sentence.words]
                analysis = {"sent_id": sentence.metadata["sent_id"], "tokens": tokens}
                uni = fst_dir / "uniparser" / f"{language}-analyses.jsonl"
                giella = fst_dir / f"{language}-giella.jsonl"
                uni.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(uni, [analysis])
                write_jsonl(giella, [analysis])
                entries[language] = {"source_sha256": file_hash(predicted),
                                     "uniparser_sha256": file_hash(uni),
                                     "giella_sha256": file_hash(giella)}
            write_json(fst_dir / "manifest.json", {"languages": entries})
            review_dir = root / "review"
            make_review(predictions, fst_dir, review_dir)
            translations = {f"{language}-fewshot-004-0001": "A rough translation."
                            for language in ("mhr", "udm")}
            translation_file = root / "translations.json"
            write_json(translation_file, translations)
            merged = merge_translations(review_dir, translation_file, root / "translated")
            self.assertEqual(len(merged["outputs"]), 4)
            result = next(read_conllu(root / "translated" / "mhr-model-review.conllu"))
            self.assertEqual(result.metadata["text_en"], "A rough translation.")
            self.assertNotIn("text_en", next(read_conllu(
                review_dir / "mhr-model-review.conllu")).metadata)
            with self.assertRaisesRegex(ValueError, "Missing English translation"):
                publish(review_dir, root / "untranslated-batch", root / "untranslated-export.json")
            exported = publish(root / "translated", root / "batch", root / "export.json")
            self.assertEqual(len(exported["files"]), 4)
            self.assertEqual({path.name for path in (root / "batch").iterdir()},
                             {f"{language}-{kind}.conllu"
                              for language in ("mhr", "udm")
                              for kind in ("fst", "model-review")})

    def test_hfst_lookup_keeps_all_readings_and_ignores_unknowns(self):
        class Result:
            stdout = "йоча\tйоча+N+Sg\nйоча\tйоч+V\n\nӱдыр\tӱдыр+?\n"

        with patch("fewshot_creation.fst_worker.subprocess.run", return_value=Result()) as call:
            rows = lookup_forms(["йоча", "ӱдыр", "йоча"], Path("model.hfstol"), "lookup")
        self.assertEqual([row["lemma"] for row in rows["йоча"]], ["йоч", "йоча"])
        self.assertEqual(rows["ӱдыр"], [])
        self.assertEqual(call.call_args.kwargs["input"], "йоча\nӱдыр\n")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment required")
    def test_parent_checkpoint_rows_survive_vocabulary_expansion(self):
        import torch
        from torch import nn
        from safetensors.torch import save_file
        from uralic_parser.training import _load_parent_weights

        class TinyParser(nn.Module):
            def __init__(self, rels: int, tags: int, lemmas: int, features: list[int]):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.rel_biaffine = nn.Module()
                self.rel_biaffine.weight = nn.Parameter(torch.randn(rels, 3, 3))
                self.upos_head = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                                               nn.Identity(), nn.Linear(2, tags))
                self.lemma_head = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                                                nn.Identity(), nn.Linear(2, lemmas))
                self.feature_heads = nn.ModuleList([
                    nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                                  nn.Identity(), nn.Linear(2, count))
                    for count in features])

        parent = TinyParser(1, 1, 1, [2])
        expanded = TinyParser(2, 2, 3, [3, 2])
        old_new_row = expanded.upos_head[4].weight[1].detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best_model" / "model.safetensors"
            path.parent.mkdir()
            save_file({key: value.detach().contiguous()
                       for key, value in parent.state_dict().items()}, str(path))
            _load_parent_weights(expanded, Path(temp),
                                 {"feats": {"Case": ["_", "Nom"]}}, "cpu")
        self.assertTrue(torch.equal(expanded.encoder.weight, parent.encoder.weight))
        self.assertTrue(torch.equal(expanded.rel_biaffine.weight[0],
                                    parent.rel_biaffine.weight[0]))
        self.assertTrue(torch.equal(expanded.upos_head[4].weight[0],
                                    parent.upos_head[4].weight[0]))
        self.assertTrue(torch.equal(expanded.upos_head[4].weight[1], old_new_row))
        self.assertTrue(torch.equal(expanded.feature_heads[0][4].weight[:2],
                                    parent.feature_heads[0][4].weight))


if __name__ == "__main__":
    unittest.main()
