"""Focused, dependency-free checks for parser data and decoding contracts."""

import itertools
from pathlib import Path
import random
import tempfile
import unittest

from uralic_parser.conllu import (Sentence, format_feats, parse_feats, read_conllu,
                                  validate_sentence, write_predictions)
from uralic_parser.lemma import apply_lemma_rule, lemma_rule
from uralic_parser.mst import decode_single_root
from uralic_parser.prepare import _split_internal
from uralic_parser.evaluation import _official_score, bootstrap_uas_las, paired_difference
from predict_fewshot import select_form_only, usable_forms
from uralic_lm.common import fingerprint
from enrich_fewshot import enrich, spaces_after


SAMPLE = Sentence([
    "# sent_id = sample:1", "# text = Ёж идёт.",
    "1\tЁж\tёж\tNOUN\tN\tCase=Nom\t2\tnsubj\t_\tSpaceAfter=No",
    "2\tидёт\tидти\tVERB\tV\tTense=Pres\t0\troot\t_\t_",
    "3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_",
])


class ParserDataTests(unittest.TestCase):
    def test_original_text_spacing_is_restored(self):
        self.assertEqual(spaces_after(["ӱдыр", ",", "йоча", "."],
                                      "ӱдыр, йоча."), [True, False, True, False])
        with self.assertRaises(ValueError):
            spaces_after(["ӱдыр", "йоча"], "ӱдыр — йоча")
        draft = Sentence(["# sent_id = mhr-fewshot-0001",
                          "# annotation_status = ZERO_SHOT_DRAFT",
                          "1\tӱдыр\tӱдыр\tNOUN\t_\t_\t0\troot\t_\t_",
                          "2\t,\t,\tPUNCT\t_\t_\t1\tpunct\t_\t_",
                          "3\tйоча\tйоча\tNOUN\t_\t_\t1\tconj\t_\t_",
                          "4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_"])
        result = enrich(draft, "ӱдыр, йоча.", "A girl, a child.")
        self.assertEqual(result.metadata["text"], "ӱдыр, йоча.")
        self.assertEqual(result.metadata["text_en"], "A girl, a child.")
        self.assertEqual([line.split("\t")[9] for line in result.lines if not line.startswith("#")],
                         ["SpaceAfter=No", "_", "SpaceAfter=No", "_"])

    def test_fewshot_selection_uses_training_forms_only(self):
        self.assertFalse(usable_forms(["thumb", "|", "right", "|", "250", "px"]))
        self.assertTrue(usable_forms(["ӱдыр", "йоча", "пӧрт", "."]))
        train_id = "mhr:wiki:" + fingerprint("train article")[:20]
        test_id = "mhr:wiki:" + fingerprint("test article")[:20]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.conllu"
            rows = []
            for article, sent_id in (("train article", "1"), ("test article", "2")):
                rows.extend([f"# sent_id = {sent_id}", f"# article = {article}",
                             "1\tӱдыр\twrong\tNOUN\tN\tCase=Nom\t_\t_\t_\t_",
                             "2\tйоча\twrong\tVERB\tV\tMood=Ind\t_\t_\t_\t_",
                             "3\tпӧрт\twrong\tX\t_\t_\t_\t_\t_\t_", ""])
            source.write_text("\n".join(rows), encoding="utf-8")
            chosen, details = select_form_only(source, "mhr", {train_id: "train",
                                                       test_id: "test"}, 1, 42, 3, 50)
        self.assertEqual(details["selection"][0]["source_id"], train_id)
        self.assertEqual([word.form for word in chosen[0].words], ["ӱдыр", "йоча", "пӧрт"])
        self.assertTrue(all(word.lemma == word.upos == word.feats == "_"
                            for word in chosen[0].words))

    def test_unicode_lemma_rule_roundtrip(self):
        for form, lemma in (("Ёж", "ёж"), ("пӧртъёс", "пӧрт"),
                            ("мӱндӱ", "Мӱндӱ"), ("ёлка", "ёлка")):
            self.assertEqual(apply_lemma_rule(form, lemma_rule(form, lemma)), lemma)

    def test_gold_tree_and_prediction_output(self):
        validate_sentence(SAMPLE)
        predicted = write_predictions(SAMPLE, [
            {"lemma": "ёж", "upos": "NOUN", "feats": "Case=Nom", "head": 2,
             "deprel": "nsubj"},
            {"lemma": "идти", "upos": "VERB", "feats": "Tense=Pres", "head": 0,
             "deprel": "root"},
            {"lemma": ".", "upos": "PUNCT", "feats": "_", "head": 2,
             "deprel": "punct"},
        ])
        self.assertIn("Ёж\tёж\tNOUN\t_\tCase=Nom\t2", predicted)
        self.assertNotIn("\tN\tCase=Nom", predicted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.conllu"
            path.write_text(predicted, encoding="utf-8")
            self.assertEqual(len(list(read_conllu(path))), 1)

    def test_cycle_is_rejected(self):
        broken = Sentence([line.replace("\t0\troot", "\t1\tconj")
                           for line in SAMPLE.lines])
        with self.assertRaises(ValueError):
            validate_sentence(broken)

    def test_features_preserve_ud_values(self):
        value = "Number=Plur,Sing|Number[psor]=Sing"
        self.assertEqual(format_feats(parse_feats(value)), value)

    def test_nenets_recordings_do_not_cross_local_splits(self):
        sentences = [Sentence([f"# doc_title = rec_{group}",
                               f"# sent_id = s_{group}_{index}",
                               "1\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_"])
                     for group in range(10) for index in range(3)]
        split = _split_internal(sentences, "yrk", 42)
        groups = [{sentence.metadata["doc_title"] for sentence in rows}
                  for rows in split.values()]
        self.assertEqual(sum(map(len, groups)), 10)
        self.assertTrue(all(a.isdisjoint(b) for a, b in itertools.combinations(groups, 2)))

    def test_single_root_decoder(self):
        rng = random.Random(7)
        for size in range(2, 12):
            for _ in range(20):
                scores = [[rng.uniform(-2, 2) for _ in range(size)]
                          for _ in range(size)]
                heads = decode_single_root(scores)
                self.assertEqual(sum(head == 0 for head in heads[1:]), 1)
                for dep in range(1, size):
                    seen = set()
                    node = dep
                    while node:
                        self.assertNotIn(node, seen)
                        seen.add(node)
                        node = heads[node]

    def test_official_ud_scorer_and_paired_articles(self):
        with tempfile.TemporaryDirectory() as directory:
            gold = Path(directory) / "gold.conllu"
            gold.write_text(SAMPLE.render(), encoding="utf-8")
            scores = _official_score(gold, gold)
            self.assertEqual(scores["UAS"]["f1"], 1.0)
            self.assertEqual(scores["BLEX"]["f1"], 1.0)
        primary = {"a": (10, 8, 7), "b": (20, 15, 13)}
        control = {"a": (10, 7, 5), "b": (20, 14, 12)}
        self.assertEqual(bootstrap_uas_las(primary, repeats=20)["articles"], 2)
        self.assertGreater(paired_difference(primary, control, repeats=20)
                           ["las"]["difference"], 0)


if __name__ == "__main__":
    unittest.main()
