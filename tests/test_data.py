import csv
from pathlib import Path
import zipfile

import pytest

from uralic_lm.data import assign_splits, canonical, clean_wikipedia, remove_overlap
from uralic_lm.inel import recover_nganasan, tei_utterances
from uralic_lm.representations import Representation


def record(identifier, text, split="train", language="nio"):
    return {"id": identifier, "source_ids": [identifier], "text": text,
            "split": split, "language": language}


def tei(text, annotation="DO_NOT_TRAIN_ON_THIS"):
    words = "".join(f"<w>{word}</w>" for word in text.split())
    return f'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><annotationBlock who="speaker">
    <u><seg type="utterance" xml:id="seg1">{words}</seg></u>
    <spanGrp type="ge"><span>{annotation}</span></spanGrp>
    <spanGrp type="fr"><span>Russian translation</span></spanGrp></annotationBlock></TEI>'''


def test_tei_surface_words_exclude_annotations():
    rows = list(tei_utterances(tei("ŋuəɁ нʼи ӱ ӥ"), "story_tei.xml"))
    assert rows[0]["text"] == "ŋuəɁ нʼи ӱ ӥ"
    assert rows[0]["recording"] == "story"
    assert rows[0]["utterance_id"] == "seg1"


def test_recovery_quarantines_unmatched_and_ambiguous(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("one_tei.xml", tei("unique first") .replace("</TEI>", tei("shared").split(">", 1)[1]))
        z.writestr("two_tei.xml", tei("shared"))
    local = tmp_path / "nganasan.txt"
    local.write_text("unique first\nshared\nunique first\nunmatched\n", encoding="utf-8")
    docs, audit, counts = recover_nganasan(local, archive)
    assert len(docs) == 1
    assert counts["exact_with_matching_neighbor_anchors"] == 1
    assert counts["unmatched"] == 1
    assert "unmatched" not in docs[0]["text"]


def test_splits_are_reproducible_and_group_duplicate_documents():
    rows = [record(str(i), f"different document {i}") for i in range(25)]
    rows.append(record("duplicate", "different document 0"))
    a = assign_splits(rows)
    b = assign_splits(list(reversed(rows)))
    assignments = lambda records: {sid: row["split"] for row in records for sid in row["source_ids"]}
    assert assignments(a) == assignments(b)
    assert assignments(a)["0"] == assignments(a)["duplicate"]
    assert set(assignments(a).values()) == {"train", "validation", "test"}


def test_protected_passages_removed_without_losing_unrelated_text():
    shared = "one two three four five six seven eight nine ten eleven twelve"
    rows = [record("heldout", shared, "test"), record("dev", shared + "\nvalidation text", "validation"),
            record("train", shared + " extras\nŊanuə maðuɁ\nexcluded example")]
    cleaned, audit = remove_overlap(rows, ["excluded example"])
    assert next(r for r in cleaned if r["id"] == "train")["text"] == "Ŋanuə maðuɁ"
    assert next(r for r in cleaned if r["id"] == "dev")["text"] == "validation text"
    assert len(audit) == 3


def test_article_revisions_cannot_cross_splits():
    rows = [record(str(i), f"document {i}") for i in range(20)]
    rows += [record("0", "another revision"), record("alias", "another revision")]
    grouped = assign_splits(rows)
    owners = [row for row in grouped if "0" in row["source_ids"] or "alias" in row["source_ids"]]
    assert len(owners) == 1
    assert set(owners[0]["source_ids"]) == {"0", "alias"}


def test_unicode_and_representation_split_inheritance():
    original = "ӱ ӧ ӥ ӝ ӟ ӵ ҥ ŋ Ɂ a\u0308 ɨ͡a нʼи"
    cleaned, _ = clean_wikipedia(original + "\nКатегорий:ignore\n[[Target|ӱ]] &amp; ӥ")
    assert cleaned == original + "\nӱ & ӥ"
    r = record("recording", original, "test")
    derived = Representation().apply(r)
    assert derived["text"] == original and derived["source_ids"] == r["source_ids"] and derived["split"] == "test"
    with pytest.raises(ValueError):
        Representation("unimplemented_transliteration").apply(r)
