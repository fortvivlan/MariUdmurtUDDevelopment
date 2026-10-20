from argparse import Namespace
import json

import pytest

from clean_target_data import LanguageFilter, clean_rows, strip_boilerplate
from fewshot_creation.refresh import apply, form_only
from uralic_parser.conllu import Sentence, read_conllu


BOILERPLATE = "Торалык Яндекс.Карте-влак ӱзгарын полышышт дене висалтын."


def sentence(text, sent_id="1", forms=None):
    return Sentence([f"# sent_id = {sent_id}", f"# text = {text}"] + [
        f"{i}\t{form}\t_\t_\t_\t_\t_\t_\t_\t_"
        for i, form in enumerate(forms or text.replace(".", " .").split(), 1)])


class Keep:
    def inspect(self, text, language):
        return {"reason": None}


def test_tab_only_sentence_separator(tmp_path):
    path = tmp_path / "input.conllu"
    path.write_text(sentence("А").render().rstrip("\n") + "\n\t\n"
                    + sentence("Б", "2").render(), encoding="utf-8")
    assert len(list(read_conllu(path))) == 2


def test_strip_boilerplate_preserves_prefix_and_identity():
    row = sentence("Ял верланен." + BOILERPLATE, forms=[
        "Ял", "верланен", ".", "Торалык", "Яндекс.Карте-влак", "ӱзгарын",
        "полышышт", "дене", "висалтын", "."])
    cleaned, changed = strip_boilerplate(row)
    assert changed
    assert cleaned.metadata == {"sent_id": "1", "text": "Ял верланен."}
    assert [word.form for word in cleaned.words] == ["Ял", "верланен", "."]


def test_strip_inside_glued_token_and_genitive_variant():
    phrase = BOILERPLATE.replace("влак ӱзгарын", "влакын ӱзгарыштын")
    row = sentence("Верланен" + phrase, forms=[
        "ВерланенТоралык", "Яндекс.Карте-влакын", "ӱзгарыштын", "полышышт",
        "дене", "висалтын", "."])
    cleaned, changed = strip_boilerplate(row)
    assert changed and cleaned.metadata["text"] == "Верланен"
    assert [word.form for word in cleaned.words] == ["Верланен"]


def test_standalone_boilerplate_removed_except_reviewed():
    row = sentence(BOILERPLATE)
    assert strip_boilerplate(row) == (None, True)
    rows, audit = clean_rows([row], "mhr", Keep(), protected_ids={"1"})
    assert rows == [row] and not audit


def test_exact_text_dedup_prefers_existing_source_identity():
    rows = [sentence("А", "1"), sentence("А", "2"), sentence("А ", "3")]
    cleaned, audit = clean_rows(rows, "udm", Keep(), preferred={"А": "2"})
    assert [row.metadata["sent_id"] for row in cleaned] == ["2", "3"]
    assert audit[0]["kept_sent_id"] == "2"


def filter_with_dictionary(known):
    classifier = LanguageFilter.__new__(LanguageFilter)
    class Model:
        def predict(self, text, k):
            return ["__label__ru", "__label__uk", "__label__bg"], [.98, .01, .005]
    classifier.model = Model()
    classifier.known = lambda word: word in known
    return classifier


def test_russian_only_requires_dictionary_evidence():
    classifier = filter_with_dictionary({"история", "сел", "и", "деревень"})
    assert classifier.inspect("История сел и деревень", "mhr")["reason"] == "russian_only"
    assert classifier.inspect("История сел и деревень кучылтыт", "mhr")["reason"] is None


def test_preserve_target_text_despite_confident_russian_lid():
    classifier = filter_with_dictionary({"пыре", "федеральной", "округе", "со", "но", "кун"})
    for text in ["Пыре федеральной округе", "Со но Кун", "Ижысь федеральной округе",
                 "Россиысь федеральной округе", "Марий ӱмбалне"]:
        assert classifier.inspect(text, "udm")["reason"] is None


def test_short_unknown_target_category_kept():
    classifier = filter_with_dictionary({"категория"})
    assert classifier.inspect("Категория:Букваос", "udm")["reason"] is None


def test_refill_forms_recompute_spacing_without_old_annotations():
    row = sentence("А, Б.", forms=["А", ",", "Б", "."])
    result = form_only(row)
    assert [word.form for word in result.words] == ["А", ",", "Б", "."]
    assert all(word.head is None and word.lemma == "_" for word in result.words)
    assert result.lines[3].endswith("SpaceAfter=No")


def test_apply_refuses_changed_source_before_backup_or_mutation(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    corpus = tmp_path / "mari.conllu"
    corpus.write_text("changed manually", encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps({
        "cleaning": str(cleaned), "original_batch": str(tmp_path / "batch")}), encoding="utf-8")
    (cleaned / "manifest.json").write_text(json.dumps({"languages": {"mhr": {
        "input": str(corpus), "input_sha256": "obsolete hash"}}}), encoding="utf-8")
    (work / "annotation-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since cleaning"):
        apply(Namespace(work=work))
    assert corpus.read_text(encoding="utf-8") == "changed manually"
    assert not (work / "backup").exists()
