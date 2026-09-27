"""Make FST consensus and parser lemma review CoNLL-U files."""

from pathlib import Path

from uralic_lm.common import file_hash, read_json, read_jsonl, utc_now, write_json, write_jsonl
from uralic_parser.conllu import Sentence, read_conllu, validate_sentence


LANGUAGES = ("mhr", "udm")


def lexical(form: str) -> bool:
    return any(character.isalpha() or character.isdigit() for character in form)


def consensus(uniparser: list[dict], giella: list[dict]) -> tuple[str | None, str]:
    first = {row.get("lemma") for row in uniparser if row.get("lemma")}
    second = {row.get("lemma") for row in giella if row.get("lemma")}
    common = first & second
    if len(common) == 1:
        return next(iter(common)), "consensus"
    if not first or not second:
        return None, "unanalyzed"
    return None, "ambiguous_common" if common else "disjoint"


def _aligned(rows: list[Sentence], analyses: list[dict], language: str) -> None:
    if len(rows) != len(analyses):
        raise ValueError(f"{language}: FST sentence count differs")
    for sentence, row in zip(rows, analyses, strict=True):
        if row["sent_id"] != sentence.metadata["sent_id"]:
            raise ValueError(f"{language}: FST sentence ID differs")
        for key in ("source_id", "source_sent_id"):
            if key in row and row[key] != sentence.metadata[key]:
                raise ValueError(f"{language}: FST {key} differs")
        if [(word.id, word.form) for word in sentence.words] != [
                (token["id"], token["form"]) for token in row["tokens"]]:
            raise ValueError(f"{language}: FST word alignment differs")


def review_sentence(sentence: Sentence, uni: dict, giella: dict) -> tuple[Sentence, Sentence, list[dict]]:
    review_lines, fst_lines, audit = [], [], []
    word_index = 0
    for line in sentence.lines:
        if line.startswith("# annotation_status = "):
            review_lines.append("# annotation_status = PARSER_FST_REVIEW")
            fst_lines.append("# annotation_status = FST_LEMMA_DRAFT")
            fst_lines.append("# lemma_source = FST_CONSENSUS")
            continue
        if line.startswith("#"):
            review_lines.append(line)
            fst_lines.append(line)
            continue
        fields = line.split("\t")
        if len(fields) != 10 or not fields[0].isdigit():
            raise ValueError("Expected ten-column syntactic word row")
        uni_token = uni["tokens"][word_index]
        giella_token = giella["tokens"][word_index]
        word_index += 1
        form, parser_lemma = fields[1], fields[2]
        if lexical(form):
            lemma, status = consensus(uni_token["analyses"], giella_token["analyses"])
            if lemma is None:
                fields[2] = parser_lemma + "?"
            elif lemma != parser_lemma:
                fields[2] = parser_lemma + "?" + lemma
        else:
            lemma, status = None, "nonlexical_exempt"
        review_lines.append("\t".join(fields))
        fst_fields = list(fields)
        fst_fields[2:9] = [lemma or "_"] + ["_"] * 6
        misc = [part for part in fields[9].split("|") if part != "_"]
        if lemma is not None:
            misc.append("LemmaSource=FST")
        fst_fields[9] = "|".join(misc) or "_"
        fst_lines.append("\t".join(fst_fields))
        audit.append({"sent_id": sentence.metadata["sent_id"], "id": word_index,
                      "form": form, "parser_lemma": parser_lemma,
                      "uniparser_lemmas": sorted({candidate.get("lemma") for candidate in
                                                  uni_token["analyses"] if candidate.get("lemma")}),
                      "giella_lemmas": sorted({candidate.get("lemma") for candidate in
                                              giella_token["analyses"] if candidate.get("lemma")}),
                      "consensus_lemma": lemma, "status": status})
    review = Sentence(review_lines)
    fst = Sentence(fst_lines)
    validate_sentence(review)
    validate_sentence(fst, require_gold=False)
    return review, fst, audit


def run(prediction_dir: Path, fst_dir: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Review output must be new/empty")
    fst_manifest = read_json(fst_dir / "manifest.json")
    prepared = {}
    for language in LANGUAGES:
        input_path = prediction_dir / f"{language}-predicted.conllu"
        entries = fst_manifest["languages"][language]
        if file_hash(input_path) != entries["source_sha256"]:
            raise ValueError(f"{language}: predictions changed since FST analysis")
        analysis_paths = {"giella": fst_dir / f"{language}-giella.jsonl",
                          "uniparser": fst_dir / "uniparser" / f"{language}-analyses.jsonl"}
        for key, path in analysis_paths.items():
            if file_hash(path) != entries[key + "_sha256"]:
                raise ValueError(f"{language}: {key} analyses changed")
        sentences = list(read_conllu(input_path))
        uni = read_jsonl(analysis_paths["uniparser"])
        giella = read_jsonl(analysis_paths["giella"])
        _aligned(sentences, uni, language)
        _aligned(sentences, giella, language)
        converted = [review_sentence(sentence, u, g)
                     for sentence, u, g in zip(sentences, uni, giella, strict=True)]
        prepared[language] = converted
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utc_now(), "prediction_dir": str(prediction_dir),
                "script_sha256": file_hash(Path(__file__)),
                "fst_dir": str(fst_dir), "fst_manifest_sha256": file_hash(fst_dir / "manifest.json"),
                "languages": {}}
    for language, converted in prepared.items():
        model_path = output / f"{language}-model-review.conllu"
        fst_path = output / f"{language}-fst.conllu"
        audit_path = output / f"{language}-lemma-audit.jsonl"
        with model_path.open("w", encoding="utf-8", newline="\n") as model, \
                fst_path.open("w", encoding="utf-8", newline="\n") as fst:
            for review, fst_sentence, _ in converted:
                model.write(review.render())
                fst.write(fst_sentence.render())
        write_jsonl(audit_path, (item for _, _, rows in converted for item in rows))
        manifest["languages"][language] = {
            "model_review": str(model_path), "model_review_sha256": file_hash(model_path),
            "fst": str(fst_path), "fst_sha256": file_hash(fst_path),
            "audit": str(audit_path), "audit_sha256": file_hash(audit_path),
            "sentences": len(converted)}
    write_json(output / "manifest.json", manifest)
    return manifest
