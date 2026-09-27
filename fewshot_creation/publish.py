"""Publish the four annotation drafts, keeping provenance outside data/."""

from pathlib import Path
import shutil

from uralic_lm.common import file_hash, read_json, utc_now, write_json
from uralic_parser.conllu import read_conllu, validate_sentence


LANGUAGES = ("mhr", "udm")
KINDS = ("model-review", "fst")


def run(review_dir: Path, output: Path, manifest_path: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Annotation output must be new or empty: {output}")
    if manifest_path.exists():
        raise ValueError(f"Publication manifest already exists: {manifest_path}")
    if output == manifest_path.parent or output in manifest_path.parents:
        raise ValueError("Publication manifest must be outside the annotation directory")
    source_manifest = review_dir / "manifest.json"
    source_manifest_hash = file_hash(source_manifest)
    source_record = read_json(source_manifest)
    prepared = {}
    for language in LANGUAGES:
        model_ids = None
        model_words = None
        model_metadata = None
        for kind in KINDS:
            name = f"{language}-{kind}.conllu"
            path = review_dir / name
            source_hash = file_hash(path)
            if "outputs" in source_record:
                recorded_hash = source_record["outputs"][name]["sha256"]
            else:
                recorded_hash = source_record["languages"][language][
                    "model_review_sha256" if kind == "model-review" else "fst_sha256"]
            if source_hash != recorded_hash:
                raise ValueError(f"Annotation file differs from its run manifest: {path}")
            rows = list(read_conllu(path))
            if not rows:
                raise ValueError(f"Empty annotation file: {path}")
            ids = [row.metadata.get("sent_id") for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate sentence IDs: {path}")
            for row in rows:
                validate_sentence(row, require_gold=kind == "model-review")
                if not row.metadata.get("text_en"):
                    raise ValueError(f"Missing English translation: {path}: {row.metadata.get('sent_id')}")
            words = [[(word.id, word.form) for word in row.words] for row in rows]
            metadata = [tuple(row.metadata.get(key) for key in
                              ("text", "text_en", "source_id", "source_sent_id"))
                        for row in rows]
            if kind == "model-review":
                model_ids, model_words, model_metadata = ids, words, metadata
            elif ids != model_ids or words != model_words or metadata != model_metadata:
                raise ValueError(f"{language}: model and FST annotation files differ")
            if file_hash(path) != source_hash:
                raise ValueError(f"Annotation file changed while being read: {path}")
            prepared[name] = {"source": path, "sha256": source_hash,
                              "sentences": len(rows)}
    if file_hash(source_manifest) != source_manifest_hash:
        raise ValueError("Review manifest changed while being read")
    output.mkdir(parents=True, exist_ok=True)
    published = {}
    for name, entry in prepared.items():
        target = output / name
        shutil.copyfile(entry["source"], target)
        if file_hash(target) != entry["sha256"]:
            raise ValueError(f"Annotation copy differs: {target}")
        published[name] = {"path": str(target), "sha256": entry["sha256"],
                           "sentences": entry["sentences"]}
    manifest = {"created_at": utc_now(), "review_dir": str(review_dir),
                "review_manifest_sha256": source_manifest_hash,
                "script_sha256": file_hash(Path(__file__)),
                "output_dir": str(output), "files": published}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest
