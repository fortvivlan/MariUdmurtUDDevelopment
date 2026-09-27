"""Create a few-shot parser cycle, FST review files, and translated drafts.

Run as ``python -m fewshot_creation.pipeline`` from the repository root.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import re

from uralic_lm.common import file_hash, read_json, utc_now, write_json


SOURCE_SPLITS = Path("data/prepared/original-v1/source_splits.json")


def _fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be new or empty: {path}")


def run_cycle(args: argparse.Namespace) -> None:
    from uralic_parser.training import TrainConfig, train
    from .dataset import prepare_targets
    from .prediction import predict_batch

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.round_id):
        raise ValueError("round_id must contain only letters, digits, underscore or hyphen")
    _fresh(args.work_dir)
    if len(args.corrected) != len(args.draft):
        raise ValueError("Each --corrected directory needs one matching --draft directory")
    targets_dir = args.work_dir / "targets"
    target_manifest = prepare_targets(args.corrected, args.draft,
                                      args.source_splits, targets_dir, args.seed)
    parent = TrainConfig.from_json(args.parent_run / "config.json")
    values = asdict(parent)
    values.update({"init_parser_run": str(args.parent_run),
                   "target_manifest": str(targets_dir / "manifest.json"),
                   "target_fraction": .5, "seed": args.seed,
                   "microbatch": 2, "accumulation": 8,
                   "max_steps": 300, "evaluate_every": 25, "log_every": 25,
                   "eval_max_sentences": 100,
                   "early_stopping_patience": 4, "encoder_lr": 2e-6,
                   "head_lr": 2e-5, "smoke": False})
    if args.training_overrides:
        overrides = read_json(args.training_overrides)
        allowed = {"target_fraction", "microbatch", "accumulation", "max_steps",
                   "evaluate_every", "log_every", "eval_max_sentences",
                   "early_stopping_patience", "encoder_lr", "head_lr",
                   "weight_decay", "warmup_fraction", "max_grad_norm",
                   "sampling_alpha", "dropout", "gradient_checkpointing",
                   "max_peak_gpu_gib", "smoke"}
        if set(overrides) - allowed:
            raise ValueError(f"Unsupported training overrides: {sorted(set(overrides) - allowed)}")
        values.update(overrides)
    config = TrainConfig(**values)
    config.validate()
    write_json(args.work_dir / "training-config.json", asdict(config))
    parser_run = args.work_dir / "parser"
    train(config, parser_run)
    predictions = args.work_dir / "predictions"
    prediction_manifest = predict_batch(
        parser_run, args.source_dir, args.source_splits, args.draft,
        targets_dir / "manifest.json", predictions, args.round_id,
        args.per_language, args.seed, args.min_words, args.max_words,
        args.batch_size, config.device)
    write_json(args.work_dir / "manifest.json", {
        "created_at": utc_now(), "round_id": args.round_id,
        "parent_parser_run": str(args.parent_run),
        "parent_checkpoint_sha256": file_hash(args.parent_run / "best_model" / "model.safetensors"),
        "target_manifest": str(targets_dir / "manifest.json"),
        "target_manifest_sha256": file_hash(targets_dir / "manifest.json"),
        "parser_run": str(parser_run), "predictions": str(predictions),
        "prediction_manifest_sha256": file_hash(predictions / "manifest.json"),
        "corrected_batches": len(args.corrected),
        "sentences_per_language": args.per_language,
        "target_counts": {lang: {key: target_manifest["languages"][lang][key]
                                 for key in ("train_sentences", "dev_sentences")}
                          for lang in ("mhr", "udm")},
        "selection": prediction_manifest["selection"],
    })


def predict_only(args: argparse.Namespace) -> None:
    from .prediction import predict_batch

    predict_batch(args.run, args.source_dir, args.source_splits, args.draft,
                  args.target_manifest, args.output, args.round_id,
                  args.per_language, args.seed, args.min_words, args.max_words,
                  args.batch_size, args.device)


def make_fst(args: argparse.Namespace) -> None:
    from .fst_worker import run as run_fst
    from .review import run as make_review

    _fresh(args.output)
    run_fst(args.predictions, args.output / "fst",
            {"mhr": args.giella_mhr_dir, "udm": args.giella_udm_dir}, args.lookup)
    make_review(args.predictions, args.output / "fst", args.output / "review")
    write_json(args.output / "manifest.json", {
        "created_at": utc_now(), "predictions": str(args.predictions),
        "fst_manifest_sha256": file_hash(args.output / "fst" / "manifest.json"),
        "review_manifest_sha256": file_hash(args.output / "review" / "manifest.json")})


def add_translations(args: argparse.Namespace) -> None:
    from .translations import run

    run(args.review, args.translations, args.output)


def publish_annotations(args: argparse.Namespace) -> None:
    from .publish import run

    run(args.review, args.output, args.manifest)


def _selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--draft", type=Path, action="append", required=True,
                        help="Prior batch draft directory; repeat in correction order")
    parser.add_argument("--source-dir", type=Path, default=Path("."))
    parser.add_argument("--source-splits", type=Path, default=SOURCE_SPLITS)
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cycle = commands.add_parser("run", help="Train and create the next form-only prediction batch")
    cycle.add_argument("--parent-run", type=Path, required=True)
    cycle.add_argument("--corrected", type=Path, action="append", required=True,
                       help="Directory with mhr-corrected.conllu and udm-corrected.conllu")
    cycle.add_argument("--work-dir", type=Path, required=True)
    cycle.add_argument("--training-overrides", type=Path)
    _selection_args(cycle)
    cycle.set_defaults(func=run_cycle)

    prediction = commands.add_parser("predict", help="Predict after a resumed training run")
    prediction.add_argument("--run", type=Path, required=True)
    prediction.add_argument("--target-manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    _selection_args(prediction)
    prediction.set_defaults(func=predict_only)

    fst = commands.add_parser("fst", help="Run FSTs and write lemma review files")
    fst.add_argument("--predictions", type=Path, required=True)
    fst.add_argument("--output", type=Path, required=True)
    fst.add_argument("--giella-mhr-dir", type=Path, required=True)
    fst.add_argument("--giella-udm-dir", type=Path, required=True)
    fst.add_argument("--lookup", default="hfst-optimized-lookup")
    fst.set_defaults(func=make_fst)

    translations = commands.add_parser("add-translations", help="Merge subagent translations")
    translations.add_argument("--review", type=Path, required=True)
    translations.add_argument("--translations", type=Path, required=True)
    translations.add_argument("--output", type=Path, required=True)
    translations.set_defaults(func=add_translations)

    publication = commands.add_parser("publish", help="Copy final CoNLL-U drafts to data/fewshot_batches")
    publication.add_argument("--review", type=Path, required=True,
                             help="Translated review directory, or a review with existing text_en")
    publication.add_argument("--output", type=Path, required=True)
    publication.add_argument("--manifest", type=Path, required=True,
                             help="Provenance manifest path under runs/, outside the annotation directory")
    publication.set_defaults(func=publish_annotations)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
