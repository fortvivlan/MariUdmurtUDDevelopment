"""Single-device MLM training with explicit local experiment records."""

from collections import Counter
from dataclasses import asdict, dataclass, field
import importlib.metadata
import logging
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from .common import file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .data import TARGETS, load_prepared
from .mlm import (autocast, build_blocks, evaluate_mlm, mask_block, pad_batch,
                  pseudo_perplexity, selected_blocks)


@dataclass
class TrainConfig:
    model: str = "cis-lmu/glot500-base"
    revision: str = "main"
    data: str = "data/prepared/original-v1"
    seed: int = 42
    eval_seed: int = 1729
    max_length: int = 256
    batch_size: int = 2
    eval_batch_size: int = 2
    accumulation_steps: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = .01
    adam_beta1: float = .9
    adam_beta2: float = .999
    adam_epsilon: float = 1e-8
    warmup_fraction: float = .06
    max_grad_norm: float = 1.
    mask_probability: float = .15
    precision: str = "bf16"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"
    early_stopping_patience: int = 2
    minimum_improvement: float = 0.
    language_weights: dict = field(default_factory=lambda: {
        "mhr": .3, "udm": .3, "nio": .3, "enets": 1 / 30, "nenets": 1 / 30, "sel": 1 / 30})
    targets: list = field(default_factory=lambda: list(TARGETS))
    num_workers: int = 0
    cpu_threads: int = 8
    logging_steps: int = 10
    checkpoint_steps: int = 100
    keep_checkpoints: int = 2
    max_train_blocks_per_language: int | None = None
    eval_max_blocks: int | None = None
    train_diagnostic_blocks: int = 64
    pseudo_perplexity_tokens: int = 0
    deterministic: bool = True

    def validate(self):
        for name in ("max_length", "batch_size", "eval_batch_size", "accumulation_steps", "epochs",
                     "early_stopping_patience", "cpu_threads", "logging_steps", "checkpoint_steps", "keep_checkpoints"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_train_blocks_per_language", "eval_max_blocks"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be null or positive")
        if self.precision not in ("fp32", "bf16") or self.device not in ("cpu", "cuda"):
            raise ValueError("Use device cpu/cuda and precision fp32/bf16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("Use precision fp32 for CPU smoke runs")
        if not 0 < self.mask_probability <= 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid masking or warmup probability")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.adam_epsilon <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.num_workers < 0 or self.pseudo_perplexity_tokens < 0 or self.train_diagnostic_blocks < 1:
            raise ValueError("Invalid worker/evaluation limits")
        if not self.targets or len(self.targets) != len(set(self.targets)):
            raise ValueError("Select distinct target languages")
        if not set(self.targets) <= set(self.language_weights):
            raise ValueError("Every target must have a positive training weight")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement cannot be negative")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.language_weights or any(not math.isfinite(w) or w <= 0 for w in self.language_weights.values()):
            raise ValueError("Sampling weights must be finite and positive")


class RunLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.logger = logging.getLogger(str(self.directory.resolve()))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handlers = [logging.FileHandler(self.directory / "console.log", encoding="utf-8"),
                         logging.StreamHandler(sys.stdout)]
        for handler in self.handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(handler)

    def event(self, event, **values):
        import json
        record = {"timestamp": utc_now(), "event": event, **values}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.logger.info("%s %s", event, json.dumps(values, ensure_ascii=False))

    def close(self):
        for handler in self.handlers:
            self.logger.removeHandler(handler)
            handler.close()


def code_identity():
    root = Path(__file__).parent.parent
    files = sorted((root / "uralic_lm").glob("*.py")) + [root / "finetune_backbone.py"]
    return {str(path.relative_to(root)): file_hash(path) for path in files if path.exists()}


def environment():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        revision, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except FileNotFoundError:
        revision = dirty = None
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            # Resolve each name through import precedence. Environments inheriting
            # system packages can expose multiple distributions of the same name.
            "packages": {name: importlib.metadata.version(name) for name in
                         sorted({d.metadata["Name"] for d in importlib.metadata.distributions()})},
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "git_revision": revision, "git_dirty_status": dirty, "implementation": code_identity()}


def identity_collate(rows):
    return rows


def epoch_schedule(blocks, weights, seed, epoch):
    groups = {}
    for language in sorted(weights):
        groups[language] = [i for i, b in enumerate(blocks) if b["language"] == language]
        if not groups[language]:
            raise ValueError(f"No training blocks for {language}")
    rng = random.Random(int(fingerprint([seed, epoch, "sampling"]), 16))
    languages = list(groups)
    schedule = []
    for language in rng.choices(languages, weights=[weights[l] for l in languages], k=len(blocks)):
        schedule.append(rng.choice(groups[language]))
    return schedule


def configure_device(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the Windows CUDA environment, or explicit CPU/fp32 smoke settings.")
    if config.device == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; select fp32.")
    torch.set_num_threads(config.cpu_threads)
    set_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device(config.device)


def get_tokenizer_and_identity(config):
    model_config = AutoConfig.from_pretrained(config.model, revision=config.revision, trust_remote_code=False)
    if model_config.model_type != "xlm-roberta":
        raise ValueError("This baseline supports XLM-R/Glot500 MLM checkpoints only.")
    resolved_revision = getattr(model_config, "_commit_hash", None)
    local = Path(config.model).is_dir()
    if not local and not resolved_revision:
        raise ValueError("Could not resolve the Hub model to an immutable revision.")
    identity = {"model": config.model, "revision": resolved_revision,
                "local_files": {p.name: file_hash(p) for p in sorted(Path(config.model).iterdir())
                                if p.is_file()} if local else None}
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=resolved_revision or config.revision,
                                              use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("A fast tokenizer with mask and padding tokens is required.")
    # XLM-R reserves pad_token_id + 1 positions before content position IDs.
    maximum = model_config.max_position_embeddings - model_config.pad_token_id - 1
    if config.max_length > maximum:
        raise ValueError(f"max_length exceeds this model's supported length ({maximum}).")
    return tokenizer, identity


def load_model(config, identity, device, checkpoint=None):
    model = AutoModelForMaskedLM.from_pretrained(
        str(checkpoint) if checkpoint else config.model,
        revision=None if checkpoint else identity["revision"] or config.revision,
        attn_implementation=config.attention_implementation, trust_remote_code=False)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if device.type == "cpu":
        # Safetensors may memory-map CPU weights. Detach their storage so Windows
        # can delete retained checkpoints while the resumed model is still alive.
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
    return model


def save_checkpoint(run, model, tokenizer, optimizer, scheduler, progress, mask_rng, signature, keep):
    checkpoint = run / "checkpoints" / f"step-{progress['step']:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    state = {"signature": signature, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "progress": progress, "mask_rng": mask_rng.getstate(), "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else None}
    temporary = checkpoint / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint / "training_state.pt")
    write_json(run / "last_checkpoint.json", {"path": str(checkpoint.relative_to(run)), **progress})
    # Only our own generated checkpoints are pruned, never arbitrary user paths.
    checkpoints = sorted((run / "checkpoints").glob("step-????????"))
    for old in checkpoints[:-keep]:
        if old != checkpoint:
            shutil.rmtree(old)
    return checkpoint


def restore_checkpoint(checkpoint, optimizer, scheduler, mask_rng, signature, device):
    # Only load locally produced checkpoints; optimizer/RNG state requires pickle.
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state["signature"] != signature:
        raise ValueError("Checkpoint experiment signature differs; start a new run.")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    mask_rng.setstate(state["mask_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if device.type != "cuda":
            raise ValueError("Resume on the original device type.")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def evaluate_targets(model, tokenizer, blocks, config, device, split, limit=None, with_pppl=False):
    result = {}
    for language in config.targets:
        chosen = selected_blocks(blocks, language, split, limit, config.eval_seed)
        metrics = evaluate_mlm(model, tokenizer, chosen, device, config.precision,
                               config.eval_batch_size, config.mask_probability, config.eval_seed)
        metrics["selection_fingerprint"] = fingerprint([b["id"] for b in chosen])
        if with_pppl and config.pseudo_perplexity_tokens:
            metrics.update(pseudo_perplexity(model, tokenizer, chosen, device, config.precision,
                                             config.pseudo_perplexity_tokens, config.eval_batch_size))
        result[language] = metrics
    return result


def run_experiment(config, output=None, resume=None, stop_after_steps=None, audit_only=False, parent_run=None):
    config.validate()
    records, manifest = load_prepared(config.data)
    tokenizer, identity = get_tokenizer_and_identity(config)
    if identity["revision"]:
        # Resumed runs use the resolved revision even if the remote branch moves.
        config.revision = identity["revision"]
    env = environment()
    signature = fingerprint({"config": asdict(config), "data": manifest["fingerprint"],
                             "model": identity, "code": env["implementation"]})
    if resume:
        run = Path(resume)
        saved = read_json(run / "experiment.json")
        if saved["signature"] != signature:
            raise ValueError("Data, model, code or settings changed. Start a new --output run with --parent-run.")
        if read_json(run / "status.json")["status"] in ("completed", "audit_only"):
            raise ValueError("This run has finished; create a new run for another experiment.")
    else:
        run = Path(output or ("runs/" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]))
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", asdict(config))
        write_json(run / "experiment.json", {"signature": signature, "created_at": utc_now(),
                   "data_fingerprint": manifest["fingerprint"], "model": identity,
                   "parent_run": str(parent_run) if parent_run else None,
                   "implementation": env["implementation"],
                   "protocol": {"mask_replacement": "80% mask / 10% random non-special / 10% unchanged; at least one target per block",
                                "scheduler": "linear decay", "optimizer": "AdamW; decay excludes biases and layer norms",
                                "checkpoint_metric": "equal mean target-language validation MLM loss",
                                "sampling": "weighted language then uniform block with replacement",
                                "determinism": "best effort; PyTorch deterministic algorithms warn_only=True"}})
        write_json(run / "environment.json", env)
        write_json(run / "dataset_manifest.json", manifest)
    log = RunLog(run)
    try:
        log.event("resume" if resume else "start", run=str(run), argv=sys.argv,
                  stop_after_steps=stop_after_steps, audit_only=audit_only)
        if resume:
            write_json(run / ("environment-resume-" + uuid.uuid4().hex[:8] + ".json"), env)
        write_json(run / "status.json", {"status": "running", "updated_at": utc_now()})
        if resume:
            token_meta = read_json(run / "tokenization.json")
            if file_hash(run / "blocks.jsonl") != token_meta["blocks_sha256"]:
                raise ValueError("Cached token blocks changed.")
            blocks = read_jsonl(run / "blocks.jsonl")
        else:
            blocks, audit = build_blocks(records, tokenizer, config.max_length)
            write_jsonl(run / "blocks.jsonl", blocks)
            tokenizer.save_pretrained(run / "tokenizer")
            token_meta = {"blocks_sha256": file_hash(run / "blocks.jsonl"), "audit": audit,
                          "vocabulary_size": len(tokenizer), "max_length": config.max_length}
            write_json(run / "tokenization.json", token_meta)
            for language, values in audit.items():
                log.event("tokenization", language=language, **{k: v for k, v in values.items() if k != "examples"})
        if audit_only:
            write_json(run / "status.json", {"status": "audit_only", "updated_at": utc_now()})
            return run
        device = configure_device(config)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_blocks = []
        for language in config.language_weights:
            train_blocks.extend(selected_blocks(blocks, language, "train", config.max_train_blocks_per_language, config.seed))
        if not train_blocks:
            raise ValueError("No training blocks")
        for language in config.targets:
            for split in ("train", "validation", "test"):
                if not selected_blocks(blocks, language, split, 1, config.seed):
                    raise ValueError(f"Missing {language}/{split} blocks")
        # Training diagnostics must come from the actual selected training pool,
        # including capped smoke experiments.
        evaluation_blocks = [b for b in blocks if b["split"] != "train"] + train_blocks
        steps_per_epoch = math.ceil(len(train_blocks) / (config.batch_size * config.accumulation_steps))
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = math.ceil(total_steps * config.warmup_fraction)
        write_json(run / "derived_settings.json", {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                   "warmup_steps": warmup_steps, "effective_batch_size": config.batch_size * config.accumulation_steps,
                   "sampling_epoch_blocks": len(train_blocks),
                   "normalized_language_weights": {k: v / sum(config.language_weights.values()) for k, v in config.language_weights.items()}})
        checkpoint = run / read_json(run / "last_checkpoint.json")["path"] if resume else None
        model = load_model(config, identity, device, checkpoint)
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": config.weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.}],
            lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
            foreach=False, fused=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        mask_rng = random.Random(config.seed)
        progress = {"epoch": 0, "sample_offset": 0, "step": 0, "best_loss": None, "bad_epochs": 0,
                    "exposure_blocks": {}, "exposure_tokens": {}}
        if resume:
            progress = restore_checkpoint(checkpoint, optimizer, scheduler, mask_rng, signature, device)
        else:
            baseline = {"validation": evaluate_targets(model, tokenizer, evaluation_blocks, config, device, "validation", config.eval_max_blocks),
                        "train_diagnostic": evaluate_targets(model, tokenizer, evaluation_blocks, config, device, "train", config.train_diagnostic_blocks,
                                                              with_pppl=True)}
            write_json(run / "baseline.json", baseline)
            log.event("baseline_validation", metrics=baseline["validation"])
            save_checkpoint(run, model, tokenizer, optimizer, scheduler, progress, mask_rng, signature, config.keep_checkpoints)
        session_start_step = progress["step"]
        session_start = time.monotonic()
        while progress["epoch"] < config.epochs and progress["bad_epochs"] < config.early_stopping_patience:
            epoch = progress["epoch"]
            schedule = epoch_schedule(train_blocks, config.language_weights, config.seed, epoch)
            remaining = [train_blocks[i] for i in schedule[progress["sample_offset"]:]]
            # A separate generator isolates DataLoader worker seeding from dropout RNG.
            loader_rng = torch.Generator().manual_seed(config.seed + epoch)
            loader = DataLoader(remaining, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, collate_fn=identity_collate, generator=loader_rng)
            iterator = iter(loader)
            while True:
                group = []
                for _ in range(config.accumulation_steps):
                    rows = next(iterator, None)
                    if rows is not None:
                        group.append(rows)
                if not group:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                # Normalize the entire accumulation group by its number of labels,
                # including the final partial group and variable-length blocks.
                masked_group = [[mask_block(b, tokenizer, config.mask_probability, mask_rng) for b in rows] for rows in group]
                label_count = sum(sum(label != -100 for label in row["labels"]) for rows in masked_group for row in rows)
                step_loss, seen_blocks, seen_tokens = 0., Counter(), Counter()
                for original, rows in zip(group, masked_group):
                    batch = pad_batch(rows, tokenizer, device)
                    nlabels = (batch["labels"] != -100).sum().item()
                    with autocast(device, config.precision):
                        output_model = model(**batch)
                        loss = output_model.loss * (nlabels / label_count)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss; last checkpoint retained.")
                    loss.backward()
                    step_loss += loss.detach().item()
                    for b in original:
                        seen_blocks[b["language"]] += 1
                        seen_tokens[b["language"]] += len(b["input_ids"]) - sum(b["special_tokens_mask"])
                    del output_model, loss, batch
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                used_learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                progress["step"] += 1
                progress["sample_offset"] += sum(len(rows) for rows in group)
                for key, values in (("exposure_blocks", seen_blocks), ("exposure_tokens", seen_tokens)):
                    for language, n in values.items():
                        progress[key][language] = progress[key].get(language, 0) + n
                if progress["step"] % config.logging_steps == 0 or progress["step"] == 1:
                    elapsed = time.monotonic() - session_start
                    log.event("train", step=progress["step"], epoch=epoch, loss=step_loss,
                              learning_rate=used_learning_rate, gradient_norm=float(gradient_norm),
                              optimizer_steps_per_second=(progress["step"] - session_start_step) / max(elapsed, 1e-9),
                              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                              exposure_blocks=progress["exposure_blocks"], exposure_tokens=progress["exposure_tokens"])
                if progress["step"] % config.checkpoint_steps == 0:
                    saved_path = save_checkpoint(run, model, tokenizer, optimizer, scheduler, progress, mask_rng, signature, config.keep_checkpoints)
                    log.event("checkpoint", step=progress["step"], path=str(saved_path))
                if stop_after_steps and progress["step"] - session_start_step >= stop_after_steps:
                    save_checkpoint(run, model, tokenizer, optimizer, scheduler, progress, mask_rng, signature, config.keep_checkpoints)
                    log.event("paused", step=progress["step"], reason="requested session step limit")
                    write_json(run / "status.json", {"status": "paused", "step": progress["step"], "updated_at": utc_now()})
                    return run
            validation = evaluate_targets(model, tokenizer, blocks, config, device, "validation", config.eval_max_blocks)
            average = sum(m["mlm_loss"] for m in validation.values()) / len(validation)
            improved = progress["best_loss"] is None or average < progress["best_loss"] - config.minimum_improvement
            if improved:
                progress["best_loss"], progress["bad_epochs"] = average, 0
                model.save_pretrained(run / "best_model", safe_serialization=True)
                tokenizer.save_pretrained(run / "best_model")
                write_json(run / "best_checkpoint.json", {"step": progress["step"], "epoch": epoch + 1,
                           "mean_target_validation_loss": average, "metrics": validation})
            else:
                progress["bad_epochs"] += 1
            progress["epoch"] += 1
            progress["sample_offset"] = 0
            log.event("validation", epoch=progress["epoch"], step=progress["step"], metrics=validation,
                      mean_target_loss=average, selected_as_best=improved)
            save_checkpoint(run, model, tokenizer, optimizer, scheduler, progress, mask_rng, signature, config.keep_checkpoints)
        del optimizer, scheduler, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary = final_evaluation(run, config, identity, tokenizer, evaluation_blocks, device)
        summary.update({"run": str(run), "signature": signature, "data_fingerprint": manifest["fingerprint"],
                        "progress": progress, "best_checkpoint": read_json(run / "best_checkpoint.json"),
                        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", {"status": "completed", "updated_at": utc_now()})
        log.event("completed", best_loss=progress["best_loss"], step=progress["step"])
        return run
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error), "updated_at": utc_now()})
        log.event("failed", error=repr(error))
        log.logger.exception("Run failed; resume from the last completed checkpoint if one exists.")
        raise
    finally:
        log.close()


def final_evaluation(run, config, identity, tokenizer, blocks, device):
    adapted_model = load_model(config, identity, device, Path(run) / "best_model")
    adapted = {split: evaluate_targets(adapted_model, tokenizer, blocks, config, device,
                                       "train" if split == "train_diagnostic" else split,
                                       config.train_diagnostic_blocks if split == "train_diagnostic" else config.eval_max_blocks,
                                       with_pppl=split != "validation")
               for split in ("validation", "test", "train_diagnostic")}
    del adapted_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base_model = load_model(config, identity, device)
    baseline = read_json(Path(run) / "baseline.json")
    baseline["test"] = evaluate_targets(base_model, tokenizer, blocks, config, device, "test", config.eval_max_blocks, with_pppl=True)
    return {"baseline": baseline, "adapted": adapted,
            "limitations": ["MLM diagnostics do not establish UD or morphology accuracy.",
                            "Base-model pretraining overlap cannot be ruled out.",
                            "Scores across different tokenizers/representations are not directly comparable."]}


def evaluate_run(run):
    run = Path(run)
    config = TrainConfig(**read_json(run / "config.json"))
    config.validate()
    _, manifest = load_prepared(config.data)
    experiment = read_json(run / "experiment.json")
    if experiment["implementation"] != code_identity():
        raise ValueError("Implementation differs from the original run; evaluate with its recorded code revision.")
    if manifest["fingerprint"] != experiment["data_fingerprint"]:
        raise ValueError("Dataset differs from original run")
    if file_hash(run / "blocks.jsonl") != read_json(run / "tokenization.json")["blocks_sha256"]:
        raise ValueError("Cached token blocks changed")
    device = configure_device(config)
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    blocks = read_jsonl(run / "blocks.jsonl")
    evaluation_blocks = [b for b in blocks if b["split"] != "train"]
    for language in config.language_weights:
        evaluation_blocks.extend(selected_blocks(blocks, language, "train", config.max_train_blocks_per_language, config.seed))
    result = final_evaluation(run, config, experiment["model"], tokenizer, evaluation_blocks, device)
    output = run / ("reevaluation-" + uuid.uuid4().hex[:8] + ".json")
    write_json(output, {"created_at": utc_now(), "environment": environment(), **result})
    return output
