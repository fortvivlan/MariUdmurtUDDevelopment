"""Local, reproducible single-GPU joint UD parser training."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

from uralic_lm.common import file_hash, fingerprint, read_json, utc_now, write_json
from .conllu import read_conllu
from .model import JointUDParser
from .vocab import build_vocab


ALL_LANGUAGES = ("fi", "hu", "myv", "mdf", "et", "koi", "kpv", "yrk")
SELECTION_LANGUAGES = ("fi", "hu", "et", "myv")


@dataclass
class TrainConfig:
    source_manifest: str = "data/ud/r2.18/prepared/manifest.json"
    backbone: str = "runs/01/best_model"
    tokenizer: str | None = None
    revision: str | None = None
    parent_run: str | None = "runs/01"
    source_languages: list[str] = field(default_factory=lambda: list(ALL_LANGUAGES))
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    max_length: int = 512
    window_overlap: int = 64
    microbatch: int = 2
    accumulation: int = 16
    max_steps: int = 4000
    evaluate_every: int = 250
    log_every: int = 25
    early_stopping_patience: int = 4
    eval_max_sentences: int = 500
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = .01
    warmup_fraction: float = .06
    max_grad_norm: float = 1.0
    sampling_alpha: float = .5
    dropout: float = .1
    gradient_checkpointing: bool = True
    max_peak_gpu_gib: float = 22.0
    smoke: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        values = read_json(path)
        unknown = set(values) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown parser config keys: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"cuda", "cpu"} or self.precision not in {"bf16", "fp32"}:
            raise ValueError("Use cuda/bf16 or cpu/fp32")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU requires fp32")
        if not self.source_languages or len(set(self.source_languages)) != len(self.source_languages):
            raise ValueError("Source languages must be distinct and nonempty")
        if set(self.source_languages) - set(ALL_LANGUAGES):
            raise ValueError("Only the eight approved UD source languages are permitted")
        for key in ("seed", "microbatch", "accumulation", "max_steps", "evaluate_every",
                    "log_every", "early_stopping_patience", "eval_max_sentences"):
            if getattr(self, key) < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be positive")
        if self.max_length < 8 or not 0 <= self.window_overlap < self.max_length - 2:
            raise ValueError("Invalid window length or overlap")
        for key in ("encoder_lr", "head_lr", "max_grad_norm", "max_peak_gpu_gib"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 <= self.weight_decay or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid weight decay or warmup")
        if not 0 < self.sampling_alpha <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid sampling alpha or dropout")


def _environment() -> dict:
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=False)
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": dict(sorted(packages.items())), "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
    }


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = list((root / "uralic_parser").glob("*.py")) + [root / "finetune_parser.py"]
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}


def _backbone_identity(config: TrainConfig) -> dict:
    path = Path(config.backbone)
    if path.is_dir():
        return {"path": str(path), "config_sha256": file_hash(path / "config.json"),
                "weights_sha256": file_hash(path / "model.safetensors"),
                "tokenizer_sha256": file_hash(path / "tokenizer.json")}
    if not config.revision or config.revision == "main":
        raise ValueError("Hub backbones require an immutable revision")
    return {"model": config.backbone, "revision": config.revision}


def _tokenizer_identity(config: TrainConfig) -> dict:
    source = config.tokenizer or config.backbone
    path = Path(source)
    if path.is_dir():
        return {"path": str(path), "tokenizer_sha256": file_hash(path / "tokenizer.json"),
                "sentencepiece_sha256": file_hash(path / "sentencepiece.bpe.model")}
    return {"model": source, "revision": config.revision}


def _events(run: Path, event: str, **values) -> None:
    record = {"time": utc_now(), "event": event, **values}
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _sampler_weights(data: dict[str, list], alpha: float) -> dict[str, float]:
    raw = {language: len(rows) ** alpha for language, rows in data.items()}
    total = sum(raw.values())
    return {language: value / total for language, value in raw.items()}


def _sample_batch(data: dict[str, list], languages: list[str], probabilities: list[float],
                  rng: random.Random, batch_size: int, exposure: Counter,
                  smoke_step: int | None = None) -> list:
    selected = []
    for position in range(batch_size):
        if smoke_step is None:
            language = rng.choices(languages, weights=probabilities, k=1)[0]
            sentence = rng.choice(data[language])
        else:
            language = languages[(smoke_step * batch_size + position) % len(languages)]
            sentence = max(data[language], key=lambda row: len(row.words))
        selected.append(sentence)
        exposure[language] += 1
    return selected


def _eval_subset(rows: list, limit: int, seed: int) -> list:
    if len(rows) <= limit:
        return rows
    return sorted(rows, key=lambda sentence: fingerprint([
        seed, sentence.metadata.get("sent_id"), sentence.metadata.get("text")
    ]))[:limit]


def _evaluate(model: JointUDParser, data: dict[str, list], batch_size: int) -> dict:
    model.eval()
    result = {}
    for language, sentences in data.items():
        correct_head = correct_both = total = 0
        for start in range(0, len(sentences), batch_size):
            part = sentences[start:start + batch_size]
            predictions = model.predict(part)
            for sentence, rows in zip(part, predictions, strict=True):
                for gold, pred in zip(sentence.words, rows, strict=True):
                    same_head = pred["head"] == gold.head
                    correct_head += same_head
                    correct_both += same_head and pred["deprel"] == gold.deprel.split(":", 1)[0]
                    total += 1
        result[language] = {"uas": correct_head / total, "las": correct_both / total,
                            "tokens": total, "sentences": len(sentences)}
    model.train()
    return result


def _optimizer(model: JointUDParser, config: TrainConfig) -> torch.optim.Optimizer:
    groups = []
    for encoder in (True, False):
        for decay in (True, False):
            parameters = []
            for name, value in model.named_parameters():
                is_encoder = name.startswith("encoder.")
                no_decay = name.endswith("bias") or "norm" in name.lower() or name == "root"
                if is_encoder == encoder and (not no_decay) == decay:
                    parameters.append(value)
            if parameters:
                groups.append({"params": parameters,
                               "lr": config.encoder_lr if encoder else config.head_lr,
                               "weight_decay": config.weight_decay if decay else 0.0})
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def _scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    warmup = max(1, round(config.max_steps * config.warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return max(1e-8, step / warmup)
        return max(0.0, (config.max_steps - step) / max(1, config.max_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _save_weights(path: Path, model: JointUDParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {key: value.detach().cpu().contiguous()
             for key, value in model.state_dict().items()}
    save_file(state, str(temporary))
    temporary.replace(path)


def _save_latest(run: Path, model: JointUDParser, optimizer, scheduler, rng,
                 progress: dict) -> None:
    # Double buffering keeps the previous complete checkpoint resumable if
    # serialization fails or the process stops between weight/state writes.
    pointer = run / "latest.json"
    previous_slot = read_json(pointer)["slot"] if pointer.exists() else 1
    slot = 1 - previous_slot
    directory = run / f"latest-{slot}"
    directory.mkdir(exist_ok=True)
    _save_weights(directory / "model.safetensors", model)
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "random_state": random.getstate(), "sampling_state": rng.getstate(),
             "torch_state": torch.get_rng_state(),
             "cuda_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "progress": progress}
    temporary = directory / "state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "state.pt")
    write_json(directory / "progress.json", progress)
    write_json(pointer, {"slot": slot, "step": progress["step"]})


def _autocast(config: TrainConfig):
    return torch.autocast("cuda", dtype=torch.bfloat16) if config.precision == "bf16" else nullcontext()


def train(config: TrainConfig, run: Path, resume: bool = False) -> dict:
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the documented local GPU environment")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    run = Path(run)
    if not resume and run.exists() and any(run.iterdir()):
        raise ValueError("Parser run directory must be new/empty")
    run.mkdir(parents=True, exist_ok=True)
    manifest = read_json(config.source_manifest)
    train_files = {language: Path(manifest["outputs"][f"{language}/train"]["path"])
                   for language in config.source_languages}
    train_data = {language: list(read_conllu(path)) for language, path in train_files.items()}
    if any(not rows for rows in train_data.values()):
        raise ValueError("At least one source training split is empty")
    dev_data = {language: _eval_subset(list(read_conllu(
                    manifest["outputs"][f"{language}/dev"]["path"])),
                    config.eval_max_sentences, config.seed)
                for language in SELECTION_LANGUAGES}
    vocab = build_vocab(train_files)
    identity = {
        "config": asdict(config), "source_fingerprint": manifest["fingerprint"],
        "backbone": _backbone_identity(config), "vocab_fingerprint": fingerprint(vocab),
        "tokenizer": _tokenizer_identity(config),
        "code": _code_hashes(),
        "selection": {language: fingerprint([sentence.metadata.get("sent_id")
                                              for sentence in rows])
                      for language, rows in dev_data.items()},
    }
    signature = fingerprint(identity)
    if resume:
        previous = read_json(run / "experiment.json")
        if previous["signature"] != signature:
            raise ValueError("Run identity changed; start a new linked parser run")
        if read_json(run / "config.json") != asdict(config):
            raise ValueError("Resume configuration changed")
    else:
        write_json(run / "config.json", asdict(config))
        write_json(run / "vocab.json", vocab)
        write_json(run / "dataset_manifest.json", manifest)
        write_json(run / "environment.json", _environment())
        write_json(run / "experiment.json", {"signature": signature, **identity})
        write_json(run / "status.json", {"status": "running", "updated_at": utc_now()})
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    model = JointUDParser(config.backbone, vocab, max_length=config.max_length,
                          window_overlap=config.window_overlap, dropout=config.dropout,
                          gradient_checkpointing=config.gradient_checkpointing,
                          revision=config.revision,
                          tokenizer_path=config.tokenizer).to(config.device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    rng = random.Random(config.seed)
    languages = list(config.source_languages)
    probabilities_by_language = _sampler_weights(train_data, config.sampling_alpha)
    probabilities = [probabilities_by_language[language] for language in languages]
    progress = {"step": 0, "best_macro_las": -1., "best_step": None,
                "bad_evaluations": 0, "exposure": {language: 0 for language in languages}}
    if resume:
        directory = run / f"latest-{read_json(run / 'latest.json')['slot']}"
        model.load_state_dict(load_file(str(directory / "model.safetensors"),
                                        device=config.device))
        state = torch.load(directory / "state.pt", map_location=config.device,
                           weights_only=False)  # Only resume a run you created/trust.
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["random_state"])
        rng.setstate(state["sampling_state"])
        torch.set_rng_state(state["torch_state"])
        if config.device == "cuda" and state["cuda_states"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_states"])
        progress = state["progress"]
        _events(run, "resume", step=progress["step"])
    else:
        _events(run, "start", source_weights=probabilities_by_language,
                effective_batch=config.microbatch * config.accumulation)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    exposure = Counter(progress["exposure"])
    try:
        for step in range(progress["step"] + 1, config.max_steps + 1):
            group_loss = 0.0
            for micro in range(config.accumulation):
                smoke_index = ((step - 1) * config.accumulation + micro) if config.smoke else None
                batch = _sample_batch(train_data, languages, probabilities, rng,
                                      config.microbatch, exposure, smoke_index)
                with _autocast(config):
                    output = model(batch)
                    loss = output["loss"] / config.accumulation
                loss.backward()
                group_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += group_loss
            progress["step"] = step
            progress["exposure"] = dict(exposure)
            peak_gib = (torch.cuda.max_memory_allocated() / 1024 ** 3
                        if config.device == "cuda" else 0.0)
            if peak_gib > config.max_peak_gpu_gib:
                raise RuntimeError(f"Peak GPU allocation {peak_gib:.2f} GiB exceeds "
                                   f"{config.max_peak_gpu_gib:.2f} GiB")
            if step % config.log_every == 0:
                _events(run, "train", step=step, mean_loss=total_loss / config.log_every,
                        peak_gpu_gib=peak_gib, exposure=dict(exposure),
                        encoder_lr=optimizer.param_groups[0]["lr"])
                total_loss = 0.0
            if step % config.evaluate_every == 0 or step == config.max_steps:
                metrics = _evaluate(model, dev_data, config.microbatch)
                macro_las = sum(value["las"] for value in metrics.values()) / len(metrics)
                improved = macro_las > progress["best_macro_las"]
                if improved:
                    progress["best_macro_las"] = macro_las
                    progress["best_step"] = step
                    progress["bad_evaluations"] = 0
                    _save_weights(run / "best_model" / "model.safetensors", model)
                    model.tokenizer.save_pretrained(run / "best_model" / "tokenizer")
                    write_json(run / "best_model" / "metrics.json", metrics)
                else:
                    progress["bad_evaluations"] += 1
                _events(run, "evaluation", step=step, macro_las=macro_las,
                        metrics=metrics, improved=improved)
                _save_latest(run, model, optimizer, scheduler, rng, progress)
                if progress["bad_evaluations"] >= config.early_stopping_patience:
                    break
        summary = {"best_step": progress["best_step"],
                   "best_macro_las": progress["best_macro_las"],
                   "last_step": progress["step"], "peak_gpu_gib": (
                       torch.cuda.max_memory_allocated() / 1024 ** 3
                       if config.device == "cuda" else None),
                   "source_weights": probabilities_by_language,
                   "exposure": dict(exposure)}
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", {"status": "complete", "updated_at": utc_now()})
        _events(run, "complete", **summary)
        return summary
    except BaseException as exc:
        write_json(run / "status.json", {"status": "failed", "error": str(exc),
                                         "updated_at": utc_now()})
        _events(run, "error", error=str(exc), step=progress["step"])
        raise
