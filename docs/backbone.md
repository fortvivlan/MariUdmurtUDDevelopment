# Backbone adaptation

## Purpose and model choice

This stage adapts `cis-lmu/glot500-base` through masked language modeling (MLM).
The shared encoder targets Meadow/Eastern Mari (`mhr`) and standard Udmurt (`udm`)
UD parsing, plus Nganasan (`nio`) morphological parsing. Enets, Nenets, and Selkup
are included for the Samoyedic/Nganasan objective. Their benefit is a hypothesis,
not an established improvement. Finnish, Hungarian, Erzya, and Moksha are not
present locally and are not silently downloaded or included.

The selected Glot500 checkpoint is an Apache-2.0 XLM-R encoder with 12 layers,
768-dimensional states, a 401,145-token vocabulary, and a 512-token input limit.
See the [model card](https://huggingface.co/cis-lmu/glot500-base) and
[Glot500 paper](https://aclanthology.org/2023.acl-long.61/).
We also investigated [mmBERT](https://arxiv.org/html/2509.06888v1), whose stronger
overall benchmarks do not establish superiority for this parser: its paper
reports UDPOS 74.0 versus XLM-R's 74.3 and discusses tokenization issues for
structured prediction. [Targeted Uralic adaptation](https://aclanthology.org/2024.findings-emnlp.918/)
is relevant methodological evidence; we did not verify an immediately usable
released checkpoint. [Zerpal-Glot500](https://huggingface.co/udmurtNLP/zerpal-glot500)
is a possible Udmurt comparison, with no verified Meadow Mari advantage.

[BaseUDParser's encoder](https://github.com/fortvivlan/BaseUDParser/blob/main/cobald_parser/encoder.py)
loads `AutoModel` and mean-pools subtokens using fast-tokenizer word IDs, which
matches the exported backbone interface. Its repository declares GPL-3.0 in
[LICENSE](https://github.com/fortvivlan/BaseUDParser/blob/main/LICENSE).
No parser code is integrated into this adaptation stage.

## Sources and preservation

| Data | Provenance and known metadata |
| --- | --- |
| `mari.csv` | Meadow/Eastern Mari Wikipedia, `article,text` CSV. The legacy extractor names `mhrwiki-2026-02-01-p2p35661.xml.bz2`; this is evidence from code, not proof of the local file's exact dump release. |
| `udmurt.csv` | Standard Udmurt Wikipedia, same schema. Legacy code names `udmwiki-2026-02-01-p1p23629.xml.bz2`; exact local release is unconfirmed. |
| `nganasan.txt` | User-extracted INEL text. Reference: Brykina, Gusev, Szeverényi, Wagner-Nagy (2025), INEL Nganasan Corpus 1.0, [DOI 10.25592/uhhfdm.17419](https://www.fdr.uni-hamburg.de/record/17419). Reference files declare CC-BY-NC-SA-4.0. Local lines are matched to this release; the local export is not relabeled as the complete release. |
| `enets.txt` | User extraction from the [INEL repository](https://www.fdr.uni-hamburg.de/communities/inel/). [Enets portal](https://inel.corpora.uni-hamburg.de/portal/corpora/enets/) describes Forest and Tundra Enets. Local release, lect selection, and exact extraction recipe are unknown. |
| `nenets.txt` | User extraction from INEL. [Nenets portal](https://inel.corpora.uni-hamburg.de/portal/corpora/nenets/) describes both Forest and Tundra Nenets; do not label this file Tundra-only. Local release and lect selection are unknown. |
| `selkup.txt` | User extraction from the [Selkup portal](https://inel.corpora.uni-hamburg.de/portal/corpora/selkup/). It cites Brykina, Orlova, Wagner-Nagy (2021), INEL Selkup Corpus 2.0, [persistent handle](https://hdl.handle.net/11022/0000-0007-F4D9-1), and links CC-BY-NC-SA-4.0. The local version/dialect subset is unconfirmed. |

Corpus licenses and the model/code license are distinct metadata. Unknown local
release information is retained explicitly rather than guessed. A JSON file
passed with `--source-metadata` can replace metadata entries by language when
more precise citations, release information, or extraction details become known.
`enets` and `nenets` are local dataset identifiers, not assertions of a single ISO
lect code; `sel` is Selkup. No raw corpora are committed.

Source files are never rewritten. Wikipedia cleaning removes recognizable markup,
category/file lines and identified main pages; the audit records removed/changed
text. This is conservative cleaning, not complete language identification or
semantic near-duplicate detection. Russian code-switching and phonetic distinctions
are retained. UTF-8 output preserves the original spelling and combining marks;
NFC/case/whitespace normalization is used only for duplicate matching.

## Setup on native Windows

Run commands from the repository root in PowerShell. Use a working Python 3.12
installation; the legacy `.venv` may contain paths from a former project location.
Create a separate environment rather than repairing it in place:

```powershell
py -3.12 -m venv .venv-lm
.\.venv-lm\Scripts\Activate.ps1
$env:HF_HOME = Join-Path $PWD ".cache/huggingface"
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The CUDA wheel requires a compatible NVIDIA driver. No FlashAttention or
bitsandbytes installation is required. Preparation/download/comparison/notebook
generation use the standard library; training needs `requirements.txt`.
Every run records the entire installed dependency list, including transitive
versions. Install `jupyterlab` and `ipykernel` in this environment if you want to
open the generated notebook. For CPU tests, install the same PyTorch version
from `https://download.pytorch.org/whl/cpu` and use explicit CPU/fp32 settings.

## Prepare once, inspect, then train

```powershell
python fetch_inel_nganasan.py
python prepare_corpora.py
python finetune_backbone.py --config configs/glot500.json --output runs/tokenizer-audit --audit-only
```

The fetch script downloads only the 119.5 MB Nganasan lite archive, validates the
published checksum, and retains it under `rawtexts/sources/`. You may supply the
same archive elsewhere with `--nganasan-archive`.

The importer reads **TEI utterance surface-word elements under `u/seg/w`**. It
does not flatten the XML or include gloss, POS, translation, or annotation tiers.
Exact local-line matches recover recording IDs. Repeated responses are assigned
only when both nearest unique-match anchors agree on a candidate recording.
Unmatched or ambiguous lines are quarantined in the mapping audit, not trained.
Multiple possible utterance spans within an identified recording remain explicit.
The initial local preparation recovered recording assignments for 30,297 of
30,472 Nganasan lines (99.4%); 163 unmatched and 12 ambiguous lines were quarantined.

Documents are deduplicated before seeded 90/5/5 splits. Article revisions and
duplicate document aliases remain together. Nganasan uses whole source texts/
recordings, not shuffled utterances; **reuse `source_splits.json` when constructing
the future morphology dataset**, excluding held-out source texts in their entirety
from LM/parser training. This does not guarantee a speaker-disjoint evaluation.
The other three exports remain training-only because their document IDs are absent.

Test passages are protected before validation passages, then both are protected
from training. Paragraphs containing a matching normalized 12-word span, or an
exact matching line of any length, are removed from lower-priority splits. This
may remove frequent boilerplate/interjections, and all removals are audited.
To exclude external parser evaluation material, prepare a **new** dataset:

```powershell
python prepare_corpora.py --output data/prepared/exclusions-v1 --exclude path/to/dev.conllu path/to/test.conllu
```

Artifacts include `documents.jsonl`, `source_splits.json`, `nganasan_mapping.jsonl`,
`cleaning_audit.jsonl`, and `manifest.json`. Existing prepared outputs are immutable.
The manifest stores inputs, exclusion hashes, provenance, transformation settings,
preparation-code hashes, split statistics, and content checksums.

Tokenizer audits show unknown-token rates, subtokens per whitespace word,
document lengths and real examples for all languages. Long documents are split
without truncating their tails, with special tokens added to every block. Short
final blocks are retained; documents/splits are never packed together.

```powershell
python -m pytest -q
python finetune_backbone.py --config configs/glot500-smoke.json --output runs/gpu-smoke
python finetune_backbone.py --config configs/glot500.json --output runs/original-001
```

The smoke run uses the real model, 256-token inputs, tiny per-language training/
evaluation samples, and one short epoch. Inspect logs, losses, tokenizer examples,
and peak GPU memory before the full run. It tests compatibility, not improvement.
The full run defaults to BF16, batch 2, accumulation 16 (effective batch 32),
gradient checkpointing, AdamW at 2e-5, weight decay .01, 6% warmup, linear decay,
15% MLM masking, five epochs maximum, and patience two. CPU workers start at zero
for Windows portability; CPU threads default to eight and are configurable.

Masking uses 80% mask / 10% random non-special / 10% unchanged replacement,
with at least one target per block. Loss is weighted by the number of masked
tokens over the whole accumulation group, including partial final groups.
An epoch samples as many blocks as the training pool contains, with replacement:
first a language according to the configured weights, then a block uniformly.
These are block probabilities, not guaranteed exact token shares; actual block
and token exposure is logged. Oversampling a small corpus can overfit it.

JSON config files override the `TrainConfig` Python defaults. All defaults are
expanded into each run's `config.json`. Make a separate config for different
paths, weights, token lengths, training budgets, or evaluation caps. Full validation
and test splits are used unless `eval_max_blocks` explicitly limits them.

## Logs, checkpoints, and resumption

Implementation validation on 2026-09-07: 13 focused tests passed on native Windows
and Linux. Real Glot500 completed the six-step smoke experiment on the RTX 4090
with Windows Python 3.12, PyTorch 2.8.0+cu129, and Transformers 4.57.6, reaching
8.81 GiB peak allocated GPU memory. A separate pause/resume experiment produced
the same evaluation metrics as uninterrupted training. The final local artifact
is `runs/gpu-verified/`; these tiny experiments establish compatibility, not
language-model or parsing quality. Full-scale training has not been run.

Each run has a unique directory (automatic if `--output` is omitted):

- `config.json`, `derived_settings.json`: every resolved setting, effective batch,
  exact optimizer-step budget, warmup length, and normalized sampling weights.
- `experiment.json`, `environment.json`, `dataset_manifest.json`: immutable Hub
  revision, local model hashes where applicable, data/representation fingerprints,
  source-code hashes, Git state, package versions, hardware, and parent run.
- `console.log`, `events.jsonl`: timestamped start/resume/error events, sampled
  exposure, losses, learning rate, gradient norm, throughput, GPU memory, evaluation,
  and checkpoint selection. `status.json` records completion/failure/pause.
- `tokenization.json`, `blocks.jsonl`, `tokenizer/`: inspectable token audit and
  cached blocks. `baseline.json`, `best_checkpoint.json`, `summary.json`: comparison
  results and selected checkpoint. `best_model/` exports the tokenizer and MLM model.
- `checkpoints/`: the latest two optimizer/RNG checkpoints by default. They include
  model, optimizer, scheduler, sampling position, and Python/NumPy/PyTorch/masking RNG
  states. Budget disk space for several copies of this roughly 400M-parameter model.

Pause deliberately after a small number of optimizer steps, then resume:

```powershell
python finetune_backbone.py --config configs/glot500-smoke.json --output runs/resume-smoke --stop-after-steps 2
python finetune_backbone.py --resume runs/resume-smoke
```

Resume uses the saved configuration and resolved model revision. Changed settings,
data, model files, or implementation are rejected; start a new linked experiment
with `--parent-run runs/original-001`. Logs append resume events and retain prior
environment records. After a crash, progress since the last complete checkpoint
must be replayed; logs can therefore contain repeated step numbers. Reproducibility
across platforms/GPU kernels is best-effort, not guaranteed bit-for-bit.
Load only checkpoints you created/trust: optimizer/RNG state is a PyTorch pickle.

```powershell
python finetune_backbone.py --evaluate-run runs/original-001
python compare_runs.py runs/original-001 runs/another-experiment --output runs/comparison.json
```

Evaluation-only execution writes a new report and preserves the original summary.
Do not use repeated final-test inspection to guide model selection.

## Evaluation and manual notebook

The original model is evaluated before training on validation and a fixed sample
from the selected training pool. Each epoch evaluates fixed, reproducible masks;
the equally weighted mean MLM loss for Mari/Udmurt/Nganasan selects the checkpoint.
Final evaluation compares that checkpoint and the original on the test split.
Both models use exactly the same selected blocks, masks, and tokenizer.

Per-language metrics are token-weighted MLM loss, top-1/top-5 masked-token accuracy,
and `exp_mlm_loss`. **Exponentiated MLM loss is not ordinary perplexity.** To enable
additional leave-one-subtoken-out pseudo-perplexity on training diagnostics and
the final test sample, set `"pseudo_perplexity_tokens": 512` in a new config.
Each scored token requires another masked forward pass; this option is off by
default. It leaves other pieces of the same word visible and uses within-block
context. Scores cannot be directly compared across different tokenizers or text
representations. See [Hugging Face's perplexity discussion](https://huggingface.co/docs/transformers/perplexity).

```powershell
python make_manual_notebook.py
```

Open `notebooks/manual_backbone_eval.ipynb`, select your completed run, and sample
validation excerpts in all three targets. Inspect token indices, choose a token
to mask, compare baseline/adapted predictions, or score candidate words with
multiple subtokens. Candidate sum/mean pseudo-log-likelihoods are not calibrated
word probabilities. The notebook refuses silent truncation and preserves source
IDs, language, split, and representation metadata in samples. It never overwrites
an existing notebook; generate a new filename if needed.

These diagnostics assess adaptation, not UD UAS/LAS or morphology accuracy.
Original Glot500 pretraining may already include some corpus material. Protecting
these local splits controls leakage introduced by this project, not historic
pretraining overlap. External evaluation data acquired later must be excluded in
a new preparation run; existing adapted weights cannot be "untrained" by filtering.

## Future script/transcription experiments

The first baseline uses identity transformation and retains phonetic distinctions.
`Representation` is the small extension point for future Cyrillic transliteration
and Nganasan transcription mappings. Implement each mapping as a named/versioned
Python transform with a version-controlled mapping table and its checksum.
Never silently strip diacritics or normalize away phonological contrasts.

Create new derived outputs carrying the **same original source IDs and split**;
all versions of a source must stay together. Include transform settings/mapping
hashes in dataset and run identities, and compare on the same source selections.
Changing tokenization or representation changes what MLM/PPPL numbers mean;
eventual downstream morphology and UD evaluations determine usefulness.

## Validation performed during implementation

Focused tests cover TEI tier isolation, source recovery/quarantine, Unicode,
duplicate/source-group splits, held-out passage exclusion, representation split
inheritance, retained token tails, fixed masks, token-weighted metrics, capped
pseudo-perplexity, notebook validity, and save/reload/resume behavior.
The tiny CPU experiment checks that interrupted/resumed training produces the
same selected model weights as uninterrupted training in that environment.
Real-corpus audits and any available GPU smoke measurements are recorded locally
under `runs/`; they are diagnostics rather than research results.
