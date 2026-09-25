# Joint UD parser transfer

## Experiment

This stage predicts LEMMA, UPOS, FEATS, HEAD and DEPREL from gold-segmented,
pre-tokenized CoNLL-U sentences. XPOS is `_`. It does **not** train on Mari or
Udmurt dependency labels. The shared XLM-R encoder starts from the local
`runs/01/best_model` MLM export. Although the parent MLM run ended after a CPU
checkpoint allocation failure, its saved best model is a complete safetensors
export from epoch 3. The parser smoke run verifies that it loads.
The original-Glot500 control uses the same local fast tokenizer as the adapted
model. Its underlying `sentencepiece.bpe.model` is byte-identical to the file
at the pinned original revision, so tokenization is shared across conditions.
This also avoids a tokenizer conversion error in the local Windows environment.

The parser's mean-subtoken encoder, biaffine arc/relation scoring and lemma edit
approach are adapted from [BaseUDParser at commit
`62c70e4`](https://github.com/fortvivlan/BaseUDParser/commit/62c70e47766b509c7c97ef53421c0193902366e6)
under GPL-3.0; see the repository `LICENSE`. The official CoNLL-2018 evaluator
is vendored in `uralic_parser/conll18_ud_eval.py` under MPL-2.0 with its license
text in `uralic_parser/MPL-2.0.txt`. The model and corpus licenses are separate.

## Data and splits

`prepare_ud.py` downloads these treebanks from the pinned
[UD 2.18 release](https://universaldependencies.org/download.html), retaining
raw files and per-treebank `LICENSE.txt` under ignored `data/ud/r2.18/raw/`.
Their exact bytes, URLs, license texts and split counts are stored in the
prepared manifest.

| Language | Treebank | Local split policy |
| --- | --- | --- |
| Finnish `fi` | TDT | Official train/dev/test |
| Hungarian `hu` | Szeged | Official train/dev/test |
| Erzya `myv` | JR | Official test retained; grouped 10% of train becomes dev |
| Moksha `mdf` | JR | Official train/test; no dev (22 training sentences) |
| Estonian `et` | EDT | Official train/dev/test |
| Komi Permyak `koi` | UH | Official train/test; no dev (34 training sentences) |
| Komi Zyrian `kpv` | Lattice | Official train/test; no dev (20 training sentences) |
| Nenets `yrk` | Tundra | Its only released file is test; repartition 80/10/10 by recording |

The Nenets local test is **not** an official UD benchmark test. Source
`brat_2018` parallel examples are kept in the same split across languages.
Checkpoint selection uses fixed development subsets of Finnish, Hungarian,
Estonian and Erzya only, with at most 500 sentences per language. Official
source tests are reserved for final diagnostics. Training samples languages
proportionally to the square root of their number of training sentences; the
run logs actual language exposure.

The existing `mari.conllu` and `udmurt.conllu` contain automatic morphology,
but no dependency heads. They are **draft inputs**, not gold evaluation data.
`make_target_ud_drafts.py` selects 200 sentences per language from distinct
Wikipedia articles in `data/prepared/original-v1/source_splits.json`'s test
split, preserving the original forms and adding stable source IDs. These drafts
must be manually corrected and adjudicated for word boundaries, LEMMA, UPOS,
FEATS, HEAD and DEPREL. Set every final sentence comment to
`# annotation_status = GOLD_ADJUDICATED` and retain `# source_id`. Use two
independent review passes and resolve disagreements before scoring. Keep the
gold files under ignored `data/ud/targets/`, and run `validate_target_ud.py`
before evaluation. The current drafts are **not** test scores.

## Local commands

Use the repository root in the documented Python 3.12/CUDA environment from
[backbone setup](backbone.md). Dependency pins are in `requirements-dev.txt`;
each run also records the versions actually installed. On this research machine
the GPU is a 24 GB RTX 4090; BF16 support is checked at startup.

```powershell
python prepare_ud.py
python make_target_ud_drafts.py
python -m pytest -q
python finetune_parser.py --config configs/parser-smoke.json --output runs/parser-smoke-004
```

Inspect the smoke `summary.json` and logs before full training. It covers all
eight languages, uses 512-subtoken windows, and fails if peak GPU allocation
exceeds 22 GiB. If it does, create a new config with `microbatch: 1` and
`accumulation: 32`, then repeat the smoke run; the effective batch remains 32.
The local `runs/parser-smoke-003` run completed on 2026-09-25 with peak allocated
GPU memory of 7.59 GiB. Its eight steps are a compatibility check, not a quality
estimate. The preceding `parser-smoke-002` checkpoint was reloaded and scored
on the small held-out Nenets split with the official evaluator. The local interpreter used Torch
2.14.0+cu132 and Transformers 4.57.6; keep one environment for the three full
conditions.
The pinned original Glot500 weights were downloaded into the local cache and
passed one GPU forward/backward compatibility check with the shared tokenizer.

```powershell
python finetune_parser.py --config configs/parser-all.json --output runs/parser-all-001
python finetune_parser.py --config configs/parser-no-nenets.json --output runs/parser-no-nenets-001
python finetune_parser.py --config configs/parser-original.json --output runs/parser-original-001
```

The three conditions use seed 42, equal hyperparameters, 4,000 optimizer steps
at most, 250-step validation, and patience four on macro source-development
LAS. The run config expands every default. Each run stores the source manifest,
model identity, code hashes, dependency versions, sampling exposure, events,
best model, and latest optimizer/RNG checkpoint. Changing settings or inputs
requires a new linked run. Resume a trusted local checkpoint with:

```powershell
python finetune_parser.py --resume runs/parser-all-001
```

This baseline uses a whole-encoder learning rate of `1e-5`, head learning rate
`1e-4`, AdamW weight decay `0.01`, 6% warmup, linear decay, BF16, gradient
checkpointing, microbatch 2 and accumulation 16. Word-aligned subtoken pooling
uses overlapping 512-token windows; no sentence is silently truncated.

## Final evaluation

After annotation, validate each target gold file and score the frozen selected
checkpoint. All three conditions must use the exact same gold checksums.

```powershell
python validate_target_ud.py --language mhr --gold data/ud/targets/mhr-gold.conllu
python validate_target_ud.py --language udm --gold data/ud/targets/udm-gold.conllu
python evaluate_parser.py --run runs/parser-all-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-all-001/target-eval-001
python evaluate_parser.py --run runs/parser-no-nenets-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-no-nenets-001/target-eval-001
python evaluate_parser.py --run runs/parser-original-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-original-001/target-eval-001
python compare_parser_runs.py --primary runs/parser-all-001/target-eval-001 --no-nenets runs/parser-no-nenets-001/target-eval-001 --original runs/parser-original-001/target-eval-001 --output runs/parser-comparison-001.json
```

The official CoNLL-2018 script scores UAS/LAS, UPOS, UFeats, Lemmas, MLAS and
BLEX. Inputs have gold word and sentence boundaries, punctuation is included,
and language-specific relation subtypes are ignored in LAS. The report includes
separate Mari/Udmurt scores and article-grouped 95% bootstrap intervals for
UAS/LAS. A one-seed difference is exploratory; repeat all conditions with new
linked runs before treating small effects as stable. The original Glot500 may
have encountered target text during its earlier pretraining; the held-out split
controls leakage introduced by this project.
