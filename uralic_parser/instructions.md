# UD parser: local PowerShell instructions

Run these commands from **PowerShell** on the Windows machine with the 24 GB
RTX 4090. The paths below match this machine; change the two variables if the
repository or interpreter moves. Use one PowerShell session so `$python` remains
set.

```powershell
Set-Location 'F:\CODE\Python\Projects\MariUdmurtUDDevelopment'
$python = 'F:\CODE\Python\Interpreters\python312\python.exe'
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $python -m pytest -q
```

The interpreter must report CUDA availability as `True`. The test suite passed
20 checks locally. Dependency pins are in `requirements-dev.txt`, while each
parser run records its actual installed versions in `environment.json`.

## Check the inputs

The prepared UD 2.18 source data and adapted backbone are already present on
this machine. Check that both exist before starting:

```powershell
Test-Path 'data\ud\r2.18\prepared\manifest.json'
Test-Path 'runs\01\best_model\model.safetensors'
```

Both commands should print `True`. If the UD manifest is missing, run
`& $python prepare_ud.py` to fetch and prepare the pinned treebanks. The
preparation command requires an empty output directory; retain the existing
prepared data and manifest for these runs. See [the parser protocol](../docs/parser.md)
for source licenses, split policies, and the Nenets local test caveat.

An eight-language GPU smoke run has already completed in
`runs/parser-smoke-003` with 7.59 GiB peak allocated memory. If code,
dependencies, or settings change, repeat the smoke check with a **new** output
directory before full training:

```powershell
& $python finetune_parser.py --config configs/parser-smoke.json --output runs/parser-smoke-004
Get-Content 'runs\parser-smoke-004\summary.json'
```

## Train the three conditions

Run these sequentially on the single GPU. Each output directory must be new
or empty. The all-source run uses Finnish, Hungarian, Erzya, Moksha, Estonian,
Komi Permyak, Komi Zyrian, and Nenets. The second run removes Nenets; the
third starts from the original pinned Glot500 weights while using the same
tokenizer as the adapted backbone. The original weights have been downloaded
into the local Hugging Face cache and passed a GPU compatibility check.

```powershell
& $python finetune_parser.py --config configs/parser-all.json --output runs/parser-all-001
if ($LASTEXITCODE -ne 0) { throw 'All-source parser run failed' }

& $python finetune_parser.py --config configs/parser-no-nenets.json --output runs/parser-no-nenets-001
if ($LASTEXITCODE -ne 0) { throw 'No-Nenets parser run failed' }

& $python finetune_parser.py --config configs/parser-original.json --output runs/parser-original-001
if ($LASTEXITCODE -ne 0) { throw 'Original-backbone parser run failed' }
```

The default is at most 4,000 optimizer steps per run, with validation every
250 steps and early stopping after four unimproved validations. Configuration,
data fingerprints, dependency versions, training events, the selected model,
and resumable optimizer state are written under each ignored `runs/` directory.
In another PowerShell window, follow a run's event log with:

```powershell
Get-Content 'runs\parser-all-001\events.jsonl' -Tail 10 -Wait
```

If a run stops **after its first validation checkpoint**, resume that same run
from `latest.json`; use the matching run directory for the other conditions:

```powershell
& $python finetune_parser.py --resume runs/parser-all-001
```

Do not reuse a run directory after changing source data, code, model, or
hyperparameters. Start a newly named run and record its relationship to the
earlier one in `parent_run` in a copied config.

## Evaluate after target annotation

The files in `data/ud/targets/drafts-v1/` are automatic **drafts**, not Mari or
Udmurt gold data. Human reviewers must correct and adjudicate 200 sentences
per language, keeping `# source_id` and setting
`# annotation_status = GOLD_ADJUDICATED`. Save the finished files at the paths
below, then validate and score the frozen selected checkpoints:

```powershell
& $python validate_target_ud.py --language mhr --gold data/ud/targets/mhr-gold.conllu
& $python validate_target_ud.py --language udm --gold data/ud/targets/udm-gold.conllu

& $python evaluate_parser.py --run runs/parser-all-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-all-001/target-eval-001
& $python evaluate_parser.py --run runs/parser-no-nenets-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-no-nenets-001/target-eval-001
& $python evaluate_parser.py --run runs/parser-original-001 --gold-mhr data/ud/targets/mhr-gold.conllu --gold-udm data/ud/targets/udm-gold.conllu --source-tests --output runs/parser-original-001/target-eval-001

& $python compare_parser_runs.py --primary runs/parser-all-001/target-eval-001 --no-nenets runs/parser-no-nenets-001/target-eval-001 --original runs/parser-original-001/target-eval-001 --output runs/parser-comparison-001.json
```

The evaluation reports separate Mari and Udmurt UAS and LAS, other UD metrics,
and article-grouped intervals. The comparison requires identical gold file
checksums across all three conditions. Full evaluation settings and the data
protocol are in [the parser documentation](../docs/parser.md).
