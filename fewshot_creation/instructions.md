# Few-shot creation: run instructions

Run commands from the repository root. The `fewshot_creation` package uses one
entry point with `run`, `predict`, `fst`, `add-translations`, and `publish` commands. Keep
every round in a new ignored `runs/` directory and keep corrected corpora under
ignored `data/`; never edit a completed run. Keep only the four CoNLL-U files
used for manual annotation in each `data/fewshot_batches/<batch>/` directory;
manifests and analyzer traces stay under `runs/`. The first round starts from the
200 Mari and 200 Udmurt drafts in `runs/parser-all-001/fewshot-004`.

## 1. Correct the current batch

Save human-adjudicated files as
`data/fewshot/corrected-004/mhr-corrected.conllu` and
`data/fewshot/corrected-004/udm-corrected.conllu`. Start from the corresponding
`data/fewshot_batches/004/*-model-review.conllu` files. Correct LEMMA, UPOS, FEATS, HEAD,
DEPREL, and any tokenization errors. Preserve `sent_id`, `source_id`,
`source_sent_id`, and the original `# text`. Set every sentence's
`# annotation_status = GOLD_ADJUDICATED`. Do not leave review notation such as
`model?FST` or `model?` in lexical lemmas. The pipeline checks IDs, source
splits, trees, and unresolved marks before training.
The four current annotation files are in `data/fewshot_batches/004/`:
`mhr-model-review.conllu`, `udm-model-review.conllu`, `mhr-fst.conllu`, and
`udm-fst.conllu`. The model review files already carry the existing `# text_en`
comments and are the starting drafts. Their FST analyses, audits, and copy
provenance remain in `runs/parser-all-001/fewshot-004-fst-review-v3/`.

## 2. Smoke check and create the next parser batch

Use the CUDA-enabled Python environment documented in [the parser
instructions](../uralic_parser/instructions.md). On the research Windows
machine, set `$python` to that interpreter. Run focused tests, then a small
GPU smoke cycle in a disposable new output directory:

```powershell
& $python -m pytest -q
& $python -m fewshot_creation.pipeline run --round-id smoke-0002 --parent-run runs/parser-all-001 --corrected data/fewshot/corrected-004 --draft runs/parser-all-001/fewshot-004 --work-dir runs/fewshot-smoke-0002 --training-overrides fewshot_creation/smoke-overrides.json --per-language 2
```

Inspect `runs/fewshot-smoke-0002/parser/summary.json` and its prediction
manifest. The smoke sampler exercises both target languages and source UD data.
The smoke round is a compatibility check and must not be annotated or
included among later `--draft` inputs. Then run the full cycle:

```powershell
& $python -m fewshot_creation.pipeline run --round-id 0002 --parent-run runs/parser-all-001 --corrected data/fewshot/corrected-004 --draft runs/parser-all-001/fewshot-004 --work-dir runs/fewshot-cycle-0002
```

The default is 200 new sentences per language. `targets/` holds cumulative
corrected train/dev files, `parser/` holds the linked trained checkpoint and
full run history, and `predictions/` holds form-only inputs, parser drafts,
source text, spacing, and `translation-request.json`. New samples use only
FORM values from `mari.conllu` and `udmurt.conllu` and exclude sentences from
the supplied `--draft` directories. The historical `fewshot-001` through
`fewshot-003` batches are deliberately ignored.

By default the few-shot training config uses seed 42, half target and half
source UD sampling, 300 maximum steps, evaluation every 25 steps, patience 4,
effective batch 16, encoder LR `2e-6`, and head LR `2e-5`. Override settings
with a JSON file passed to `--training-overrides`; the resolved config is saved
in the run. Each corrected batch contributes 10% to its language's development
set until that set reaches 500 sentences. Development articles never enter
training or later sampling.

If training stops after a checkpoint, resume with the existing parser command,
then use the `predict` subcommand with the same selection arguments and a new
prediction output directory:

```powershell
& $python finetune_parser.py --resume runs/fewshot-cycle-0002/parser
& $python -m fewshot_creation.pipeline predict --round-id 0002 --run runs/fewshot-cycle-0002/parser --target-manifest runs/fewshot-cycle-0002/targets/manifest.json --draft runs/parser-all-001/fewshot-004 --output runs/fewshot-cycle-0002/predictions
```

## 3. Run the two FST analyzers

Create a separate Python virtual environment for the pinned UniParser packages.
GiellaLT analyzers are compiled `.hfstol` data files, not Python packages.
Under WSL/Linux, the following also works when `sudo` or `python3-venv` is
unavailable:

```bash
python3 -m venv --without-pip .venv-fst
python3 -m pip --python .venv-fst install -r fewshot_creation/requirements-fst.txt
source .venv-fst/bin/activate
mkdir -p data/fst_tools
(cd data/fst_tools && apt-get download hfst && dpkg-deb -x hfst_*.deb .)
data/fst_tools/usr/bin/hfst-optimized-lookup --version
```

Obtain the **normative** `analyser-gt-norm.hfstol`, `metadata.json`, and
license file for both [Mari (`mhr`)](https://models.uralicnlp.com/nightly/mhr/)
and [Udmurt (`udm`)](https://models.uralicnlp.com/nightly/udm/) from the
UralicNLP model catalog. Put each trio under `data/fst_models/mhr/` and
`data/fst_models/udm/`, respectively. If a catalog snapshot omits a license,
copy the license from its linked GiellaLT language repository and record that
source in `source.json`. Keep the source URL and download date
in an optional `source.json` in each directory, for example
`{"artifact_url":"https://models.uralicnlp.com/nightly/mhr/analyser-gt-norm.hfstol","downloaded_at":"2026-09-28","release":"unknown nightly build"}`.
If omitted, the manifest explicitly records an unknown snapshot date and
release. The FST manifest records binary, metadata,
license, and input hashes, plus the HFST and UniParser versions. Do not replace
these files between cycles without making a new snapshot.
The current local snapshots label the license `CC BY` in `metadata.json`, while
their bundled `LICENCE.txt` contains LGPL-3.0 text; retain both records when
sharing results or models.

```bash
python -m fewshot_creation.pipeline fst \
  --predictions runs/fewshot-cycle-0002/predictions \
  --output runs/fewshot-cycle-0002/fst-review \
  --giella-mhr-dir data/fst_models/mhr \
  --giella-udm-dir data/fst_models/udm \
  --lookup data/fst_tools/usr/bin/hfst-optimized-lookup
```

Run the FST command from WSL if the parser was trained in Windows; both see
the shared repository. `fst-review/fst/` retains all UniParser and GiellaLT
analyses. `fst-review/review/` contains `mhr-fst.conllu`, `udm-fst.conllu`,
`mhr-model-review.conllu`, `udm-model-review.conllu`, and lemma audits. A unique
lemma common to both analyzers is marked `LemmaSource=FST` in the FST file.
The model review file keeps equal lemmas plain, writes `model?FST` for a
disagreement, and `model?` if consensus is absent or ambiguous. Punctuation
and symbol-only forms are exempt. Both files remain drafts for human review.

## 4. Add English translations separately

Give a translation subagent `predictions/translation-request.json`. Ask it to
return a UTF-8 JSON object with **exactly the same `sent_id` keys**, each value
one rough English sentence on one line. These translations are annotation aids,
not model inputs or gold labels. Save the result outside tracked files, then
merge it with the same pipeline entry point:

```powershell
& $python -m fewshot_creation.pipeline add-translations --review runs/fewshot-cycle-0002/fst-review/review --translations data/fewshot/translations-0002.json --output runs/fewshot-cycle-0002/translated
```

The command checks ID coverage and writes translated copies of all four
CoNLL-U files with `# text_en`; the untranslated files remain intact.

Publish the four files for manual annotation. The `publish` command verifies
sentence alignment and translations, copies only the CoNLL-U files into
`data/fewshot_batches`, and saves checksums in a separate run manifest:

```powershell
& $python -m fewshot_creation.pipeline publish --review runs/fewshot-cycle-0002/translated --output data/fewshot_batches/0002 --manifest runs/fewshot-cycle-0002/annotation-export.json
```

## 5. Correct the new batch and repeat

Use `data/fewshot_batches/0002/{mhr,udm}-model-review.conllu` as annotation
drafts. Resolve every
question-mark lemma, inspect both FST analyses when useful, correct all UD
fields, and set `GOLD_ADJUDICATED`. Save the pair as
`data/fewshot/corrected-0002/{mhr,udm}-corrected.conllu`. In the next round,
list **all** corrected and corresponding draft directories in chronological
order, one pair per batch. For example:

```powershell
& $python -m fewshot_creation.pipeline run --round-id 0003 --parent-run runs/fewshot-cycle-0002/parser --corrected data/fewshot/corrected-004 --draft runs/parser-all-001/fewshot-004 --corrected data/fewshot/corrected-0002 --draft runs/fewshot-cycle-0002/predictions --work-dir runs/fewshot-cycle-0003
```

The `--draft` history prevents sentence reuse; a different sentence from an
earlier training article remains eligible. The fixed test articles remain
reserved for [final parser evaluation](../docs/parser.md).
