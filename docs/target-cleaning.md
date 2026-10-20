# Target corpus cleaning and batch 004

The current working corpora are `mari.conllu` and `udmurt.conllu`. The local
Wikipedia extract release is unknown; original source corpora under `rawtexts/`
are untouched. This operation filters existing local derivatives and does not
change their source licenses or article split assignments.

| Working corpus | Before | After | Russian entries removed | Ukrainian entries removed | Exact duplicates removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mari | 202,478 | 122,324 | 29,875 | 1,035 | 49,243 |
| Udmurt | 91,417 | 74,642 | 3,233 | 0 | 13,542 |

Russian filtering combines the local fastText `lid.176.ftz` classifier with
OpenCorpora dictionary membership via pymorphy3. Language-specific letters,
common target words, Udmurt suffixes, and unknown lowercase words protect mixed
or target-language text. Named entities are allowed in otherwise dictionary-
supported Russian sentences. Explicit Ukrainian-script citations with strong
Ukrainian LID evidence are removed from Mari. This is conservative heuristic
filtering, not gold language adjudication; uncertain short phrases, unknown
names, and other foreign-language text can remain. Russian quotes inside Mari
or Udmurt sentences are retained. No external service receives corpus text.

Deduplication compares exact `# text` strings after boilerplate removal; no
Unicode normalization, case folding, fuzzy matching, or global sentence ID
renumbering is used. Surviving batch source IDs are preferred when choosing
between duplicate full-corpus entries. Other ties retain the first occurrence.

Mari Yandex boilerplate and its observed spelling variants are stripped from
1,874 source entries; surrounding text is retained, and one standalone entry
is removed. Token spans, glued tokens, and word IDs are adjusted. Altered
sentences keep their original source identity. The audit records original and
cleaned text. Original annotations on changed full-corpus rows are unset;
those rows are still automatic drafts.

The annotation directory still contains only its four CoNLL-U files. Both
languages now have 200 unique sentences. Mari retained 156 current entries,
including 20 with boilerplate removed, and added 44 (`0201`–`0244`). Udmurt
retained 192 entries and added 8 (`0201`–`0208`). Existing IDs were preserved;
deleted IDs were not reused. New selections use seed 42, 5–35 words, distinct
training articles, and exclude all original batch source sentences and all
retained batch articles. Validation/test articles are excluded.

The 136 unchanged Mari review entries and 192 unchanged Udmurt review entries
retain their annotation lines. This includes every Mari entry through `0036`
and the manually separated Yandex sentence `0015-1`. That single reviewed
boilerplate example remains only in the batch. Changed and added entries have
fresh predictions from the existing `parser-all-001` checkpoint, GiellaLT and
UniParser analyses, lemma review marks, and rough English annotation aids.
All four batch files agree on sentence IDs, source IDs, forms, and text.

The preserved Mari `0036` has two tokens with HEAD=0 (`лийын`, `шоген`). Its
existing annotation is untouched; resolve this issue before gold validation.
The CoNLL-U reader now treats whitespace-only lines as sentence separators,
preventing the tab-only line between `0037` and `0039` from merging them.

Provenance is kept outside the annotation directory:

- `runs/target-cleaning-005/`: cleaned corpora, retained review entries,
  per-sentence removal/edit audits, model hash, dependency versions, and counts.
- `runs/fewshot-004-cleaned-002/`: selections, model drafts, FST analyses,
  translations, staged annotation files, validation report, and application log.
- `runs/fewshot-004-cleaned-002/backup/`: exact original full-corpus bytes and
  the four original annotation files under `batch/`.

For the next correction/training cycle, use
`--draft runs/fewshot-004-cleaned-002/predictions`. Completed historical runs
and their manifests remain unchanged. Backbone adaptation outputs and earlier
parser checkpoints were not regenerated and still describe their original
inputs; future training must record the cleaned input fingerprints.

To repeat on a new snapshot, install `fewshot_creation/requirements-cleaning.txt`
and retain the classifier artifact and its checksum. Use fresh run directories:

```bash
python clean_target_data.py --lid-model /path/to/lid.176.ftz --output runs/target-cleaning-NEW
python -m fewshot_creation.refresh prepare --cleaned runs/target-cleaning-NEW --history runs/parser-all-001/fewshot-004 --output runs/fewshot-refresh-NEW
python -m fewshot_creation.refresh predict --work runs/fewshot-refresh-NEW
```

Then run the regular `fewshot_creation.pipeline fst` command on the new
`predictions/` directory. Translate exactly the IDs in `translation-request.json`
and pass the resulting JSON to `refresh assemble --work ... --review ...
--translations ...`. Review the staged `annotation/` and validate corpus and
batch invariants before `refresh apply --work ...`. Apply checks input/output
hashes, backs up originals, and replaces the working files. Repeating this
recipe later requires the matching current draft history and review cutoff;
the defaults describe batch 004's current repair.
