"""Generate a local, Git-ignored notebook; all model logic stays in Python."""

import argparse
from pathlib import Path
from uralic_lm.common import write_json


def notebook():
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
                "id": f"cell-{len(cells):02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Manual backbone inspection
Compare Glot500 and its adapted checkpoint on Meadow Mari (`mhr`), Udmurt (`udm`),
and Nganasan (`nio`). Choose a completed run below. Start with validation examples;
keep final test data out of interactive model selection. Predictions concern
subtokens, not necessarily words. These diagnostics do not measure parser accuracy.
""")
    add("code", """from pathlib import Path
import os, sys
from pprint import pprint
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'finetune_backbone.py').exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from uralic_lm.inspection import Inspector, sample_excerpts, compare_predictions, training_events
from uralic_lm.common import read_json
RUN = Path('runs/REPLACE_WITH_RUN_DIRECTORY')
pprint(read_json(RUN / 'config.json'))
pprint(read_json(RUN / 'best_checkpoint.json'))
""")
    add("code", """examples = {lang: sample_excerpts(RUN, lang, count=5) for lang in ('mhr', 'udm', 'nio')}
for lang, rows in examples.items():
    print('\\n' + lang)
    for i, row in enumerate(rows):
        print(i, row['source_ids'], row['text'])
""")
    add("markdown", """Choose a language and excerpt. Display token indices before selecting a mask.
The default comparison masks the same middle subtoken in both checkpoints.
Use `token_index` to select another displayed position, or supply one literal
`<mask>` in your own text (a single mask predicts one subtoken).
""")
    add("code", """LANGUAGE = 'mhr'
EXAMPLE_INDEX = 0
sample = examples[LANGUAGE][EXAMPLE_INDEX]
TEXT = sample['text']  # Replace with your own text if desired.
inspector = Inspector(RUN, 'adapted')
pprint(inspector.tokens(TEXT))
inspector.close()
TOKEN_INDEX = None
pprint({'source': sample, 'comparison': compare_predictions(RUN, TEXT, TOKEN_INDEX)})
""")
    add("markdown", """## Candidate-word comparison
Supply complete candidates between a prefix and suffix, with spaces included where
appropriate. The example below selects a real word from the excerpt; add your own
alternatives. Scores mask each candidate subtoken individually, leaving other pieces
visible. Higher scores are better within this context, but they are not normalized
word probabilities. Compare both summed and mean scores and inspect token counts.
""")
    add("code", """words = TEXT.split()
word_index = len(words) // 2
PREFIX = ' '.join(words[:word_index]) + ' '
SUFFIX = ' ' + ' '.join(words[word_index + 1:])
CANDIDATES = [words[word_index]]  # Add linguistically meaningful alternatives.
for checkpoint in ('baseline', 'adapted'):
    inspector = Inspector(RUN, checkpoint)
    try:
        print(checkpoint)
        pprint(inspector.candidates(PREFIX, SUFFIX, CANDIDATES))
    finally:
        inspector.close()
""")
    add("code", """pprint(read_json(RUN / 'summary.json'))
training_events(RUN)[-10:]  # Learning rate, losses, sampling exposure, memory.
""")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/manual_backbone_eval.ipynb"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Notebook exists; choose another --output to preserve your edits.")
    write_json(args.output, notebook())
    print(args.output)


if __name__ == "__main__":
    main()
