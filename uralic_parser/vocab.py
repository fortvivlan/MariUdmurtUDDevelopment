"""Shared labels for the eight supervised source languages."""

from collections import Counter, defaultdict
from pathlib import Path

from .conllu import parse_feats, read_conllu, validate_sentence
from .lemma import COPY_RULE, lemma_rule


def build_vocab(train_files: dict[str, Path]) -> dict:
    upos, deprel, rules = set(), set(), Counter()
    features: dict[str, set[str]] = defaultdict(set)
    for path in train_files.values():
        for sentence in read_conllu(path):
            validate_sentence(sentence)
            for word in sentence.words:
                if word.upos != "_":
                    upos.add(word.upos)
                deprel.add(word.deprel.split(":", 1)[0])
                if word.lemma != "_":
                    rules[lemma_rule(word.form, word.lemma)] += 1
                for key, value in parse_feats(word.feats).items():
                    features[key].add(value)
    if not upos or not deprel:
        raise ValueError("Empty parser vocabulary")
    return {
        "upos": sorted(upos), "deprel": sorted(deprel),
        "lemma_rules": [COPY_RULE] + sorted((rule for rule in rules if rule != COPY_RULE),
                                            key=lambda rule: (-rules[rule], rule)),
        "feats": {key: ["_"] + sorted(values) for key, values in sorted(features.items())},
        "lemma_rule_counts": dict(rules),
    }
