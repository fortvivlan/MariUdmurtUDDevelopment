"""Reversible Unicode lemma edits adapted from BaseUDParser's rule approach."""

from difflib import SequenceMatcher
import json


COPY_RULE = json.dumps([0, 0, "", ""], ensure_ascii=False, separators=(",", ":"))


def lemma_rule(word: str, lemma: str) -> str:
    if lemma == "_":
        raise ValueError("Missing lemma has no edit rule")
    match = SequenceMatcher(None, word, lemma, autojunk=False).find_longest_match(
        0, len(word), 0, len(lemma)
    )
    return json.dumps([
        match.a, len(word) - match.a - match.size,
        lemma[:match.b], lemma[match.b + match.size:],
    ], ensure_ascii=False, separators=(",", ":"))


def apply_lemma_rule(word: str, rule: str) -> str:
    try:
        cut_prefix, cut_suffix, add_prefix, add_suffix = json.loads(rule)
        if cut_prefix < 0 or cut_suffix < 0 or cut_prefix + cut_suffix > len(word):
            return word
        end = len(word) - cut_suffix if cut_suffix else len(word)
        return add_prefix + word[cut_prefix:end] + add_suffix
    except (TypeError, ValueError, json.JSONDecodeError):
        return word
