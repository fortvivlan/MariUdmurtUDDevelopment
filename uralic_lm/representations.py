"""Versioned representation interface; the baseline preserves original spelling.

Future transforms must return derived records with the SAME source_ids and split.
Mapping tables belong in version control and their hashes in the specification.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Representation:
    name: str = "identity"
    version: str = "1"
    mapping_sha256: str | None = None

    def apply(self, record):
        if self != Representation():
            raise ValueError("Only identity v1 is implemented; no implicit transliteration.")
        return {**record, "representation": asdict(self)}
