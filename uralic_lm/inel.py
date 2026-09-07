"""Recover Nganasan provenance from TEI surface words, never annotation tiers."""

from collections import defaultdict, Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

NS = {"t": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
NGANASAN_URL = "https://www.fdr.uni-hamburg.de/record/17419/files/nganasan-1.0-lite.zip?download=1"
NGANASAN_MD5 = "5ad2dc3aa08b5e9d7844e64ebb5ad1e0"
NGANASAN_SOURCE = {
    "url": "https://www.fdr.uni-hamburg.de/record/17419",
    "doi": "10.25592/uhhfdm.17419", "version": "1.0", "date": "2025-05-02",
    "license": "CC-BY-NC-SA-4.0",
    "citation": "Brykina, Gusev, Szeverényi, Wagner-Nagy (2025). INEL Nganasan Corpus 1.0.",
}


def tei_utterances(xml, member):
    """Yield only words inside utterance segments under <u>, excluding spanGrp."""
    root = ET.fromstring(xml)
    recording = Path(member).stem.removesuffix("_tei")
    for block in root.findall(".//t:annotationBlock", NS):
        for segment in block.findall('./t:u//t:seg[@type="utterance"]', NS):
            words = ["".join(w.itertext()).strip() for w in segment.findall(".//t:w", NS)]
            text = " ".join(w for w in words if w)
            if text:
                yield {"text": text, "recording": recording,
                       "utterance_id": segment.get(XML_ID), "speaker": block.get("who"),
                       "member": member, "tier": "TEI u/seg[@type=utterance]/w"}


def recover_nganasan(local_path, archive):
    index = defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if name.endswith("_tei.xml"):
                for item in tei_utterances(source.read(name), name):
                    index[item["text"]].append(item)
    lines = Path(local_path).read_text(encoding="utf-8-sig").splitlines()
    matches = [index.get(line.strip(), []) for line in lines]
    candidates = [{m["recording"] for m in rows} for rows in matches]
    # Unique exact matches anchor context. Repeated short responses may be assigned
    # only when the closest anchors on BOTH sides agree and match a source record.
    left, previous = [], None
    for docs in candidates:
        left.append(previous)
        if len(docs) == 1:
            previous = next(iter(docs))
    right, following = [None] * len(lines), None
    for i in range(len(lines) - 1, -1, -1):
        right[i] = following
        if len(candidates[i]) == 1:
            following = next(iter(candidates[i]))
    documents, audit = defaultdict(list), []
    for i, (line, docs, rows) in enumerate(zip(lines, candidates, matches)):
        selected, method = None, "unmatched"
        if len(docs) == 1:
            selected, method = next(iter(docs)), "unique_exact_recording"
        elif len(docs) > 1:
            method = "ambiguous_recording"
            if left[i] == right[i] and left[i] in docs:
                selected, method = left[i], "exact_with_matching_neighbor_anchors"
        entry = {"line": i + 1, "text": line, "method": method,
                 "recording": selected, "candidate_recordings": sorted(docs)}
        if selected:
            entry["source_spans"] = [r for r in rows if r["recording"] == selected]
            documents[selected].append(entry)
        audit.append(entry)
    result = []
    for recording, rows in sorted(documents.items()):
        result.append({"id": "nio:" + recording, "language": "nio",
                       "source_ids": ["nio:" + recording], "source": "nganasan.txt",
                       "text": "\n".join(r["text"] for r in rows),
                       "local_lines": [r["line"] for r in rows],
                       "recording_id": recording})
    return result, audit, dict(Counter(r["method"] for r in audit))
