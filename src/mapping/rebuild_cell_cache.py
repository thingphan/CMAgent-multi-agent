#!/usr/bin/env python3
"""Build an audited cell cache from CL labels and scoped synonyms; preserve gene/tissue caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mapping.ontology_mapper import normalize_cell_mention

FIELDS = ["normalized_term", "identifier", "canonical_term", "mapping_source"]


def ontology_entries(path: Path):
    content = path.read_text(encoding="utf-8")
    version = re.search(r"^data-version: (.+)$", content, re.M)
    labels, entries = {}, []
    for block in re.split(r"\n\[Term\]\n", content)[1:]:
        ident = re.search(r"^id: (CL:\d{7})$", block, re.M)
        name = re.search(r"^name: (.+)$", block, re.M)
        if not ident or not name or re.search(r"^is_obsolete: true$", block, re.M):
            continue
        identifier, label = ident[1].replace(":", "_"), name[1]
        labels[identifier] = label
        entries.append((label, identifier, "cell_ontology_label", "LABEL"))
        for match in re.finditer(r'^synonym: ("(?:\\.|[^"\\])*") (EXACT|BROAD|NARROW|RELATED)\b', block, re.M):
            alias = json.loads(match[1])
            source = "cell_ontology_exact_synonym" if match[2] == "EXACT" else "cell_ontology_" + match[2].lower() + "_synonym"
            entries.append((alias, identifier, source, match[2]))
    return labels, entries, version[1] if version else "unknown"


def build_cache(source: Path, ontology: Path, output: Path):
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output directory so the original cache is preserved.")
    labels, entries, version = ontology_entries(ontology)
    candidates, by_name, known = [], defaultdict(list), set()
    ontology_name_ids = defaultdict(set)
    for alias, identifier, source_type, scope in entries:
        key = normalize_cell_mention(alias)
        if not key:
            continue
        item = dict(zip(FIELDS, [key, identifier, labels[identifier], source_type]))
        candidates.append(item)
        known.add(key)
        ontology_name_ids[key].add(identifier)
        if scope in {"LABEL", "EXACT"}:
            by_name[key].append((0 if scope == "LABEL" else 1, item))
    with (source / "cell_terms.csv").open(encoding="utf-8-sig", newline="") as handle:
        old = list(csv.DictReader(handle))
    for row in old:
        key = normalize_cell_mention(row["normalized_term"])
        identifier = row["identifier"].replace(":", "_")
        if identifier not in labels or not row["mapping_source"].startswith("cellmarker_"):
            continue
        item = {**row, "normalized_term": key, "identifier": identifier, "canonical_term": labels[identifier]}
        if key in known and identifier not in ontology_name_ids[key]:
            continue
        # Custom CellMarker names remain searchable, but a majority conflict or
        # a non-exact ontology synonym cannot establish an exact local mapping.
        candidates.append(item)
        if key not in known and "conflict" not in row["mapping_source"]:
            by_name[key].append((2, item))
    deterministic, ambiguous = {}, []
    for key, values in sorted(by_name.items()):
        rank = min(rank for rank, _ in values)
        preferred = [item for priority, item in values if priority == rank]
        identifiers = {item["identifier"] for item in preferred}
        if len(identifiers) == 1 and key not in {"alpha cell", "beta cell"}:
            deterministic[key] = sorted(preferred, key=lambda r: (r["mapping_source"], r["canonical_term"]))[0]
        else:
            ambiguous.append({"normalized_term": key, "candidate_ids": "|".join(sorted(identifiers)),
                              "reason": "context_required" if key in {"alpha cell", "beta cell"} else "multiple_exact_ids"})
    audit = []
    for row in old:
        key = normalize_cell_mention(row["normalized_term"])
        new = deterministic.get(key, {})
        old_id, new_id = row["identifier"].replace(":", "_"), new.get("identifier", "")
        if old_id != new_id or row["canonical_term"] != new.get("canonical_term", ""):
            audit.append({"old_term": row["normalized_term"], "normalized_term": key,
                          "old_id": old_id, "new_id": new_id,
                          "old_label": row["canonical_term"], "new_label": new.get("canonical_term", ""),
                          "old_source": row["mapping_source"], "new_source": new.get("mapping_source", ""),
                          "action": "corrected" if new_id else "requires_agent_or_context"})
    output.mkdir(parents=True, exist_ok=True)
    def write(name, rows, fields):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write("cell_terms.csv", list(deterministic.values()), FIELDS)
    unique_candidates = {tuple(row[k] for k in FIELDS): row for row in candidates}
    write("cell_candidate_terms.csv", list(unique_candidates.values()), FIELDS)
    write("cell_ambiguous_terms.csv", ambiguous, ["normalized_term", "candidate_ids", "reason"])
    write("cell_cache_changes.csv", audit, ["old_term", "normalized_term", "old_id", "new_id", "old_label", "new_label", "old_source", "new_source", "action"])
    for name in ["gene_terms.csv", "gene_exact_terms.csv", "tissue_terms.csv"]:
        shutil.copy2(source / name, output / name)
    metadata = {"schema_version": 2, "ontology_version": version, "ontology_source": str(ontology.resolve()),
                "ontology_sha256": hashlib.sha256(ontology.read_bytes()).hexdigest(),
                "normalizer_source_sha256": hashlib.sha256((ROOT / "src/mapping/ontology_mapper.py").read_bytes()).hexdigest(),
                "original_cache": str(source.resolve()), "active_cl_ids": len(labels),
                "deterministic_cell_terms": len(deterministic), "candidate_terms": len(unique_candidates),
                "ambiguous_terms": len(ambiguous), "changed_cache_entries": len(audit),
                "policy": "labels and unambiguous EXACT synonyms; BROAD/RELATED/NARROW are candidate-only; context-sensitive alpha/beta require organ context"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.source_cache, args.ontology, args.output_cache), ensure_ascii=False, indent=2))
