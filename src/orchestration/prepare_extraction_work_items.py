#!/usr/bin/env python3
"""Controller-owned preparation of extraction work items.

This is deterministic: it joins verified PMID metadata and materializes dictionary hints
before the extraction agent is invoked.  It never calls an LLM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pmid(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return str(int(number)) if pd.notna(number) and float(number).is_integer() else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-dictionary-hints", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"row_id", "PMID", "sentence"}
    if missing := required - set(data):
        raise ValueError(f"Input missing columns: {', '.join(sorted(missing))}")
    metadata: dict[str, tuple[str, str]] = {}
    if args.metadata and args.metadata.exists():
        frame = pd.read_csv(args.metadata, dtype=str).fillna("")
        if missing := {"PMID", "Species", "Tissue"} - set(frame):
            raise ValueError(f"Metadata missing columns: {', '.join(sorted(missing))}")
        for row in frame.itertuples(index=False):
            key, values = pmid(row.PMID), (str(row.Species).strip(), str(row.Tissue).strip())
            if key in metadata and metadata[key] != values:
                raise ValueError(f"Conflicting metadata rows for PMID {key}")
            metadata[key] = values
    data["controller_species"] = data["PMID"].map(lambda value: metadata.get(pmid(value), ("", ""))[0])
    data["controller_tissue"] = data["PMID"].map(lambda value: metadata.get(pmid(value), ("", ""))[1])
    if args.disable_dictionary_hints:
        data["dictionary_hint_block"] = ""
    else:
        from src.extraction.dictionary_hints import DictionaryHintProvider
        hints = DictionaryHintProvider(max_per_type=12)
        data["dictionary_hint_block"] = data["sentence"].map(hints.prompt_block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Prepared extraction work items: {len(data):,}")


if __name__ == "__main__":
    main()
