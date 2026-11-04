#!/usr/bin/env python3
"""Prepare extracted relations for production ontology and gene mapping.

Species and tissue are document metadata supplied before extraction and carried in the
relation artifact.  This production step intentionally has no gold/candidate input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "row_id", "PMID", "Species", "Tissue", "Cell_type", "Cell type (subtype)",
    "Polarity", "Marker mention", "Evidence",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    relations = pd.read_csv(args.relations, dtype=str).fillna("")
    if missing := REQUIRED_COLUMNS - set(relations.columns):
        raise ValueError(f"Relations missing columns: {', '.join(sorted(missing))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    relations.to_csv(args.output, index=False)
    print(f"Prepared production mapping work items: {len(relations):,}")
    print(f"Rows with species metadata: {relations['Species'].str.strip().ne('').sum():,}")
    print(f"Rows with tissue metadata: {relations['Tissue'].str.strip().ne('').sum():,}")


if __name__ == "__main__":
    main()
