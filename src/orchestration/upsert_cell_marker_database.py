#!/usr/bin/env python3
"""Upsert one run's final mapped relations into the canonical production database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = ("Species", "Tissue", "CellOntology_ID", "NCBI_GeneID", "Polarity")
REQUIRED_COLUMNS = {
    *IDENTITY_COLUMNS, "PMID", "Evidence", "Cell mention", "Broad cell mention", "Marker mention",
}
SOURCE_COLUMNS = {
    "PMID": "Source_PMIDs",
    "Evidence": "Source_evidence",
    "Cell mention": "Source_cell_mentions",
    "Broad cell mention": "Source_broad_cell_mentions",
    "Marker mention": "Source_marker_mentions",
    "source_run": "Source_runs",
}


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None or pd.isna(value) else str(value)).strip().casefold()


def database_key(row: pd.Series) -> str:
    fields = tuple(clean(row[column]) for column in IDENTITY_COLUMNS)
    return hashlib.sha256("\x1f".join(fields).encode()).hexdigest()


def values_from(row: pd.Series, raw_column: str, aggregate_column: str) -> list[str]:
    aggregate = row.get(aggregate_column, "")
    if clean(aggregate):
        try:
            decoded = json.loads(str(aggregate))
            if isinstance(decoded, list):
                return [str(value) for value in decoded if clean(value)]
        except json.JSONDecodeError:
            pass
    value = row.get(raw_column, "")
    return [str(value)] if clean(value) else []


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = clean(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Mapped input missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    # A canonical ID database must not invent identity from textual mentions.
    frame = frame[frame["CellOntology_ID"].map(clean).ne("") & frame["NCBI_GeneID"].map(clean).ne("")].copy()
    frame["database_key"] = frame.apply(database_key, axis=1)
    return frame


def upsert(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    incoming = prepare(incoming)
    existing = prepare(existing) if not existing.empty else existing
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    records: list[dict[str, object]] = []
    for _, group in combined.groupby("database_key", sort=False):
        # Last record contains the newest normalized labels/symbol.  Keep all source
        # papers, evidence, and original mentions as JSON arrays for provenance.
        record = group.iloc[-1].to_dict()
        for raw_column, aggregate_column in SOURCE_COLUMNS.items():
            values = [value for _, row in group.iterrows() for value in values_from(row, raw_column, aggregate_column)]
            record[aggregate_column] = json.dumps(unique(values), ensure_ascii=False)
        record["Normalized_Cell_Name"] = str(record.get("Cell_Mapping_Term", ""))
        record["Normalized_Broad_Cell_Name"] = str(record.get("Broad_Cell_Mapping_Term", ""))
        record["Normalized_Gene_ID"] = str(record["NCBI_GeneID"])
        record["Normalized_Gene_Symbol"] = str(record.get("Official_Symbol", ""))
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Final cell_gene_agent.csv for one run.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    incoming = pd.read_csv(args.input, dtype=str).fillna("")
    incoming["source_run"] = args.run_name
    existing = pd.read_csv(args.database, dtype=str).fillna("") if args.database.exists() else pd.DataFrame()
    result = upsert(existing, incoming)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.database.with_suffix(args.database.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, args.database)
    skipped = len(incoming) - len(prepare(incoming))
    print(f"Canonical database: {len(existing):,} -> {len(result):,} rows")
    print(f"Skipped without both canonical IDs: {skipped:,}")
    print(f"Wrote: {args.database}")


if __name__ == "__main__":
    main()
