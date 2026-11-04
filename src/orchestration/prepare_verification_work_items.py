#!/usr/bin/env python3
"""Controller-owned candidate-local context and risk-gate preparation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RISK_PATTERNS: dict[str, str] = {
    "MOTIF_OR_TF_ACTIVITY": r"\b(?:motif|transcription factor|tf) activity\b",
    "REGULATORY_ELEMENT_ONLY": r"\b(?:chromatin accessibility|enhancer|promoter)\b",
    "BINDING_OR_CHIP_ONLY": r"\b(?:chip(?:-seq)?|binding)\b",
    "REPORTER_ACTIVITY_ONLY": r"\b(?:luciferase|reporter) activity\b",
    "INTERVENTION_NOT_EXPRESSION": r"\b(?:inhibition|inhibitor|knockdown|knockout|silencing|overexpression|treated with)\b",
    "CELL_ABUNDANCE_OR_INFILTRATION": r"\b(?:infiltration|recruitment|abundance)\b",
    "FUNCTION_OR_DIFFERENTIATION_ONLY": r"\b(?:differentiation|cell fate|crosstalk|interaction)\b",
    "TISSUE_ONLY_EXPRESSION": r"\b(?:tissue expression|whole tissue)\b",
    "GENE_SET_OR_PATHWAY_ONLY": r"\b(?:gene set|gene signature|genes? associated with|pathway)\b",
}


def text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def positions(sentence: str, mention: str) -> list[tuple[int, int]]:
    return [] if not sentence or not mention else [(m.start(), m.end()) for m in re.finditer(re.escape(mention), sentence, flags=re.I)]


def candidate_context(row: pd.Series, flank: int = 100) -> str:
    cell, marker = text(row.get("Cell mention", "")), text(row.get("Marker mention", ""))
    for source in dict.fromkeys(value for value in (text(row.get("Relation evidence", "")), text(row.get("Evidence", ""))) if value):
        cells, markers = positions(source, cell), positions(source, marker)
        if cells and markers:
            first, second = min(((a, b) for a in cells for b in markers), key=lambda pair: abs((pair[0][0] + pair[0][1]) - (pair[1][0] + pair[1][1])))
            return source[max(0, min(first[0], second[0]) - flank):min(len(source), max(first[1], second[1]) + flank)]
    return ""


def risk_flags(row: pd.Series, context: str) -> list[str]:
    flags = [name for name, pattern in RISK_PATTERNS.items() if re.search(pattern, context, flags=re.I)]
    cells, markers = positions(context, text(row.get("Cell mention", ""))), positions(context, text(row.get("Marker mention", "")))
    if cells and markers:
        first, second = min(((a, b) for a in cells for b in markers), key=lambda pair: abs((pair[0][0] + pair[0][1]) - (pair[1][0] + pair[1][1])))
        between = context[min(first[1], second[1]):max(first[0], second[0])]
        if re.search(r"[;]|\b(?:whereas|respectively)\b", between, flags=re.I):
            flags.append("SCOPE_SEPARATED_COOCCURRENCE")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype=str).fillna("")
    data["Candidate local context"] = data.apply(candidate_context, axis=1)
    data["Risk flags"] = data.apply(lambda row: "|".join(risk_flags(row, row["Candidate local context"])), axis=1)
    data["Verification route hint"] = data["Risk flags"].map(lambda flags: "LLM_HIGH_RISK" if flags else "AUTO_KEEP_LOW_RISK")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Prepared verification work items: {len(data):,}; high risk: {int(data['Risk flags'].ne('').sum()):,}")


if __name__ == "__main__":
    main()
