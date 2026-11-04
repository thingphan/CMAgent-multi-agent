#!/usr/bin/env python3
"""Fail-open, selective verifier for high-risk cell-marker candidates.

Low-risk extractor proposals are retained without an LLM call.  Only proposals whose
sentence contains a known false-positive context are sent to the verifier, which may
remove a proposal only when that context clearly does not assert the proposed relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.orchestration.prepare_verification_work_items import RISK_PATTERNS, candidate_context, risk_flags
PROMPT_VERSION = "relation-verifier-v3-selective-fail-open"
SYSTEM_PROMPT = """You verify a high-risk proposed biomedical cell-marker relation.
Use only the supplied sentence and proposal; do not use outside biological knowledge.
Keep a proposal only when the sentence directly asserts its cell-marker relation through
an explicit relation cue. Do not keep it merely because the cell and marker co-occur or
because a cell-marker interpretation seems biologically plausible. Return only the JSON schema."""
RESPONSE_SCHEMA = {
    "name": "selective_relation_verification",
    "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "assertion_type": {"type": "string", "enum": [
                "EXPRESSION", "POS_NEG", "MARKER_OF", "MARKER_LIST",
                "MARKER_MODIFIED", "NONE",
            ]},
            "assertion_span": {"type": "string"},
            "reason_code": {"type": "string", "enum": [
                "DIRECT_RELATION_OR_UNCERTAIN", "MOTIF_OR_TF_ACTIVITY",
                "REGULATORY_ELEMENT_ONLY", "BINDING_OR_CHIP_ONLY",
                "REPORTER_ACTIVITY_ONLY", "INTERVENTION_NOT_EXPRESSION",
                "CELL_ABUNDANCE_OR_INFILTRATION", "FUNCTION_OR_DIFFERENTIATION_ONLY",
                "TISSUE_ONLY_EXPRESSION", "GENE_SET_OR_PATHWAY_ONLY", "SCOPE_SEPARATED_COOCCURRENCE",
                "MERE_COOCCURRENCE", "WRONG_SCOPE", "OTHER_CLEAR_FP",
            ]},
            "reason": {"type": "string"},
        },
        "required": ["assertion_type", "assertion_span", "reason_code", "reason"],
    },
}

DECISION_COLUMNS = [
    "relation_id", "row_id", "PMID", "decision", "assertion_type", "assertion_span", "reason_code", "reason",
    "verification_route", "risk_flags", "agent_status", "error", "prompt_version",
    "model", "run_signature", "Cell mention", "Broad cell mention", "Marker mention",
    "Polarity", "Relation evidence", "Evidence",
]


def text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def relation_id(row: pd.Series) -> str:
    fields = ["row_id", "PMID", "Cell mention", "Broad cell mention", "Marker mention", "Polarity", "Relation evidence"]
    return hashlib.sha256("\x1f".join(text(row.get(field, "")) for field in fields).encode()).hexdigest()[:24]


def signature(model: str) -> str:
    payload = {"version": PROMPT_VERSION, "model": model, "system": SYSTEM_PROMPT, "schema": RESPONSE_SCHEMA, "risk": RISK_PATTERNS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def prompt(row: pd.Series, flags: list[str], correction: str = "") -> str:
    context = text(row.get("Candidate local context", "")) or candidate_context(row)
    return f"""[Sentence]\n{text(row.get('Evidence', ''))}\n\n[Proposed relation]\nCell mention: {text(row.get('Cell mention', ''))}\nBroad cell mention: {text(row.get('Broad cell mention', ''))}\nMarker mention: {text(row.get('Marker mention', ''))}\nPolarity: {text(row.get('Polarity', ''))}\nEvidence span: {text(row.get('Relation evidence', ''))}\n\n[Candidate-local context]\n{context}\n\n[Risk contexts that triggered review]\n{', '.join(flags)}\n\n[Required decision order]\n1. Inspect the candidate-local context first. Identify whether it directly asserts this cell-marker relation through one of: expression/enrichment (EXPRESSION), positivity or negativity (POS_NEG), marker-of wording (MARKER_OF), a cell-linked marker list (MARKER_LIST), or a marker-modified cell phrase (MARKER_MODIFIED). Set assertion_span to an exact sentence span that contains both the proposed cell and marker.\n2. If an assertion type is present, risk words elsewhere in the sentence cannot override this direct assertion.\n3. Only when no direct assertion exists, set assertion_type to NONE and assertion_span to empty. This applies when the proposal rests only on motif/TF activity, chromatin or regulatory-element activity, ChIP/binding, reporter activity, intervention, abundance/infiltration, interaction/function/differentiation, tissue-only expression, gene-set/pathway membership, scope-separated co-occurrence, or co-occurrence.\n\nThe pipeline derives KEEP from a non-NONE assertion type and CLEAR_REJECT from NONE. Do not create or correct relations.{correction}"""


def validate(payload: object, row: pd.Series) -> dict[str, str]:
    required = {"assertion_type", "assertion_span", "reason_code", "reason"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid verifier response fields")
    assertion_type, assertion_span, code, reason = (
        text(payload[key]) for key in ("assertion_type", "assertion_span", "reason_code", "reason")
    )
    if assertion_type not in {"EXPRESSION", "POS_NEG", "MARKER_OF", "MARKER_LIST", "MARKER_MODIFIED", "NONE"} or not reason:
        raise ValueError("invalid verifier decision")
    if assertion_type == "NONE":
        if assertion_span:
            raise ValueError("NONE assertion type requires an empty span")
        decision = "CLEAR_REJECT"
    else:
        if not assertion_span:
            raise ValueError("direct assertion type requires an assertion span")
        evidence = text(row.get("Evidence", ""))
        if assertion_span.casefold() not in evidence.casefold():
            raise ValueError("assertion span is not an exact sentence span")
        if text(row.get("Cell mention", "")).casefold() not in assertion_span.casefold() or text(row.get("Marker mention", "")).casefold() not in assertion_span.casefold():
            raise ValueError("assertion span must contain the proposed cell and marker")
        decision = "KEEP"
    return {"decision": decision, "assertion_type": assertion_type, "assertion_span": assertion_span, "reason_code": code, "reason": reason}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    relations = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"row_id", "PMID", "Cell mention", "Marker mention", "Polarity", "Relation evidence", "Evidence"}
    missing = required - set(relations)
    if missing:
        raise ValueError(f"Input missing columns: {', '.join(sorted(missing))}")
    relations = relations.copy()
    relations["relation_id"] = relations.apply(relation_id, axis=1)
    run_signature = signature(args.model)

    history = pd.DataFrame(columns=DECISION_COLUMNS)
    if args.decisions_output.exists() and not args.overwrite:
        history = pd.read_csv(args.decisions_output, dtype=str).fillna("")
    for column in DECISION_COLUMNS:
        if column not in history:
            history[column] = ""
    history = history[DECISION_COLUMNS]
    previous = history[history.run_signature.eq(run_signature)].drop_duplicates("relation_id", keep="last")
    # A fail-open record protects recall for the current run, but it is not a
    # completed semantic decision.  Re-submit only these rows when the verifier
    # is resumed; genuine KEEP/CLEAR_REJECT decisions remain terminal.
    done = set(previous.loc[previous["verification_route"].ne("API_FAIL_OPEN"), "relation_id"])
    pending = relations[~relations.relation_id.isin(done)]

    api_key = None
    client: Any = None
    records: list[dict[str, str]] = []

    def checkpoint() -> None:
        """Persist progress so an interrupted API run resumes without duplicate calls."""
        nonlocal history, records
        if not records:
            return
        history = pd.concat([history, pd.DataFrame(records, columns=DECISION_COLUMNS)], ignore_index=True)
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.decisions_output, index=False)
        records = []

    for number, (_, row) in enumerate(pending.iterrows(), 1):
        prepared_flags = text(row.get("Risk flags", ""))
        flags = [value for value in prepared_flags.split("|") if value] if prepared_flags else risk_flags(row, candidate_context(row))
        record = {column: text(row.get(column, "")) for column in DECISION_COLUMNS}
        record.update({"relation_id": row.relation_id, "row_id": text(row.row_id), "PMID": text(row.PMID), "prompt_version": PROMPT_VERSION, "model": args.model, "run_signature": run_signature, "risk_flags": "|".join(flags)})
        if not flags:
            record.update({"decision": "KEEP", "reason_code": "DIRECT_RELATION_OR_UNCERTAIN", "reason": "Low-risk candidate; retained without LLM review.", "verification_route": "AUTO_KEEP_LOW_RISK", "agent_status": "KEEP", "error": ""})
        else:
            record["verification_route"] = "LLM_HIGH_RISK"
            try:
                if client is None:
                    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
                    if not api_key:
                        from dotenv import load_dotenv
                        load_dotenv(ROOT / ".env")
                        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError("Set OPENAI_API_KEY or VITE_OPENAI_API_KEY before verification.")
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                last_error = ""
                for attempt in range(1, args.max_attempts + 1):
                    try:
                        correction = "" if not last_error else f"\n[Correction required]\nYour previous assertion was invalid: {last_error}. Return a corrected assertion for the same candidate."
                        response = client.chat.completions.create(model=args.model, temperature=0.0, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt(row, flags, correction)}], response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA})
                        record.update(validate(json.loads(response.choices[0].message.content or "{}"), row))
                        record.update({"agent_status": record["decision"], "error": ""})
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.max_attempts:
                            time.sleep(2 ** (attempt - 1))
                else:
                    record.update({"decision": "KEEP", "reason_code": "DIRECT_RELATION_OR_UNCERTAIN", "reason": "Verifier unavailable; retained by fail-open policy.", "verification_route": "API_FAIL_OPEN", "agent_status": "KEEP", "error": last_error})
            except Exception as exc:
                record.update({"decision": "KEEP", "reason_code": "DIRECT_RELATION_OR_UNCERTAIN", "reason": "Verifier unavailable; retained by fail-open policy.", "verification_route": "API_FAIL_OPEN", "agent_status": "KEEP", "error": f"{type(exc).__name__}: {exc}"})
        records.append(record)
        if number % 25 == 0:
            checkpoint()
            print(f"Selective verifier: {number:,}/{len(pending):,}", flush=True)
    checkpoint()

    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.decisions_output, index=False)
    latest = history[history.run_signature.eq(run_signature)].drop_duplicates("relation_id", keep="last")
    final = relations.merge(latest[["relation_id", "decision", "reason_code", "reason", "verification_route", "risk_flags"]], on="relation_id", how="left")
    args.verified_output.parent.mkdir(parents=True, exist_ok=True)
    final[final.decision.eq("KEEP")].drop(columns="relation_id").to_csv(args.verified_output, index=False)
    print(f"Decisions: {latest.decision.value_counts().to_dict()}")
    print(f"Routes: {latest.verification_route.value_counts().to_dict()}")
    print(f"Verified relations written: {int(final.decision.eq('KEEP').sum()):,}")


if __name__ == "__main__":
    main()
