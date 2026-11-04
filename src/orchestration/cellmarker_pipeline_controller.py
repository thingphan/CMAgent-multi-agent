#!/usr/bin/env python3
"""Resumable controller for the CellMarker multi-agent pipeline.

The controller owns stage ordering, artifacts, retries, and hand-offs.  Agents own only
their specialised LLM decision plus validation of that decision; deterministic helpers
run inside the stage that owns their input contract.  Terminal rows are never submitted
again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
# Use the interpreter that launched the controller instead of a machine-specific,
# stale conda path.
PYTHON = Path(sys.executable)
DEFAULT_METADATA = ROOT / "data/processed/pmid_metadata.csv"
DEFAULT_DATABASE = ROOT / "data/processed/cell_marker_database.csv"
DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///data/processed/cell_marker.db"
)
DEFAULT_CACHE_DIR = Path(os.environ.get("CMAGENT_RESOURCES_DIR", ROOT / "resources")) / "local_entity_cache_v3"
RUNS_DIR = Path(os.environ.get("CMAGENT_RUNS_DIR", ROOT / "results/full/runs"))
TERMINAL_EXTRACTION = {"SUCCESS", "NO_ASSOCIATION", "SKIPPED_ICL_EXAMPLE"}
TERMINAL_AGENT = {"ACCEPT", "REJECT", "REVIEW", "CORRECT", "UNMAPPED", "NEEDS_REVIEW"}
RELATION_COLUMNS = [
    "row_id", "PMID", "Species", "Tissue", "Cell_type", "Cell type (subtype)",
    "Cell marker", "Negative marker", "Polarity", "Confidence", "Cell mention",
    "Broad cell mention", "Marker mention", "Relation evidence", "Evidence", "icl_mode",
    "icl_example_ids", "icl_retrieval_scores", "run_signature", "prompt_version", "model",
]
# Keep the zero-relation artifact compatible with downstream mapping consumers.
MAPPED_COLUMNS = [
    "row_id", "Species", "NCBITaxon_ID", "Tissue", "UBERON_ID", "Cell_type", "Subtype",
    "Cell_Mapping_Term", "Cell_Mapping_Source", "CellOntology_ID",
    "Broad_Cell_Mapping_Term", "Broad_Cell_Mapping_Source", "Broad_CellOntology_ID",
    "Original_Marker", "Polarity", "Confidence", "Marker mention", "Gene_Mapping_Source",
    "Cell mention", "Broad cell mention", "Official_Symbol", "NCBI_GeneID", "PMID", "Evidence",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() and path.stat().st_size else pd.DataFrame()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    print("\n[controller] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def extraction_pending(input_path: Path, status_path: Path) -> int:
    inputs = read_csv(input_path)
    status = read_csv(status_path)
    if status.empty:
        return len(inputs)
    latest = status.drop_duplicates("row_id", keep="last")
    done = set(latest.loc[latest["status"].isin(TERMINAL_EXTRACTION), "row_id"])
    return int((~inputs["row_id"].astype(str).isin(done)).sum())


def decision_pending(relations_path: Path, decisions_path: Path) -> int:
    relations = read_csv(relations_path)
    if relations.empty:
        return 0
    decisions = read_csv(decisions_path)
    if decisions.empty:
        return len(relations)
    if "relation_id" not in decisions:
        return len(relations)
    latest = decisions.drop_duplicates("relation_id", keep="last")
    return int((~latest.get("agent_status", pd.Series(dtype=str)).isin(TERMINAL_AGENT)).sum())


def verification_pending(relations_path: Path, decisions_path: Path) -> int:
    """Count relations without a semantic verifier decision.

    ``API_FAIL_OPEN`` protects recall for an interrupted invocation, but is deliberately
    retryable.  KEEP and CLEAR_REJECT (and auto-kept low-risk rows) are final.
    """
    relations = read_csv(relations_path)
    if relations.empty:
        return 0
    decisions = read_csv(decisions_path)
    if decisions.empty or "relation_id" not in decisions:
        return len(relations)
    latest = decisions.drop_duplicates("relation_id", keep="last")
    completed = set(latest.loc[latest.get("verification_route", pd.Series(dtype=str)).ne("API_FAIL_OPEN"), "relation_id"])
    fields = ("row_id", "PMID", "Cell mention", "Broad cell mention", "Marker mention", "Polarity", "Relation evidence")
    relation_ids = relations.apply(
        lambda row: hashlib.sha256("\x1f".join(str(row.get(field, "")) for field in fields).encode()).hexdigest()[:24],
        axis=1,
    )
    return int((~relation_ids.isin(completed)).sum())


def normalization_pending(decisions_path: Path) -> int:
    decisions = read_csv(decisions_path)
    if decisions.empty:
        return 0
    latest = decisions.drop_duplicates("decision_key", keep="last")
    return int((~latest.get("agent_status", pd.Series(dtype=str)).isin(TERMINAL_AGENT)).sum())


def retry_stage(name: str, action, pending, cycles: int, manifest: dict) -> None:
    for cycle in range(1, cycles + 1):
        before = pending()
        if before == 0:
            print(f"[controller] {name}: complete")
            return
        print(f"[controller] {name}: {before:,} retryable items (cycle {cycle}/{cycles})")
        try:
            action()
            manifest["stages"][name] = {"cycle": cycle, "pending_after": pending(), "last_error": ""}
        except subprocess.CalledProcessError as exc:
            manifest["stages"][name] = {"cycle": cycle, "pending_after": pending(), "last_error": str(exc)}
            print(f"[controller] {name} command failed; pending items will be retried.", flush=True)
        if pending() == 0:
            return
        time.sleep(min(60, 2 ** cycle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--database-output", type=Path, default=DEFAULT_DATABASE,
        help="Canonical CSV database updated from each completed run.",
    )
    parser.add_argument(
        "--database-url", default=DEFAULT_DATABASE_URL,
        help="Relational database URL for the public website (PostgreSQL in production).",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--mapping-dir-name",
        default="mapping",
        help="Subdirectory under the run directory for mapping artifacts (default: mapping).",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=Path("src/extraction/extraction_agent_v2_6.py"),
        help="Extraction-agent entry point; defaults to the confirmed v2.6 agent.",
    )
    parser.add_argument(
        "--icl-mode",
        choices=("retrieval", "all", "none"),
        default="none",
        help="ICL mode passed to the selected extractor; v2.6 production default is none.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dictionary-hints", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--normalization-mode", choices=("agents", "local_only"), default="agents")
    parser.add_argument(
        "--verifier-mode",
        choices=("disabled", "selective"),
        default="disabled",
        help="Optional post-extraction verifier. selective reviews only high-risk candidates and fails open.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts inside one agent invocation.")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=3,
        help=(
            "Maximum controller re-invocations for unresolved API/validation rows. "
            "Together with --max-attempts this bounds each row's total model calls "
            "(default: 3 x 3 = 9)."
        ),
    )
    args = parser.parse_args()
    if args.max_attempts < 1 or args.max_cycles < 1 or args.workers < 1:
        raise ValueError("workers, max-attempts, and max-cycles must be positive")
    if not args.mapping_dir_name or Path(args.mapping_dir_name).name != args.mapping_dir_name:
        raise ValueError("--mapping-dir-name must be a single directory name")
    if not args.extractor.exists():
        raise FileNotFoundError(f"Extractor script not found: {args.extractor}")

    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("--run-name must be a single directory name")
    run_dir = RUNS_DIR / args.run_name
    extraction_dir = run_dir / "extraction"
    mapping_dir = run_dir / args.mapping_dir_name
    for directory in (extraction_dir, mapping_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "controller_manifest.json"
    manifest = {"run_name": args.run_name, "input": str(args.input), "model": args.model, "stages": {}}
    write_manifest(manifest_path, manifest)

    relations = extraction_dir / "relations.csv"
    unique_relations = extraction_dir / "relations_deduplicated.csv"
    status = extraction_dir / "status.csv"
    extraction_work_items = extraction_dir / "extraction_work_items.csv"
    verifier_decisions = extraction_dir / "selective_verifier_decisions.csv"
    verified_relations = extraction_dir / "selective_verified_relations.csv"
    verification_work_items = extraction_dir / "verification_work_items.csv"
    mapping_input = mapping_dir / "relations_for_mapping.csv"
    mapped_local = mapping_dir / "local_mapped.csv"
    mapped_cell = mapping_dir / "cell_agent.csv"
    mapped_final = mapping_dir / "cell_gene_agent.csv"
    cell_decisions = mapping_dir / "cell_decisions.csv"
    gene_decisions = mapping_dir / "gene_decisions.csv"

    database_action = lambda: run([
        str(PYTHON), "src/orchestration/upsert_cell_marker_database.py",
        "--input", str(mapped_final), "--database", str(args.database_output), "--run-name", args.run_name,
    ])
    relational_database_action = lambda: run([
        str(PYTHON), "src/orchestration/ingest_marker_database.py",
        "--input", str(mapped_final), "--run-name", args.run_name, "--database-url", args.database_url,
    ])

    work_item_command = [
        str(PYTHON), "src/orchestration/prepare_extraction_work_items.py",
        "--input", str(args.input), "--metadata", str(args.metadata), "--output", str(extraction_work_items),
    ]
    if args.dictionary_hints == "disabled":
        work_item_command.append("--disable-dictionary-hints")
    run(work_item_command)
    manifest["stages"]["extraction_work_items"] = {"output": str(extraction_work_items)}
    retry_stage(
        "extractor",
        lambda: run([
            str(PYTHON), str(args.extractor), "--input", str(args.input), "--work-items", str(extraction_work_items),
            "--relations-output", str(relations), "--status-output", str(status), "--model", args.model,
            "--workers", str(args.workers), "--max-attempts", str(args.max_attempts),
            "--icl-mode", args.icl_mode,
        ]),
        lambda: extraction_pending(args.input, status), args.max_cycles, manifest,
    )
    write_manifest(manifest_path, manifest)

    remaining_extraction = extraction_pending(args.input, status)
    if remaining_extraction:
        manifest["stages"]["mapping"] = {
            "skipped": f"Extraction remains incomplete: {remaining_extraction} retryable rows."
        }
        write_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Extraction is incomplete ({remaining_extraction} retryable rows); "
            "verification and mapping will not run. Re-run the controller to continue."
        )

    # The extractor only creates relations.csv when it has at least one relation.
    # Materialize its schema so an all-NO_ASSOCIATION run remains a valid run.
    if not relations.exists():
        pd.DataFrame(columns=RELATION_COLUMNS).to_csv(relations, index=False)

    run([
        str(PYTHON), "src/orchestration/deduplicate_relation_artifact.py",
        "--input", str(relations), "--output", str(unique_relations),
    ])
    relations = unique_relations
    manifest["stages"]["relation_deduplication"] = {"output": str(unique_relations)}

    if read_csv(relations).empty:
        # The wrapper always evaluates the selective-verifier artifact, so preserve
        # the same empty relation contract even though no verifier invocation occurs.
        pd.DataFrame(columns=RELATION_COLUMNS).to_csv(verified_relations, index=False)
        pd.DataFrame(columns=MAPPED_COLUMNS).to_csv(mapped_final, index=False)
        manifest["stages"]["verifier"] = {"skipped": "No extracted relations to verify."}
        manifest["stages"]["mapping"] = {"skipped": "No extracted relations to map.", "output": str(mapped_final)}
        write_manifest(manifest_path, manifest)
        print(f"\n[controller] finished with no extracted relations: {run_dir}")
        return

    relations_for_mapping = relations
    if args.verifier_mode == "selective":
        run([
            str(PYTHON), "src/orchestration/prepare_verification_work_items.py",
            "--input", str(relations), "--output", str(verification_work_items),
        ])
        verifier_action = lambda: run([
                str(PYTHON), "src/verification/verification_agent_selective.py",
                "--input", str(verification_work_items), "--decisions-output", str(verifier_decisions),
                "--verified-output", str(verified_relations), "--model", args.model,
                "--max-attempts", str(args.max_attempts),
            ])
        retry_stage(
            "verifier", verifier_action,
            lambda: verification_pending(verification_work_items, verifier_decisions),
            args.max_cycles, manifest,
        )
        relations_for_mapping = verified_relations
        manifest["stages"]["verifier"] = {
            "mode": "selective", "work_items": str(verification_work_items), "decisions": str(verifier_decisions),
            "verified_relations": str(verified_relations),
        }
    else:
        manifest["stages"]["verifier"] = {"skipped": "Verifier disabled for this run."}
    write_manifest(manifest_path, manifest)

    run([
        str(PYTHON), "src/orchestration/prepare_mapping_work_items.py",
        "--relations", str(relations_for_mapping), "--output", str(mapping_input),
    ])
    if mapping_input.exists():
        run([
            str(PYTHON), "src/mapping/ontology_mapper.py", "--input", str(mapping_input), "--output", str(mapped_local),
            "--cache-dir", str(DEFAULT_CACHE_DIR),
        ])
        if args.normalization_mode == "local_only":
            shutil.copyfile(mapped_local, mapped_final)
            database_action()
            relational_database_action()
            manifest["stages"]["normalization"] = {"mode": "local_only", "output": str(mapped_final)}
            manifest["stages"]["database"] = {"csv_output": str(args.database_output), "url": args.database_url}
            write_manifest(manifest_path, manifest)
            print("[controller] normalization agents skipped: local mapping only")
            print(f"\n[controller] finished: {run_dir}")
            return
        cell_action = lambda: run([
                str(PYTHON), "src/mapping/cell_normalization_agent.py", "--input", str(mapped_local), "--output", str(mapped_cell),
                "--decisions-output", str(cell_decisions), "--model", args.model, "--max-attempts", str(args.max_attempts),
            ])
        cell_action()
        retry_stage("cell_normalizer", cell_action, lambda: normalization_pending(cell_decisions), args.max_cycles, manifest)
        gene_action = lambda: run([
                str(PYTHON), "src/mapping/gene_normalization_agent.py", "--input", str(mapped_cell), "--output", str(mapped_final),
                "--decisions-output", str(gene_decisions), "--model", args.model, "--max-attempts", str(args.max_attempts),
            ])
        gene_action()
        retry_stage("gene_normalizer", gene_action, lambda: normalization_pending(gene_decisions), args.max_cycles, manifest)
        database_action()
        relational_database_action()
        manifest["stages"]["database"] = {"csv_output": str(args.database_output), "url": args.database_url}
    else:
        manifest["stages"]["mapping"] = {"skipped": "No extraction relation artifact was created."}
    write_manifest(manifest_path, manifest)
    print(f"\n[controller] finished: {run_dir}")


if __name__ == "__main__":
    main()
