#!/usr/bin/env python3
"""Run the production CellMarker v2.6 zero-shot + strict verifier pipeline.

This bundle deliberately excludes the experimental secondary restorative reviewer.
All relation polarities are preserved through extraction, verification, and mapping.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[2]
CONTROLLER = BUNDLE / "controller.py"
EXTRACTION_AGENT = BUNDLE / "extraction_agent.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--database-output", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mapping-dir-name", default="mapping")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--dictionary-hints", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--verifier-mode", choices=("selective", "disabled"), default="selective")
    parser.add_argument("--normalization-mode", choices=("agents", "local_only"), default="agents")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required_paths = ((args.input, "input"), (CONTROLLER, "controller"), (EXTRACTION_AGENT, "extraction agent"))
    if args.metadata is not None:
        required_paths += ((args.metadata, "metadata"),)
    for path, label in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    command = [
        sys.executable, str(CONTROLLER),
        "--input", str(args.input),
        "--run-name", args.run_name,
        "--mapping-dir-name", args.mapping_dir_name,
        "--extractor", str(EXTRACTION_AGENT),
        "--icl-mode", "none", "--dictionary-hints", args.dictionary_hints,
        "--verifier-mode", args.verifier_mode,
        "--normalization-mode", args.normalization_mode,
        "--model", args.model,
        "--workers", str(args.workers),
        "--max-attempts", str(args.max_attempts),
        "--max-cycles", str(args.max_cycles),
    ]
    if args.metadata is not None:
        command.extend(("--metadata", str(args.metadata)))
    if args.database_output is not None:
        command.extend(("--database-output", str(args.database_output)))
    if args.database_url is not None:
        command.extend(("--database-url", args.database_url))
    print("[production-v2.6] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
