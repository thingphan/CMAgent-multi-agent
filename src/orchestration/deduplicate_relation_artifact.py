#!/usr/bin/env python3
"""Deduplicate resumable extraction relation artifacts before downstream stages."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

KEY = ["row_id", "Cell mention", "Broad cell mention", "Marker mention", "Polarity", "Relation evidence"]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype=str).fillna("")
    if missing := set(KEY) - set(frame):
        raise ValueError(f"Input missing columns: {', '.join(sorted(missing))}")
    result = frame.drop_duplicates(KEY, keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Deduplicated relations: {len(frame):,} -> {len(result):,}")

if __name__ == "__main__":
    main()
