from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mapping.ontology_mapper import normalize_term, normalize_cell_mention
from src.mapping.normalization_runtime import run_tasks


DEFAULT_INPUT = ROOT / "results/full/mapped/normalization_input.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("CMAGENT_RESOURCES_DIR", ROOT / "resources")) / "local_entity_cache_v3"
DEFAULT_ALIASES = ROOT / "data/mapping/cell_normalization_aliases.json"
DEFAULT_OUTPUT = ROOT / "results/full/mapped/normalization_cell_agent.csv"
DEFAULT_DECISIONS = ROOT / "results/full/mapped/normalization_cell_agent_decisions.csv"
PROMPT_VERSION = "cell-normalization-agent-v4-check-candidate-reason"

SYSTEM_PROMPT = """You normalize extracted biomedical cell mentions to Cell Ontology candidates.
Use only the candidates provided. Never invent an ID, label, synonym, or biological fact.
Choose ACCEPT only when the sentence evidence supports one candidate clearly. Otherwise
return NEEDS_REVIEW or UNMAPPED."""

RESPONSE_SCHEMA = {
    "name": "cell_normalization_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["ACCEPT", "NEEDS_REVIEW", "UNMAPPED"],
            },
            "candidate_index": {"type": "integer", "minimum": 0},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "candidate_index", "confidence", "reason"],
    },
}

DECISION_COLUMNS = [
    "decision_key",
    "Species",
    "Cell_type",
    "Subtype",
    "Evidence",
    "candidate_count",
    "candidate_ids",
    "candidate_labels",
    "decision",
    "candidate_index",
    "confidence",
    "reason",
    "agent_status",
    "error",
    "run_signature",
    "prompt_version",
    "model",
]


@dataclass(frozen=True)
class Candidate:
    identifier: str
    label: str
    source: str
    score: float


def decision_key(row: pd.Series) -> str:
    values = [
        normalize_term(row.get("Species", "")),
        normalize_cell_mention(row.get("Cell_type", "")),
        normalize_cell_mention(row.get("Subtype", "")),
    ]
    return "|".join(values)


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


class CellCandidateRetriever:
    def __init__(self, cache_dir: Path, aliases_path: Path) -> None:
        candidate_path = cache_dir / "cell_candidate_terms.csv"
        cache = pd.read_csv(candidate_path if candidate_path.exists() else cache_dir / "cell_terms.csv", dtype=str).fillna("")
        cache = cache[cache["identifier"].str.startswith("CL_")].copy()
        source_rank = {
            "cell_ontology_label": 0,
            "cell_ontology_exact_synonym": 1,
            "cellmarker_cell_name": 1,
            "cellmarker_cell_class": 2,
        }
        cache["source_rank"] = cache["mapping_source"].map(source_rank).fillna(9)
        cache = cache.sort_values(["identifier", "source_rank", "canonical_term"])
        self.candidates = (
            cache.drop_duplicates("identifier", keep="first")
            [["identifier", "canonical_term", "mapping_source"]]
            .rename(columns={"canonical_term": "label", "mapping_source": "source"})
            .reset_index(drop=True)
        )
        self.search_terms = cache["normalized_term"].astype(str).tolist()
        candidate_lookup = {
            identifier: index
            for index, identifier in enumerate(self.candidates["identifier"].astype(str))
        }
        self.search_candidate_index = [
            candidate_lookup[str(identifier)] for identifier in cache["identifier"]
        ]
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.word_matrix = self.word.fit_transform(self.search_terms)
        self.char_matrix = self.char.fit_transform(self.search_terms)
        with aliases_path.open(encoding="utf-8") as handle:
            raw_aliases = json.load(handle)
        self.aliases = {
            normalize_cell_mention(alias): [normalize_cell_mention(value) for value in values]
            for alias, values in raw_aliases.items()
        }

    def retrieve(self, subtype: str, broad: str, top_k: int) -> list[Candidate]:
        mention_terms = [normalize_cell_mention(subtype), normalize_cell_mention(broad)]
        queries = list(mention_terms)
        for mention in mention_terms:
            queries.extend(self.aliases.get(mention, []))
        scores = np.zeros(len(self.search_terms), dtype=float)
        for rank, query in enumerate(query for query in queries if query):
            word_query = self.word.transform([query])
            char_query = self.char.transform([query])
            lexical = 0.6 * cosine_similarity(word_query, self.word_matrix)[0]
            lexical += 0.4 * cosine_similarity(char_query, self.char_matrix)[0]
            scores = np.maximum(scores, lexical + (0.25 if rank >= 2 else 0.0))
            exact = np.asarray([term == query for term in self.search_terms])
            scores[exact] = np.maximum(scores[exact], 2.0)

        candidate_scores: dict[int, float] = {}
        for term_index, score in enumerate(scores):
            candidate_index = self.search_candidate_index[term_index]
            candidate_scores[candidate_index] = max(candidate_scores.get(candidate_index, 0.0), float(score))
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], self.candidates.iloc[item[0]]["label"]),
        )[:top_k]
        return [
            Candidate(
                identifier=str(self.candidates.iloc[index]["identifier"]),
                label=str(self.candidates.iloc[index]["label"]),
                source=str(self.candidates.iloc[index]["source"]),
                score=score,
            )
            for index, score in ranked
        ]


def build_prompt(row: pd.Series, candidates: list[Candidate]) -> str:
    candidate_lines = "\n".join(
        f"{index}. {candidate.label} | {candidate.identifier}"
        for index, candidate in enumerate(candidates, 1)
    )
    return f"""[Extracted mention]
Species: {row['Species']}
Known tissue: {row.get('Tissue', '')}
Most specific cell mention: {row['Subtype']}
Broad cell mention: {row['Cell_type']}
Sentence evidence: {row['Evidence']}

[Candidate Cell Ontology terms]
{candidate_lines}

[Decision]
Return ACCEPT with one candidate_index only when the mention and evidence clearly support it.
If the most specific mention is more specific than the broad cell mention, do not ACCEPT
a candidate that only normalizes the broad cell; return NEEDS_REVIEW instead.
Return NEEDS_REVIEW when several candidates remain plausible. Return UNMAPPED when none fit.
For NEEDS_REVIEW or UNMAPPED set candidate_index to 0."""


def validate_decision(payload: object, candidates: list[Candidate]) -> tuple[dict[str, object] | None, str]:
    if not isinstance(payload, dict):
        return None, "response is not an object"
    required = {"decision", "candidate_index", "confidence", "reason"}
    if set(payload) != required:
        return None, "response fields do not match schema"
    decision = payload.get("decision")
    candidate_index = payload.get("candidate_index")
    confidence = payload.get("confidence")
    if decision not in {"ACCEPT", "NEEDS_REVIEW", "UNMAPPED"}:
        return None, "invalid decision"
    if confidence not in {"high", "medium", "low"}:
        return None, "invalid confidence"
    if type(candidate_index) is not int:
        return None, "candidate_index is not an integer"
    if decision == "ACCEPT" and not 1 <= candidate_index <= len(candidates):
        return None, "ACCEPT must select a candidate"
    if decision != "ACCEPT" and candidate_index != 0:
        return None, "non-accept decision must use candidate_index 0"
    if decision == "ACCEPT":
        reason_ids = {value.upper().replace(":", "_") for value in
                      re.findall(r"\bCL[_:]\d{7}\b", str(payload.get("reason", "")), flags=re.I)}
        selected_id = candidates[candidate_index - 1].identifier.upper().replace(":", "_")
        if reason_ids and selected_id not in reason_ids:
            return None, "selected candidate ID disagrees with the ID in the explanation"
    return payload, ""


def run_signature(retriever: CellCandidateRetriever, model: str, top_k: int) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "top_k": top_k,
        "system_prompt": SYSTEM_PROMPT,
        "schema": RESPONSE_SCHEMA,
        "candidate_ids": retriever.candidates["identifier"].tolist(),
        "candidate_labels": retriever.candidates["label"].tolist(),
        "search_terms": retriever.search_terms,
        "search_candidate_index": retriever.search_candidate_index,
        "aliases": retriever.aliases,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def terminal_decisions(path: Path, signature: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "run_signature" not in frame:
        raise ValueError("Existing decision file has no run signature; use --overwrite.")
    signatures = set(frame["run_signature"].astype(str))
    if signatures and signatures != {signature}:
        raise ValueError("Existing decisions belong to a different configuration; use --overwrite.")
    latest = frame.drop_duplicates("decision_key", keep="last")
    completed = latest["agent_status"].isin(["ACCEPT", "UNMAPPED", "NEEDS_REVIEW"])
    return set(latest.loc[completed, "decision_key"])


def apply_decisions(mapping: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    result = mapping.copy()
    result["decision_key"] = result.apply(decision_key, axis=1)
    latest = decisions.drop_duplicates("decision_key", keep="last").set_index("decision_key")
    prefix = "Cell"
    result[f"{prefix}_Agent_Status"] = result["decision_key"].map(latest["agent_status"]).fillna("NOT_REQUESTED")
    result[f"{prefix}_Agent_Confidence"] = result["decision_key"].map(latest["confidence"]).fillna("")
    result[f"{prefix}_Agent_Reason"] = result["decision_key"].map(latest["reason"]).fillna("")
    result[f"{prefix}_Agent_Candidate_ID"] = ""
    accepted = result[f"{prefix}_Agent_Status"].eq("ACCEPT")
    for index in result[accepted].index:
        decision = latest.loc[result.at[index, "decision_key"]]
        candidate_ids = str(decision["candidate_ids"]).split("|")
        candidate_labels = str(decision["candidate_labels"]).split("|")
        selected = int(decision["candidate_index"]) - 1
        if not 0 <= selected < len(candidate_ids):
            continue
        result.at[index, "CellOntology_ID"] = candidate_ids[selected]
        result.at[index, "Cell_Mapping_Term"] = candidate_labels[selected]
        result.at[index, "Cell_Mapping_Source"] = "agent_candidate_high"
        result.at[index, f"{prefix}_Agent_Candidate_ID"] = candidate_ids[selected]
    return result.drop(columns="decision_key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained LLM cell normalization over local CL candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions-output", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.max_attempts < 1 or args.workers < 1:
        raise ValueError("--top-k, --max-attempts and --workers must be positive")
    mapping = pd.read_csv(args.input).fillna("")
    target_id_column = "CellOntology_ID"
    required = {"Species", "Cell_type", "Subtype", target_id_column, "Evidence"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Input missing columns: {', '.join(sorted(missing))}")
    retriever = CellCandidateRetriever(args.cache_dir, args.aliases)
    signature = run_signature(retriever, args.model, args.top_k)
    unmapped = mapping[mapping[target_id_column].eq("")].copy()
    unmapped["decision_key"] = unmapped.apply(decision_key, axis=1)
    tasks = unmapped.drop_duplicates("decision_key", keep="first")
    if args.limit is not None:
        tasks = tasks.head(args.limit)

    if args.validate_only:
        candidate_counts = [len(retriever.retrieve(row.Subtype, "", args.top_k)) for row in tasks.itertuples(index=False)]
        print(f"Unmapped relations:      {len(unmapped):,}")
        print(f"Unique agent decisions: {len(tasks):,}")
        print(f"Candidate pool:         {len(retriever.candidates):,}")
        print(f"Candidates per task:    min={min(candidate_counts, default=0)}, max={max(candidate_counts, default=0)}")
        print(f"Run signature:          {signature}")
        print("Validation passed; no API calls were made.")
        return

    if args.overwrite:
        args.decisions_output.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)
    done = terminal_decisions(args.decisions_output, signature)
    tasks = tasks[~tasks["decision_key"].isin(done)].copy()
    print(f"Specific cell normalization agent: {len(tasks):,} pending decisions, {len(done):,} already completed")
    if not tasks.empty:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv(ROOT / ".env")
                api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or VITE_OPENAI_API_KEY before agent normalization.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        def process(row):
            candidates = retriever.retrieve(row["Subtype"], "", args.top_k)
            record: dict[str, object] = {
                "decision_key": row["decision_key"],
                "Species": row["Species"],
                "Cell_type": row["Cell_type"],
                "Subtype": row["Subtype"],
                "Evidence": row["Evidence"],
                "candidate_count": len(candidates),
                "candidate_ids": "|".join(candidate.identifier for candidate in candidates),
                "candidate_labels": "|".join(candidate.label for candidate in candidates),
                "decision": "",
                "candidate_index": 0,
                "confidence": "",
                "reason": "",
                "agent_status": "API_ERROR",
                "error": "",
                "run_signature": signature,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
            }
            for attempt in range(1, args.max_attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_prompt(row, candidates) +
                             (f"\n[Correction required]\n{record['error']}. Select a valid candidate index for ACCEPT, or use NEEDS_REVIEW/UNMAPPED with index 0." if record["error"] else "")},
                        ],
                        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                        temperature=0.0,
                    )
                    payload = json.loads(response.choices[0].message.content or "{}")
                    validated, error = validate_decision(payload, candidates)
                    if validated is None:
                        raise ValueError(error)
                    record.update(validated)
                    record["error"] = ""
                    if validated["decision"] == "ACCEPT" and validated["confidence"] == "high":
                        record["agent_status"] = "ACCEPT"
                    elif validated["decision"] == "UNMAPPED":
                        record["agent_status"] = "UNMAPPED"
                    else:
                        record["agent_status"] = "NEEDS_REVIEW"
                    break
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        for count, record in enumerate(run_tasks(tasks, process, args.workers), 1):
            append_csv(args.decisions_output, [record])
            if count % 10 == 0 or count == len(tasks):
                print(f"[{count:,}/{len(tasks):,}] decisions completed")

    decisions = pd.read_csv(args.decisions_output).fillna("") if args.decisions_output.exists() else pd.DataFrame(columns=DECISION_COLUMNS)
    output = apply_decisions(mapping, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Decisions: {args.decisions_output}")
    print(f"Mapped output: {args.output}")


if __name__ == "__main__":
    main()
