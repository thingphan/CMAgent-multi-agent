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

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mapping.ontology_mapper import normalize_term
from src.mapping.normalization_runtime import run_tasks


DEFAULT_INPUT = ROOT / "results/full/mapped/normalization_cell_agent.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("CMAGENT_RESOURCES_DIR", ROOT / "resources")) / "local_entity_cache_v3"
DEFAULT_ALIASES = ROOT / "data/mapping/gene_normalization_aliases.json"
DEFAULT_OUTPUT = ROOT / "results/full/mapped/normalization_cell_gene_agent.csv"
DEFAULT_DECISIONS = ROOT / "results/full/mapped/normalization_gene_agent_decisions.csv"
PROMPT_VERSION = "gene-normalization-agent-v3-check-candidate-reason"

SYSTEM_PROMPT = """You normalize extracted gene or protein marker mentions to NCBI Gene candidates.
Use only the species-specific candidates provided. Never invent a gene symbol, Gene ID,
alias, or biological fact. ACCEPT only when the sentence evidence clearly supports one
candidate. A protein/CD name may be accepted only when its gene-symbol equivalence is
shown by a provided candidate. Otherwise return NEEDS_REVIEW or UNMAPPED."""

RESPONSE_SCHEMA = {
    "name": "gene_normalization_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["ACCEPT", "NEEDS_REVIEW", "UNMAPPED"]},
            "candidate_index": {"type": "integer", "minimum": 0},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "candidate_index", "confidence", "reason"],
    },
}

DECISION_COLUMNS = [
    "decision_key", "Species", "Original_Marker", "Evidence", "candidate_count",
    "candidate_ids", "candidate_symbols", "decision", "candidate_index", "confidence",
    "reason", "agent_status", "error", "run_signature", "prompt_version", "model",
]


@dataclass(frozen=True)
class GeneCandidate:
    identifier: str
    symbol: str
    score: float


def decision_key(row: pd.Series) -> str:
    return "|".join([normalize_term(row.get("Species", "")), str(row.get("Original_Marker", "")).strip()])


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS, extrasaction="ignore")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


class GeneCandidateRetriever:
    def __init__(self, cache_dir: Path, aliases_path: Path) -> None:
        self.cache_signature = hashlib.sha256((cache_dir / "gene_terms.csv").read_bytes() +
                                              (cache_dir / "gene_exact_terms.csv").read_bytes()).hexdigest()
        terms = pd.read_csv(cache_dir / "gene_terms.csv", dtype=str).fillna("")
        exact_terms = pd.read_csv(cache_dir / "gene_exact_terms.csv", dtype=str).fillna("")
        exact_terms = exact_terms.rename(columns={"exact_term": "normalized_term"})
        exact_terms["normalized_term"] = exact_terms["normalized_term"].map(normalize_term)
        terms = pd.concat([terms, exact_terms], ignore_index=True)
        terms = terms[terms["species"].isin(["Human", "Mouse"]) & terms["identifier"].str.fullmatch(r"\d+")].copy()
        source_rank = {"cellmarker_symbol": 0, "cellmarker_marker": 1, "cellmarker_gene_name": 2}
        terms["source_rank"] = terms["mapping_source"].map(source_rank).fillna(9)
        terms = terms.sort_values(["species", "identifier", "source_rank", "canonical_term"])
        self.terms_by_species: dict[str, pd.DataFrame] = {}
        self.word: dict[str, TfidfVectorizer] = {}
        self.char: dict[str, TfidfVectorizer] = {}
        self.word_matrix: dict[str, object] = {}
        self.char_matrix: dict[str, object] = {}
        self.candidates: dict[str, pd.DataFrame] = {}
        self.candidate_index: dict[str, list[int]] = {}
        for species, frame in terms.groupby("species", sort=False):
            candidates = (
                frame.drop_duplicates("identifier", keep="first")[["identifier", "canonical_term"]]
                .rename(columns={"canonical_term": "symbol"})
                .reset_index(drop=True)
            )
            lookup = {identifier: index for index, identifier in enumerate(candidates["identifier"].astype(str))}
            self.terms_by_species[species] = frame.reset_index(drop=True)
            self.candidates[species] = candidates
            self.candidate_index[species] = [lookup[str(identifier)] for identifier in frame["identifier"]]
            self.word[species] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
            self.char[species] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
            search_terms = frame["normalized_term"].astype(str).tolist()
            self.word_matrix[species] = self.word[species].fit_transform(search_terms)
            self.char_matrix[species] = self.char[species].fit_transform(search_terms)
        with aliases_path.open(encoding="utf-8") as handle:
            raw_aliases = json.load(handle)
        self.aliases = {normalize_term(key): [str(value).strip() for value in values] for key, values in raw_aliases.items()}

    def retrieve(self, species: str, marker: str, top_k: int) -> list[GeneCandidate]:
        if species not in self.terms_by_species:
            return []
        queries = [normalize_term(marker)]
        queries.extend(normalize_term(value) for value in self.aliases.get(normalize_term(marker), []))
        frame = self.terms_by_species[species]
        scores = np.zeros(len(frame), dtype=float)
        for rank, query in enumerate(query for query in queries if query):
            lexical = 0.6 * cosine_similarity(self.word[species].transform([query]), self.word_matrix[species])[0]
            lexical += 0.4 * cosine_similarity(self.char[species].transform([query]), self.char_matrix[species])[0]
            scores = np.maximum(scores, lexical + (0.25 if rank else 0.0))
            # An exact cache term is more reliable than lexical similarity alone.
            exact_matches = self.terms_by_species[species]["normalized_term"].eq(query).to_numpy()
            scores[exact_matches] = np.maximum(scores[exact_matches], 2.0 + (0.25 if rank else 0.0))
        by_candidate: dict[int, float] = {}
        for index, score in enumerate(scores):
            candidate = self.candidate_index[species][index]
            by_candidate[candidate] = max(by_candidate.get(candidate, 0.0), float(score))
        ranked = sorted(by_candidate.items(), key=lambda item: (-item[1], self.candidates[species].iloc[item[0]]["symbol"]))[:top_k]
        return [GeneCandidate(str(self.candidates[species].iloc[index]["identifier"]), str(self.candidates[species].iloc[index]["symbol"]), score) for index, score in ranked]


def build_prompt(row: pd.Series, candidates: list[GeneCandidate]) -> str:
    candidate_lines = "\n".join(f"{index}. {item.symbol} | NCBI Gene ID {item.identifier}" for index, item in enumerate(candidates, 1))
    return f"""[Extracted marker]\nSpecies: {row['Species']}\nMarker mention: {row['Original_Marker']}\nSentence evidence: {row['Evidence']}\n\n[Candidate genes]\n{candidate_lines}\n\n[Decision]\nReturn ACCEPT with one candidate_index only when the mention and evidence clearly support it. Return NEEDS_REVIEW when candidates remain plausible. Return UNMAPPED when the mention is not a gene/protein marker or none fit. For NEEDS_REVIEW or UNMAPPED set candidate_index to 0."""


def validate_decision(payload: object, candidates: list[GeneCandidate]) -> tuple[dict[str, object] | None, str]:
    if not isinstance(payload, dict) or set(payload) != {"decision", "candidate_index", "confidence", "reason"}:
        return None, "response fields do not match schema"
    decision, index, confidence = payload["decision"], payload["candidate_index"], payload["confidence"]
    if decision not in {"ACCEPT", "NEEDS_REVIEW", "UNMAPPED"} or confidence not in {"high", "medium", "low"}:
        return None, "invalid decision or confidence"
    if type(index) is not int:
        return None, "candidate_index is not an integer"
    if decision == "ACCEPT" and not 1 <= index <= len(candidates):
        return None, "ACCEPT must select a candidate"
    if decision != "ACCEPT" and index != 0:
        return None, "non-accept decision must use candidate_index 0"
    if decision == "ACCEPT":
        mentioned = set(re.findall(r"\b(?:NCBI\s+)?(?:gene\s+ID|geneid|Entrez\s+(?:gene\s+)?ID)\s*[:=#]?\s*(\d+)\b",
                                   str(payload.get("reason", "")), flags=re.I))
        if mentioned and str(candidates[index - 1].identifier) not in mentioned:
            return None, "selected candidate ID disagrees with the gene ID in the explanation"
    return payload, ""


def signature(retriever: GeneCandidateRetriever, model: str, top_k: int) -> str:
    payload = {"prompt_version": PROMPT_VERSION, "model": model, "top_k": top_k, "schema": RESPONSE_SCHEMA,
               "aliases": retriever.aliases, "cache_signature": retriever.cache_signature}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def apply_decisions(mapping: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    result = mapping.copy()
    result["decision_key"] = result.apply(decision_key, axis=1)
    latest = decisions.drop_duplicates("decision_key", keep="last").set_index("decision_key")
    result["Gene_Agent_Status"] = result["decision_key"].map(latest["agent_status"]).fillna("NOT_REQUESTED")
    result["Gene_Agent_Confidence"] = result["decision_key"].map(latest["confidence"]).fillna("")
    result["Gene_Agent_Reason"] = result["decision_key"].map(latest["reason"]).fillna("")
    result["Gene_Agent_Candidate_ID"] = ""
    for index in result.index[result["Gene_Agent_Status"].eq("ACCEPT")]:
        decision = latest.loc[result.at[index, "decision_key"]]
        ids = str(decision["candidate_ids"]).split("|")
        symbols = str(decision["candidate_symbols"]).split("|")
        selected = int(decision["candidate_index"]) - 1
        if 0 <= selected < len(ids):
            result.at[index, "NCBI_GeneID"] = ids[selected]
            result.at[index, "Official_Symbol"] = symbols[selected]
            result.at[index, "Gene_Mapping_Source"] = "agent_candidate_high"
            result.at[index, "Gene_Agent_Candidate_ID"] = ids[selected]
    return result.drop(columns="decision_key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained LLM gene normalization over local NCBI Gene candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions-output", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.top_k < 1:
        raise ValueError("workers, max-attempts and top-k must be positive")
    if args.top_k < 1 or args.max_attempts < 1:
        raise ValueError("--top-k and --max-attempts must be positive")
    mapping = pd.read_csv(args.input).fillna("")
    required = {"Species", "Original_Marker", "NCBI_GeneID", "Evidence"}
    if missing := required - set(mapping.columns):
        raise ValueError(f"Input missing columns: {', '.join(sorted(missing))}")
    retriever = GeneCandidateRetriever(args.cache_dir, args.aliases)
    run_signature = signature(retriever, args.model, args.top_k)
    tasks = mapping[mapping["NCBI_GeneID"].eq("")].copy()
    tasks["decision_key"] = tasks.apply(decision_key, axis=1)
    tasks = tasks.drop_duplicates("decision_key", keep="first")
    if args.validate_only:
        counts = [len(retriever.retrieve(row.Species, row.Original_Marker, args.top_k)) for row in tasks.itertuples(index=False)]
        print(f"Unmapped gene relations: {mapping['NCBI_GeneID'].eq('').sum():,}")
        print(f"Unique agent decisions:  {len(tasks):,}")
        print(f"Candidates per task:     min={min(counts, default=0)}, max={max(counts, default=0)}")
        print("Validation passed; no API calls were made.")
        return
    if args.overwrite:
        args.decisions_output.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)
    done: set[str] = set()
    if args.decisions_output.exists():
        previous = pd.read_csv(args.decisions_output, dtype=str).fillna("")
        if set(previous.get("run_signature", [])) - {run_signature}:
            raise ValueError("Existing decisions belong to a different configuration; use --overwrite.")
        done = set(previous.loc[previous["agent_status"].isin(["ACCEPT", "NEEDS_REVIEW", "UNMAPPED"]), "decision_key"])
    tasks = tasks[~tasks["decision_key"].isin(done)]
    print(f"Gene normalization agent: {len(tasks):,} pending decisions, {len(done):,} already completed")
    if not tasks.empty:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or VITE_OPENAI_API_KEY before agent normalization.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        def process(row):
            candidates = retriever.retrieve(row["Species"], row["Original_Marker"], args.top_k)
            record: dict[str, object] = {"decision_key": row["decision_key"], "Species": row["Species"], "Original_Marker": row["Original_Marker"], "Evidence": row["Evidence"], "candidate_count": len(candidates), "candidate_ids": "|".join(item.identifier for item in candidates), "candidate_symbols": "|".join(item.symbol for item in candidates), "decision": "", "candidate_index": 0, "confidence": "", "reason": "", "agent_status": "API_ERROR", "error": "", "run_signature": run_signature, "prompt_version": PROMPT_VERSION, "model": args.model}
            for _ in range(args.max_attempts):
                try:
                    correction = f"\n[Correction required]\n{record['error']}. ACCEPT requires an index from 1 to {len(candidates)}; otherwise choose NEEDS_REVIEW/UNMAPPED with index 0." if record["error"] else ""
                    response = client.chat.completions.create(model=args.model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(row, candidates) + correction}], response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA}, temperature=0.0)
                    validated, error = validate_decision(json.loads(response.choices[0].message.content or "{}"), candidates)
                    if validated is None:
                        raise ValueError(error)
                    record.update(validated)
                    record["error"] = ""
                    record["agent_status"] = "ACCEPT" if validated["decision"] == "ACCEPT" and validated["confidence"] == "high" else ("UNMAPPED" if validated["decision"] == "UNMAPPED" else "NEEDS_REVIEW")
                    break
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        for count, record in enumerate(run_tasks(tasks, process, args.workers), 1):
            append_csv(args.decisions_output, [record])
            print(f"[{count:,}/{len(tasks):,}] decisions completed")
    decisions = pd.read_csv(args.decisions_output).fillna("") if args.decisions_output.exists() else pd.DataFrame(columns=DECISION_COLUMNS)
    output = apply_decisions(mapping, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Decisions: {args.decisions_output}")
    print(f"Mapped output: {args.output}")


if __name__ == "__main__":
    main()
