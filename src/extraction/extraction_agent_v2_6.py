"""Prompt3 v2.6: standalone zero-shot cell-marker relation extractor.

This version uses the revised v2.6 prompt and v2.5 dictionary span hints. It
does not load few-shot examples unless an explicit --icl path is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "prompt3-v2.6-dictionary-hints-zero-shot"
# Preserve v2.5's verifier-first design: enforce the response contract here and
# let src/verification/verification_agent_selective.py judge relation semantics.
VALIDATION_MODE = "structure_only"
DEFAULT_INPUT = ROOT / "data/processed/input_sentences.csv"
DEFAULT_ICL: Path | None = None
DEFAULT_METADATA = ROOT / "data/processed/pmid_metadata.csv"
DEFAULT_RELATIONS = ROOT / "results/full/extraction/prompt3_v2_6_relations.csv"
DEFAULT_STATUS = ROOT / "results/full/extraction/prompt3_v2_6_status.csv"


_dictionary_provider: Any | None = None


def dictionary_provider() -> Any:
    """Build the v2.5 dictionary-hint provider only when an API prompt needs it."""
    global _dictionary_provider
    if _dictionary_provider is None:
        try:
            from .dictionary_hints import DictionaryHintProvider
        except ImportError:  # Script execution from src/extraction.
            from dictionary_hints import DictionaryHintProvider

        _dictionary_provider = DictionaryHintProvider(max_per_type=12)
    return _dictionary_provider

RELATION_COLUMNS = [
    "row_id",
    "PMID",
    "Species",
    "Tissue",
    "Cell_type",
    "Cell type (subtype)",
    "Cell marker",
    "Negative marker",
    "Polarity",
    "Confidence",
    "Cell mention",
    "Broad cell mention",
    "Marker mention",
    "Relation evidence",
    "Evidence",
    "icl_mode",
    "icl_example_ids",
    "icl_retrieval_scores",
    "run_signature",
    "prompt_version",
    "model",
]

STATUS_COLUMNS = [
    "row_id",
    "PMID",
    "status",
    "relation_count",
    "attempts",
    "error",
    "raw_response",
    "auto_repair_count",
    "auto_filter_count",
    "auto_scope_repair_count",
    "Evidence",
    "icl_mode",
    "icl_example_ids",
    "icl_retrieval_scores",
    "run_signature",
    "prompt_version",
    "model",
]

RESPONSE_SCHEMA = {'name': 'cell_marker_relation_extraction',
 'strict': True,
 'schema': {'type': 'object',
            'additionalProperties': False,
            'properties': {'relations': {'type': 'array',
                                         'items': {'type': 'object',
                                                   'additionalProperties': False,
                                                   'properties': {'cell_mention': {'type': 'string'},
                                                                  'broad_cell_mention': {'type': 'string'},
                                                                  'marker_mention': {'type': 'string'},
                                                                  'polarity': {'type': 'string',
                                                                               'enum': ['positive',
                                                                                        'negative',
                                                                                        'uncertain']},
                                                                  'relation_evidence': {'type': 'string'},
                                                                  'confidence': {'type': 'string',
                                                                                 'enum': ['high',
                                                                                          'medium',
                                                                                          'low']}},
                                                   'required': ['cell_mention',
                                                                'broad_cell_mention',
                                                                'marker_mention',
                                                                'polarity',
                                                                'relation_evidence',
                                                                'confidence']}}},
            'required': ['relations']}}

SYSTEM_PROMPT = '''You extract explicit cell-marker relations from biomedical sentences.
Return only data conforming to the supplied JSON schema. Do not use outside biological
knowledge to create entities or relations.'''

TASK_RULES = '''Extract text-supported cell-marker relation candidates.
Copy exact spans, using the most specific written cell span as cell_mention. CELL must be a written biological cell
population, subtype, state, lineage, or cell line, not tissue, disease, organism, or a bare
cluster. MARKER must be a written gene, protein, transcript, miRNA, surface antigen, or
reporter, not a pathway, signature, module, generic marker phrase, or cluster label.

Return a pair only when grammar links marker and cell through expression, enrichment,
positivity/negativity, marker-of wording, a cell-linked marker list, or a marker-modified
cell phrase. A marker written inside a marker-modified cell name (e.g. CD8+ T cells) is
also an explicit relation.

For broad_cell_mention, use a broader cell phrase explicitly contained in cell_mention
when available (e.g. neural progenitor cells -> progenitor cells; CD8+ T cells -> T cells);
otherwise return "". Do not infer a broad cell from biological knowledge.

Do not emit co-occurrence alone. Preserve list, contrast, parenthesis, exception, and
respectively scope. relation_evidence is one contiguous exact span supporting both mentions.
Dictionary hints may be supplied only to notice exact written spans; they are never proof of a relation.
Return {"relations":[]} when unsupported.'''


@dataclass(frozen=True)
class ValidationResult:
    relations: list[dict[str, str]]
    errors: list[str]


@dataclass(frozen=True)
class RetrievalResult:
    examples: list[dict[str, Any]]
    scores: list[float]

    @property
    def example_ids(self) -> list[str]:
        return [str(example["example_id"]) for example in self.examples]


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _pmid(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return _text(value)


def _nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def sentence_identity(value: str) -> str:
    text = _nfkc(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def sentence_identities_overlap(first: str, second: str) -> bool:
    if first == second:
        return True
    # Captions and copied evidence are often shortened; catch substantial containment.
    return min(len(first), len(second)) >= 40 and (first in second or second in first)


def exact_substring(mention: str, sentence: str) -> bool:
    return bool(mention) and _nfkc(mention) in _nfkc(sentence)


def validate_response(payload: object, sentence: str) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult([], ["response must be a JSON object"])
    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        return ValidationResult([], ["relations must be a JSON array"])

    required = {
        "cell_mention",
        "broad_cell_mention",
        "marker_mention",
        "polarity",
        "relation_evidence",
        "confidence",
    }
    allowed_polarities = {"positive", "negative", "uncertain"}
    allowed_confidences = {"high", "medium", "low"}
    validated: list[dict[str, str]] = []
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for index, raw in enumerate(raw_relations):
        prefix = f"relations[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = required - set(raw)
        extra = set(raw) - required
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"{prefix} unexpected fields: {', '.join(sorted(extra))}")
        if missing or extra:
            continue

        relation = {key: _text(raw[key]) for key in required}
        cell = relation["cell_mention"]
        broad = relation["broad_cell_mention"]
        marker = relation["marker_mention"]
        evidence = relation["relation_evidence"]

        if not cell:
            errors.append(f"{prefix}.cell_mention is empty")
        elif not exact_substring(cell, sentence):
            errors.append(f"{prefix}.cell_mention is not an exact sentence span: {cell!r}")
        if broad and not exact_substring(broad, sentence):
            errors.append(
                f"{prefix}.broad_cell_mention is not an exact sentence span: {broad!r}"
            )
        if not marker:
            errors.append(f"{prefix}.marker_mention is empty")
        elif not exact_substring(marker, sentence):
            errors.append(
                f"{prefix}.marker_mention is not an exact sentence span: {marker!r}"
            )
        if not evidence:
            errors.append(f"{prefix}.relation_evidence is empty")
        elif not exact_substring(evidence, sentence):
            errors.append(f"{prefix}.relation_evidence is not an exact sentence span")
        else:
            if cell and not exact_substring(cell, evidence):
                errors.append(f"{prefix}.relation_evidence does not contain cell_mention")
            if marker and not exact_substring(marker, evidence):
                errors.append(f"{prefix}.relation_evidence does not contain marker_mention")
        if relation["polarity"] not in allowed_polarities:
            errors.append(f"{prefix}.polarity is invalid: {relation['polarity']!r}")
        if relation["confidence"] not in allowed_confidences:
            errors.append(f"{prefix}.confidence is invalid: {relation['confidence']!r}")

        duplicate_key = (cell, marker, relation["polarity"])
        if duplicate_key in seen:
            errors.append(f"{prefix} duplicates an earlier relation")
        seen.add(duplicate_key)
        validated.append(relation)

    return ValidationResult(validated, errors)


def validate_structure(payload: object) -> ValidationResult:
    """Validate the response contract without judging extraction semantics.

    This mode deliberately preserves model-proposed spans for a separate verifier.
    """
    if not isinstance(payload, dict):
        return ValidationResult([], ["response must be a JSON object"])
    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        return ValidationResult([], ["relations must be a JSON array"])

    required = {
        "cell_mention", "broad_cell_mention", "marker_mention", "polarity",
        "relation_evidence", "confidence",
    }
    allowed_polarities = {"positive", "negative", "uncertain"}
    allowed_confidences = {"high", "medium", "low"}
    relations: list[dict[str, str]] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_relations):
        prefix = f"relations[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing, extra = required - set(raw), set(raw) - required
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"{prefix} unexpected fields: {', '.join(sorted(extra))}")
        if missing or extra:
            continue
        relation = {key: _text(raw[key]) for key in required}
        # A list response can contain one malformed placeholder alongside valid
        # relations.  Do not reject the complete sentence for that isolated
        # placeholder: it cannot form a cell-marker pair and is safely omitted.
        if not relation["cell_mention"] or not relation["marker_mention"]:
            continue
        if relation["polarity"] not in allowed_polarities:
            errors.append(f"{prefix}.polarity is invalid: {relation['polarity']!r}")
        if relation["confidence"] not in allowed_confidences:
            errors.append(f"{prefix}.confidence is invalid: {relation['confidence']!r}")
        relations.append(relation)
    return ValidationResult(relations, errors)


def repair_relation_evidence(payload: object, sentence: str) -> tuple[object, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return payload, 0
    repaired = 0
    for relation in payload["relations"]:
        if not isinstance(relation, dict):
            continue
        cell = _text(relation.get("cell_mention"))
        marker = _text(relation.get("marker_mention"))
        evidence = _text(relation.get("relation_evidence"))
        mentions_are_grounded = exact_substring(cell, sentence) and exact_substring(
            marker, sentence
        )
        evidence_is_valid = (
            exact_substring(evidence, sentence)
            and exact_substring(cell, evidence)
            and exact_substring(marker, evidence)
        )
        if mentions_are_grounded and not evidence_is_valid:
            relation["relation_evidence"] = sentence
            repaired += 1
    return payload, repaired


def repair_mention_formatting(payload: object, sentence: str) -> tuple[object, int]:
    """Restore exact source casing/spacing without changing entity wording."""
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return payload, 0
    repaired = 0
    for relation in payload["relations"]:
        if not isinstance(relation, dict):
            continue
        for field in ("cell_mention", "broad_cell_mention", "marker_mention"):
            mention = _text(relation.get(field))
            if not mention or exact_substring(mention, sentence):
                continue
            pattern = re.escape(mention)
            pattern = pattern.replace(r"\ ", r"\s*")
            pattern = pattern.replace(r"\+", r"\s*\+\s*")
            pattern = pattern.replace(r"\-", r"[-‐‑–—]")
            matches = list(re.finditer(pattern, sentence, flags=re.IGNORECASE))
            if len(matches) == 1:
                relation[field] = matches[0].group(0)
                repaired += 1
    return payload, repaired


def filter_invalid_entity_types(
    payload: object, sentence: str = ""
) -> tuple[object, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return payload, 0
    kept = []
    changed = 0
    seen: set[tuple[str, str, str]] = set()
    for relation in payload["relations"]:
        if not isinstance(relation, dict):
            kept.append(relation)
            continue
        cell = _text(relation.get("cell_mention"))
        marker = _text(relation.get("marker_mention"))
        cluster_pattern = r"(?:clusters?\s*[#0-9, -]*(?:and\s*[#0-9, -]+)?|cl\s*\d+)"
        bare_cluster = bool(re.fullmatch(cluster_pattern, cell.casefold()))
        generic_marker = bool(
            re.fullmatch(
                r"(?:typical |canonical |known |established )?[a-z -]*markers?",
                marker.casefold(),
            )
        )
        if generic_marker:
            changed += 1
            continue

        if bare_cluster:
            broad = _text(relation.get("broad_cell_mention"))
            broad_is_biological = bool(
                broad
                and not re.fullmatch(cluster_pattern, broad.casefold())
                and (not sentence or exact_substring(broad, sentence))
            )
            if not broad_is_biological:
                changed += 1
                continue
            relation = dict(relation)
            relation["cell_mention"] = broad
            relation["broad_cell_mention"] = ""
            changed += 1

        key = (
            _text(relation.get("cell_mention")).casefold(),
            _text(relation.get("marker_mention")).casefold(),
            _text(relation.get("polarity")).casefold(),
        )
        if key in seen:
            changed += 1
            continue
        seen.add(key)
        kept.append(relation)
    payload["relations"] = kept
    return payload, changed


def repair_explicit_exception_scope(payload: object, sentence: str) -> tuple[object, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return payload, 0
    matches = re.findall(
        r"not\s+in\s+[\"“]*([^\"”]+?)[\"”]*\s+except\s+for\s+"
        r"([A-Za-z0-9][A-Za-z0-9.+/_-]*)",
        sentence,
        flags=re.IGNORECASE,
    )
    repaired = 0
    for cell, marker in matches:
        cell = cell.strip()
        marker = marker.strip()
        if not exact_substring(cell, sentence) or not exact_substring(marker, sentence):
            continue
        existing = payload["relations"]
        matching = [
            relation
            for relation in existing
            if isinstance(relation, dict)
            and _text(relation.get("marker_mention")).casefold() == marker.casefold()
        ]
        broad = next(
            (
                _text(relation.get("broad_cell_mention"))
                for relation in matching
                if _text(relation.get("cell_mention")) == cell
            ),
            "",
        )
        corrected = {
            "cell_mention": cell,
            "broad_cell_mention": broad,
            "marker_mention": marker,
            "polarity": "positive",
            "relation_evidence": sentence,
            "confidence": "high",
        }
        replacement = [
            relation
            for relation in existing
            if not (
                isinstance(relation, dict)
                and _text(relation.get("marker_mention")).casefold() == marker.casefold()
            )
        ]
        replacement.append(corrected)
        if replacement != existing:
            payload["relations"] = replacement
            repaired += 1
    return payload, repaired


def load_icl_examples(path: Path, minimum_count: int = 1) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ICL file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    examples = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(examples, list):
        raise ValueError("ICL JSON must contain an 'examples' array.")
    if len(examples) < minimum_count:
        raise ValueError(
            f"ICL file must contain at least {minimum_count} examples; found {len(examples)}."
        )

    seen_ids: set[str] = set()
    seen_sentences: set[str] = set()
    validated_examples = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"ICL example {index + 1} must be an object.")
        example_id = _text(example.get("example_id"))
        sentence = _text(example.get("sentence"))
        if not example_id or example_id in seen_ids:
            raise ValueError(f"ICL example {index + 1} has a missing or duplicate example_id.")
        if not sentence or sentence in seen_sentences:
            raise ValueError(f"ICL example {index + 1} has a missing or duplicate sentence.")
        result = validate_response({"relations": example.get("relations")}, sentence)
        if result.errors:
            raise ValueError(
                f"ICL example {example_id} is invalid: {'; '.join(result.errors)}"
            )
        seen_ids.add(example_id)
        seen_sentences.add(sentence)
        validated_examples.append(
            {
                "example_id": example_id,
                "sentence": sentence,
                "relations": result.relations,
            }
        )
    return validated_examples


def retrieval_pattern_features(
    sentence: str, relations: list[dict[str, str]] | None = None
) -> set[str]:
    text = sentence.casefold()
    features: set[str] = set()
    if re.search(
        r"\b(?:not|negative|low|lo|lacking|without|absent|decreased|reduced|less)\b|"
        r"did not|failed to express",
        text,
    ):
        features.add("negative")
    if re.search(
        r"\b(?:compared|comparison|whereas|while|differential|moderate|whether|similar)\b",
        text,
    ):
        features.add("comparison")
    if "respectively" in text:
        features.add("respectively")
    if sentence.count(",") >= 2:
        features.add("multi_marker")
    if re.search(r"\([A-Za-z][A-Za-z0-9.+/ -]{1,18}\)", sentence):
        features.add("alias")
    if re.search(r"\b(?:subset|subcluster|cluster|population)\b", text):
        features.add("subtype")

    if relations is not None:
        polarities = {relation["polarity"] for relation in relations}
        features.update(polarities)
        if not relations:
            features.add("no_relation")
        if len(relations) >= 3:
            features.add("multi_marker")
        cells = {relation["cell_mention"] for relation in relations}
        if len(cells) >= 2:
            features.add("multi_cell")
    return features


class IclRetriever:
    METHOD = "hybrid_tfidf_mmr_v1"

    def __init__(
        self,
        examples: list[dict[str, Any]],
        word_weight: float = 0.65,
        char_weight: float = 0.35,
        pattern_weight: float = 0.15,
        diversity_weight: float = 0.12,
    ) -> None:
        if not examples:
            raise ValueError("At least one ICL example is required for retrieval.")
        self.examples = sorted(examples, key=lambda example: str(example["example_id"]))
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.pattern_weight = pattern_weight
        self.diversity_weight = diversity_weight
        sentences = [str(example["sentence"]) for example in self.examples]
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w.+/-]+\b",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(sentences)
        self.char_matrix = self.char_vectorizer.fit_transform(sentences)
        self.patterns = [
            retrieval_pattern_features(str(example["sentence"]), example["relations"])
            for example in self.examples
        ]
        self.example_similarity = (
            self.word_weight * cosine_similarity(self.word_matrix)
            + self.char_weight * cosine_similarity(self.char_matrix)
        )

    def _pattern_scores(self, sentence: str) -> np.ndarray:
        query_features = retrieval_pattern_features(sentence)
        if not query_features:
            return np.zeros(len(self.examples), dtype=float)
        scores = []
        for example_features in self.patterns:
            overlap = len(query_features & example_features)
            scores.append(overlap / len(query_features))
        return np.asarray(scores, dtype=float)

    def retrieve(self, sentence: str, top_k: int = 4) -> RetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        top_k = min(top_k, len(self.examples))
        word_query = self.word_vectorizer.transform([sentence])
        char_query = self.char_vectorizer.transform([sentence])
        lexical = (
            self.word_weight * cosine_similarity(word_query, self.word_matrix)[0]
            + self.char_weight * cosine_similarity(char_query, self.char_matrix)[0]
        )
        relevance = (1.0 - self.pattern_weight) * lexical + self.pattern_weight * self._pattern_scores(sentence)

        selected: list[int] = []
        remaining = set(range(len(self.examples)))
        while remaining and len(selected) < top_k:
            ranked = []
            for index in remaining:
                redundancy = (
                    max(self.example_similarity[index, chosen] for chosen in selected)
                    if selected
                    else 0.0
                )
                mmr_score = relevance[index] - self.diversity_weight * redundancy
                ranked.append((mmr_score, relevance[index], str(self.examples[index]["example_id"]), index))
            _, _, _, best_index = min(
                ranked, key=lambda item: (-item[0], -item[1], item[2])
            )
            selected.append(best_index)
            remaining.remove(best_index)

        return RetrievalResult(
            examples=[self.examples[index] for index in selected],
            scores=[float(relevance[index]) for index in selected],
        )


def format_icl_block(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for example in examples:
        answer = json.dumps(
            {"relations": example["relations"]}, ensure_ascii=False, separators=(",", ":")
        )
        blocks.append(f"Sentence: {example['sentence']}\nAnswer: {answer}")
    return "\n\n".join(blocks)


def build_user_prompt(
    sentence: str,
    examples: list[dict[str, Any]],
    icl_mode: str = "retrieval",
    species: str = "",
    tissue: str = "",
    validation_errors: list[str] | None = None,
    dictionary_hint_block: str | None = None,
) -> str:
    metadata_lines = []
    if species:
        metadata_lines.append(f"Known species: {species}")
    if tissue:
        metadata_lines.append(f"Known tissue: {tissue}")
    metadata = "\n".join(metadata_lines) or "No verified document metadata supplied."
    repair = ""
    if validation_errors:
        repair = (
            "\n[Correction required]\nThe previous response failed validation:\n- "
            + "\n- ".join(validation_errors)
            + "\nReturn a corrected extraction from the original sentence.\n"
        )
    examples_block = ""
    if icl_mode != "none":
        example_heading = (
            "Retrieved reviewed examples" if icl_mode == "retrieval" else "Fixed reviewed examples"
        )
        examples_block = f"\n[{example_heading}]\n{format_icl_block(examples)}\n"
    prompt = f"""{TASK_RULES}
{examples_block}

[Verified document metadata]
{metadata}

[Input sentence]
{sentence}
{repair}"""
    return f"{prompt}{dictionary_provider().prompt_block(sentence) if dictionary_hint_block is None else dictionary_hint_block}"


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    metadata = pd.read_csv(path)
    required = {"PMID", "Species", "Tissue"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata file missing columns: {', '.join(sorted(missing))}")
    result: dict[str, dict[str, str]] = {}
    for row in metadata.itertuples(index=False):
        pmid = _pmid(row.PMID)
        values = {"species": _text(row.Species), "tissue": _text(row.Tissue)}
        if pmid in result and result[pmid] != values:
            raise ValueError(f"Conflicting metadata rows for PMID {pmid}.")
        result[pmid] = values
    return result


def relation_records(
    row: pd.Series,
    relations: list[dict[str, str]],
    metadata: dict[str, str],
    model: str,
    icl_mode: str,
    icl_example_ids: list[str],
    icl_scores: list[float],
    run_signature: str,
) -> list[dict[str, object]]:
    records = []
    for relation in relations:
        polarity = relation["polarity"]
        marker = relation["marker_mention"]
        broad = relation["broad_cell_mention"]
        cell = relation["cell_mention"]
        records.append(
            {
                "row_id": int(row["row_id"]),
                "PMID": _pmid(row["PMID"]),
                "Species": metadata.get("species", ""),
                "Tissue": metadata.get("tissue", ""),
                "Cell_type": broad,
                "Cell type (subtype)": cell,
                "Cell marker": marker if polarity == "positive" else "",
                "Negative marker": marker if polarity == "negative" else "",
                "Polarity": polarity,
                "Confidence": relation["confidence"],
                "Cell mention": cell,
                "Broad cell mention": broad,
                "Marker mention": marker,
                "Relation evidence": relation["relation_evidence"],
                "Evidence": _text(row["sentence"]),
                "icl_mode": icl_mode,
                "icl_example_ids": "|".join(icl_example_ids),
                "icl_retrieval_scores": "|".join(f"{score:.6f}" for score in icl_scores),
                "run_signature": run_signature,
                "prompt_version": PROMPT_VERSION,
                "model": model,
            }
        )
    return records


def call_model(client: Any, model: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("model returned an empty response")
    return json.loads(content)


def process_row(
    row: pd.Series,
    client: Any,
    model: str,
    examples: list[dict[str, Any]],
    icl_mode: str,
    icl_scores: list[float],
    run_signature: str,
    metadata: dict[str, str],
    max_attempts: int,
    dictionary_hint_block: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    errors: list[str] = []
    raw_response = ""
    auto_repair_count = 0
    auto_filter_count = 0
    auto_scope_repair_count = 0
    sentence = _text(row["sentence"])
    icl_example_ids = [str(example["example_id"]) for example in examples]
    for attempt in range(1, max_attempts + 1):
        try:
            prompt = build_user_prompt(
                sentence,
                examples,
                icl_mode=icl_mode,
                species=metadata.get("species", ""),
                tissue=metadata.get("tissue", ""),
                validation_errors=errors or None,
                dictionary_hint_block=dictionary_hint_block,
            )
            payload = call_model(client, model, prompt)
            raw_response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if VALIDATION_MODE == "strict":
                payload, mention_repair_count = repair_mention_formatting(payload, sentence)
                payload, auto_scope_repair_count = repair_explicit_exception_scope(
                    payload, sentence
                )
                payload, auto_filter_count = filter_invalid_entity_types(payload, sentence)
                payload, evidence_repair_count = repair_relation_evidence(payload, sentence)
                auto_repair_count = mention_repair_count + evidence_repair_count
                validation = validate_response(payload, sentence)
            elif VALIDATION_MODE == "structure_only":
                validation = validate_structure(payload)
            else:
                raise ValueError(f"Unknown VALIDATION_MODE: {VALIDATION_MODE}")
            if validation.errors:
                errors = validation.errors
                if attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                status = "VALIDATION_ERROR"
                relations: list[dict[str, object]] = []
            else:
                relations = relation_records(
                    row,
                    validation.relations,
                    metadata,
                    model,
                    icl_mode,
                    icl_example_ids,
                    icl_scores,
                    run_signature,
                )
                status = "SUCCESS" if relations else "NO_ASSOCIATION"
            return relations, {
                "row_id": int(row["row_id"]),
                "PMID": _pmid(row["PMID"]),
                "status": status,
                "relation_count": len(relations),
                "attempts": attempt,
                "error": "" if status in {"SUCCESS", "NO_ASSOCIATION"} else "; ".join(errors),
                "raw_response": raw_response,
                "auto_repair_count": auto_repair_count,
                "auto_filter_count": auto_filter_count,
                "auto_scope_repair_count": auto_scope_repair_count,
                "Evidence": sentence,
                "icl_mode": icl_mode,
                "icl_example_ids": "|".join(icl_example_ids),
                "icl_retrieval_scores": "|".join(f"{score:.6f}" for score in icl_scores),
                "run_signature": run_signature,
                "prompt_version": PROMPT_VERSION,
                "model": model,
            }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            errors = [f"PARSE_ERROR: {exc}"]
        except Exception as exc:  # API and transport exceptions are retried uniformly.
            errors = [f"API_ERROR: {type(exc).__name__}: {exc}"]
        if attempt < max_attempts:
            time.sleep(2 ** (attempt - 1))

    final_status = "PARSE_ERROR" if errors and errors[0].startswith("PARSE_ERROR") else "API_ERROR"
    return [], {
        "row_id": int(row["row_id"]),
        "PMID": _pmid(row["PMID"]),
        "status": final_status,
        "relation_count": 0,
        "attempts": max_attempts,
        "error": "; ".join(errors),
        "raw_response": raw_response,
        "auto_repair_count": auto_repair_count,
        "auto_filter_count": auto_filter_count,
        "auto_scope_repair_count": auto_scope_repair_count,
        "Evidence": sentence,
        "icl_mode": icl_mode,
        "icl_example_ids": "|".join(icl_example_ids),
        "icl_retrieval_scores": "|".join(f"{score:.6f}" for score in icl_scores),
        "run_signature": run_signature,
        "prompt_version": PROMPT_VERSION,
        "model": model,
    }


def append_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        if rows:
            writer.writerows(rows)
        handle.flush()


def completed_row_ids(status_path: Path, run_signature: str | None = None) -> set[int]:
    if not status_path.exists() or status_path.stat().st_size == 0:
        return set()
    status = pd.read_csv(status_path)
    if run_signature is not None:
        if "run_signature" not in status.columns:
            raise ValueError(
                "Existing status file has no run_signature; use --overwrite or another output path."
            )
        signatures = set(status["run_signature"].dropna().astype(str))
        if signatures and signatures != {run_signature}:
            raise ValueError(
                "Existing status file belongs to a different model/ICL configuration; "
                "use --overwrite or another output path."
            )
    status = status[
        status["status"].isin({"SUCCESS", "NO_ASSOCIATION", "SKIPPED_ICL_EXAMPLE"})
    ]
    return set(pd.to_numeric(status["row_id"], errors="coerce").dropna().astype(int))


def prepare_input(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"row_id", "PMID", "sentence"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {', '.join(sorted(missing))}")
    if "pseudo_label" in data.columns:
        data = data[data["pseudo_label"].astype(str).str.upper().eq("POS")].copy()
    if data["row_id"].duplicated().any():
        raise ValueError("Input row_id values must be unique.")
    return data.reset_index(drop=True)


def select_icl_examples(
    examples: list[dict[str, Any]],
    retriever: IclRetriever | None,
    sentence: str,
    mode: str,
    top_k: int,
) -> RetrievalResult:
    if mode == "none":
        return RetrievalResult(examples=[], scores=[])
    if mode == "all":
        return RetrievalResult(examples=examples, scores=[])
    if retriever is None:
        raise ValueError("retriever is required when icl_mode is retrieval")
    return retriever.retrieve(sentence, top_k=top_k)


def build_run_signature(
    examples: list[dict[str, Any]], model: str, mode: str, top_k: int
) -> str:
    effective_top_k = 0 if mode == "none" else (len(examples) if mode == "all" else min(top_k, len(examples)))
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "icl_mode": mode,
        "icl_top_k": effective_top_k,
        "examples": examples if mode != "none" else [],
        "system_prompt": SYSTEM_PROMPT,
        "task_rules": TASK_RULES,
        "response_schema": RESPONSE_SCHEMA,
        "retrieval_method": IclRetriever.METHOD if mode == "retrieval" else mode,
        "dictionary": "mgb+local",
        "dictionary_max_hints": 12,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone, validated Prompt3 v2.6 zero-shot extraction.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--icl", type=Path, default=DEFAULT_ICL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--work-items", type=Path, help="Controller-prepared input with metadata and dictionary_hint_block.")
    parser.add_argument("--relations-output", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--icl-mode",
        choices=["retrieval", "all", "none"],
        default="none",
        help="Omit ICL by default; an explicit --icl file may be used for retrieval or all modes.",
    )
    parser.add_argument(
        "--icl-top-k",
        type=int,
        default=4,
        help="Number of examples selected in retrieval mode.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate input, metadata, and all ICL annotations without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.icl_top_k < 1:
        raise ValueError("--workers, --max-attempts, and --icl-top-k must be positive integers.")

    examples = [] if args.icl_mode == "none" else load_icl_examples(args.icl)
    retriever = IclRetriever(examples) if args.icl_mode == "retrieval" else None
    run_signature = build_run_signature(
        examples, args.model, args.icl_mode, args.icl_top_k
    )
    data = prepare_input(args.work_items or args.input)
    metadata_map = load_metadata(args.metadata)
    icl_sentence_keys = (
        {sentence_identity(example["sentence"]) for example in examples}
        if args.icl_mode != "none"
        else set()
    )
    input_sentence_keys = data["sentence"].astype(str).map(sentence_identity)

    leaked_mask = input_sentence_keys.map(
        lambda key: any(
            sentence_identities_overlap(key, icl_key) for icl_key in icl_sentence_keys
        )
    )
    overlapping_rows = leaked_mask.sum()
    if args.validate_only:
        print(f"Input rows:             {len(data):,}")
        print(f"Validated ICL examples: {len(examples):,}")
        print(f"Metadata PMIDs:         {len(metadata_map):,}")
        print(f"ICL/input overlaps:     {overlapping_rows:,} (these rows will be skipped)")
        print(f"ICL mode:               {args.icl_mode}")
        print(f"Run signature:          {run_signature}")
        if retriever is not None:
            from collections import Counter

            usage: Counter[str] = Counter()
            for sentence in data["sentence"].astype(str).head(200):
                usage.update(retriever.retrieve(sentence, args.icl_top_k).example_ids)
            print(f"Retrieval top-k:        {min(args.icl_top_k, len(examples))}")
            print(
                "Top selected examples:  "
                + ", ".join(f"{example_id}={count}" for example_id, count in usage.most_common(5))
            )
        elif args.icl_mode == "none":
            print("Retrieval top-k:        0 (ICL omitted)")
        print("Validation passed; no API calls were made.")
        return

    if args.overwrite:
        args.relations_output.unlink(missing_ok=True)
        args.status_output.unlink(missing_ok=True)
    done = completed_row_ids(args.status_output, run_signature)
    data = data[~data["row_id"].astype(int).isin(done)].copy()

    leaked = data[leaked_mask]
    for _, row in leaked.iterrows():
        append_csv(
            args.status_output,
            [{
                "row_id": int(row["row_id"]),
                "PMID": _pmid(row["PMID"]),
                "status": "SKIPPED_ICL_EXAMPLE",
                "relation_count": 0,
                "attempts": 0,
                "error": "Input sentence is present in the fixed ICL file.",
                "raw_response": "",
                "auto_repair_count": 0,
                "auto_filter_count": 0,
                "auto_scope_repair_count": 0,
                "Evidence": _text(row["sentence"]),
                "icl_mode": args.icl_mode,
                "icl_example_ids": "",
                "icl_retrieval_scores": "",
                "run_signature": run_signature,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
            }],
            STATUS_COLUMNS,
        )
    data = data[~leaked_mask].copy()
    if args.limit is not None:
        data = data.head(args.limit)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VITE_OPENAI_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "VITE_OPENAI_API_KEY"
            )
        except ImportError:
            pass
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or VITE_OPENAI_API_KEY before extraction.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package in the extraction environment.") from exc
    client = OpenAI(api_key=api_key)

    print(
        f"Prompt3 v2.6: {len(data):,} pending rows, {len(done):,} already completed, "
        f"{len(examples)} ICL pool examples, mode={args.icl_mode}, top_k={args.icl_top_k}"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for _, row in data.iterrows():
            pmid_metadata = {
                "species": _text(row.get("controller_species", "")),
                "tissue": _text(row.get("controller_tissue", "")),
            } if args.work_items else metadata_map.get(_pmid(row["PMID"]), {})
            selection = select_icl_examples(
                examples,
                retriever,
                _text(row["sentence"]),
                args.icl_mode,
                args.icl_top_k,
            )
            future = executor.submit(
                process_row,
                row,
                client,
                args.model,
                selection.examples,
                args.icl_mode,
                selection.scores,
                run_signature,
                pmid_metadata,
                args.max_attempts,
                _text(row.get("dictionary_hint_block", "")),
            )
            futures[future] = int(row["row_id"])

        for count, future in enumerate(as_completed(futures), 1):
            relations, status = future.result()
            append_csv(args.relations_output, relations, RELATION_COLUMNS)
            append_csv(args.status_output, [status], STATUS_COLUMNS)
            if count % 25 == 0 or count == len(futures):
                print(f"[{count:,}/{len(futures):,}] rows completed")

    print(f"Relations: {args.relations_output}")
    print(f"Status:    {args.status_output}")


if __name__ == "__main__":
    main()
