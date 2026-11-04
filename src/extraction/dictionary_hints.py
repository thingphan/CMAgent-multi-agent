"""Lightweight dictionary evidence for Prompt3 extraction.

Dictionary hits are hints only: they never create relations or IDs by themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
# External lexical resources only; no classifier implementation or model is needed.
RESOURCES = Path(os.environ.get("CMAGENT_RESOURCES_DIR", ROOT / "resources"))
MGB_DIR = RESOURCES / "marker_dictionary"
CACHE_DIR = RESOURCES / "local_entity_cache"


class DictionaryHintProvider:
    def __init__(self, max_per_type: int = 12) -> None:
        self.max_per_type = max_per_type
        self.nlp = spacy.blank("en")
        ruler = self.nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})

        mgb_genes = pd.read_csv(MGB_DIR / "gene_collect.csv", encoding="utf-8-sig")["gene_collect"]
        cache_genes = pd.read_csv(CACHE_DIR / "gene_exact_terms.csv")["exact_term"]
        cache_cells = pd.read_csv(CACHE_DIR / "cell_terms.csv")["canonical_term"]
        mgb_cells = pd.read_table(MGB_DIR / "cell_ontology_cellname.txt", header=None, encoding="utf-8-sig").iloc[:, 1]

        gene_terms = self._terms(mgb_genes) | self._terms(cache_genes)
        cell_terms = self._cell_variants(self._terms(cache_cells) | self._terms(mgb_cells))
        ruler.add_patterns([{"label": "DICT_GENE", "pattern": value} for value in sorted(gene_terms)])
        ruler.add_patterns([{"label": "DICT_CELL", "pattern": value} for value in sorted(cell_terms)])

    @staticmethod
    def _terms(values: pd.Series) -> set[str]:
        return {
            value.strip()
            for value in values.dropna().astype(str)
            if value.strip() and len(value.strip()) > 1
        }

    @staticmethod
    def _cell_variants(terms: set[str]) -> set[str]:
        variants = set(terms)
        for term in terms:
            if term.endswith(" cell"):
                variants.add(f"{term}s")
            elif term.endswith("s") and len(term) > 3:
                variants.add(term[:-1])
        return variants

    def hits(self, sentence: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {"cell": [], "gene": []}
        for entity in self.nlp(sentence).ents:
            key = "cell" if entity.label_ == "DICT_CELL" else "gene"
            if key == "gene" and entity.text.casefold() in STOP_WORDS:
                continue
            if entity.text not in found[key]:
                found[key].append(entity.text)
        return {key: values[: self.max_per_type] for key, values in found.items()}

    def prompt_block(self, sentence: str) -> str:
        hits = self.hits(sentence)
        if not hits["cell"] and not hits["gene"]:
            return ""
        cells = "; ".join(hits["cell"]) or "none"
        genes = "; ".join(hits["gene"]) or "none"
        return (
            "\n[Dictionary span hints]\n"
            f"Possible CELL spans written in the sentence: {cells}\n"
            f"Possible MARKER spans written in the sentence: {genes}\n"
            "These are lexical hints only. They are not relation evidence and may be wrong; "
            "do not output them unless the sentence itself supports a relation.\n"
        )
