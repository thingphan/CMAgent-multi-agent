from __future__ import annotations

import argparse
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/full/extraction/relations.csv"
DEFAULT_OUTPUT = ROOT / "results/full/mapped/ontology_mapped.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("CMAGENT_RESOURCES_DIR", ROOT / "resources")) / "local_entity_cache_v3"

SPECIES = {
    "human": {
        "label": "Human",
        "taxon_id": "NCBITaxon:9606",
        "taxid": 9606,
    },
    "mouse": {
        "label": "Mouse",
        "taxon_id": "NCBITaxon:10090",
        "taxid": 10090,
    },
}


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json"})
    return session


SESSION = build_session()


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def ontology_match_term(value: object) -> str:
    text = normalize_term(value)
    text = re.sub(r"\bcells\b", "cell", text)
    text = re.sub(r"\b([a-z]+)blasts\b", r"\1blast", text)
    plurals = {
        "macrophages": "macrophage",
        "monocytes": "monocyte",
        "lymphocytes": "lymphocyte",
        "neutrophils": "neutrophil",
        "fibroblasts": "fibroblast",
        "astrocytes": "astrocyte",
        "myocytes": "myocyte",
        "adipocytes": "adipocyte",
        "osteocytes": "osteocyte",
        "chondrocytes": "chondrocyte",
        "erythrocytes": "erythrocyte",
        "megakaryocytes": "megakaryocyte",
        "hepatocytes": "hepatocyte",
        "keratinocytes": "keratinocyte",
        "melanocytes": "melanocyte",
        "pericytes": "pericyte",
        "pneumocytes": "pneumocyte",
        "cardiomyocytes": "cardiomyocyte",
    }
    text = " ".join(plurals.get(token, token) for token in text.split())
    return text


def normalize_cell_mention(value: object) -> str:
    """Normalize spelling and exact abbreviations without dropping subtype detail."""
    term = ontology_match_term(value)
    term = term.translate(str.maketrans({"−": "-", "–": "-", "‐": "-"}))
    term = re.sub(r"(?<=\+)(?=[tb]\s*cell\b)", " ", term)
    term = re.sub(r"\b([tb])[- ]cell\b", r"\1 cell", term)
    term = re.sub(r"([αβ])[- ]?cell\b", lambda m: {"α": "alpha", "β": "beta"}[m[1]] + " cell", term)
    term = re.sub(r"\bγ\s*[/\-]?\s*δ\s*t\s*cell\b", "gamma-delta t cell", term)
    term = re.sub(r"\bgamma\s*[/ ]\s*delta\s+t\s*cell\b", "gamma-delta t cell", term)
    term = re.sub(r"\bmicroglia\b", "microglial cell", term)
    # A population is a collection of the stated cell type, not a new subtype.
    term = re.sub(r"(?<=cell) populations?$", "", term)
    aliases = {
        "treg": "regulatory t cell", "tregs": "regulatory t cell",
        "regulatory t cell (treg)": "regulatory t cell",
        "regulatory t cell (tregs)": "regulatory t cell",
        "pdcs": "pdc",
        "natural killer (nk) cell": "natural killer cell",
    }
    if term in aliases:
        return aliases[term]
    return re.sub(r"\btregs?\b", "regulatory t cell", term)


_CELL_HEADS = (
    "plasmacytoid dendritic cell", "natural killer t cell", "natural killer cell",
    "gamma-delta t cell", "dendritic cell", "endothelial cell", "epithelial cell",
    "smooth muscle cell", "microglial cell", "t cell", "b cell", "macrophage",
    "neutrophil", "monocyte", "fibroblast", "myofibroblast", "astrocyte",
    "oligodendrocyte", "pericyte", "neuron",
)


def broad_cell_head(value: object) -> str:
    """Extract a written cell head for broad mapping only; retain all subtype text."""
    term = normalize_cell_mention(value)
    for head in _CELL_HEADS:
        if term != head and not term.endswith(" " + head):
            continue
        prefix = term[:-len(head)].strip()
        # Do not infer a type from negated or compound cell mentions.
        if re.search(r"(?:^|\s)(?:non[-\s]|not\b|without\b|absence\b|depletion\b|devoid\b|no\b|and\b|or\b|versus\b|vs\b)", prefix):
            return ""
        if "/" in prefix or ";" in prefix:
            return ""
        return head
    return ""


def normalize_species(value: object) -> dict[str, Any] | None:
    species = normalize_term(value)
    if species == "human":
        return SPECIES["human"]
    if species == "mouse":
        return SPECIES["mouse"]
    return None


class LocalMappingCache:
    """Exact, prevalidated entity mappings stored in project-local CSV files."""

    def __init__(
        self,
        cell_terms: dict[str, tuple[str, str, str]],
        gene_exact_terms: dict[tuple[str, str], tuple[str, str, str]],
        gene_terms: dict[tuple[str, str], tuple[str, str, str]],
        tissue_terms: dict[str, tuple[str, str, str]],
    ) -> None:
        self.cell_terms = cell_terms
        self.gene_exact_terms = gene_exact_terms
        self.gene_terms = gene_terms
        self.tissue_terms = tissue_terms

    @classmethod
    def from_dir(cls, directory: Path) -> "LocalMappingCache":
        required = {
            "cell_terms.csv",
            "gene_exact_terms.csv",
            "gene_terms.csv",
            "tissue_terms.csv",
        }
        missing = sorted(name for name in required if not (directory / name).exists())
        if missing:
            raise FileNotFoundError(
                f"Local mapping cache is incomplete in {directory}: {', '.join(missing)}"
            )

        cells = pd.read_csv(directory / "cell_terms.csv", dtype=str).fillna("")
        exact_genes = pd.read_csv(directory / "gene_exact_terms.csv", dtype=str).fillna("")
        genes = pd.read_csv(directory / "gene_terms.csv", dtype=str).fillna("")
        tissues = pd.read_csv(directory / "tissue_terms.csv", dtype=str).fillna("")

        source_rank = {"cell_ontology_label": 0, "cell_ontology_exact_synonym": 1,
                       "cell_ontology_synonym": 2, "cellmarker_cell_name": 3,
                       "cellmarker_cell_class": 4}
        cells["lookup_key"] = cells["normalized_term"].map(normalize_cell_mention)
        cells["source_rank"] = cells["mapping_source"].map(source_rank).fillna(9)
        cell_terms = {}
        for key, group in cells.groupby("lookup_key", sort=False):
            best = group[group["source_rank"].eq(group["source_rank"].min())]
            if key and best["identifier"].nunique() == 1:
                row = best.iloc[0]
                cell_terms[key] = (row.identifier, row.canonical_term, row.mapping_source)
        gene_exact_terms = {
            (row.species, row.exact_term): (
                row.identifier,
                row.canonical_term,
                row.mapping_source,
            )
            for row in exact_genes.itertuples(index=False)
        }
        gene_terms = {
            (row.species, row.normalized_term): (
                row.identifier,
                row.canonical_term,
                row.mapping_source,
            )
            for row in genes.itertuples(index=False)
        }
        tissue_terms = {
            row.normalized_term: (row.identifier, row.canonical_term, row.mapping_source)
            for row in tissues.itertuples(index=False)
        }
        return cls(cell_terms, gene_exact_terms, gene_terms, tissue_terms)

    def _cell_lookup(self, mention: str, tissue: str = "", evidence: str = "") -> tuple[str, str, str] | None:
        key = normalize_cell_mention(mention)
        # Alpha/beta are reused across organs. A unique ontology synonym alone
        # must not force a pancreatic mention onto a retinal/pituitary ID.
        if key in {"alpha cell", "beta cell"}:
            context = normalize_term(tissue + " " + evidence)
            if re.search(r"\b(?:pancrea\w*|islet\w*)\b", context):
                target = "pancreatic alpha cell" if key == "alpha cell" else "pancreatic beta cell"
                result = self.cell_terms.get(target)
                if result:
                    return result[0], result[1], "cell_ontology_context_pancreas"
            return None
        return self.cell_terms.get(key)

    def cell_id(self, cell_type: str, subtype: str, tissue: str = "", evidence: str = "") -> tuple[str, str, str]:
        result = self._cell_lookup(subtype, tissue, evidence)
        if result:
            identifier, term, source = result
            return identifier, term, f"local_{source}"
        # Broad cell normalization is independent. Do not hide a failed
        # subtype normalization by writing the broad ID here.
        return "", "", "unmapped_specific"

    def broad_cell_id(self, cell_type: str, tissue: str = "", evidence: str = "", subtype: str = "") -> tuple[str, str, str]:
        """Map the written broad-cell mention without using the subtype fallback."""
        result = self._cell_lookup(cell_type, tissue, evidence)
        if result and result[0] != "CL_0000000":
            identifier, term, source = result
            return identifier, term, f"local_{source}"
        exact_specific = self._cell_lookup(subtype, tissue, evidence) if subtype else None
        # Do not amplify a CellMarker-only name-to-ID assignment into a second
        # field without ontology support: some source labels have conflicting IDs.
        if (exact_specific and exact_specific[0] != "CL_0000000"
                and exact_specific[2].startswith("cell_ontology_")):
            return exact_specific[0], exact_specific[1], "local_broad_from_exact_specific"
        head = broad_cell_head(subtype)
        inferred = self._cell_lookup(head, tissue, evidence) if head else None
        if inferred:
            identifier, term, _ = inferred
            return identifier, term, "local_broad_head_from_specific"
        if result:
            identifier, term, source = result
            return identifier, term, f"local_{source}"
        return "", "", "unmapped"

    def gene_info(self, marker: str, species: str) -> tuple[str, str, str]:
        exact_key = (species, str(marker).strip())
        if exact_key in self.gene_exact_terms:
            identifier, symbol, source = self.gene_exact_terms[exact_key]
            return symbol, identifier, f"local_{source}_exact"
        key = (species, normalize_term(marker))
        if key not in self.gene_terms:
            return "", "", "unmapped"
        identifier, symbol, source = self.gene_terms[key]
        return symbol, identifier, f"local_{source}"

    def tissue_id(self, tissue: str) -> tuple[str, str, str]:
        key = ontology_match_term(tissue)
        if key not in self.tissue_terms:
            return "", "", "unmapped"
        identifier, term, source = self.tissue_terms[key]
        return identifier, term, f"local_{source}"


def flatten_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@lru_cache(maxsize=None)
def get_ontology_id(term: str, ontology: str, prefix: str) -> str:
    target = normalize_term(term)
    if not target:
        return ""

    query_term = ontology_match_term(target)

    params = {
        "q": query_term,
        "ontology": ontology,
        "queryFields": "label,synonym",
        "fieldList": "label,synonym,short_form,obo_id,ontology_name,is_obsolete,type",
        "exact": "true",
        "local": "true",
        "type": "class",
        "rows": 50,
    }

    try:
        response = SESSION.get("https://www.ebi.ac.uk/ols4/api/search", params=params, timeout=15)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
    except (requests.RequestException, ValueError):
        return ""

    label_ids = set()
    synonym_ids = set()
    for doc in docs:
        short_form = str(doc.get("short_form", ""))
        ontology_name = normalize_term(doc.get("ontology_name", ""))
        is_obsolete = doc.get("is_obsolete", False)

        if ontology_name != ontology:
            continue
        if not short_form.startswith(prefix):
            continue
        if is_obsolete in (True, "true", "True"):
            continue
        if ontology_match_term(doc.get("label", "")) == query_term:
            label_ids.add(short_form)
            continue

        synonyms = flatten_strings(doc.get("synonym"))
        synonyms.extend(flatten_strings(doc.get("synonyms")))
        if query_term in {ontology_match_term(value) for value in synonyms}:
            synonym_ids.add(short_form)

    if len(label_ids) == 1:
        return next(iter(label_ids))
    if len(label_ids) > 1:
        return ""
    return next(iter(synonym_ids)) if len(synonym_ids) == 1 else ""


@lru_cache(maxsize=None)
def get_tissue_id(tissue: str) -> str:
    return get_ontology_id(tissue, "uberon", "UBERON_")


@lru_cache(maxsize=None)
def get_cell_id(cell_type: str, subtype: str) -> tuple[str, str, str]:
    subtype = normalize_term(subtype)

    if subtype:
        subtype_id = get_ontology_id(subtype, "cl", "CL_")
        if subtype_id:
            return subtype_id, subtype, "subtype_exact"
    return "", "", "unmapped_specific"


@lru_cache(maxsize=None)
def get_broad_cell_id(cell_type: str) -> tuple[str, str, str]:
    """Map only the broad mention, independently of the specific cell mention."""
    cell_type = normalize_term(cell_type)
    if not cell_type:
        return "", "", "unmapped"
    cell_type_id = get_ontology_id(cell_type, "cl", "CL_")
    return (
        (cell_type_id, cell_type, "broad_cell_exact")
        if cell_type_id
        else ("", "", "unmapped")
    )


def aliases_from_hit(hit: dict[str, Any]) -> list[str]:
    return flatten_strings(hit.get("alias"))


@lru_cache(maxsize=None)
def get_gene_info(marker: str, taxid: int) -> tuple[str, str, str]:
    marker = str(marker).strip()
    target = normalize_term(marker)
    if not target or taxid not in (9606, 10090):
        return "", "", "unmapped"

    queries = (
        (f'symbol:"{marker}"', "symbol_exact"),
        (f'alias:"{marker}"', "alias_exact"),
    )

    for query, source in queries:
        params = {
            "q": query,
            "species": taxid,
            "fields": "symbol,alias,entrezgene,taxid",
            "size": 20,
        }
        try:
            response = SESSION.get("https://mygene.info/v3/query", params=params, timeout=15)
            response.raise_for_status()
            hits = response.json().get("hits", [])
        except (requests.RequestException, ValueError):
            return "", "", "unmapped"

        valid: dict[str, tuple[str, str]] = {}
        for hit in hits:
            try:
                hit_taxid = int(hit.get("taxid"))
            except (TypeError, ValueError):
                continue
            if hit_taxid != taxid:
                continue

            symbol = str(hit.get("symbol", "")).strip()
            gene_id = str(hit.get("entrezgene", "")).strip()
            if not symbol or not gene_id:
                continue

            if source == "symbol_exact":
                matched = normalize_term(symbol) == target
            else:
                matched = target in {normalize_term(alias) for alias in aliases_from_hit(hit)}
            if matched:
                valid[gene_id] = (symbol, gene_id)

        if len(valid) == 1:
            symbol, gene_id = next(iter(valid.values()))
            return symbol, gene_id, source
        if len(valid) > 1:
            return "", "", "ambiguous"

    return "", "", "unmapped"


def split_markers(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [marker.strip() for marker in str(value).split(",") if marker.strip()]


def prefetch_mappings(df: pd.DataFrame, workers: int) -> None:
    tissues = sorted({str(value) for value in df["Tissue"].dropna() if str(value).strip()})
    cell_pairs = sorted(
        {
            (
                "" if pd.isna(row.Cell_type) else str(row.Cell_type),
                "" if pd.isna(row.Subtype) else str(row.Subtype),
            )
            for row in df[["Cell_type", "Subtype"]].itertuples(index=False)
        }
    )
    gene_pairs = sorted(
        {
            (str(row.Positive_Marker), int(row.Species_Info["taxid"]))
            for row in df[["Positive_Marker", "Species_Info"]].itertuples(index=False)
        }
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(get_tissue_id, tissues))
        list(executor.map(lambda pair: get_cell_id(*pair), cell_pairs))
        list(executor.map(lambda pair: get_gene_info(*pair), gene_pairs))


def map_dataframe(
    df: pd.DataFrame, workers: int = 8, local_cache: LocalMappingCache | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "Species",
        "Tissue",
        "Cell_type",
        "Cell type (subtype)",
        "PMID",
        "Evidence",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    stats = {"input_rows": len(df)}
    df = df.copy()
    df = df.rename(columns={"Cell type (subtype)": "Subtype"})
    # Preserve decision provenance through the mapping boundary.  Space-containing
    # relation columns are stabilized before itertuples is used below.
    for source, target in {
        "verification_route": "Verification_Route",
        "reason_code": "Verifier_Reason_Code",
        "prompt_version": "Prompt_Version",
        "model": "Extraction_Model",
    }.items():
        df[target] = df[source] if source in df else ""
    # Prompt3 stores a marker for every polarity in Marker mention.  Prefer it
    # so negative and uncertain relations survive mapping; retain compatibility
    # with older CSVs that only populated Cell marker/Negative marker.
    if "Marker mention" in df.columns:
        df["Extracted_Marker"] = df["Marker mention"]
    else:
        positive = df.get("Cell marker", pd.Series("", index=df.index)).fillna("")
        negative = df.get("Negative marker", pd.Series("", index=df.index)).fillna("")
        df["Extracted_Marker"] = positive.where(positive.astype(str).str.strip().ne(""), negative)
    df["Species_Info"] = df["Species"].apply(normalize_species)
    stats["unsupported_species_rows"] = int(df["Species_Info"].isna().sum())
    df = df[df["Species_Info"].notna()].copy()

    df["Extracted_Marker"] = df["Extracted_Marker"].apply(split_markers)
    df = df.explode("Extracted_Marker", ignore_index=True)
    df = df[
        df["Extracted_Marker"].notna()
        & df["Extracted_Marker"].astype(str).str.strip().ne("")
    ]
    stats["marker_rows"] = len(df)

    if local_cache is None:
        prefetch_mappings(df, workers)

    mapped_rows = []
    for row in df.itertuples(index=False):
        species_info = row.Species_Info
        species = species_info["label"]
        tissue = row.Tissue
        cell_type = row.Cell_type
        subtype = row.Subtype
        marker = row.Extracted_Marker

        if local_cache is None:
            tissue_id = get_tissue_id(str(tissue)) if pd.notna(tissue) else ""
            cell_id, cell_mapping_term, cell_mapping_source = get_cell_id(cell_type, subtype)
            broad_cell_id, broad_cell_mapping_term, broad_cell_mapping_source = get_broad_cell_id(cell_type)
            gene_symbol, gene_id, gene_mapping_source = get_gene_info(
                str(marker), species_info["taxid"]
            )
        else:
            tissue_id, _, _ = local_cache.tissue_id(str(tissue)) if pd.notna(tissue) else ("", "", "")
            cell_id, cell_mapping_term, cell_mapping_source = local_cache.cell_id(
                cell_type, subtype, str(tissue), str(row.Evidence)
            )
            broad_cell_id, broad_cell_mapping_term, broad_cell_mapping_source = local_cache.broad_cell_id(
                cell_type, str(tissue), str(row.Evidence), subtype
            )
            gene_symbol, gene_id, gene_mapping_source = local_cache.gene_info(
                str(marker), species
            )

        mapped_rows.append(
            {
                "row_id": getattr(row, "row_id", ""),
                "Species": species,
                "NCBITaxon_ID": species_info["taxon_id"],
                "Tissue": tissue,
                "UBERON_ID": tissue_id,
                "Cell_type": cell_type,
                "Subtype": subtype,
                "Cell_Mapping_Term": cell_mapping_term,
                "Cell_Mapping_Source": cell_mapping_source,
                "CellOntology_ID": cell_id,
                "Broad_Cell_Mapping_Term": broad_cell_mapping_term,
                "Broad_Cell_Mapping_Source": broad_cell_mapping_source,
                "Broad_CellOntology_ID": broad_cell_id,
                "Original_Marker": marker,
                "Polarity": getattr(row, "Polarity", ""),
                "Confidence": getattr(row, "Confidence", ""),
                # These are the original extracted spans carried through the
                # mapping contract.  ``itertuples`` renames space-containing
                # source columns, so use their already-normalized equivalents.
                "Cell mention": subtype,
                "Broad cell mention": cell_type,
                "Marker mention": marker,
                "Gene_Mapping_Source": gene_mapping_source,
                "Official_Symbol": gene_symbol,
                "NCBI_GeneID": gene_id,
                "PMID": row.PMID,
                "Evidence": row.Evidence,
                "Verification_Route": row.Verification_Route,
                "Verifier_Reason_Code": row.Verifier_Reason_Code,
                "Prompt_Version": row.Prompt_Version,
                "Extraction_Model": row.Extraction_Model,
            }
        )

    result = pd.DataFrame(mapped_rows)
    stats["output_rows"] = len(result)
    stats["mapped_tissues"] = int(result["UBERON_ID"].ne("").sum()) if len(result) else 0
    stats["mapped_cells"] = int(result["CellOntology_ID"].ne("").sum()) if len(result) else 0
    stats["mapped_broad_cells"] = int(result["Broad_CellOntology_ID"].ne("").sum()) if len(result) else 0
    stats["mapped_genes"] = int(result["NCBI_GeneID"].ne("").sum()) if len(result) else 0
    stats["mapping_backend"] = "local_cache" if local_cache is not None else "online_api"
    return result, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict ontology and gene standardization.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local strict entity cache (defaults to the v3 cache).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    local_cache = LocalMappingCache.from_dir(args.cache_dir) if args.cache_dir else None
    result, stats = map_dataframe(df, workers=args.workers, local_cache=local_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote strict mappings to: {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")


if __name__ == "__main__":
    main()
