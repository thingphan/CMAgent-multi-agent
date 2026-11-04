#!/usr/bin/env python3
"""Idempotently ingest final mapped relations into the relational marker database."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.backend.database import make_engine
from webapp.backend.migrations import ensure_schema
from webapp.backend.models import Cell, Gene, IngestionRun, MarkerRelation, Publication, RelationEvidence, Species, Tissue


REQUIRED = {
    "Species", "NCBITaxon_ID", "Tissue", "UBERON_ID", "CellOntology_ID", "Cell_Mapping_Term",
    "NCBI_GeneID", "Official_Symbol", "Polarity", "PMID", "Evidence", "Cell mention",
    "Broad cell mention", "Marker mention",
}


def value(row: pd.Series, column: str) -> str:
    raw = row.get(column, "")
    return "" if raw is None or pd.isna(raw) else str(raw).strip()


def one(session: Session, model, column, key: str):
    return session.scalar(select(model).where(column == key))


def get_species(session: Session, row: pd.Series) -> Species:
    taxon = value(row, "NCBITaxon_ID")
    species = one(session, Species, Species.ncbi_taxon_id, taxon)
    if species is None:
        label = value(row, "Species")
        species = Species(ncbi_taxon_id=taxon, scientific_name=label, display_name=label)
        session.add(species)
        session.flush()
    return species


def get_tissue(session: Session, row: pd.Series) -> Tissue | None:
    uberon = value(row, "UBERON_ID")
    if not uberon:
        return None
    tissue = one(session, Tissue, Tissue.uberon_id, uberon)
    if tissue is None:
        label = value(row, "Tissue")
        tissue = Tissue(uberon_id=uberon, normalized_name=label, display_name=label)
        session.add(tissue)
        session.flush()
    return tissue


def get_cell(session: Session, row: pd.Series) -> Cell:
    cl_id = value(row, "CellOntology_ID")
    broad_id = value(row, "Broad_CellOntology_ID")
    broad_cell = None
    if broad_id and broad_id != cl_id:
        broad_cell = one(session, Cell, Cell.cell_ontology_id, broad_id)
        if broad_cell is None:
            broad_cell = Cell(
                cell_ontology_id=broad_id,
                normalized_name=value(row, "Broad_Cell_Mapping_Term") or value(row, "Cell_type"),
                mapping_source=value(row, "Broad_Cell_Mapping_Source"),
            )
            session.add(broad_cell)
            session.flush()
    cell = one(session, Cell, Cell.cell_ontology_id, cl_id)
    if cell is None:
        cell = Cell(
            cell_ontology_id=cl_id,
            normalized_name=value(row, "Cell_Mapping_Term") or value(row, "Subtype"),
            mapping_source=value(row, "Cell_Mapping_Source"),
            broad_cell_id=broad_cell.id if broad_cell else None,
        )
        session.add(cell)
        session.flush()
    elif broad_cell is not None and cell.broad_cell_id is None:
        cell.broad_cell_id = broad_cell.id
    return cell


def get_gene(session: Session, row: pd.Series, species: Species) -> Gene:
    gene_id = value(row, "NCBI_GeneID")
    gene = one(session, Gene, Gene.ncbi_gene_id, gene_id)
    if gene is None:
        gene = Gene(
            ncbi_gene_id=gene_id,
            species_id=species.id,
            official_symbol=value(row, "Official_Symbol") or value(row, "Original_Marker"),
            mapping_source=value(row, "Gene_Mapping_Source"),
        )
        session.add(gene)
        session.flush()
    return gene


def get_publication(session: Session, row: pd.Series) -> Publication | None:
    pmid = value(row, "PMID")
    if not pmid:
        return None
    publication = one(session, Publication, Publication.pmid, pmid)
    if publication is None:
        publication = Publication(pmid=pmid)
        session.add(publication)
        session.flush()
    return publication


def get_relation(session: Session, species: Species, tissue: Tissue | None, cell: Cell, gene: Gene, row: pd.Series) -> tuple[MarkerRelation, bool]:
    statement = select(MarkerRelation).where(
        MarkerRelation.species_id == species.id,
        MarkerRelation.cell_id == cell.id,
        MarkerRelation.gene_id == gene.id,
        MarkerRelation.polarity == value(row, "Polarity"),
    )
    statement = statement.where(MarkerRelation.tissue_id.is_(None)) if tissue is None else statement.where(MarkerRelation.tissue_id == tissue.id)
    relation = session.scalar(statement)
    if relation is not None:
        return relation, False
    relation = MarkerRelation(
        species_id=species.id, tissue_id=tissue.id if tissue else None, cell_id=cell.id, gene_id=gene.id,
        polarity=value(row, "Polarity"), best_confidence=value(row, "Confidence"),
    )
    session.add(relation)
    session.flush()
    return relation, True


def ingest(session: Session, frame: pd.DataFrame, run_name: str) -> dict[str, int]:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Mapped input missing columns: {', '.join(sorted(missing))}")
    run = session.scalar(select(IngestionRun).where(IngestionRun.run_name == run_name))
    if run is None:
        run = IngestionRun(run_name=run_name, input_count=len(frame))
        session.add(run)
    else:
        run.status = "running"
    run.data_version = "CMagent-1.0"
    run.schema_version = "2"
    # Replaying a run is safe: relation and evidence lookups are idempotent, while
    # newly appended input rows are still ingested into the same canonical database.
    stats = {"input": len(frame), "relations": 0, "evidence": 0, "skipped": 0}
    for _, row in frame.iterrows():
        if not value(row, "CellOntology_ID") or not value(row, "NCBI_GeneID") or not value(row, "NCBITaxon_ID"):
            stats["skipped"] += 1
            continue
        species = get_species(session, row)
        tissue = get_tissue(session, row)
        cell = get_cell(session, row)
        gene = get_gene(session, row, species)
        relation, created = get_relation(session, species, tissue, cell, gene, row)
        stats["relations"] += int(created)
        publication = get_publication(session, row)
        evidence = session.scalar(select(RelationEvidence).where(
            RelationEvidence.relation_id == relation.id,
            RelationEvidence.publication_id == (publication.id if publication else None),
            RelationEvidence.evidence_sentence == value(row, "Evidence"),
            RelationEvidence.cell_mention == value(row, "Cell mention"),
            RelationEvidence.broad_cell_mention == value(row, "Broad cell mention"),
            RelationEvidence.marker_mention == value(row, "Marker mention"),
        ))
        if evidence is None:
            session.add(RelationEvidence(
                relation_id=relation.id, publication_id=publication.id if publication else None,
                evidence_sentence=value(row, "Evidence"), cell_mention=value(row, "Cell mention"),
                broad_cell_mention=value(row, "Broad cell mention"), marker_mention=value(row, "Marker mention"),
                confidence=value(row, "Confidence"), source_run=run_name,
                verification_route=value(row, "Verification_Route"), verifier_reason_code=value(row, "Verifier_Reason_Code"),
                prompt_version=value(row, "Prompt_Version"), model=value(row, "Extraction_Model"),
                cell_mapping_source=value(row, "Cell_Mapping_Source"), gene_mapping_source=value(row, "Gene_Mapping_Source"),
            ))
            stats["evidence"] += 1
    run.status = "success"
    run.input_count = stats["input"]
    run.inserted_relations = stats["relations"]
    run.inserted_evidence = stats["evidence"]
    run.skipped_unmapped = stats["skipped"]
    run.finished_at = datetime.now(timezone.utc)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--database-url", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype=str).fillna("")
    engine = make_engine(args.database_url)
    ensure_schema(engine)
    with Session(engine) as session:
        try:
            stats = ingest(session, frame, args.run_name)
            session.commit()
        except Exception:
            session.rollback()
            raise
    print("Relational database ingestion: " + ", ".join(f"{key}={value:,}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
