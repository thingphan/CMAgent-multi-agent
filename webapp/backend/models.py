from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Species(Base):
    __tablename__ = "species"
    id: Mapped[int] = mapped_column(primary_key=True)
    ncbi_taxon_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Tissue(Base):
    __tablename__ = "tissues"
    id: Mapped[int] = mapped_column(primary_key=True)
    uberon_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True, index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255))


class Cell(Base):
    __tablename__ = "cells"
    id: Mapped[int] = mapped_column(primary_key=True)
    cell_ontology_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    normalized_name: Mapped[str] = mapped_column(String(512), index=True)
    broad_cell_id: Mapped[int | None] = mapped_column(ForeignKey("cells.id"), nullable=True)
    mapping_source: Mapped[str] = mapped_column(String(128), default="")
    broad_cell: Mapped["Cell | None"] = relationship(remote_side="Cell.id")


class Gene(Base):
    __tablename__ = "genes"
    id: Mapped[int] = mapped_column(primary_key=True)
    ncbi_gene_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    species_id: Mapped[int] = mapped_column(ForeignKey("species.id"), index=True)
    official_symbol: Mapped[str] = mapped_column(String(255), index=True)
    official_name: Mapped[str] = mapped_column(String(1024), default="")
    mapping_source: Mapped[str] = mapped_column(String(128), default="")
    species: Mapped[Species] = relationship()


class Publication(Base):
    __tablename__ = "publications"
    id: Mapped[int] = mapped_column(primary_key=True)
    pmid: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    journal: Mapped[str] = mapped_column(String(512), default="")
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str] = mapped_column(String(512), default="")
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)


class MarkerRelation(Base):
    __tablename__ = "marker_relations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    species_id: Mapped[int] = mapped_column(ForeignKey("species.id"), index=True)
    tissue_id: Mapped[int | None] = mapped_column(ForeignKey("tissues.id"), nullable=True, index=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("cells.id"), index=True)
    gene_id: Mapped[int] = mapped_column(ForeignKey("genes.id"), index=True)
    polarity: Mapped[str] = mapped_column(String(16), index=True)
    __table_args__ = (
        UniqueConstraint("species_id", "tissue_id", "cell_id", "gene_id", "polarity", name="uq_relation_identity"),
        # PostgreSQL considers NULL values distinct in a normal UNIQUE constraint.
        # This expression index keeps unknown-tissue relations idempotent as well.
        Index("uq_relation_identity_nullsafe", "species_id", func.coalesce(tissue_id, -1), "cell_id", "gene_id", "polarity", unique=True),
    )
    best_confidence: Mapped[str] = mapped_column(String(16), default="")
    is_suppressed: Mapped[bool] = mapped_column(default=False, index=True)
    suppression_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    species: Mapped[Species] = relationship()
    tissue: Mapped[Tissue | None] = relationship()
    cell: Mapped[Cell] = relationship()
    gene: Mapped[Gene] = relationship()
    evidence: Mapped[list["RelationEvidence"]] = relationship(back_populates="relation", cascade="all, delete-orphan")


class RelationEvidence(Base):
    __tablename__ = "relation_evidence"
    __table_args__ = (UniqueConstraint("relation_id", "publication_id", "evidence_sentence", "cell_mention", "broad_cell_mention", "marker_mention", name="uq_evidence_source"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    relation_id: Mapped[str] = mapped_column(ForeignKey("marker_relations.id"), index=True)
    publication_id: Mapped[int | None] = mapped_column(ForeignKey("publications.id"), nullable=True, index=True)
    evidence_sentence: Mapped[str] = mapped_column(Text)
    cell_mention: Mapped[str] = mapped_column(String(512), default="")
    broad_cell_mention: Mapped[str] = mapped_column(String(512), default="")
    marker_mention: Mapped[str] = mapped_column(String(512), default="")
    confidence: Mapped[str] = mapped_column(String(16), default="")
    source_run: Mapped[str] = mapped_column(String(255), default="")
    verification_route: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verifier_reason_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cell_mapping_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gene_mapping_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    relation: Mapped[MarkerRelation] = relationship(back_populates="evidence")
    publication: Mapped[Publication | None] = relationship()


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    input_count: Mapped[int] = mapped_column(Integer, default=0)
    inserted_relations: Mapped[int] = mapped_column(Integer, default=0)
    inserted_evidence: Mapped[int] = mapped_column(Integer, default=0)
    skipped_unmapped: Mapped[int] = mapped_column(Integer, default=0)
    data_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CorrectionAuditLog(Base):
    __tablename__ = "correction_audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    relation_id: Mapped[str] = mapped_column(ForeignKey("marker_relations.id"), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    actor: Mapped[str] = mapped_column(String(128), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
