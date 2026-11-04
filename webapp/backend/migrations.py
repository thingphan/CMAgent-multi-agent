"""Small forward-only schema migration helper for local and Docker deployments."""
from __future__ import annotations

from sqlalchemy import inspect, text

from .database import Base


ADDITIVE_COLUMNS = {
    "publications": {"abstract": "TEXT"},
    "relation_evidence": {
        "verification_route": "VARCHAR(64)", "verifier_reason_code": "VARCHAR(128)",
        "prompt_version": "VARCHAR(128)", "model": "VARCHAR(128)",
        "cell_mapping_source": "VARCHAR(128)", "gene_mapping_source": "VARCHAR(128)",
    },
    "ingestion_runs": {"data_version": "VARCHAR(64)", "schema_version": "VARCHAR(32)"},
    "marker_relations": {"is_suppressed": "BOOLEAN DEFAULT FALSE", "suppression_reason": "TEXT"},
}


def ensure_schema(engine) -> None:
    """Create new tables and add nullable provenance columns to existing databases."""
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, columns in ADDITIVE_COLUMNS.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
