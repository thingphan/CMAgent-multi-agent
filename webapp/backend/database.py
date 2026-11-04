from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def database_url() -> str:
    """Return PostgreSQL in deployment and a local SQLite database for development."""
    return os.environ.get("DATABASE_URL", "sqlite:///data/processed/cell_marker.db")


def make_engine(url: str | None = None):
    return create_engine(url or database_url(), future=True, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
