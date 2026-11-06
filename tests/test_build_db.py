"""Tests for housing.build_db."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from housing.build_db import build_database, load_sales
from housing.clean import clean_ppr
from housing.ingest import COLUMN_NAMES, load_raw_ppr

HEADER = ",".join(COLUMN_NAMES)
ROWS = [
    '"01/01/2010","5 Braemor Drive, Churchtown, Co.Dublin","Dublin","","€343,000.00","No","No","Second-Hand Dwelling house /Apartment",""',
    '"01/01/2010","5 Braemor Drive, Churchtown, Co.Dublin","Dublin","","€343,000.00","No","No","Second-Hand Dwelling house /Apartment",""',
    '"03/01/2010","1 Bóthar na Trá, Gaillimh","Galway","H91 ABC1","€185,000.00","No","Yes","Teach/Árasán Cónaithe Nua",""',
]


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """A tiny cp1252 PPR CSV with one exact duplicate row."""
    path = tmp_path / "ppr.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="cp1252")
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A database path inside a folder that doesn't exist yet."""
    return tmp_path / "processed" / "housing.duckdb"


def test_build_database_creates_sales_table(raw_csv: Path, db_path: Path) -> None:
    row_count = build_database(raw_csv, db_path)

    assert db_path.exists()
    assert row_count == 2  # exact duplicate removed
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = [name for (name,) in con.execute("SHOW TABLES").fetchall()]
        (db_rows,) = con.execute("SELECT COUNT(*) FROM sales").fetchone()
    assert tables == ["sales"]
    assert db_rows == 2


def test_build_database_replaces_existing_table(raw_csv: Path, db_path: Path) -> None:
    build_database(raw_csv, db_path)
    row_count = build_database(raw_csv, db_path)

    assert row_count == 2
    assert len(load_sales(db_path)) == 2


def test_load_sales_matches_clean_ppr(raw_csv: Path, db_path: Path) -> None:
    build_database(raw_csv, db_path)

    loaded = load_sales(db_path)
    expected = clean_ppr(load_raw_ppr(raw_csv))

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_sales_missing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m housing.build_db"):
        load_sales(tmp_path / "missing.duckdb")
