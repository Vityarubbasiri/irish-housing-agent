"""Tests for housing.ingest."""

from pathlib import Path

import pytest

from housing.ingest import COLUMN_NAMES, load_raw_ppr

HEADER = ",".join(COLUMN_NAMES)
ROWS = [
    '"01/01/2010","5 Braemor Drive, Churchtown, Co.Dublin","Dublin","","€343,000.00","No","No","Second-Hand Dwelling house /Apartment",""',
    '"03/01/2010","1 Bóthar na Trá, Gaillimh","Galway","H91 ABC1","€185,000.00","No","Yes","Teach/Árasán Cónaithe Nua",""',
]


def write_cp1252_csv(path: Path, header: str, rows: list[str]) -> Path:
    """Write a tiny CSV encoded as cp1252, like the real PPR file."""
    path.write_text("\n".join([header, *rows]) + "\n", encoding="cp1252")
    return path


def test_load_raw_ppr_renames_columns(tmp_path: Path) -> None:
    csv = write_cp1252_csv(tmp_path / "ppr.csv", HEADER, ROWS)

    df = load_raw_ppr(csv)

    assert list(df.columns) == list(COLUMN_NAMES.values())
    assert len(df) == 2


def test_load_raw_ppr_keeps_text_and_decodes_cp1252(tmp_path: Path) -> None:
    csv = write_cp1252_csv(tmp_path / "ppr.csv", HEADER, ROWS)

    df = load_raw_ppr(csv)

    # Price stays as raw text (no cleaning yet), € decoded correctly
    assert df.loc[0, "price"] == "€343,000.00"
    assert df.loc[0, "sale_date"] == "01/01/2010"
    # Irish characters decoded correctly
    assert df.loc[1, "address"] == "1 Bóthar na Trá, Gaillimh"
    assert df.loc[1, "description"] == "Teach/Árasán Cónaithe Nua"


def test_load_raw_ppr_rejects_unexpected_headers(tmp_path: Path) -> None:
    bad_header = HEADER.replace("County", "Region")
    csv = write_cp1252_csv(tmp_path / "bad.csv", bad_header, ROWS)

    with pytest.raises(ValueError, match="Unexpected PPR columns"):
        load_raw_ppr(csv)
