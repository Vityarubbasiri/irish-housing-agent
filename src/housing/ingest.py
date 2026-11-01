"""Load the raw Property Price Register (PPR) data."""

from pathlib import Path

import pandas as pd

# Project root is three levels up: src/housing/ingest.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PPR_PATH = PROJECT_ROOT / "data" / "raw" / "PPR-ALL.csv"

# Original PPR headers -> snake_case names, in file order
COLUMN_NAMES = {
    "Date of Sale (dd/mm/yyyy)": "sale_date",
    "Address": "address",
    "County": "county",
    "Eircode": "eircode",
    "Price (€)": "price",
    "Not Full Market Price": "not_full_market_price",
    "VAT Exclusive": "vat_exclusive",
    "Description of Property": "description",
    "Property Size Description": "size_description",
}


def load_raw_ppr(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw PPR CSV with all columns as text and snake_case column names.

    Defaults to data/raw/PPR-ALL.csv. Raises ValueError if the headers
    don't match the expected PPR columns.
    """
    path = RAW_PPR_PATH if path is None else Path(path)
    df = pd.read_csv(path, encoding="cp1252", low_memory=False, dtype=str)

    if list(df.columns) != list(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected PPR columns in {path}.\n"
            f"Expected: {list(COLUMN_NAMES)}\n"
            f"Got:      {list(df.columns)}"
        )

    return df.rename(columns=COLUMN_NAMES)
