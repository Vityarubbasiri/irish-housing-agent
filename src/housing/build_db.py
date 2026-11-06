"""Build the DuckDB database of cleaned PPR sales, and load it back.

Run from the terminal with: python -m housing.build_db
"""

import time
from pathlib import Path

import duckdb
import pandas as pd

from housing.clean import clean_ppr
from housing.ingest import PROJECT_ROOT, load_raw_ppr

DB_PATH = PROJECT_ROOT / "data" / "processed" / "housing.duckdb"
TABLE_NAME = "sales"


def build_database(raw_path: str | Path | None = None, db_path: str | Path | None = None) -> int:
    """Load and clean the raw PPR CSV, save it as table 'sales', and return the row count.

    Defaults to data/raw/PPR-ALL.csv and data/processed/housing.duckdb.
    Creates the folder if needed and replaces the table if it exists.
    """
    db_path = DB_PATH if db_path is None else Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    sales = clean_ppr(load_raw_ppr(raw_path))

    with duckdb.connect(str(db_path)) as con:
        con.register("clean_sales", sales)
        con.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM clean_sales")
        con.unregister("clean_sales")
        (row_count,) = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()

    return row_count


def load_sales(db_path: str | Path | None = None) -> pd.DataFrame:
    """Load the 'sales' table from the DuckDB database (read-only) as a DataFrame."""
    db_path = DB_PATH if db_path is None else Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Run `python -m housing.build_db` first."
        )

    with duckdb.connect(str(db_path), read_only=True) as con:
        relation = con.table(TABLE_NAME)
        sales = relation.df()
        # An all-empty VARCHAR column comes back as object dtype; make it str like the rest
        text_columns = [
            name for name, dtype in zip(relation.columns, relation.dtypes) if dtype == "VARCHAR"
        ]

    return sales.astype({name: "str" for name in text_columns})


if __name__ == "__main__":
    start = time.perf_counter()
    rows = build_database()
    seconds = time.perf_counter() - start
    print(f"Saved {rows:,} rows to table '{TABLE_NAME}' in {DB_PATH} (took {seconds:.1f} s)")
