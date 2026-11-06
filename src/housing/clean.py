"""Clean the raw PPR data, following the cleaning rules in CLAUDE.md."""

import pandas as pd

# Columns that together identify the "same sale" for near-duplicate checks
NEAR_DUPLICATE_KEY = ["sale_date", "address", "price"]

YES_NO = {"Yes": True, "No": False}

VAT_RATE = 0.135
BULK_PRICE_THRESHOLD = 5_000_000
LOW_PRICE_THRESHOLD = 20_000

# Bulk keywords (rule 6). Singular "apartment"/"apt"/"block", plus "apartments",
# "apts" and "folio", are deliberately excluded: on the real data they mostly
# match ordinary single homes.
BULK_KEYWORDS = r"(?i)\bblocks\b|\bunits\b|(?:&|\band)\s*others\b"
NUMBER_RANGE = r"\b(?P<start>\d+)\s*-\s*(?P<end>\d+)\b"
# A single home at the start of the address, e.g. "Apt 9, ..." or "Apartment no. 7, ..."
SINGLE_HOME_PREFIX = (
    r"(?i)^\s*(?:apt|apartment|flat|unit)\b\.?\s*(?:no\.?\s*|number\s*)?\d+[a-z]?\b(?!\s*-)"
)
MIN_BULK_RANGE_SPAN = 2  # end - start >= 2 means the range covers 3+ numbers


def parse_sale_date(dates: pd.Series) -> pd.Series:
    """Convert dd/mm/yyyy text (Irish, day-first) to datetime. Raises on bad dates."""
    parsed = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    bad = dates[parsed.isna()]
    if not bad.empty:
        raise ValueError(f"Could not parse {len(bad)} sale dates, e.g. {bad.iloc[0]!r}")
    return parsed


def parse_price(prices: pd.Series) -> pd.Series:
    """Convert text like '€343,000.00' to float. Raises on bad prices."""
    stripped = prices.str.replace("€", "", regex=False).str.replace(",", "", regex=False)
    parsed = pd.to_numeric(stripped, errors="coerce").astype(float)
    bad = prices[parsed.isna()]
    if not bad.empty:
        raise ValueError(f"Could not parse {len(bad)} prices, e.g. {bad.iloc[0]!r}")
    return parsed


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows identical in every column, keeping the first (rule 2)."""
    return df.drop_duplicates(keep="first")


def flag_possible_duplicates(df: pd.DataFrame) -> pd.Series:
    """Flag every row sharing date + address + price with another row (rule 3).

    Run after drop_exact_duplicates, so flagged rows differ in some other column.
    """
    return df.duplicated(subset=NEAR_DUPLICATE_KEY, keep=False)


def to_property_type(descriptions: pd.Series) -> pd.Series:
    """Map Description of Property to 'New' / 'Second-Hand', incl. Irish versions (rule 4).

    Anything containing 'Nua' is New (covers the broken-encoding row).
    Raises on descriptions that match neither.
    """
    is_new = descriptions.str.startswith("New") | descriptions.str.contains("Nua", regex=False)
    is_second_hand = descriptions.str.startswith("Second-Hand") | descriptions.str.contains(
        "Atháimhe", regex=False
    )

    property_type = pd.Series(pd.NA, index=descriptions.index, dtype="object")
    property_type[is_new.fillna(False)] = "New"
    property_type[is_second_hand.fillna(False)] = "Second-Hand"

    unknown = descriptions[property_type.isna()]
    if not unknown.empty:
        raise ValueError(
            f"{len(unknown)} descriptions are neither New nor Second-Hand, "
            f"e.g. {unknown.iloc[0]!r}"
        )
    return property_type.astype("str")


def yes_no_to_bool(values: pd.Series) -> pd.Series:
    """Convert 'Yes' / 'No' text to True / False. Raises on any other value."""
    unexpected = values[~values.isin(YES_NO.keys())]
    if not unexpected.empty:
        raise ValueError(
            f"{len(unexpected)} values are not 'Yes'/'No', e.g. {unexpected.iloc[0]!r}"
        )
    return values.map(YES_NO).astype(bool)


def flag_new_no_vat(property_type: pd.Series, vat_exclusive: pd.Series) -> pd.Series:
    """Flag New properties whose price is not VAT-exclusive (rule 5)."""
    return (property_type == "New") & ~vat_exclusive


def add_vat(price: pd.Series, vat_exclusive: pd.Series) -> pd.Series:
    """Return price including VAT: add 13.5% where vat_exclusive, else unchanged (rule 5)."""
    return price.where(~vat_exclusive, price * (1 + VAT_RATE)).round(2)


def has_bulk_unit_range(addresses: pd.Series) -> pd.Series:
    """True where the address has a number range covering 3+ numbers (e.g. '1-150').

    Ignored when the address starts with a single 'Apt/Apartment/Flat/Unit N',
    since that is one home in a building with a ranged street number.
    """
    ranges = addresses.str.extractall(NUMBER_RANGE).astype(int)
    is_wide = (ranges["end"] - ranges["start"]) >= MIN_BULK_RANGE_SPAN
    has_wide_range = is_wide.groupby(level=0).any().reindex(addresses.index, fill_value=False)
    is_single_home = addresses.str.contains(SINGLE_HOME_PREFIX, regex=True, na=False)
    return has_wide_range & ~is_single_home


def flag_bulk_suspect(price: pd.Series, addresses: pd.Series) -> pd.Series:
    """Flag likely bulk sales: price > €5m or a bulk pattern in the address (rule 6).

    clean_ppr passes price_incl_vat as the price.
    """
    has_keyword = addresses.str.contains(BULK_KEYWORDS, regex=True, na=False)
    return (price > BULK_PRICE_THRESHOLD) | has_keyword | has_bulk_unit_range(addresses)


def flag_low_price_suspect(price: pd.Series) -> pd.Series:
    """Flag prices below €20,000 (rule 7). clean_ppr passes price_incl_vat as the price."""
    return price < LOW_PRICE_THRESHOLD


def flag_model_eligible(df: pd.DataFrame) -> pd.Series:
    """True where none of the exclusion flags are set (rule 9)."""
    excluded = (
        df["not_full_market_price"]
        | df["is_bulk_suspect"]
        | df["is_low_price_suspect"]
        | df["is_new_no_vat"]
    )
    return ~excluded


def clean_ppr(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the DataFrame from load_raw_ppr() (rules 1–9).

    Removes exact duplicates, parses dates and prices, adds property_type,
    converts Yes/No columns to booleans, adds price_incl_vat, and adds the
    is_* flags. Flagged rows are kept. The input DataFrame is not changed.
    """
    df = drop_exact_duplicates(raw.copy())
    df["is_possible_duplicate"] = flag_possible_duplicates(df)

    df["sale_date"] = parse_sale_date(df["sale_date"])
    df["price"] = parse_price(df["price"])
    df["property_type"] = to_property_type(df["description"])
    df["not_full_market_price"] = yes_no_to_bool(df["not_full_market_price"])
    df["vat_exclusive"] = yes_no_to_bool(df["vat_exclusive"])

    df["is_new_no_vat"] = flag_new_no_vat(df["property_type"], df["vat_exclusive"])
    df["price_incl_vat"] = add_vat(df["price"], df["vat_exclusive"])
    df["is_bulk_suspect"] = flag_bulk_suspect(df["price_incl_vat"], df["address"])
    df["is_low_price_suspect"] = flag_low_price_suspect(df["price_incl_vat"])
    df["is_model_eligible"] = flag_model_eligible(df)

    return df.reset_index(drop=True)
