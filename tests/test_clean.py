"""Tests for housing.clean."""

import pandas as pd
import pytest

from housing.clean import (
    add_vat,
    clean_ppr,
    drop_exact_duplicates,
    flag_bulk_suspect,
    flag_low_price_suspect,
    flag_model_eligible,
    flag_new_no_vat,
    flag_possible_duplicates,
    has_bulk_unit_range,
    parse_price,
    parse_sale_date,
    to_property_type,
    yes_no_to_bool,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a tiny raw-PPR-style DataFrame (all text), filling unspecified columns."""
    defaults = {
        "sale_date": "01/01/2010",
        "address": "1 Main Street, Athlone",
        "county": "Westmeath",
        "eircode": None,
        "price": "€200,000.00",
        "not_full_market_price": "No",
        "vat_exclusive": "No",
        "description": "Second-Hand Dwelling house /Apartment",
        "size_description": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows], dtype=str)


# --- parse_sale_date ---

def test_parse_sale_date_is_day_first() -> None:
    dates = pd.Series(["03/01/2010", "23/10/2026"])

    parsed = parse_sale_date(dates)

    assert parsed[0] == pd.Timestamp(2010, 1, 3)  # 3 January, not 1 March
    assert parsed[1] == pd.Timestamp(2026, 10, 23)


def test_parse_sale_date_rejects_bad_date() -> None:
    with pytest.raises(ValueError, match="sale dates"):
        parse_sale_date(pd.Series(["01/01/2010", "2010-01-01"]))


# --- parse_price ---

def test_parse_price_strips_euro_and_commas() -> None:
    prices = pd.Series(["€343,000.00", "€1,250,000.50", "€5,000.00"])

    assert parse_price(prices).tolist() == [343000.0, 1250000.5, 5000.0]


def test_parse_price_rejects_bad_price() -> None:
    with pytest.raises(ValueError, match="prices"):
        parse_price(pd.Series(["€343,000.00", "unknown"]))


# --- drop_exact_duplicates ---

def test_drop_exact_duplicates_keeps_first() -> None:
    raw = make_raw([
        {"address": "A"},
        {"address": "A"},  # exact copy of row 0
        {"address": "B"},
    ])

    result = drop_exact_duplicates(raw)

    assert result.index.tolist() == [0, 2]


def test_drop_exact_duplicates_keeps_rows_differing_in_one_column() -> None:
    raw = make_raw([{"county": "Dublin"}, {"county": "Meath"}])

    assert len(drop_exact_duplicates(raw)) == 2


# --- flag_possible_duplicates ---

def test_flag_possible_duplicates_flags_every_row_in_group() -> None:
    raw = make_raw([
        {"address": "A", "county": "Dublin"},
        {"address": "A", "county": "Meath"},       # same date+address+price as row 0
        {"address": "A", "price": "€1.00"},        # different price
        {"address": "B"},                          # different address
        {"address": "A", "sale_date": "02/01/2010"},  # different date
    ])

    assert flag_possible_duplicates(raw).tolist() == [True, True, False, False, False]


# --- to_property_type ---

def test_to_property_type_maps_all_known_descriptions() -> None:
    descriptions = pd.Series([
        "Second-Hand Dwelling house /Apartment",
        "New Dwelling house /Apartment",
        "Teach/Árasán Cónaithe Atháimhe",
        "Teach/Árasán Cónaithe Nua",
        "Teach/?ras?n C?naithe Nua",  # broken-encoding row in the real data
    ])

    assert to_property_type(descriptions).tolist() == [
        "Second-Hand", "New", "Second-Hand", "New", "New",
    ]


def test_to_property_type_is_str_dtype() -> None:
    result = to_property_type(pd.Series(["New Dwelling house /Apartment"], dtype=str))

    assert result.dtype == pd.Series(["x"], dtype=str).dtype


def test_to_property_type_rejects_unknown_description() -> None:
    with pytest.raises(ValueError, match="neither New nor Second-Hand"):
        to_property_type(pd.Series(["New Dwelling house /Apartment", "Castle"]))


# --- yes_no_to_bool ---

def test_yes_no_to_bool_converts_values() -> None:
    result = yes_no_to_bool(pd.Series(["Yes", "No", "No"]))

    assert result.tolist() == [True, False, False]
    assert result.dtype == bool


def test_yes_no_to_bool_rejects_other_values() -> None:
    with pytest.raises(ValueError, match="not 'Yes'/'No'"):
        yes_no_to_bool(pd.Series(["Yes", "yes"]))


# --- flag_new_no_vat / add_vat (rule 5) ---

def test_flag_new_no_vat() -> None:
    property_type = pd.Series(["New", "New", "Second-Hand", "Second-Hand"])
    vat_exclusive = pd.Series([True, False, True, False])

    assert flag_new_no_vat(property_type, vat_exclusive).tolist() == [False, True, False, False]


def test_add_vat_only_where_vat_exclusive() -> None:
    price = pd.Series([200000.0, 200000.0, 100.0])
    vat_exclusive = pd.Series([True, False, True])

    assert add_vat(price, vat_exclusive).tolist() == [227000.0, 200000.0, 113.5]


def test_add_vat_rounds_to_cents() -> None:
    result = add_vat(pd.Series([123456.78]), pd.Series([True]))

    assert result[0] == 140123.45  # 123456.78 * 1.135 = 140123.4453


# --- has_bulk_unit_range / flag_bulk_suspect (rule 6) ---

@pytest.mark.parametrize(
    "address, expected",
    [
        ("Units 1-37 Eastgate Hall, Clondalkin", True),
        ("APTS 1 - 19, ST NICHOLAS HOUSE, CHURCH ST", True),
        ("1-3 Main Street, Ennis", True),  # range covers 3 numbers
        ("12-14 Main Street, Ennis", True),
        ("78-79 Ard Abhainn, Knockraha", False),  # covers only 2 numbers
        ("5-6 Maypole Crescent, Knocklong", False),
        ("Apartment 9, 1-4 Sussex Street, Dun Laoghaire", False),  # one home in the building
        ("APT 9, ALDBOROUGH COURT, 1-4 NORTH STRAND RD", False),
        ("Apartment no. 7, 44-48 Connaught Street", False),
        ("Apartment number 97, 52-58 Station Road", False),
        ("Apt 1-5, 88 North Main Street", True),  # "Apt 1-5" is a range, not one home
        ("1 Main Street, 2-4 Back Lane", True),  # range not at the start still counts
        ("5 Braemor Drive, Churchtown", False),
    ],
)
def test_has_bulk_unit_range(address: str, expected: bool) -> None:
    assert has_bulk_unit_range(pd.Series([address])).tolist() == [expected]


@pytest.mark.parametrize(
    "address, expected",
    [
        # Bulk keywords
        ("Site for Blocks B and C, Woodbrook, Shankill", True),
        ("48 Units at Cronins Wood, Spa Road, Killarney", True),
        ("1 2 3 4 and others, Rathgoggin Place", True),
        ("24 25 26, & others  Rathgoggin Place", True),
        ("Units 1-37 Eastgate Hall, Clondalkin", True),
        # Ordinary single homes that must NOT be flagged
        ("Apartment 5, Block B, Honeypark", False),
        ("Apt 12, Taney Court, Dundrum", False),
        ("APT 3 CASTLEPARK APARTMENTS, LEIGHLIN ROAD", False),
        ("12 HOWTH LODGE APTS, CLARMONT ROAD", False),
        ("Property Described in Folio 12058F., Co Kilkenny", False),
        ("Unit 4, The Mill, Kilkenny", False),
        ("12 Blockhouse Road, Cork", False),  # "block" inside another word
        ("5 Braemor Drive, Churchtown", False),
    ],
)
def test_flag_bulk_suspect_address(address: str, expected: bool) -> None:
    result = flag_bulk_suspect(pd.Series([300000.0]), pd.Series([address]))

    assert result.tolist() == [expected]


def test_flag_bulk_suspect_price_threshold() -> None:
    price = pd.Series([5_000_000.0, 5_000_000.01, 12_000_000.0])
    addresses = pd.Series(["1 Main Street"] * 3)

    assert flag_bulk_suspect(price, addresses).tolist() == [False, True, True]


# --- flag_low_price_suspect (rule 7) ---

def test_flag_low_price_suspect_threshold() -> None:
    price = pd.Series([5001.0, 19999.99, 20000.0, 250000.0])

    assert flag_low_price_suspect(price).tolist() == [True, True, False, False]


# --- flag_model_eligible (rule 9) ---

def test_flag_model_eligible_excludes_any_flag() -> None:
    df = pd.DataFrame({
        "not_full_market_price": [False, True, False, False, False],
        "is_bulk_suspect":       [False, False, True, False, False],
        "is_low_price_suspect":  [False, False, False, True, False],
        "is_new_no_vat":         [False, False, False, False, True],
    })

    assert flag_model_eligible(df).tolist() == [True, False, False, False, False]


def test_flag_model_eligible_ignores_possible_duplicate() -> None:
    df = pd.DataFrame({
        "not_full_market_price": [False],
        "is_bulk_suspect": [False],
        "is_low_price_suspect": [False],
        "is_new_no_vat": [False],
        "is_possible_duplicate": [True],
    })

    assert flag_model_eligible(df).tolist() == [True]


# --- clean_ppr (all steps together) ---

def test_clean_ppr_end_to_end() -> None:
    raw = make_raw([
        {"address": "A", "sale_date": "03/01/2010", "price": "€343,000.00",
         "vat_exclusive": "Yes", "description": "New Dwelling house /Apartment"},
        {"address": "A", "sale_date": "03/01/2010", "price": "€343,000.00",
         "vat_exclusive": "Yes", "description": "New Dwelling house /Apartment"},  # exact dup
        {"address": "B", "not_full_market_price": "Yes"},
        {"address": "B", "county": "Meath"},  # near-dup of row 2
    ])

    df = clean_ppr(raw)

    assert len(df) == 3
    assert df.index.tolist() == [0, 1, 2]
    assert df["sale_date"].tolist() == [
        pd.Timestamp(2010, 1, 3), pd.Timestamp(2010, 1, 1), pd.Timestamp(2010, 1, 1),
    ]
    assert df["price"].tolist() == [343000.0, 200000.0, 200000.0]
    assert df["is_possible_duplicate"].tolist() == [False, True, True]
    assert df["property_type"].tolist() == ["New", "Second-Hand", "Second-Hand"]
    assert df["vat_exclusive"].tolist() == [True, False, False]
    assert df["not_full_market_price"].tolist() == [False, True, False]
    assert "description" in df.columns  # original column kept
    # Part 2: VAT and flags
    assert df["is_new_no_vat"].tolist() == [False, False, False]
    assert df["price_incl_vat"].tolist() == [389305.0, 200000.0, 200000.0]
    assert df["is_bulk_suspect"].tolist() == [False, False, False]
    assert df["is_low_price_suspect"].tolist() == [False, False, False]
    assert df["is_model_eligible"].tolist() == [True, False, True]  # row 1 is not full market price


def test_clean_ppr_price_flags_use_price_incl_vat() -> None:
    raw = make_raw([
        # €4.5m + 13.5% VAT = €5,107,500 -> bulk
        {"address": "A", "price": "€4,500,000.00", "vat_exclusive": "Yes",
         "description": "New Dwelling house /Apartment"},
        # €4.5m, no VAT added -> not bulk
        {"address": "B", "price": "€4,500,000.00"},
        # €18,000 + 13.5% VAT = €20,430 -> not low price
        {"address": "C", "price": "€18,000.00", "vat_exclusive": "Yes",
         "description": "New Dwelling house /Apartment"},
        # €18,000, no VAT added -> low price
        {"address": "D", "price": "€18,000.00"},
    ])

    df = clean_ppr(raw)

    assert df["is_bulk_suspect"].tolist() == [True, False, False, False]
    assert df["is_low_price_suspect"].tolist() == [False, False, False, True]
    assert df["is_model_eligible"].tolist() == [False, True, True, False]


def test_clean_ppr_does_not_change_input() -> None:
    raw = make_raw([{"address": "A"}, {"address": "A"}])
    before = raw.copy()

    clean_ppr(raw)

    pd.testing.assert_frame_equal(raw, before)
