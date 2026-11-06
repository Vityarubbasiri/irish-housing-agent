# Irish Housing Market Agent

## Project goal
A portfolio project: analyse Ireland's Property Price Register (PPR), build a price model,
and an LLM agent that answers questions about Irish property prices using the data.
Audience: recruiters and hiring managers for data scientist roles in Ireland.

## Tech stack
- Python 3.12, virtual environment in `.venv` (Windows, PowerShell)
- pandas, duckdb, matplotlib, jupyter
- Later: scikit-learn / LightGBM, Streamlit, pytest

## Project structure
- `data/raw/PPR-ALL.csv` — original download. NEVER modify or overwrite.
- `data/processed/` — cleaned outputs (gitignored)
- `notebooks/` — exploration only. `01_explore_ppr.ipynb` contains my findings (54 notes).
- `src/housing/` — reusable, tested code (ingest, clean, features, model, agent)
- `tests/` — pytest tests

## Data facts (from my exploration)
- Encoding: `cp1252`. Load with `low_memory=False`.
- 814,413 rows, 9 columns, all text. Sales from 1 Jan 2010 to 23 Oct 2026.
- Dates are dd/mm/yyyy (Irish, day-first) — always use format="%d/%m/%Y".
- Price is text like "€343,000.00" — strip € and commas, convert to float.
- Eircode only 31% present; Property Size Description only 6.5% present.

## Cleaning rules (decided — follow these exactly)
1. Rename columns to snake_case (sale_date, address, county, eircode, price, not_full_market_price, vat_exclusive, description, size_description).
2. Remove exact duplicate rows (all original columns identical), keep first. Expect 1,092 removed.
3. Flag (do NOT remove) near-duplicates: same date + address + price but another column differs → `is_possible_duplicate`.
4. Map Description of Property to `property_type` = "New" / "Second-Hand". Irish versions: "Teach/Árasán Cónaithe Atháimhe" = Second-Hand; anything containing "Nua" = New (includes one broken-encoding row).
5. VAT: where vat_exclusive == "Yes", add 13.5% → `price_incl_vat`. Otherwise price_incl_vat = price.
   Do NOT add VAT where property_type == "New" and vat_exclusive == "No" — flag those as `is_new_no_vat` (2,377 rows, odd/cheap).
   (2,387 in the raw file; 10 of those are exact duplicates removed by rule 2.)
6. Flag `is_bulk_suspect`: price_incl_vat > €5,000,000 OR address matches a bulk pattern (case-insensitive, whole words):
   - "blocks" (plural), "units" (plural), "& others" / "and others"
   - a number range covering 3+ numbers (e.g. "1-3", "1-150"), UNLESS the address starts with a single "Apt / Apartment / Flat / Unit + number" (one home in a building with a ranged street number, e.g. "Apartment 9, 1-4 Sussex Street").
   Do NOT use singular "apartment", "apt", "block", nor "apartments", "apts", "folio" — checked on real data, they mostly flag ordinary single homes (development names, block letters, folio-described properties).
7. Flag `is_low_price_suspect`: price_incl_vat < €20,000.
8. Keep `not_full_market_price` as a boolean.
9. Add `is_model_eligible` = not any of: not_full_market_price, is_bulk_suspect, is_low_price_suspect, is_new_no_vat.
   Never delete flagged rows — only exclude them in analysis/modelling.

## Coding rules
- Put logic in functions inside `src/housing/`, with type hints and short docstrings.
- Write pytest tests for every cleaning function (small hand-made DataFrames, not the full file).
- Never use random train/test splits for modelling — always split by time.
- Don't install new packages without asking me first.
- Don't commit data files.

## How to work with me
- I'm learning: work in small steps, one task at a time.
- Before writing code, briefly explain your plan. After, explain what you changed in plain language.
- If anything in my rules looks wrong or unclear, ask instead of guessing.