"""Clean DLD property-level transactions into a modelling table of residential SALES.

Run from project root:
    python3 src/clean_sales.py data/raw/transactions_2026-10-23_17-55-06_0001.csv [more parts...]
Output: data/processed/sales_clean.parquet and reports/cleaning_log.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed/sales_clean.parquet")
LOG = Path("reports/cleaning_log.txt")

# ---- Cleaning rules (change here, then re-run) ----
SALE_PROCEDURES = ["Sell", "Sell - Pre registration", "Delayed Sell"]   # excludes "Sell Development"
PROPERTY_TYPES = ["Unit", "Villa"]
SUB_TYPES = ["Flat", "Villa", "Stacked Townhouses"]
START_DATE, END_DATE = "2020-01-01", pd.Timestamp.today().normalize()
MIN_PRICE, MAX_PRICE = 150_000, 100_000_000        # AED
MIN_AREA, MAX_AREA = 15, 2_000                      # sqm
MIN_PPSM, MAX_PPSM = 2_000, 80_000                  # AED per sqm (computed from price / area)

USE = ["transaction_id", "instance_date", "trans_group_en", "procedure_name_en", "actual_worth",
       "procedure_area", "property_type_en", "property_sub_type_en", "property_usage_en", "reg_type_en",
       "rooms_en", "has_parking", "area_name_en", "master_project_en", "project_name_en",
       "building_name_en", "nearest_metro_en", "nearest_mall_en", "nearest_landmark_en"]
# Deliberately NOT used: meter_sale_price (= price / area -> data leakage)


def rooms_to_number(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    out = s.str.extract(r"^(\d+)\s*B/R", expand=False).astype("Float64")
    out = out.mask(s.str.lower() == "studio", 0)
    out = out.mask(s.str.upper() == "PENTHOUSE", np.nan)     # no bedroom count given
    return out.astype(float)


def clean_chunk(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    def step(name, mask):
        log[name] = log.get(name, 0) + int((~mask).sum())
        return df[mask]

    log["rows_read"] = log.get("rows_read", 0) + len(df)
    df = step("dropped: not a Sales transaction", df["trans_group_en"].eq("Sales"))
    df = step("dropped: procedure not an ordinary sale", df["procedure_name_en"].isin(SALE_PROCEDURES))
    df = step("dropped: not unit/villa", df["property_type_en"].isin(PROPERTY_TYPES))
    df = step("dropped: not residential flat/villa/townhouse",
              df["property_usage_en"].eq("Residential") & df["property_sub_type_en"].isin(SUB_TYPES))
    df = df.copy()
    df["date"] = pd.to_datetime(df["instance_date"], format="%Y-%m-%d", errors="coerce")
    df = step("dropped: bad or out-of-range date", df["date"].between(START_DATE, END_DATE))
    df = df.copy()
    df["price_aed"] = pd.to_numeric(df["actual_worth"], errors="coerce")
    df["area_sqm"] = pd.to_numeric(df["procedure_area"], errors="coerce")
    df = step("dropped: price outside range", df["price_aed"].between(MIN_PRICE, MAX_PRICE))
    df = step("dropped: area outside range", df["area_sqm"].between(MIN_AREA, MAX_AREA))
    df = df.copy()
    df["ppsm"] = df["price_aed"] / df["area_sqm"]
    df = step("dropped: price/sqm outside range", df["ppsm"].between(MIN_PPSM, MAX_PPSM))
    return df


def main(paths):
    log, parts = {}, []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=USE, chunksize=200_000, low_memory=False):
            parts.append(clean_chunk(chunk, log))
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(subset="transaction_id")
    log["dropped: duplicate transaction_id"] = n0 - len(df)

    df["rooms"] = rooms_to_number(df["rooms_en"])
    df["is_offplan"] = df["reg_type_en"].str.contains("Off-Plan", na=False).astype(int)
    df["has_parking"] = pd.to_numeric(df["has_parking"], errors="coerce")
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    keep = ["transaction_id", "date", "year", "month", "price_aed", "area_sqm", "ppsm", "rooms",
            "is_offplan", "has_parking", "property_type_en", "property_sub_type_en", "area_name_en",
            "master_project_en", "project_name_en", "building_name_en", "nearest_metro_en",
            "nearest_mall_en", "nearest_landmark_en"]
    df = df[keep].rename(columns={"property_type_en": "property_type", "property_sub_type_en": "sub_type",
                                  "area_name_en": "area", "master_project_en": "master_project",
                                  "project_name_en": "project", "building_name_en": "building",
                                  "nearest_metro_en": "nearest_metro", "nearest_mall_en": "nearest_mall",
                                  "nearest_landmark_en": "nearest_landmark"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    lines = [f"{k}: {v:,}" for k, v in log.items()]
    lines += ["", f"Rows kept: {len(df):,}", f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}",
              "", "Rows per year:", df["year"].value_counts().sort_index().to_string(),
              "", "Share off-plan: " + f"{df['is_offplan'].mean():.1%}",
              "Missing rooms: " + f"{df['rooms'].isna().mean():.1%}",
              "", "Price (AED) / area (sqm) / ppsm summary:",
              df[["price_aed", "area_sqm", "ppsm"]].describe(percentiles=[.01, .5, .99]).round(0).to_string()]
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 src/clean_sales.py <csv> [<csv> ...]")
    main(sys.argv[1:])
