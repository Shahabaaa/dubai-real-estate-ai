"""Load raw Dubai transaction CSVs, standardise columns, clean, and save to data/processed/."""
import re
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/transactions_clean.parquet")


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Put your downloaded CSV file(s) in data/raw/")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [snake(c) for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # Column names differ by dataset: after inspecting df.columns, rename here, e.g.
    # df = df.rename(columns={"actual_worth": "price_aed", "area_name_en": "area"})
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    for c in [c for c in df.columns if any(k in c for k in ("price", "worth", "amount"))]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


if __name__ == "__main__":
    raw = load_raw()
    print("Columns:", list(raw.columns))
    print(raw.head())
    cleaned = clean(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT, index=False)
    print(f"Saved {len(cleaned):,} rows to {OUT}")
