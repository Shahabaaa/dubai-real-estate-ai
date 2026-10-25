"""Clean the DLD quarterly market-summary CSV (aggregate data, not property-level).

Input : data/raw/Real_Estate_Transactions_-_Emirate_of_Dubai_*.csv
Output: data/processed/market_quarterly.csv  (one row per period x transaction title x property type)
"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/market_quarterly.csv")


def load_market() -> pd.DataFrame:
    f = sorted(RAW.glob("Real_Estate_Transactions*.csv"))[-1]
    df = pd.read_csv(f)
    # Keep the English columns only (the file has Arabic duplicates)
    df = df[["Year", "Quarter_Number", "Title", "Type", "Description", "Value"]]
    df = df.rename(columns={"Quarter_Number": "quarter", "Year": "year",
                            "Title": "transaction", "Type": "property_type"})
    # Long -> wide: Number (count) and Value (AED) side by side
    wide = (df.pivot_table(index=["year", "quarter", "transaction", "property_type"],
                           columns="Description", values="Value", aggfunc="sum")
              .reset_index()
              .rename(columns={"Number": "n_transactions", "Value": "value_aed"}))
    wide["avg_value_aed"] = wide["value_aed"] / wide["n_transactions"]
    wide["period"] = pd.PeriodIndex.from_fields(year=wide["year"], quarter=wide["quarter"], freq="Q")
    wide["period_end"] = wide["period"].dt.end_time.dt.normalize()
    return wide.sort_values(["transaction", "property_type", "period"]).reset_index(drop=True)


def missing_periods(df: pd.DataFrame) -> list:
    full = pd.period_range(df["period"].min(), df["period"].max(), freq="Q")
    return [str(p) for p in full.difference(pd.PeriodIndex(df["period"].unique()))]


if __name__ == "__main__":
    m = load_market()
    print(m.head(), "\n")
    print("Periods:", m["period"].min(), "->", m["period"].max())
    print("Missing quarters:", missing_periods(m))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.assign(period=m["period"].astype(str)).to_csv(OUT, index=False)
    print(f"Saved {len(m)} rows to {OUT}")
