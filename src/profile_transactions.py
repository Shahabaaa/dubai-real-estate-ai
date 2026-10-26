"""Profile the big DLD transactions CSV in chunks (safe on a laptop).

Run from project root:
    python3 src/profile_transactions.py data/raw/transactions_2026-10-23_17-55-06_0001.csv
Prints a summary and saves it to reports/transactions_profile.txt
"""
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

CAT_COLS = ["trans_group_en", "procedure_name_en", "property_type_en", "property_sub_type_en",
            "reg_type_en", "rooms_en", "property_usage_en", "has_parking"]
NUM_COLS = ["actual_worth", "meter_sale_price", "procedure_area", "rent_value"]
USE = CAT_COLS + NUM_COLS + ["instance_date", "area_name_en", "transaction_id"]


def main(path: str, chunksize: int = 200_000) -> None:
    counters = {c: Counter() for c in CAT_COLS}
    n = 0
    nulls = Counter()
    dmin = dmax = None
    nums = []
    areas = Counter()
    ids = 0
    for chunk in pd.read_csv(path, usecols=USE, chunksize=chunksize, low_memory=False):
        n += len(chunk)
        nulls.update(chunk.isna().sum().to_dict())
        for c in CAT_COLS:
            counters[c].update(chunk[c].fillna("<missing>").astype(str).value_counts().to_dict())
        areas.update(chunk["area_name_en"].fillna("<missing>").value_counts().to_dict())
        d = pd.to_datetime(chunk["instance_date"], errors="coerce", dayfirst=True)
        dmin = d.min() if dmin is None else min(dmin, d.min())
        dmax = d.max() if dmax is None else max(dmax, d.max())
        nums.append(chunk[NUM_COLS].sample(min(len(chunk), 20_000), random_state=0))
        ids += chunk["transaction_id"].nunique()

    out = [f"Rows: {n:,}", f"Date range (instance_date): {dmin} -> {dmax}", "",
           "Missing values (% of rows):"]
    out += [f"  {c}: {100 * nulls[c] / n:.1f}%" for c in USE]
    for c in CAT_COLS:
        out += ["", f"{c} (top 15):"]
        out += [f"  {k}: {v:,}" for k, v in counters[c].most_common(15)]
    out += ["", "Top 15 areas:"] + [f"  {k}: {v:,}" for k, v in areas.most_common(15)]
    out += ["", "Numeric summary (sample):", pd.concat(nums).describe(percentiles=[.01, .5, .99]).round(1).to_string()]
    text = "\n".join(out)
    print(text)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/transactions_profile.txt").write_text(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 src/profile_transactions.py <path-to-csv>")
    main(sys.argv[1])
