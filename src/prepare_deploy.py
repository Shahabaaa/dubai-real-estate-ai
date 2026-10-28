"""Prepare everything the web app needs, so the app never touches the big raw data.

Run from the project root (after clean_sales.py and train_price_model.py):
    python3 src/prepare_deploy.py

Creates small summary files in app/data/ and compresses the saved model.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA = Path("data/processed/sales_clean.parquet")
OUT = Path("app/data")
MODEL = Path("models/price_model.joblib")


def sync_requirements() -> None:
    """Pin scikit-learn, pandas and numpy in requirements.txt to the versions installed here (they trained the model)."""
    import sklearn
    pins = {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    req = Path("requirements.txt")
    lines = req.read_text().splitlines() if req.exists() else []
    out, seen = [], set()
    for line in lines:
        name = line.split("==")[0].strip()
        if name in pins and "==" in line:
            out.append(f"{name}=={pins[name]}")
            seen.add(name)
        else:
            out.append(line)
    out += [f"{n}=={v}" for n, v in pins.items() if n not in seen]
    req.write_text("\n".join(out) + "\n")
    print("requirements.txt now pins:", pins)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA, columns=["date", "price_aed", "ppsm", "is_offplan", "property_type", "area", "sub_type", "master_project", "project", "building", "nearest_metro", "nearest_mall", "nearest_landmark"])
    last = df["date"].max()
    recent = df[df["date"] >= last - pd.DateOffset(months=12)]

    # Area statistics over the last 12 months
    g = recent.groupby("area")
    stats = pd.DataFrame({
        "n_sales": g.size(),
        "median_price_aed": g["price_aed"].median(),
        "median_ppsm": g["ppsm"].median(),
        "p25_ppsm": g["ppsm"].quantile(0.25),
        "p75_ppsm": g["ppsm"].quantile(0.75),
        "offplan_share": g["is_offplan"].mean(),
    }).reset_index()
    stats = stats[stats["n_sales"] >= 100].sort_values("n_sales", ascending=False).round(3)
    stats.to_csv(OUT / "area_stats.csv", index=False)

    # Typical context (project, building, nearby metro/mall) for each area and property type, last 24 months.
    # The model learns property type mostly through this context, so the app must supply it.
    ctx_cols = ["master_project", "building", "nearest_metro", "nearest_mall", "nearest_landmark"]
    d = df[df["date"] >= last - pd.DateOffset(months=24)].copy()
    for c in ["area", "sub_type", "project"] + ctx_cols:
        d[c] = d[c].fillna("Unknown").astype(str)

    def mode(x):
        return x.value_counts().index[0]

    ctx = d.groupby(["area", "sub_type", "project"]).agg(
        n_sales=("project", "size"), median_ppsm=("ppsm", "median"),
        **{c: (c, mode) for c in ctx_cols}).reset_index()
    ctx = ctx[ctx["n_sales"] >= 10]
    ctx["rank"] = ctx.groupby(["area", "sub_type"])["n_sales"].rank(method="first", ascending=False)
    ctx[ctx["rank"] <= 12].round(1).to_csv(OUT / "area_context.csv", index=False)

    # Quarterly trend (partial quarters are dropped so the latest point is not misleading)
    q = df.assign(period=df["date"].dt.to_period("Q"),
                  status=np.where(df["is_offplan"] == 1, "Off-plan", "Existing"))
    q = q[q["period"].dt.end_time <= last]
    trend = (q.groupby(["period", "property_type", "status"])
               .agg(n_sales=("ppsm", "size"), median_ppsm=("ppsm", "median")).reset_index())
    trend["quarter"] = trend["period"].astype(str)
    trend[["quarter", "property_type", "status", "n_sales", "median_ppsm"]].round(1).to_csv(
        OUT / "trend.csv", index=False)

    summary = {
        "n_sales": int(len(df)),
        "first_date": str(df["date"].min().date()), "last_date": str(last.date()),
        "median_ppsm_12m": float(recent["ppsm"].median()),
        "median_price_12m": float(recent["price_aed"].median()),
        "offplan_share_12m": float(recent["is_offplan"].mean()),
        "n_areas": int(df["area"].nunique()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote app/data/area_stats.csv, area_context.csv, trend.csv, summary.json")
    print(json.dumps(summary, indent=2))

    # Compress the model so it is small enough for GitHub (limit 100 MB per file)
    bundle = joblib.load(MODEL)
    joblib.dump(bundle, MODEL, compress=3)
    mb = MODEL.stat().st_size / 1e6
    print(f"Model file size after compression: {mb:.1f} MB")
    if mb > 90:
        print("WARNING: model is close to or above GitHub's 100 MB limit. Tell Claude and we will shrink it.")
    sync_requirements()


if __name__ == "__main__":
    main()
