"""Explain the price model with Shapley-value attributions (permutation sampling).

For a property, each feature's contribution is its average marginal effect on the predicted price when
features are switched from a "typical recent sale" (the background) to this property's values, over many random
orderings. This is the same idea as SHAP's permutation explainer, and it works with any model (no libomp needed).

Run from the project root:  python3 src/explain.py
Needs: data/processed/sales_clean.parquet and models/price_model.joblib
Creates in reports/: explain_global.png, explain_groups.png, explain_examples.png, explain_importance.csv
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, MODEL, REPORTS = Path("data/processed/sales_clean.parquet"), Path("models/price_model.joblib"), Path("reports")
N_EXPLAIN, N_BACKGROUND, N_PERM, SEED = 300, 150, 16, 0
TEST_START = pd.Timestamp("2026-01-01")
CATS = ["property_type", "sub_type", "area", "master_project", "project", "building",
        "nearest_metro", "nearest_mall", "nearest_landmark"]
LABELS = {"area_sqm": "Size (sq m)", "rooms": "Bedrooms", "is_offplan": "Off-plan status", "has_parking": "Parking",
          "t": "Sale date (market level)", "month": "Month of year", "property_type": "Unit or villa",
          "sub_type": "Sub-type", "area": "Area", "master_project": "Master project", "project": "Project",
          "building": "Building", "nearest_metro": "Nearest metro", "nearest_mall": "Nearest mall",
          "nearest_landmark": "Nearest landmark"}
GROUPS = {"Location": ["area", "master_project", "project", "building", "nearest_metro", "nearest_mall",
                       "nearest_landmark"],
          "Size and layout": ["area_sqm", "rooms", "sub_type", "property_type", "has_parking"],
          "Off-plan or ready": ["is_offplan"], "Timing": ["t", "month"]}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["t"] = (df["year"] - 2020) * 12 + df["month"]
    for c in CATS:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


def to_frame(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = raw[bundle["features"]].copy()
    for c in bundle["features"]:
        if c not in CATS:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CATS:
        levels = bundle["levels"][c]
        if "Other" in levels:
            X[c] = X[c].where(X[c].isin(levels), "Other")
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


def explain_rows(bundle, rows, background, rng):
    feats, F = bundle["features"], len(bundle["features"])
    xs, bs = rows[feats].to_numpy(dtype=object), background[feats].to_numpy(dtype=object)
    contrib, base, pred = np.zeros((len(xs), F)), np.zeros(len(xs)), np.zeros(len(xs))
    for i, x in enumerate(xs):
        blocks, orders = [], []
        for _ in range(N_PERM):
            order, b = rng.permutation(F), bs[rng.integers(len(bs))]
            M = np.zeros((F + 1, F), dtype=bool)
            for j in range(1, F + 1):
                M[j, order[:j]] = True
            blocks.append(np.where(M, x, b))
            orders.append(order)
        big = pd.DataFrame(np.vstack(blocks), columns=feats)
        yhat = bundle["model"].predict(to_frame(big, bundle)).reshape(N_PERM, F + 1)
        d = np.diff(yhat, axis=1)
        for k, order in enumerate(orders):
            contrib[i, order] += d[k]
        contrib[i] /= N_PERM
        base[i], pred[i] = yhat[:, 0].mean(), yhat[:, -1].mean()
    return contrib, base, pred


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nice_value(feat, v):
    if isinstance(v, (float, np.floating, int, np.integer)):
        if np.isnan(v):
            return "n/a"
        if feat == "is_offplan":
            return "Off-plan" if v == 1 else "Ready"
        if feat == "has_parking":
            return "Yes" if v == 1 else "No"
        if feat == "rooms":
            return "Studio" if v == 0 else f"{v:g}"
        if feat == "t":
            return f"{MONTHS[int(v - 1) % 12]} {2020 + int(v - 1) // 12}"
        if feat == "month":
            return MONTHS[int(v) - 1]
        return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.1f}"
    return str(v)[:28]


def waterfall(ax, feats, c, base_log, raw_row, title, top=7):
    order = np.argsort(-np.abs(c))
    shown, rest = order[:top], order[top:]
    steps = [(f"{LABELS[feats[i]]} = {nice_value(feats[i], raw_row[feats[i]])}", c[i]) for i in shown]
    if len(rest):
        steps.append(("All other features", c[rest].sum()))
    cum, level = base_log, np.expm1(base_log)
    ax.barh(len(steps), level, color="#8aa0a8")
    ax.text(level * 1.01, len(steps), f"AED {level:,.0f}", va="center", fontsize=8)
    levels_seen = [level]
    labels = ["Typical recent sale"]
    ypos = [len(steps)]
    for k, (name, val) in enumerate(steps):
        new = np.expm1(cum + val)
        ax.barh(len(steps) - 1 - k, new - level, left=level, color="#d4a24c" if val > 0 else "#0e7c86")
        ax.text(max(level, new) * 1.01, len(steps) - 1 - k, f"{new - level:+,.0f}", va="center", fontsize=8)
        cum, level = cum + val, new
        levels_seen.append(level)
        labels.append(name); ypos.append(len(steps) - 1 - k)
    ax.set_xlim(min(levels_seen) * 0.85, max(levels_seen) * 1.12)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{title}\nEstimated price: AED {level:,.0f}", fontsize=10)
    ax.set_xlabel("AED"); ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")


if __name__ == "__main__":
    t0, rng = time.time(), np.random.default_rng(SEED)
    REPORTS.mkdir(exist_ok=True)
    bundle, df = joblib.load(MODEL), load_data()
    feats = bundle["features"]
    test = df[df["date"] >= TEST_START]
    bg_pool = df[(df["date"] >= TEST_START - pd.DateOffset(years=1)) & (df["date"] < TEST_START)]
    rows = test.sample(min(N_EXPLAIN, len(test)), random_state=SEED).reset_index(drop=True)
    background = bg_pool.sample(min(N_BACKGROUND, len(bg_pool)), random_state=SEED)
    print(f"Explaining {len(rows)} properties from 2026 against {len(background)} typical 2025 sales...")
    contrib, base, pred = explain_rows(bundle, rows, background, rng)
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes.")

    # Global importance: average absolute effect on the price, in percent
    imp = pd.Series(np.abs(contrib).mean(axis=0) * 100, index=feats).rename("avg_impact_percent")
    imp.sort_values(ascending=False).round(2).rename(LABELS).to_csv(REPORTS / "explain_importance.csv")
    ax = imp.sort_values().rename(LABELS).plot.barh(figsize=(7, 6), color="#0e7c86")
    ax.set_xlabel("Average effect on the estimated price (%)")
    ax.set_title("What moves the estimate? (2026 sales)")
    plt.tight_layout(); plt.savefig(REPORTS / "explain_global.png", dpi=130); plt.close()

    grp = pd.Series({g: np.abs(contrib[:, [feats.index(f) for f in fs if f in feats]]).sum(axis=1).mean() * 100
                     for g, fs in GROUPS.items()}).sort_values()
    ax = grp.plot.barh(figsize=(7, 3.2), color="#d4a24c")
    ax.set_xlabel("Combined average effect on the estimated price (%)"); ax.set_title("Grouped by theme")
    plt.tight_layout(); plt.savefig(REPORTS / "explain_groups.png", dpi=130); plt.close()

    # Three worked examples: a lower-priced, a mid-priced and a high-priced sale
    prices = rows["price_aed"].to_numpy()
    picks = [int(np.argmin(np.abs(prices - np.quantile(prices, q)))) for q in (0.25, 0.5, 0.9)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 4.2 * len(picks)))
    for ax, i, name in zip(np.atleast_1d(axes), picks, ["Lower-priced sale", "Mid-priced sale", "High-priced sale"]):
        waterfall(ax, feats, contrib[i], base[i], rows.iloc[i], name)
    plt.tight_layout(); plt.savefig(REPORTS / "explain_examples.png", dpi=130)
    print("\nAverage effect on the estimated price (%):")
    print(imp.sort_values(ascending=False).round(1).rename(LABELS).to_string())
    print("\nSaved charts and explain_importance.csv in reports/")
