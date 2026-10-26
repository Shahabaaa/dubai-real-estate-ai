"""Train and evaluate a property price model against a simple baseline.

Uses LightGBM if it can be loaded; otherwise falls back to scikit-learn's
HistGradientBoostingRegressor (no extra system libraries needed).

Run from project root:  python3 src/train_price_model.py
Needs: data/processed/sales_clean.parquet  (from clean_sales.py)
Split (by time, never random): train = 2020-01 .. 2025-09, validation = 2025-10 .. 2025-12,
                               test  = 2026-01 onward
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

try:
    import lightgbm as lgb
    USE_LGBM = True
except (ImportError, OSError) as e:          # OSError = missing libomp on Mac
    print(f"LightGBM not available ({type(e).__name__}); using scikit-learn HistGradientBoosting instead.")
    USE_LGBM = False

DATA = Path("data/processed/sales_clean.parquet")
REPORTS, MODELS = Path("reports"), Path("models")
VAL_START, TEST_START = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-01-01")
MODEL_NAME = "LightGBM" if USE_LGBM else "HistGradientBoosting"
MAX_CATS = 200       # scikit-learn allows at most 255 categories per feature; rarer ones become "Other"

CATS = ["property_type", "sub_type", "area", "master_project", "project", "building",
        "nearest_metro", "nearest_mall", "nearest_landmark"]
NUMS = ["area_sqm", "rooms", "is_offplan", "has_parking", "t", "month"]
FEATURES = NUMS + CATS          # NOTE: ppsm / meter_sale_price are never used (leakage)


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["t"] = (df["year"] - 2020) * 12 + df["month"]            # months since 2020-01
    for c in CATS:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


def category_levels(train: pd.DataFrame) -> dict:
    levels = {}
    for c in CATS:
        vc = train[c].value_counts()
        keep = list(vc.index) if USE_LGBM else list(vc.index[:MAX_CATS])
        levels[c] = sorted(set(keep) | ({"Other"} if not USE_LGBM else set()))
    return levels


def to_model_frame(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATS:
        if not USE_LGBM:
            X[c] = X[c].where(X[c].isin(levels[c]), "Other")
        X[c] = pd.Categorical(X[c], categories=levels[c])       # unseen categories -> NaN (LightGBM)
    return X


def build_model():
    if USE_LGBM:
        return lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 min_data_per_group=50, cat_smooth=20, random_state=42, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.06, max_leaf_nodes=127,
                                         min_samples_leaf=50, categorical_features="from_dtype",
                                         early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=30, random_state=42)


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Median price per sqm over the last 12 months of training data, by area x off-plan, times size."""
    recent = train[train["date"] >= train["date"].max() - pd.DateOffset(months=12)]
    by_key = recent.groupby(["area", "is_offplan"])["ppsm"].median()
    by_type = recent.groupby("property_type")["ppsm"].median()
    key = pd.MultiIndex.from_frame(test[["area", "is_offplan"]])
    ppsm = pd.Series(by_key.reindex(key).values, index=test.index)
    ppsm = ppsm.fillna(test["property_type"].map(by_type)).fillna(recent["ppsm"].median())
    return ppsm * test["area_sqm"]


def metrics(y, p) -> dict:
    ape = np.abs(p - y) / y
    return {"MAE_AED": np.mean(np.abs(p - y)), "MedianAE_AED": np.median(np.abs(p - y)),
            "MAPE_%": 100 * ape.mean(), "MedianAPE_%": 100 * np.median(ape),
            "within_10%": 100 * (ape <= .10).mean(), "within_20%": 100 * (ape <= .20).mean(),
            "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), "n": len(y)}


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = load()
    train = df[df["date"] < VAL_START]
    val = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)]
    test = df[df["date"] >= TEST_START]
    print(f"model: {MODEL_NAME} | train {len(train):,} | validation {len(val):,} | test {len(test):,}")

    levels = category_levels(train)
    Xtr, Xva, Xte = (to_model_frame(d, levels) for d in (train, val, test))
    model = build_model()
    ytr, yva = np.log1p(train["price_aed"]), np.log1p(val["price_aed"])
    if USE_LGBM:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        print("best iteration:", model.best_iteration_)
    else:
        model.fit(Xtr, ytr)
        print("iterations used:", model.n_iter_)

    y = test["price_aed"].to_numpy()
    p_model = np.expm1(model.predict(Xte))
    p_base = baseline(pd.concat([train, val]), test).to_numpy()
    rows = {"Baseline (median AED/sqm x size)": metrics(y, p_base), MODEL_NAME: metrics(y, p_model)}
    for name, mask in [("Unit", test["property_type"] == "Unit"), ("Villa", test["property_type"] == "Villa"),
                       ("Off-plan", test["is_offplan"] == 1), ("Existing", test["is_offplan"] == 0)]:
        m = mask.to_numpy()
        rows[f"{MODEL_NAME} - {name}"] = metrics(y[m], p_model[m])
    res = pd.DataFrame(rows).T.round(1)
    res["R2"] = pd.DataFrame(rows).T["R2"].round(3)
    print(res.to_string())
    res.to_csv(REPORTS / "model_metrics.csv")

    joblib.dump({"model": model, "levels": levels, "features": FEATURES, "model_name": MODEL_NAME},
                MODELS / "price_model.joblib")

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES)
        title = "Feature importance (LightGBM split count)"
    else:
        idx = np.random.default_rng(0).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
        pi = permutation_importance(model, Xte.iloc[idx], np.log1p(y[idx]), n_repeats=3, random_state=0, n_jobs=1)
        imp = pd.Series(pi.importances_mean, index=FEATURES)
        title = "Permutation importance (drop in score when shuffled)"
    imp.sort_values().plot.barh(figsize=(7, 6), title=title)
    plt.tight_layout(); plt.savefig(REPORTS / "feature_importance.png", dpi=130); plt.close()

    pos = np.random.default_rng(0).choice(len(test), size=min(5000, len(test)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(y[pos], p_model[pos], s=4, alpha=.3)
    lim = [1e5, 3e7]; plt.plot(lim, lim, "r--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Actual price (AED)"); plt.ylabel("Predicted price (AED)"); plt.title("Test set: predicted vs actual")
    plt.tight_layout(); plt.savefig(REPORTS / "pred_vs_actual.png", dpi=130)
    print("Saved model and reports.")
