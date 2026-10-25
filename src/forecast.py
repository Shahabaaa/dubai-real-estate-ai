"""Quarterly Dubai sales forecasting: baselines vs a log-linear trend+season model.

Run from project root:  python3 src/forecast.py
Needs: data/processed/market_quarterly.csv (made by clean_market.py)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

IN = Path("data/processed/market_quarterly.csv")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)
HOLDOUT = 4          # last 4 OBSERVED quarters are the test set
HORIZON = 4          # quarters to forecast beyond the data


def load_series() -> pd.Series:
    m = pd.read_csv(IN)
    s = m[m.transaction == "Sales"].groupby("period")["n_transactions"].sum()
    s.index = pd.PeriodIndex(s.index, freq="Q")
    full = pd.period_range(s.index.min(), s.index.max(), freq="Q")
    return s.reindex(full)          # missing quarters stay NaN (never interpolated)


def features(idx: pd.PeriodIndex, t0: pd.Period) -> pd.DataFrame:
    X = pd.DataFrame(index=idx)
    X["t"] = [(p - t0).n for p in idx]
    for q in (2, 3, 4):
        X[f"q{q}"] = (idx.quarter == q).astype(int)
    return X


def fit_predict(train: pd.Series, future_idx: pd.PeriodIndex) -> pd.Series:
    tr = train.dropna()
    t0 = train.index[0]
    model = Ridge(alpha=1.0).fit(features(tr.index, t0), np.log(tr.values))
    return pd.Series(np.exp(model.predict(features(future_idx, t0))), index=future_idx)


def seasonal_naive(train: pd.Series, future_idx: pd.PeriodIndex) -> pd.Series:
    out = {}
    for p in future_idx:
        v, k = np.nan, 1
        while np.isnan(v) and k <= 8:        # same quarter, most recent year available
            prev = p - 4 * k
            v = train.get(prev, np.nan); k += 1
        out[p] = v
    return pd.Series(out)


def naive(train: pd.Series, future_idx: pd.PeriodIndex) -> pd.Series:
    return pd.Series(train.dropna().iloc[-1], index=future_idx)


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~np.isnan(yhat)
    return {"MAE": np.mean(np.abs(y[ok] - yhat[ok])),
            "MAPE_%": 100 * np.mean(np.abs((y[ok] - yhat[ok]) / y[ok])), "n": int(ok.sum())}


if __name__ == "__main__":
    s = load_series()
    obs = s.dropna()
    test_idx = obs.index[-HOLDOUT:]
    train = s[s.index < test_idx[0]]
    print(f"Train: {train.dropna().index[0]} -> {train.dropna().index[-1]} ({train.notna().sum()} quarters)")
    print(f"Test : {list(map(str, test_idx))}\n")

    preds = {"Naive (last value)": naive(train, test_idx),
             "Seasonal naive": seasonal_naive(train, test_idx),
             "Log-linear trend+season": fit_predict(train, test_idx)}
    rows = {k: metrics(obs[test_idx], v.reindex(test_idx)) for k, v in preds.items()}
    res = pd.DataFrame(rows).T.round(1)
    print(res, "\n")
    res.to_csv(OUT / "forecast_metrics.csv")

    # Refit on all observed data and forecast the next quarters
    fut = pd.period_range(s.index[-1] + 1, periods=HORIZON, freq="Q")
    fc = fit_predict(s, fut)
    print("Forecast (number of sales transactions):")
    print(fc.round(0).astype(int).rename(index=str))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index.to_timestamp(), s.values, marker="o", label="Actual (gaps = missing quarters)")
    ax.plot(fut.to_timestamp(), fc.values, marker="o", ls="--", color="tab:red", label="Forecast")
    for k, v in preds.items():
        ax.plot(test_idx.to_timestamp(), v.reindex(test_idx).values, ls=":", label=f"Test: {k}")
    ax.set_ylabel("Sales transactions per quarter"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Dubai quarterly sales transactions: forecast")
    plt.tight_layout(); plt.savefig(OUT / "forecast.png", dpi=130)
