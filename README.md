# Dubai Real Estate Intelligence Platform

A data science project on the Dubai property market. It forecasts market activity, estimates the price of
an individual apartment or villa, and serves the estimate through a small web app.

**Status: Phases 1 and 2 complete (market forecast, price model, Streamlit app).**

## Roadmap

- [x] **Phase 1:** Clean the quarterly market summary and forecast sales volume
- [x] **Phase 2:** Property-level price model, evaluated on unseen 2026 sales, with a Streamlit app
- [ ] **Phase 3:** Explainability (SHAP) and a RAG assistant over market data and regulations
- [ ] **Phase 4:** Public deployment and dashboard

## Data

| Dataset | Used for | Notes |
|---|---|---|
| DLD real estate transactions, quarterly summary | Phase 1 forecast | Aggregate counts and values, 2016 Q1 to 2026 Q3. All of 2024 and 2025 Q3 are missing in the source. |
| DLD real estate transactions, property-level | Phase 2 price model | About 904k raw records (one part of a multi-part download). |

Raw files are not included in this repo. Download them from the official Dubai Land Department / Dubai Pulse
portals into `data/raw/` and check each dataset's terms before reuse.

### Cleaning (Phase 2)

From 903,704 raw records, `src/clean_sales.py` keeps **336,945** residential sales (flats, villas, stacked
townhouses) from 2020-01 to 2026-10. It removes mortgages and gifts, developer bulk deals, impossible dates
(some records are dated the year 1420 or in the future), and implausible prices and sizes. Every rule is a
constant at the top of the script, and the number of rows dropped at each step is saved in
`reports/cleaning_log.txt`. 67.9% of the remaining sales are off-plan.

`meter_sale_price` is deliberately not used as a feature: it is the price divided by the area, so using it
would leak the answer into the model.

## Phase 2 results: price model

The model predicts the log of the sale price from size, rooms, off-plan or existing status, parking, location
(area, project, building), nearest metro, mall and landmark, and the sale month.

**Evaluation is split by time, not at random.** Training uses sales up to September 2025, validation is
Q4 2025, and the test set is all 54,521 sales from 2026, which the model never saw.

| Metric (2026 test set) | Baseline | Gradient boosting model |
|---|---|---|
| Mean absolute error | AED 384,186 | **AED 299,170** |
| Median absolute error | AED 161,969 | **AED 128,152** |
| Median % error | 13.1% | **9.8%** |
| Estimates within 10% of actual price | 39.8% | **50.7%** |
| Estimates within 20% of actual price | 68.2% | **77.4%** |
| R² | 0.824 | **0.884** |

The baseline is the median price per sq m over the last 12 months in the same area and off-plan status,
multiplied by the property size.

| Segment | Median % error | R² | Test sales |
|---|---|---|---|
| Units | 9.9% | 0.883 | 49,749 |
| Villas | 8.8% | 0.809 | 4,772 |
| Off-plan | 9.7% | 0.883 | 41,426 |
| Existing | 10.2% | 0.889 | 13,095 |

Charts: `reports/pred_vs_actual.png` (predicted vs actual price) and `reports/feature_importance.png`
(permutation importance).

### Limitations

- About one in four estimates is off by more than 20%. The data has no floor, view, finish or condition.
- Prices are registered transaction values, not listing prices.
- The model was trained on 2020 to 2025 and tested on 2026; market regime changes (such as the 2026 slowdown
  seen in the quarterly data) can reduce accuracy. Retraining on recent data is needed to keep it current.
- Villa results rest on only 4,772 test sales.
- Training hit the 800-iteration cap, so tuning could improve results slightly.
- The model is scikit-learn's `HistGradientBoostingRegressor`. The script uses LightGBM automatically when it
  can be loaded. It could not be loaded on the Intel Mac this was developed on.

## Phase 1 results: quarterly sales forecast

On the last four observed quarters, a trend-plus-season model reaches 16.1% MAPE, against 24.8% for repeating
the last value and 23.7% for the seasonal naive baseline. It does not capture the 2026 slowdown, so it is a
baseline, not a forecast to rely on. See `reports/forecast.png`.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# put the CSV files in data/raw/
python3 src/clean_market.py                                  # Phase 1 data
python3 src/forecast.py                                      # Phase 1 forecast
python3 src/clean_sales.py data/raw/<transactions_file>.csv  # Phase 2 data
python3 src/train_price_model.py                             # train and evaluate
streamlit run app/streamlit_app.py                           # price estimator app
```

## Structure

```
src/clean_market.py        quarterly market summary cleaning
src/forecast.py            Phase 1 forecasting
src/profile_transactions.py  chunked profile of the large transactions file
src/clean_sales.py         property-level cleaning
src/train_price_model.py   training, evaluation, charts
src/predict.py             load the model and estimate one price
app/streamlit_app.py       web app
reports/                   metrics, charts, cleaning log
```
