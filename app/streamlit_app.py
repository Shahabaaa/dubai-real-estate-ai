"""Dubai Property Price Estimator and Market Explorer (Streamlit).
Run from the project root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "app")])
import charts  # noqa: E402
from predict import MODEL_PATH, load_bundle, predict_price  # noqa: E402

DATA, REPORTS = ROOT / "app" / "data", ROOT / "reports"
st.set_page_config(page_title="Dubai Property Price Estimator", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#0b3c49 0%,#0e7c86 55%,#d4a24c 150%);padding:2rem 2.2rem;
      border-radius:18px;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.2rem;color:#fff}
.hero p{margin:.4rem 0 0;color:#e6f2f3;font-size:1.05rem}
.kpi{background:#f4f8f9;border:1px solid #dbe7ea;border-radius:14px;padding:.9rem 1.1rem}
.kpi .v{font-size:1.6rem;font-weight:700;color:#0b3c49}
.kpi .l{font-size:.78rem;color:#5b6f75;text-transform:uppercase;letter-spacing:.05em}
.price{background:linear-gradient(120deg,#0b3c49,#0e7c86);border-radius:16px;padding:1.3rem 1.6rem;margin-bottom:.8rem}
.price .l{color:#cfe6e8;font-size:.9rem}
.price .v{color:#fff;font-size:2.5rem;font-weight:800;line-height:1.15}
.price .s{color:#f1d9a6;font-size:.95rem}
</style>
<div class="hero"><h1>Dubai Property Price Estimator</h1>
<p>Machine-learning estimates and market insight built from hundreds of thousands of Dubai Land Department
sales records (2020 to 2026).</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def read_csv(name: str):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_summary():
    p = DATA / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python3 src/train_price_model.py` first.")
    st.stop()

bundle, stats, trend, summary = get_bundle(), read_csv("area_stats.csv"), read_csv("trend.csv"), read_summary()
ctx = read_csv("area_context.csv")
if stats is not None and stats.empty:
    stats = None

if summary:
    k = st.columns(4)
    kpi(k[0], f"{summary['n_sales']:,}", "Sales analysed")
    kpi(k[1], f"AED {summary['median_ppsm_12m']:,.0f}", "Median price per sq m, last 12 months")
    kpi(k[2], f"{summary['offplan_share_12m']:.0%}", "Off-plan share, last 12 months")
    kpi(k[3], f"{summary['n_areas']}", "Areas covered")
    st.write("")

tab_est, tab_mkt, tab_model = st.tabs(["💰 Price estimator", "📊 Market explorer", "🧪 About the model"])

# ---------------------------------------------------------------- Estimator
with tab_est:
    levels = bundle["levels"]
    areas = [a for a in levels["area"] if a not in ("Unknown", "Other")]
    names = {"Flat": "Apartment", "Villa": "Villa", "Stacked Townhouses": "Townhouse"}
    sub_types = [s for s in levels["sub_type"] if s in names]
    defaults = {"Flat": (80.0, 1), "Villa": (350.0, 4), "Stacked Townhouses": (200.0, 3)}   # size, bedrooms

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("Describe the property")
        kind = st.radio("Property type", [names[s] for s in sub_types], horizontal=True)
        sub_type = {v: k for k, v in names.items()}[kind]
        ctx_t = ctx[ctx["sub_type"] == sub_type] if ctx is not None else None
        area_options = [a for a in areas if ctx_t is not None and a in set(ctx_t["area"])] or areas
        default = area_options.index("Business Bay") if "Business Bay" in area_options else 0
        area = st.selectbox("Area", area_options, index=default, key=f"area_{sub_type}")
        extra, project_row = {}, None
        if ctx_t is not None:
            opts = ctx_t[ctx_t["area"] == area].sort_values("rank")
            if len(opts):
                labels = [f"{'Not specified' if r.project == 'Unknown' else r.project} ({int(r.n_sales)} recent sales)"
                          for r in opts.itertuples()]
                choice = st.selectbox("Project", labels, key=f"project_{sub_type}_{area}",
                                      help="Most active first. The project sets the building and the nearby metro "
                                           "and mall that the model sees.")
                project_row = opts.iloc[labels.index(choice)]
                extra = {c: project_row[c] for c in ["master_project", "project", "building", "nearest_metro",
                                                     "nearest_mall", "nearest_landmark"]}
            else:
                st.warning(f"No recent {kind.lower()} sales found in {area}, so this estimate is less reliable.")
        status = st.radio("Status", ["Off-plan", "Existing (ready)"], horizontal=True)
        d_size, d_rooms = defaults[sub_type]
        size_sqm = st.number_input("Size (sq m)", min_value=15.0, max_value=2000.0, value=d_size, step=5.0,
                                   key=f"size_{sub_type}")
        rooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5, 6, 7], index=d_rooms, key=f"rooms_{sub_type}",
                             format_func=lambda r: "Studio" if r == 0 else str(r))
        parking = st.checkbox("Includes parking", value=True)

    today = date.today()
    price = predict_price(bundle, area=area, property_type="Villa" if sub_type != "Flat" else "Unit",
                          sub_type=sub_type, area_sqm=size_sqm, rooms=rooms,
                          is_offplan=int(status == "Off-plan"), has_parking=int(parking),
                          year=today.year, month=today.month, **extra)
    ppsm = price / size_sqm

    with right:
        st.markdown(f'<div class="price"><div class="l">Estimated price</div>'
                    f'<div class="v">AED {price:,.0f}</div>'
                    f'<div class="s">Likely range AED {price * 0.9:,.0f} to {price * 1.1:,.0f} &nbsp;|&nbsp; '
                    f'AED {ppsm:,.0f} per sq m</div></div>', unsafe_allow_html=True)
        st.plotly_chart(charts.range_chart(price), width="stretch")
        if stats is not None and (stats["area"] == area).any():
            row = stats[stats["area"] == area].iloc[0]
            diff = ppsm / row["median_ppsm"] - 1
            st.plotly_chart(charts.ppsm_position(row, ppsm, area), width="stretch")
            st.caption(f"This estimate is {abs(diff):.0%} {'above' if diff > 0 else 'below'} the median of "
                       f"AED {row['median_ppsm']:,.0f} per sq m for {area} over the last 12 months.")
    st.info("On 2026 test data, about half of the model's estimates were within 10% of the actual registered "
            "price and about three in four within 20%. The model cannot see floor, view or condition, so use "
            "this as a starting point, not a valuation.")

# ------------------------------------------------------------------ Market
with tab_mkt:
    if stats is None or trend is None:
        st.warning("Market data files are missing. Run `python3 src/prepare_deploy.py`.")
    else:
        c1, c2 = st.columns([2, 1])
        metric = c1.radio("Compare areas by", ["Median price per sq m", "Median price"], horizontal=True)
        top_n = c2.slider("Number of areas", 5, 30, 15)
        st.plotly_chart(charts.area_bar(stats, metric, top_n), width="stretch")
        st.plotly_chart(charts.bubble(stats), width="stretch")
        ptype = st.radio("Property type", sorted(trend["property_type"].unique()), horizontal=True)
        t1, t2 = st.columns(2)
        t1.plotly_chart(charts.trend_line(trend, ptype), width="stretch")
        t2.plotly_chart(charts.volume_bar(trend, ptype), width="stretch")
        st.caption("Areas with at least 100 sales in the last 12 months. Quarterly charts exclude the "
                   "current, incomplete quarter.")

# ------------------------------------------------------------------- Model
with tab_model:
    st.subheader("How the model was evaluated")
    st.write("Trained on sales up to September 2025 and tested on every sale from 2026, so the test is a "
             "genuine look into the future. A simple baseline (recent median price per sq m in the same area, "
             "times size) is shown for comparison.")
    mfile = REPORTS / "model_metrics.csv"
    if mfile.exists():
        m = pd.read_csv(mfile, index_col=0)
        show = m[["MAE_AED", "MedianAPE_%", "within_10%", "within_20%", "R2", "n"]].rename(columns={
            "MAE_AED": "Mean error (AED)", "MedianAPE_%": "Median error (%)", "within_10%": "Within 10% (%)",
            "within_20%": "Within 20% (%)", "n": "Test sales"})
        st.dataframe(show, width="stretch")
    i1, i2 = st.columns(2)
    for col, fname, cap in [(i1, "pred_vs_actual.png", "Predicted vs actual price, 2026 test sales"),
                            (i2, "feature_importance.png", "What drives the estimate")]:
        if (REPORTS / fname).exists():
            col.image(str(REPORTS / fname), caption=cap, width="stretch")
    st.subheader("Why the model gives a particular price")
    st.write("Shapley-value attributions on 300 random 2026 sales: how much each feature moves the estimate "
             "compared with a typical recent sale.")
    e1, e2 = st.columns(2)
    for col, fname, cap in [(e1, "explain_global.png", "Average effect of each feature on the estimate"),
                            (e2, "explain_examples.png", "Three worked examples")]:
        if (REPORTS / fname).exists():
            col.image(str(REPORTS / fname), caption=cap, width="stretch")
    st.markdown("""
**Limitations**
- About one in four estimates is off by more than 20%.
- Prices are registered transaction values, not asking prices.
- The market changed during 2026, so the model should be retrained regularly.
- No floor, view, finish or condition information is available in the data.
""")
