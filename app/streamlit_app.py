"""Streamlit app: estimate the price of a Dubai apartment or villa.
Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from predict import MODEL_PATH, load_bundle, predict_price  # noqa: E402

st.set_page_config(page_title="Dubai Property Price Estimator", page_icon="🏙️")
st.title("Dubai Property Price Estimator")
st.caption("Machine-learning estimate trained on Dubai Land Department sales records (2020 to 2026).")


@st.cache_resource
def get_bundle():
    return load_bundle()


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python3 src/train_price_model.py` first.")
    st.stop()

bundle = get_bundle()
levels = bundle["levels"]
areas = [a for a in levels["area"] if a not in ("Unknown", "Other")]
sub_types = [s for s in levels["sub_type"] if s not in ("Unknown", "Other")]

c1, c2 = st.columns(2)
with c1:
    area = st.selectbox("Area", areas, index=areas.index("Business Bay") if "Business Bay" in areas else 0)
    sub_type = st.selectbox("Property type", sub_types,
                            index=sub_types.index("Flat") if "Flat" in sub_types else 0)
    status = st.radio("Status", ["Off-plan", "Existing (ready)"], horizontal=True)
with c2:
    size_sqm = st.number_input("Size (sq m)", min_value=15.0, max_value=2000.0, value=80.0, step=5.0)
    rooms = st.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5, 6, 7], index=1, format_func=lambda r: "Studio" if r == 0 else str(r))
    parking = st.checkbox("Includes parking", value=True)

today = date.today()
if st.button("Estimate price", type="primary"):
    property_type = "Villa" if sub_type in ("Villa", "Stacked Townhouses") else "Unit"
    price = predict_price(bundle, area=area, property_type=property_type, sub_type=sub_type,
                          area_sqm=size_sqm, rooms=rooms, is_offplan=int(status == "Off-plan"),
                          has_parking=int(parking), year=today.year, month=today.month)
    st.metric("Estimated price", f"AED {price:,.0f}")
    st.write(f"About AED {price / size_sqm:,.0f} per sq m.")
    st.info("On 2026 test data, about half of the model's estimates were within 10% of the actual "
            "registered price, and about three in four were within 20%. The model cannot see floor, "
            "view or condition, so treat this as a starting point, not a valuation.")
