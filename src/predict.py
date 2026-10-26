"""Load the trained price model and predict the price of one property."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "price_model.joblib"
CATS = ["property_type", "sub_type", "area", "master_project", "project", "building",
        "nearest_metro", "nearest_mall", "nearest_landmark"]


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_price(bundle: dict, *, area: str, property_type: str, sub_type: str, area_sqm: float,
                  rooms: float, is_offplan: int, has_parking: int, year: int, month: int,
                  **extra_categories) -> float:
    """Return the estimated price in AED. Unspecified categorical inputs are treated as 'Unknown'."""
    row = {c: "Unknown" for c in CATS}
    row.update({"area": area, "property_type": property_type, "sub_type": sub_type}, **extra_categories)
    row.update({"area_sqm": area_sqm, "rooms": rooms, "is_offplan": is_offplan, "has_parking": has_parking,
                "t": (year - 2020) * 12 + month, "month": month})
    X = pd.DataFrame([row])[bundle["features"]]
    for c in CATS:
        levels = bundle["levels"][c]
        if "Other" in levels:                       # scikit-learn model: rare/unseen values -> "Other"
            X[c] = X[c].where(X[c].isin(levels), "Other")
        X[c] = pd.Categorical(X[c], categories=levels)
    return float(np.expm1(bundle["model"].predict(X))[0])
