import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Cook County Home Price Estimator", layout="centered")

# ---------------------------------------------------------------------------
# Optional friendly names. Fill these in from the Cook County Assessor's own
# documentation and confirm every entry against the source before you publish.
# Anything left out falls back to the raw code.
# ---------------------------------------------------------------------------
TOWNSHIP_NAMES = {}   # example format: {"70": "Township name"}
CLASS_LABELS = {}     # example format: {"203": "Description of property class"}


@st.cache_resource
def load_model():
    model = joblib.load(HERE / "price_model.joblib")
    meta = json.loads((HERE / "model_meta.json").read_text())
    return model, meta


model, meta = load_model()
rng, dflt = meta["ranges"], meta["defaults"]
SALE_YEAR, SALE_MONTH = meta["assumed_sale_year"], meta["assumed_sale_month"]


def township_label(code):
    name = TOWNSHIP_NAMES.get(code)
    return f"{name} ({code})" if name else f"Township {code}"


def class_label(code):
    desc = CLASS_LABELS.get(code)
    return f"{code}: {desc}" if desc else f"Class {code}"


# ---------------------------------------------------------------------------
# Page text. Replace with your group's own wording for the purpose and the
# intended user, since the assignment asks for those on the site.
# ---------------------------------------------------------------------------
st.title("Cook County Home Price Estimator")
st.write(
    "Enter a home's features to get an estimated sale price. "
    "Built for real estate agents who are pricing a listing."
)

with st.form("home_features"):
    c1, c2 = st.columns(2)
    with c1:
        building_sqft = st.number_input("Building size (sq ft)", 200, 20000,
                                        int(dflt["building_sqft"]), step=50)
        bedrooms = st.number_input("Bedrooms", 0, 15, int(dflt["bedrooms"]))
        half_bathrooms = st.number_input("Half bathrooms", 0, 6, int(dflt["half_bathrooms"]))
        year_built = st.number_input("Year built", 1800, SALE_YEAR, int(dflt["year_built"]))
    with c2:
        lot_sqft = st.number_input("Lot size (sq ft)", 500, 200000,
                                   int(dflt["lot_sqft"]), step=250)
        full_bathrooms = st.number_input("Full bathrooms", 0, 10, int(dflt["full_bathrooms"]))
        township = st.selectbox("Township", meta["townships"], format_func=township_label)
        prop_class = st.selectbox("Property class", meta["classes"], format_func=class_label)
    submitted = st.form_submit_button("Estimate price", type="primary")

if submitted:
    row = pd.DataFrame([{
        "building_sqft": building_sqft,
        "lot_sqft": lot_sqft,
        "bedrooms": bedrooms,
        "full_bathrooms": full_bathrooms,
        "half_bathrooms": half_bathrooms,
        "home_age_at_sale": SALE_YEAR - year_built,
        "sale_year": SALE_YEAR,
        "sale_month": SALE_MONTH,
        "township_code": str(township),
        "property_class": str(prop_class),
    }])
    pred = float(model.predict(row)[0])

    st.subheader("Estimate")
    if pred <= 0:
        st.error("These inputs fall outside what the model was trained on, so it can't produce a price. "
                 "Check the building size, lot size, and year built.")
    else:
        low, high = pred * meta["ratio_low"], pred * meta["ratio_high"]
        st.metric("Estimated sale price", f"${pred:,.0f}")
        st.write(f"Typical range: **${low:,.0f}** to **${high:,.0f}**")
        st.caption(
            "On homes the model had not seen before, about 8 in 10 actual sale prices "
            "fell inside a range like this around the estimate."
        )

    # Warn when an input is outside the data the model learned from
    checks = {
        "Building size": (building_sqft, rng["building_sqft"]),
        "Lot size": (lot_sqft, rng["lot_sqft"]),
        "Bedrooms": (bedrooms, rng["bedrooms"]),
        "Full bathrooms": (full_bathrooms, rng["full_bathrooms"]),
        "Half bathrooms": (half_bathrooms, rng["half_bathrooms"]),
        "Year built": (year_built, rng["year_built"]),
    }
    outside = [f"{k} ({v:,})" for k, (v, (lo, hi)) in checks.items() if v < lo or v > hi]
    if outside:
        st.warning("These inputs are outside the range of homes the model was trained on, "
                   "so treat the estimate with extra caution: " + ", ".join(outside) + ".")

with st.expander("About this tool and its limits"):
    st.write(
        f"- **Model:** {meta['model_name']}, trained on {meta['training_rows']:,} Cook County "
        f"home sales from {meta['data_years']}.\n"
        f"- **Accuracy on {meta['test_rows']:,} sales it had not seen:** average miss of "
        f"${meta['test_mae']:,.0f}, R² of {meta['test_r2']:.2f}.\n"
        f"- **Market timing:** estimates assume a sale in {SALE_MONTH}/{SALE_YEAR}, the most recent "
        "period in the training data.\n"
        "- **Not covered:** condominiums, homes with more than one building on the parcel, and "
        "homes outside the property classes the model was trained on.\n"
        "- **Not included in the model:** basement, garage, condition, finishes, and neighborhood "
        "within a township.\n"
        "- This is an estimate to support pricing, not an appraisal."
    )
