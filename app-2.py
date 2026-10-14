import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Cook County Home Price Estimator", layout="centered")

# ---------------------------------------------------------------------------
# Reference labels.
# Property classes: short wording based on the Cook County Assessor's "Definitions
# for the Classifications of Real Property" (revised 12/16/24).
# Townships: code-to-name list from the legend of the Cook County Assessor's
# "CCAO Triennial Assessment Cycle" township map.
# ---------------------------------------------------------------------------
CLASS_LABELS = {
    "202": "One-story residence, up to 999 sq ft",
    "203": "One-story residence, 1,000 to 1,800 sq ft",
    "204": "One-story residence, 1,801 sq ft and over",
    "205": "Two-or-more story residence, over 62 years old, up to 2,200 sq ft",
    "206": "Two-or-more story residence, over 62 years old, 2,201 to 4,999 sq ft",
    "207": "Two-or-more story residence, up to 62 years old, up to 2,000 sq ft",
    "208": "Two-or-more story residence, up to 62 years old, 3,801 to 4,999 sq ft",
    "209": "Two-or-more story residence, any age, 5,000 sq ft and over",
    "210": "Old-style row house (townhome), over 62 years old",
    "234": "Split-level residence with lower level below grade, any age or size",
    "278": "Two-or-more story residence, up to 62 years old, 2,001 to 3,800 sq ft",
    "295": "Individually owned townhome or row house, up to 62 years old",
}

TOWNSHIP_NAMES = {
    "10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet",
    "15": "Cicero", "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont",
    "20": "Leyden", "21": "Lyons", "22": "Maine", "23": "New Trier", "24": "Niles",
    "25": "Northfield", "26": "Norwood Park", "27": "Oak Park", "28": "Orland", "29": "Palatine",
    "30": "Palos", "31": "Proviso", "32": "Rich", "33": "River Forest", "34": "Riverside",
    "35": "Schaumburg", "36": "Stickney", "37": "Thornton", "38": "Wheeling", "39": "Worth",
    "70": "Hyde Park", "71": "Jefferson", "72": "Lake", "73": "Lake View", "74": "North Chicago",
    "75": "Rogers Park", "76": "South Chicago", "77": "West Chicago",
}


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

with st.expander("Reference: township and property class codes"):
    st.write("**Property classes** (the Cook County Assessor's code for the type of home)")
    st.dataframe(
        pd.DataFrame({"Code": meta["classes"],
                      "What it means": [CLASS_LABELS.get(c, "Not listed") for c in meta["classes"]]}),
        hide_index=True, use_container_width=True)
    st.caption("Source: Cook County Assessor's Office, Definitions for the Classifications of Real Property "
               "(revised 12/16/24). The class is the one recorded at the time of sale.")
    st.write("**Townships** (the Assessor's code for each township)")
    st.dataframe(
        pd.DataFrame({"Code": meta["townships"],
                      "Township": [TOWNSHIP_NAMES.get(t, "Name not loaded") for t in meta["townships"]]}),
        hide_index=True, use_container_width=True)
    st.caption("Source: Cook County Assessor's Office, township legend of the Triennial Assessment Cycle map.")

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
