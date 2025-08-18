import io
import math
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Config & Constants
# =========================
APP_TITLE = "Profit Pulse — Phase 1 Diagnostic"
PASSWORD = "PP-PULSE"  # Keep in sync with your internal standard if needed

# Traffic-light thresholds
THR = {
    "gm_green": 0.30,
    "disc_reb_green": 0.08,
    "returns_green": 0.02,
    "data_green": 0.90,
}

# Large file guardrails
MAX_FILE_BYTES = 250 * 1024 * 1024  # 250 MB
MAX_ROWS = 1_000_000

st.set_page_config(page_title=APP_TITLE, layout="wide")

# Theme helpers
COLOR = {
    "green": "#1a9c5d",
    "amber": "#f2a700",
    "red": "#d13c3c",
    "blue": "#1f4b99",
    "slate": "#3c4758",
    "teal": "#008b8b",
}

# =========================
# UI: Header & About
# =========================
st.title(APP_TITLE)
st.caption("Private-use diagnostic. Operates on anonymized CSV data only. No PII.")

# =========================
# Password Gate
# =========================
with st.sidebar:
    st.subheader("Access")
    pw = st.text_input("Password", type="password")
    authed = pw == PASSWORD
    if pw and not authed:
        st.error("That’s not the correct password. I can’t proceed.")

if not authed:
    st.stop()

# =========================
# Mode Selection & Guidance
# =========================
st.sidebar.subheader("Diagnostic Mode")
mode = st.sidebar.radio(
    "Choose scope",
    ["Quick (Transactional only)", "Full (Transactional + optional masters)"],
    index=0,
)

st.markdown(
    """
**Expected CSV columns (anonymized IDs only):**

**Required (Quick):**
- `date` (YYYY-MM-DD)
- `product_id`
- `quantity` (numeric)
- `unit_price` (numeric)
- `unit_cost` (numeric)

**Optional (improves accuracy):**
- `customer_id`
- `discount` (per line, absolute)
- `rebate` (per line, absolute)
- `currency` (e.g., USD/EUR) — currency math is **flagged** unless clarified
- `returns_qty`, `returns_amount`

> Upload CSVs only to keep installs light. If your file is huge, we’ll auto-aggregate by `product_id × month` (and `customer_id × month` if present).
"""
)

# =========================
# File Uploaders
# =========================
st.subheader("Upload Data")
txn = st.file_uploader("Transactional CSV (required)", type=["csv"], key="txn")
price = None
cts = None
if mode.startswith("Full"):
    price = st.file_uploader("Optional: Price/Master CSV", type=["csv"], key="price")
    cts = st.file_uploader("Optional: Cost-to-Serve CSV", type=["csv"], key="cts")

period = st.radio("Rolling period", ["3 months", "6 months", "12 months", "Full dataset"], index=2)

# =========================
# Utility Functions
# =========================
def parse_period_filter(df, col="date"):
    if period == "Full dataset":
        return df
    months = int(period.split()[0])
    max_date = pd.to_datetime(df[col], errors="coerce").max()
    if pd.isna(max_date):
        return df
    cutoff = (max_date.to_period("M") - (months - 1)).to_timestamp()
    return df[df[col] >= cutoff]

def traffic(value, green_thresh, reverse=False):
    """
    Returns (label, color) where green >= thresh (or <= if reverse=True).
    """
    if pd.isna(value):
        return ("N/A", COLOR["amber"])
    if reverse:
        if value <= green_thresh:
            return ("Good", COLOR["green"])
        elif value <= green_thresh * 1.5:
            return ("Watch", COLOR["amber"])
        else:
            return ("Risk", COLOR["red"])
    else:
        if value >= green_thresh:
            return ("Good", COLOR["green"])
        elif value >= green_thresh * 0.7:
            return ("Watch", COLOR["amber"])
        else:
            return ("Risk", COLOR["red"])

def read_csv_safely(file, required_cols, sample_rows=10_000):
    """
    Try to read quickly. If file is very large or rows exceed threshold,
    return (df_or_agg, used_agg_mode: bool)
    """
    if file.size and file.size > MAX_FILE_BYTES:
        return aggregate_in_chunks(file, required_cols), True

    file.seek(0)
    sample = pd.read_csv(file, nrows=sample_rows, low_memory=False)

    total_rows_estimate = None
    if file.size:
        avg_row_size = max(1, int(file.size / max(1, len(sample))))
        total_rows_estimate = int(file.size / avg_row_size)

    if total_rows_estimate and total_rows_estimate > MAX_ROWS:
        file.seek(0)
        return aggregate_in_chunks(file, required_cols), True

    file.seek(0)
    df = pd.read_csv(file, low_memory=False)
    return df, False

def aggregate_in_chunks(file, required_cols):
    file.seek(0)
    chunks = pd.read_csv(file, chunksize=200_000, low_memory=False)
    agg = None

    for ch in chunks:
        ch_cols = {c.lower().strip(): c for c in ch.columns}
        ch.rename(columns={v: k for k, v in ch_cols.items()}, inplace=True)

        for col in ["date", "quantity", "unit_price", "unit_cost"]:
            if col not in ch.columns:
                ch[col] = np.nan

        ch["date"] = pd.to_datetime(ch.get("date"), errors="coerce")
        ch["month"] = ch["date"].dt.to_period("M").dt.to_timestamp()
        ch["discount"] = pd.to_numeric(ch.get("discount", 0), errors="coerce").fillna(0.0)
        ch["rebate"] = pd.to_numeric(ch.get("rebate", 0), errors="coerce").fillna(0.0)
        ch["quantity"] = pd.to_numeric(ch.get("quantity"), errors="coerce").fillna(0.0)
        ch["unit_price"] = pd.to_numeric(ch.get("unit_price"), errors="coerce").fillna(0.0)
        ch["unit_cost"] = pd.to_numeric(ch.get("unit_cost"), errors="coerce").fillna(0.0)

        ch["extended_price"] = ch["quantity"] * ch["unit_price"] - ch["discount"] - ch["rebate"]
        ch["cogs"] = ch["quantity"] * ch["unit_cost"]
        ch["gross_margin"] = ch["extended_price"] - ch["cogs"]

        group_keys = ["product_id", "month"]
        if "customer_id" in ch.columns:
            group_keys = ["customer_id"] + group_keys

        g = ch.groupby(group_keys, dropna=False).agg(
            quantity=("quantity", "sum"),
            revenue=("extended_price", "sum"),
            cogs=("cogs", "sum"),
            discount=("discount", "sum"),
            rebate=("rebate", "sum"),
            txn_rows=("unit_price", "count"),
        ).reset_index()

        g["gross_margin"] = g["revenue"] - g["cogs"]
        g["gm_pct"] = np.where(g["revenue"] != 0, g["gross_margin"] / g["revenue"], np.nan)

        agg = g if agg is None else pd.concat([agg, g], ignore_index=True)

    if agg is not None:
        by = [c for c in ["customer_id", "product_id", "month"] if c in agg.columns]
        agg = agg.groupby(by, dropna=False, as_index=False).sum(numeric_only=True)
        agg["gross_margin"] = agg["revenue"] - agg["cogs"]
        agg["gm_pct"] = np.where(agg["revenue"] != 0, agg["gross_margin"] / agg["revenue"], np.nan)

    return agg if agg is not None else pd.DataFrame()

def normalize_and_derive(df):
    cols = {c.lower().strip(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in cols.items()})

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    for num in ["quantity", "unit_price", "unit_cost", "discount", "rebate", "returns_qty", "returns_amount"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")

    df["discount"] = df.get("discount", 0).fillna(0.0)
    df["rebate"] = df.get("rebate", 0).fillna(0.0)
    if "returns_qty" in df.columns:
        df["returns_qty"] = df["returns_qty"].fillna(0.0)
    if "returns_amount" in df.columns:
        df["returns_amount"] = df["returns_amount"].fillna(0.0)

    df["extended_price"] = df["quantity"] * df["unit_price"] - df["discount"] - df["rebate"]
    df["cogs"] = df["quantity"] * df["unit_cost"]
    df["gross_margin"] = df["extended_price"] - df["cogs"]
    df["gm_pct"] = np.where(df["extended_price"] != 0, df["gross_margin"] / df["extended_price"], np.nan)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df

def data_completeness_score(df):
    required = ["date", "product_id", "quantity", "unit_price", "unit_cost"]
    available = df[required]
    completeness = 1.0 - (available.isna().sum().sum() / available.size)
    return max(0.0, min(1.0, completeness))

def remediation_tier(missing_flags):
    score = 0
    for k, v in missing_flags.items():
        score += 2 if v else 0
    if score >= 6:
        return "C"
    if score >= 3:
        return "B"
    return "A"

def render_kpis(summary):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${summary['revenue']:,.0f}")
    c2.metric("Gross Margin", f"${summary['gross_margin']:,.0f}")
    gm_label, gm_color = traffic(summary["gm_pct"], THR["gm_green"])
    c3.markdown(f"<div style='color:{gm_color};font-weight:600'>GM%: {summary['gm_pct']:.1%} · {gm_label}</div>", unsafe_allow_html=True)

    disc_reb = summary.get("disc_reb_pct")
    lbl, col = traffic(disc_reb, THR["disc_reb_green"], reverse=True)
    c4.markdown(f"<div style='color:{col};font-weight:600'>Discount+Rebate: {disc_reb:.1%} · {lbl}</div>", unsafe_allow_html=True)

# =========================
# Processing
# =========================
if txn is None:
    st.info("Upload transactional CSV to begin.")
    st.stop()

required_cols = ["date", "product_id", "quantity", "unit_price", "unit_cost"]

with st.spinner("Reading data..."):
    df, aggregated = read_csv_safely(txn, required_cols)

if df.empty:
    st.error("The uploaded file appears to be empty or unreadable.")
    st.stop()

if aggregated:
    st.warning("Large file detected. Switched to aggregation mode (by month × product, and customer if present).")

with st.spinner("Normalizing & deriving metrics..."):
    df = normalize_and_derive(df)
    df = parse_period_filter(df)

# =========================
# Validation & Remediation
# =========================
missing_flags = {
    "date_missing": df["date"].isna().any(),
    "product_id_missing": "product_id" not in df.columns or df["product_id"].isna().any(),
    "qty_missing": df["quantity"].isna().any(),
    "price_missing": df["unit_price"].isna().any(),
    "cost_missing": df["unit_cost"].isna().any(),
    "currency_unhandled": "currency" in df.columns,
}
tier = remediation_tier(missing_flags)
data_score = data_completeness_score(df)

st.subheader("Data Remediation Plan")
st.write(f"**Tier {tier}** — data completeness: **{data_score:.0%}**")
issues = []
if missing_flags["date_missing"]: issues.append("Some `date` values are missing or invalid.")
if missing_flags["product_id_missing"]: issues.append("`product_id` missing or contains nulls.")
if missing_flags["qty_missing"]: issues.append("Some `quantity` values are missing/invalid.")
if missing_flags["price_missing"]: issues.append("Some `unit_price` values are missing/invalid.")
if missing_flags["cost_missing"]: issues.append("Some `unit_cost` values are missing/invalid.")
if missing_flags["currency_unhandled"]: issues.append("`currency` present — multi-currency not normalized (flagged).")

if issues:
    st.error("• " + "\n• ".join(issues))
else:
    st.success("No critical structural issues detected in required fields.")

st.caption("Next steps: supply missing fields; confirm currency handling; provide returns/discount/rebate detail if available.")

# =========================
# Dashboard Readout
# =========================
st.subheader("Dashboard")

summary = {}
summary["revenue"] = float(df["extended_price"].sum())
summary["cogs"] = float(df["cogs"].sum())
summary["gross_margin"] = summary["revenue"] - summary["cogs"]
summary["gm_pct"] = (summary["gross_margin"] / summary["revenue"]) if summary["revenue"] else np.nan
disc_reb_total = float(df["discount"].sum() + df["rebate"].sum())
summary["disc_reb_pct"] = (disc_reb_total / (summary["revenue"] + disc_reb_total)) if (summary["revenue"] + disc_reb_total) else np.nan
render_kpis(summary)

st.markdown("**Traffic-light checks**")
checks = pd.DataFrame([
    {
        "Metric": "GM%",
        "Value": f"{summary['gm_pct']:.1%}" if not pd.isna(summary["gm_pct"]) else "N/A",
        "Status": traffic(summary["gm_pct"], THR["gm_green"])[0],
    },
    {
        "Metric": "Discount+Rebate %",
        "Value": f"{summary['disc_reb_pct']:.1%}" if not pd.isna(summary["disc_reb_pct"]) else "N/A",
        "Status": traffic(summary["disc_reb_pct"], THR["disc_reb_green"], reverse=True)[0],
    },
    {
        "Metric": "Returns %",
        "Value": (
            f"{(df['returns_amount'].sum() / summary['revenue']):.1%}"
            if "returns_amount" in df.columns and summary["revenue"]
            else "N/A"
        ),
        "Status": (
            traffic((df["returns_amount"].sum() / summary["revenue"]), THR["returns_green"], reverse=True)[0]
            if "returns_amount" in df.columns and summary["revenue"]
            else "N/A"
        ),
    },
    {
        "Metric": "Data completeness",
        "Value": f"{data_score:.0%}",
        "Status": traffic(data_score, THR["data_green"])[0],
    },
])
st.dataframe(checks, use_container_width=True)


st.markdown("**GM% by Month**")
gm_by_m = df.groupby("month", as_index=False).agg(revenue=("extended_price", "sum"), cogs=("cogs", "sum"))
gm_by_m["gm_pct"] = np.where(gm_by_m["revenue"] != 0, (gm_by_m["revenue"] - gm_by_m["cogs"]) / gm_by_m["revenue"], np.nan)

chart = alt.Chart(gm_by_m).mark_line(point=True, color=COLOR["blue"]).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("gm_pct:Q", title="GM%", axis=alt.Axis(format="%")),
    tooltip=["month", alt.Tooltip("gm_pct:Q", format=".1%"), alt.Tooltip("revenue:Q", format="$.2s")]
).properties(height=280)
st.altair_chart(chart, use_container_width=True)

st.markdown("**Top/Bottom Products by GM%**")
if "product_id" in df.columns:
    prod = df.groupby("product_id", as_index=False).agg(revenue=("extended_price", "sum"), cogs=("cogs", "sum"))
    prod["gm_pct"] = np.where(prod["revenue"] != 0, (prod["revenue"] - prod["cogs"]) / prod["revenue"], np.nan)
    top5 = prod.sort_values("gm_pct", ascending=False).head(5)
    bot5 = prod.sort_values("gm_pct", ascending=True).head(5)
    c1, c2 = st.columns(2)
    c1.write("**Top 5**"); c1.dataframe(top5, use_container_width=True)
    c2.write("**Bottom 5**"); c2.dataframe(bot5, use_container_width=True)

st.subheader("Briefing Deck (Summary)")
bullets = [
    f"GM% over selected period: {summary['gm_pct']:.1%}" if not pd.isna(summary['gm_pct']) else "GM% not computable.",
    f"Discount+Rebate load: {summary['disc_reb_pct']:.1%}" if not pd.isna(summary['disc_reb_pct']) else "Discount/Rebate % not available.",
    f"Data completeness: {data_score:.0%} (Tier {tier}).",
    "Large-file aggregation was applied." if aggregated else "Full-grain data used.",
]
st.write("• " + "\n• ".join(bullets))

st.subheader("Opportunities & Risks")
opp = []
risk = []
if not pd.isna(summary["disc_reb_pct"]) and summary["disc_reb_pct"] > THR["disc_reb_green"]:
    opp.append("Tighten discount/rebate policies; target ≤ 8% of pre-discount revenue.")
if not pd.isna(summary["gm_pct"]) and summary["gm_pct"] < THR["gm_green"]:
    opp.append("Review pricing and cost-to-serve on low-GM% products; consider price floors.")
if "returns_amount" in df.columns and summary["revenue"]:
    ret_pct = df["returns_amount"].sum() / summary["revenue"]
    if ret_pct > THR["returns_green"]:
        opp.append("Reduce returns via quality/fit improvements; audit high-return SKUs.")
if data_score < THR["data_green"]:
    risk.append("Data gaps may distort margin metrics; prioritize remediation before decisions.")
if missing_flags["currency_unhandled"]:
    risk.append("Multi-currency present without normalization; GM% may be skewed.")

st.write("**Top 3–5 Levers**")
st.write("• " + "\n• ".join(opp[:5]) if opp else "No obvious levers beyond standard hygiene.")
st.write("**Key Risks**")
st.write("• " + "\n• ".join(risk[:5]) if risk else "No major risks detected.")

st.subheader("Questions for First Client Call")
qs = [
    "How are discounts and rebates recorded (per-line vs. end-of-period)?",
    "Any multi-currency transactions? If so, what FX policy should be applied?",
    "Are returns netted in revenue or recorded separately?",
    "Any non-product revenue or freight in unit_price/unit_cost?",
    "Do we have cost-to-serve or fulfillment cost drivers for Phase 2?",
]
st.write("• " + "\n• ".join(qs))

st.subheader("Assumptions & Formulas")
st.markdown(
    """
**Assumptions**
- Anonymized IDs only; PII excluded.
- Currency not normalized unless explicitly stated.
- Returns excluded unless provided as `returns_amount` or `returns_qty`.
- Aggregation mode uses `product_id × month` (and `customer_id` if present).

**Formulas**
- `extended_price = quantity * unit_price − discount − rebate`
- `cogs = quantity * unit_cost`
- `gross_margin = extended_price − cogs`
- `gm% = gross_margin / extended_price`
- `discount+rebate % = (discount + rebate) / (extended_price + discount + rebate)`
"""
)
