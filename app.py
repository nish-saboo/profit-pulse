import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Config & Constants
# =========================
APP_TITLE = "Profit Pulse — Phase 1 Diagnostic"
PASSWORD = "PP-PULSE"  # Must match your internal standard

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

PII_PATTERNS = ("name", "email", "phone", "mobile", "address", "tax", "ssn")

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
- `currency` (e.g., USD/EUR) — currency math is **flagged** unless acknowledged
- `returns_qty`, `returns_amount`

> Upload CSVs only to keep installs light. If your file is huge, we’ll auto-aggregate by `product_id × month` (and `customer_id × month` if present).
"""
)

# CSV templates for easy onboarding
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

tmpl_txn = pd.DataFrame({
    "date": ["2024-01-15", "2024-01-20"],
    "product_id": ["P001", "P002"],
    "customer_id": ["C001", "C002"],
    "quantity": [10, 5],
    "unit_price": [25.0, 40.0],
    "unit_cost": [12.0, 22.0],
    "discount": [5.0, 0.0],
    "rebate": [0.0, 2.0],
})
tmpl_prod = pd.DataFrame({
    "product_id": ["P001", "P002"],
    "category_code": ["CAT-A", "CAT-B"],
    "brand_code": ["BR-1", "BR-2"],
    "size_tier": ["S", "M"],
})
tmpl_cust = pd.DataFrame({
    "customer_id": ["C001", "C002"],
    "segment_code": ["SMB", "ENT"],
    "region_code": ["NA", "EU"],
    "channel_code": ["DTC", "RESELLER"],
})

c_t1, c_t2, c_t3 = st.columns(3)
with c_t1:
    st.download_button("Download Transactional Template", data=df_to_csv_bytes(tmpl_txn), file_name="transactional_template.csv", mime="text/csv")
with c_t2:
    st.download_button("Download Product Master Template", data=df_to_csv_bytes(tmpl_prod), file_name="product_master_template.csv", mime="text/csv")
with c_t3:
    st.download_button("Download Customer Master Template", data=df_to_csv_bytes(tmpl_cust), file_name="customer_master_template.csv", mime="text/csv")

# =========================
# File Uploaders
# =========================
st.subheader("Upload Data")
txn = st.file_uploader("Transactional CSV (required)", type=["csv"], key="txn")

price = None
cts = None
prod_master = None
cust_master = None
if mode.startswith("Full"):
    price = st.file_uploader("Optional: Price/Master CSV", type=["csv"], key="price")
    cts = st.file_uploader("Optional: Cost-to-Serve CSV", type=["csv"], key="cts")
    prod_master = st.file_uploader("Optional: Product Master CSV", type=["csv"], key="prod_master")
    cust_master = st.file_uploader("Optional: Customer Master CSV", type=["csv"], key="cust_master")

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

    # Peek a small sample to detect columns cheaply
    file.seek(0)
    try:
        sample = pd.read_csv(file, nrows=sample_rows, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV sample: {e}")

    # Estimate total rows if file size known
    total_rows_estimate = None
    if file.size and len(sample) > 0:
        avg_row_size = max(1, int(file.size / max(1, len(sample))))
        total_rows_estimate = int(file.size / avg_row_size)

    # If too big, aggregate
    if total_rows_estimate and total_rows_estimate > MAX_ROWS:
        file.seek(0)
        return aggregate_in_chunks(file, required_cols), True

    # Not too big: read all
    file.seek(0)
    df = pd.read_csv(file, low_memory=False)
    return df, False

def aggregate_in_chunks(file, required_cols):
    """
    Chunked aggregation by product_id×month (and customer_id×month if available).
    """
    file.seek(0)
    chunks = pd.read_csv(file, chunksize=200_000, low_memory=False)
    agg = None

    for ch in chunks:
        ch_cols = {c.lower().strip(): c for c in ch.columns}
        ch.rename(columns={v: k for k, v in ch_cols.items()}, inplace=True)

        # Derivations
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

    # Combine duplicate keys after concatenation
    if agg is not None:
        by = [c for c in ["customer_id", "product_id", "month"] if c in agg.columns]
        agg = agg.groupby(by, dropna=False, as_index=False).sum(numeric_only=True)
        agg["gross_margin"] = agg["revenue"] - agg["cogs"]
        agg["gm_pct"] = np.where(agg["revenue"] != 0, agg["gross_margin"] / agg["revenue"], np.nan)

    return agg if agg is not None else pd.DataFrame()

def normalize_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names to lower snake
    cols = {c.lower().strip(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in cols.items()})

    # Parse types
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    numeric_cols = [
        "quantity", "unit_price", "unit_cost",
        "discount", "rebate",
        "returns_qty", "returns_amount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure discount/rebate columns always exist as numeric Series
    df["discount"] = df["discount"].fillna(0.0) if "discount" in df.columns else 0.0
    df["rebate"]   = df["rebate"].fillna(0.0)   if "rebate" in df.columns else 0.0

    # Returns columns optional, ensure numeric fallback
    df["returns_qty"] = df["returns_qty"].fillna(0.0) if "returns_qty" in df.columns else 0.0
    df["returns_amount"] = df["returns_amount"].fillna(0.0) if "returns_amount" in df.columns else 0.0

    # Derivations
    for base_col in ["quantity", "unit_price", "unit_cost"]:
        if base_col not in df.columns:
            df[base_col] = 0.0
        else:
            df[base_col] = df[base_col].fillna(0.0)

    df["extended_price"] = df["quantity"] * df["unit_price"] - df["discount"] - df["rebate"]
    df["cogs"] = df["quantity"] * df["unit_cost"]
    df["gross_margin"] = df["extended_price"] - df["cogs"]
    df["gm_pct"] = np.where(df["extended_price"] != 0, df["gross_margin"] / df["extended_price"], np.nan)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df

def data_completeness_score(df):
    required = ["date", "product_id", "quantity", "unit_price", "unit_cost"]
    avail = df[[c for c in required if c in df.columns]]
    if avail.empty:
        return 0.0
    completeness = 1.0 - (avail.isna().sum().sum() / (len(avail.columns) * len(avail)))
    return float(max(0.0, min(1.0, completeness)))

def remediation_tier(missing_flags):
    # A=minor, B=moderate, C=major issues
    score = 0
    for v in missing_flags.values():
        score += 2 if v else 0
    if score >= 6: return "C"
    if score >= 3: return "B"
    return "A"

def read_master_csv(file, key_cols, allowed_extra=None):
    """
    Reads a master CSV, normalizes columns, drops PII-like columns,
    and returns only key + allowed attributes.
    """
    if file is None:
        return None
    file.seek(0)
    m = pd.read_csv(file, low_memory=False)

    # normalize col names
    m.columns = [c.lower().strip() for c in m.columns]

    # drop PII-like columns
    drop_cols = [c for c in m.columns if any(pat in c for pat in PII_PATTERNS)]
    m = m.drop(columns=drop_cols, errors="ignore")

    # ensure key columns present
    missing_keys = [k for k in key_cols if k not in m.columns]
    if missing_keys:
        st.warning(f"Master file is missing key columns: {missing_keys}. Skipping.")
        return None

    # keep only keys + allowed extras
    cols_to_keep = list(key_cols)
    if allowed_extra:
        cols_to_keep += [c for c in allowed_extra if c in m.columns]
    m = m[cols_to_keep].drop_duplicates()

    return m

def scan_for_pii_columns(columns):
    flagged = [c for c in columns if any(pat in c for pat in PII_PATTERNS)]
    if flagged:
        st.warning(f"Potential PII-like column names detected and will be ignored where applicable: {flagged}")

def generate_rule_based_insights(df, summary, thresholds):
    insights = []

    # Overall GM%
    gm = summary.get("gm_pct", np.nan)
    if not pd.isna(gm):
        if gm < thresholds["gm_green"]:
            insights.append(f"GM% at {gm:.1%} is below the {thresholds['gm_green']:.0%} target — check price floors and unit costs.")
        else:
            insights.append(f"GM% at {gm:.1%} meets/exceeds target — maintain pricing discipline.")

    # Discounts + rebates
    dr = summary.get("disc_reb_pct", np.nan)
    if not pd.isna(dr):
        if dr > thresholds["disc_reb_green"]:
            insights.append(f"Discount+Rebate load {dr:.1%} exceeds target {thresholds['disc_reb_green']:.0%} — tighten exceptions and approvals.")
        else:
            insights.append(f"Discount+Rebate load {dr:.1%} is within tolerance.")

    # Returns
    if "returns_amount" in df.columns and summary["revenue"]:
        ret_pct = df["returns_amount"].sum() / summary["revenue"]
        if ret_pct > thresholds["returns_green"]:
            insights.append(f"Returns at {ret_pct:.1%} are elevated — audit top-return SKUs and causes.")
        else:
            insights.append(f"Returns at {ret_pct:.1%} look manageable.")

    # Product outliers
    if "product_id" in df.columns:
        prod = df.groupby("product_id", as_index=False).agg(
            revenue=("extended_price", "sum"),
            cogs=("cogs", "sum"),
            qty=("quantity", "sum")
        )
        prod["gm_pct"] = np.where(prod["revenue"] != 0, (prod["revenue"] - prod["cogs"]) / prod["revenue"], np.nan)
        high_rev_low_gm = prod[(prod["revenue"] > prod["revenue"].median()) & (prod["gm_pct"] < prod["gm_pct"].median())]
        if not high_rev_low_gm.empty:
            insights.append(f"{len(high_rev_low_gm)} high-revenue SKUs have below-median GM% — candidates for price review or cost reduction.")

    # Customer concentration (if present)
    if "customer_id" in df.columns:
        cust = df.groupby("customer_id", as_index=False)["extended_price"].sum().sort_values("extended_price", ascending=False)
        top10_share = cust["extended_price"].head(10).sum() / max(1e-9, cust["extended_price"].sum())
        insights.append(f"Top 10 customers contribute {top10_share:.1%} of revenue — monitor concentration risk.")

    # Data quality
    if "currency" in df.columns:
        insights.append("Multi-currency detected — FX not normalized; compare GM% by currency or confirm policy.")

    return insights[:8]

def pvm_bridge(df):
    m = df.groupby("month", as_index=False).agg(
        revenue=("extended_price", "sum"),
        qty=("quantity", "sum")
    ).sort_values("month")
    if len(m) < 2:
        return None, None

    # Compare last vs prior month
    cur, prev = m.iloc[-1], m.iloc[-2]
    # Average prices
    p_prev = prev["revenue"] / prev["qty"] if prev["qty"] else 0.0
    p_cur  = cur["revenue"] / cur["qty"] if cur["qty"] else 0.0

    # Effects
    volume_eff = (cur["qty"] - prev["qty"]) * (p_prev)
    price_eff  = (p_cur - p_prev) * (cur["qty"])
    mix_eff    = (cur["revenue"] - prev["revenue"]) - volume_eff - price_eff
    total_delta = cur["revenue"] - prev["revenue"]

    steps = pd.DataFrame({
        "label": ["Prev Rev", "Volume", "Price", "Mix", "Cur Rev"],
        "value": [prev["revenue"], volume_eff, price_eff, mix_eff, cur["revenue"]],
        "type": ["base", "posneg", "posneg", "posneg", "base"]
    })
    return steps, total_delta

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

# Column mapping UI if required columns are missing / differently named
st.subheader("Schema Check & Column Mapping")
df.columns = [c.lower().strip() for c in df.columns]
scan_for_pii_columns(df.columns)

missing_reqs = [c for c in required_cols if c not in df.columns]
if missing_reqs:
    st.warning(f"Missing required columns: {missing_reqs}. Map them below if your file uses different names.")
    with st.expander("Map your columns", expanded=True):
        col_map = {}
        for req in required_cols:
            options = ["-- none --"] + list(df.columns)
            sel = st.selectbox(f"Map **{req}** to:", options, index=options.index(req) if req in df.columns else 0, key=f"map_{req}")
            if sel != "-- none --":
                col_map[req] = sel

        if st.button("Apply Mapping"):
            # Invert collisions by renaming current reqs to temp
            for req, src in col_map.items():
                if req in df.columns and req != src:
                    df.rename(columns={req: f"__old_{req}__"}, inplace=True)
            # Now rename selected sources to required names
            df.rename(columns={src: req for req, src in col_map.items()}, inplace=True)
            st.experimental_rerun()

# Re-check requireds
still_missing = [c for c in required_cols if c not in df.columns]
if still_missing:
    st.error(f"Still missing required columns after mapping: {still_missing}. Please fix your file and re-upload.")
    st.stop()

if aggregated:
    st.warning("Large file detected. Switched to aggregation mode (by month × product, and customer if present).")

with st.spinner("Normalizing & deriving metrics..."):
    df = normalize_and_derive(df)
    df = parse_period_filter(df)

# ---- Merge Product & Customer Masters (optional) ----
if prod_master is not None:
    pm_allowed = [
        "category_code", "family_code", "brand_code",
        "line_code", "size_tier", "uom", "launch_year"
    ]
    pm = read_master_csv(prod_master, key_cols=["product_id"], allowed_extra=pm_allowed)
    if pm is not None:
        try:
            df = df.merge(pm, on="product_id", how="left", validate="m:1")
            st.success("Product Master merged (keys: product_id).")
        except Exception as e:
            st.warning(f"Could not merge Product Master: {e}")

if cust_master is not None:
    cm_allowed = [
        "segment_code", "region_code", "channel_code",
        "tier_code", "size_tier"
    ]
    cm = read_master_csv(cust_master, key_cols=["customer_id"], allowed_extra=cm_allowed)
    if cm is not None:
        if "customer_id" not in df.columns:
            st.warning("Transactional file lacks `customer_id`; Customer Master skipped.")
        else:
            try:
                df = df.merge(cm, on="customer_id", how="left", validate="m:1")
                st.success("Customer Master merged (keys: customer_id).")
            except Exception as e:
                st.warning(f"Could not merge Customer Master: {e}")

# Optional filters if attributes added
with st.sidebar.expander("Filters (optional)", expanded=False):
    if "category_code" in df.columns:
        cats = sorted([c for c in df["category_code"].dropna().unique()])
        sel = st.multiselect("Category", cats)
        if sel:
            df = df[df["category_code"].isin(sel)]
    if "segment_code" in df.columns:
        segs = sorted([c for c in df["segment_code"].dropna().unique()])
        sel2 = st.multiselect("Customer Segment", segs)
        if sel2:
            df = df[df["segment_code"].isin(sel2)]

# =========================
# Validation & Remediation
# =========================
missing_flags = {
    "date_missing": df["date"].isna().any(),
    "product_id_missing": "product_id" not in df.columns or df["product_id"].isna().any(),
    "qty_missing": df["quantity"].isna().any(),
    "price_missing": df["unit_price"].isna().any(),
    "cost_missing": df["unit_cost"].isna().any(),
    "currency_unhandled": "currency" in df.columns,  # present but unnormalized
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

# KPIs
summary = {}
summary["revenue"] = float(df["extended_price"].sum())
summary["cogs"] = float(df["cogs"].sum())
summary["gross_margin"] = summary["revenue"] - summary["cogs"]
summary["gm_pct"] = (summary["gross_margin"] / summary["revenue"]) if summary["revenue"] else np.nan
disc_reb_total = float(df["discount"].sum() + df["rebate"].sum())
summary["disc_reb_pct"] = (disc_reb_total / (summary["revenue"] + disc_reb_total)) if (summary["revenue"] + disc_reb_total) else np.nan

# Currency acknowledgement if present
if "currency" in df.columns:
    st.info("⚠️ Currency column detected. FX normalization is not applied. Confirm if values are comparable across currencies.")
render_cols = st.columns(4)
with render_cols[0]:
    st.metric("Revenue", f"${summary['revenue']:,.0f}")
with render_cols[1]:
    st.metric("Gross Margin", f"${summary['gross_margin']:,.0f}")
with render_cols[2]:
    if not pd.isna(summary["gm_pct"]):
        st.metric("GM%", f"{summary['gm_pct']:.1%}")
    else:
        st.metric("GM%", "N/A")
with render_cols[3]:
    if not pd.isna(summary["disc_reb_pct"]):
        st.metric("Discount+Rebate %", f"{summary['disc_reb_pct']:.1%}")
    else:
        st.metric("Discount+Rebate %", "N/A")

# Traffic-light table
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

# Charts
st.markdown("**GM% by Month**")
gm_by_m = df.groupby("month", as_index=False).agg(
    revenue=("extended_price", "sum"),
    cogs=("cogs", "sum"),
)
gm_by_m["gm_pct"] = np.where(gm_by_m["revenue"] != 0, (gm_by_m["revenue"] - gm_by_m["cogs"]) / gm_by_m["revenue"], np.nan)
chart = alt.Chart(gm_by_m).mark_line(point=True, color=COLOR["blue"]).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("gm_pct:Q", title="GM%", axis=alt.Axis(format="%")),
    tooltip=["month", alt.Tooltip("gm_pct:Q", format=".1%"), alt.Tooltip("revenue:Q", format="$.2s")]
).properties(height=280)
st.altair_chart(chart, use_container_width=True)

st.markdown("**Top/Bottom Products by GM%**")
if "product_id" in df.columns:
    prod = df.groupby("product_id", as_index=False).agg(
        revenue=("extended_price", "sum"),
        cogs=("cogs", "sum")
    )
    prod["gm_pct"] = np.where(prod["revenue"] != 0, (prod["revenue"] - prod["cogs"]) / prod["revenue"], np.nan)
    top5 = prod.sort_values("gm_pct", ascending=False).head(5)
    bot5 = prod.sort_values("gm_pct", ascending=True).head(5)
    c1, c2 = st.columns(2)
    c1.write("**Top 5**"); c1.dataframe(top5, use_container_width=True)
    c2.write("**Bottom 5**"); c2.dataframe(bot5, use_container_width=True)

# =========================
# Automated Insights
# =========================
st.subheader("Automated Insights (Local)")
rb_insights = generate_rule_based_insights(df, summary, THR)
st.write("• " + "\n• ".join(rb_insights) if rb_insights else "No notable rule-based findings from current data.")

# Revenue Bridge (P–V–M)
steps, total_delta = pvm_bridge(df)
if steps is not None:
    st.subheader("Revenue Bridge (MoM P–V–M)")
    # Build a simple waterfall using start+height bars
    base_start = steps.iloc[0]["value"]
    wf = []
    cur = base_start
    for _, row in steps.iterrows():
        if row["type"] == "posneg":
            start = cur
            end = cur + row["value"]
            wf.append({"label": row["label"], "start": min(start, end), "height": abs(row["value"]), "color": COLOR["green"] if row["value"] >= 0 else COLOR["red"]})
            cur = end
        else:
            wf.append({"label": row["label"], "start": row["value"], "height": 0.0, "color": COLOR["blue"]})
    wf_df = pd.DataFrame(wf)
    chart = alt.Chart(wf_df).mark_bar().encode(
        x=alt.X("label:N", title=None),
        y=alt.Y("start:Q", title="Revenue"),
        y2="y2:Q",
        color=alt.Color("color:N", scale=None)
    ).transform_calculate(
        y2="datum.start + datum.height"
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Δ Revenue vs prior month: {total_delta:,.0f}")

# Quality & Outliers
st.subheader("Quality & Outliers")
flags = []
neg_gm = df[df["gross_margin"] < 0]
if not neg_gm.empty:
    flags.append(f"{len(neg_gm)} rows with negative gross margin (check costs/prices).")
high_disc = df[(df["discount"] + df["rebate"]) > (0.25 * (df["quantity"] * df["unit_price"]).replace(0, np.nan))]
if not high_disc.empty:
    flags.append(f"{len(high_disc)} rows where discounts+rebates exceed 25% of list value.")
gm_by_m2 = gm_by_m.copy()
if len(gm_by_m2) >= 3:
    gm_last = gm_by_m2["gm_pct"].iloc[-1]
    gm_prev = gm_by_m2["gm_pct"].iloc[-2]
    if pd.notna(gm_last) and pd.notna(gm_prev) and abs(gm_last - gm_prev) > 0.05:
        flags.append(f"GM% moved {gm_last - gm_prev:+.1%} month-over-month — investigate drivers (price, mix, cost).")
st.write("• " + "\n• ".join(flags) if flags else "No outlier flags found with current thresholds.")

# =========================
# Briefing Deck (Bullets)
# =========================
st.subheader("Briefing Deck (Summary)")
bullets = [
    f"GM% over selected period: {summary['gm_pct']:.1%}" if not pd.isna(summary['gm_pct']) else "GM% not computable due to missing revenue.",
    f"Discount+Rebate load: {summary['disc_reb_pct']:.1%}" if not pd.isna(summary['disc_reb_pct']) else "Discount/Rebate % not available.",
    f"Data completeness: {data_score:.0%} (Tier {tier}).",
    "Large-file aggregation was applied." if aggregated else "Full-grain data used.",
]
st.write("• " + "\n• ".join(bullets))

# =========================
# AI Narration (ChatGPT) — optional
# =========================
with st.expander("AI Narration (ChatGPT) — optional", expanded=False):
    enable_ai = st.checkbox("Enable ChatGPT narration (aggregated data only)", value=False)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0) if enable_ai else None

    default_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    api_key = st.text_input("OpenAI API Key (kept local; not stored)", type="password", value=default_key or "") if enable_ai else None

    include_samples = st.checkbox(
        "Include a few aggregated examples (never raw PII/rows)",
        value=False
    ) if enable_ai else False

    if enable_ai:
        if not api_key:
            st.info("Enter your OpenAI API key (or set st.secrets['OPENAI_API_KEY']).")
        else:
            # Build prompt from aggregates + rule-based insights
            prompt = f"""
You are a pricing and profitability analyst. Given aggregate metrics below, produce:
1) A 5–8 bullet executive narrative.
2) Top 3 actions with sizing logic (directional is fine).
3) 3–5 risks and watchouts.
4) 3 follow-up questions to ask the client.

Metrics:
- Revenue: ${summary.get('revenue', 0):,.0f}
- Gross Margin: ${summary.get('gross_margin', 0):,.0f}
- GM%: {summary.get('gm_pct')}
- Discount+Rebate %: {summary.get('disc_reb_pct')}
- Data completeness: {data_score:.0%}

Local rule-based notes:
- {chr(10).join(rb_insights or [])}

Instructions:
- Do not invent data.
- Keep language concise and actionable.
- Avoid PII or specific names; refer to anonymized groups only.
""".strip()

            # Optional mini context of top products (aggregated, anonymized)
            extra_context = None
            if include_samples and "product_id" in df.columns:
                prod = df.groupby("product_id", as_index=False).agg(
                    revenue=("extended_price", "sum"),
                    cogs=("cogs", "sum")
                )
                prod["gm_pct"] = np.where(prod["revenue"] != 0, (prod["revenue"] - prod["cogs"]) / prod["revenue"], np.nan)
                tiny = prod.sort_values("revenue", ascending=False).head(5)[["product_id", "revenue", "gm_pct"]]
                extra_context = tiny.to_dict(orient="records")

            try:
                import requests
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                messages = [
                    {"role": "system", "content": "You are an expert B2B profitability consultant."},
                    {"role": "user", "content": prompt},
                ]
                if extra_context is not None:
                    messages.append({"role": "user", "content": "Aggregated examples (top products): " + json.dumps(extra_context)})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 600,
                }

                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    st.subheader("AI Narrative")
                    st.write(content)
                else:
                    st.error(f"LLM call failed [{resp.status_code}]: {resp.text[:400]}")
            except Exception as e:
                st.error(f"LLM request error: {e}")

# =========================
# Opportunities & Risks
# =========================
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
st.write("• " + "\n• ".join(risk[:5]) if risk else "No major risks detected from available fields.")

# =========================
# Questions for First Client Call
# =========================
st.subheader("Questions for First Client Call")
qs = [
    "How are discounts and rebates recorded (per-line vs. end-of-period)?",
    "Any multi-currency transactions? If so, what FX policy should be applied?",
    "Are returns netted in revenue or recorded separately (returns_qty/amount)?",
    "Any non-product revenue or freight in unit_price/unit_cost?",
    "Do we have cost-to-serve or fulfillment cost drivers for Phase 2?",
]
st.write("• " + "\n• ".join(qs))

# =========================
# Assumptions & Formulas
# =========================
st.subheader("Assumptions & Formulas")
st.markdown(
    """
**Assumptions**
- Anonymized IDs only; PII is excluded.
- Currency not normalized unless explicitly stated.
- Returns are excluded unless provided as `returns_amount` or `returns_qty`.
- Aggregation mode uses `product_id × month` (and `customer_id` if present).

**Formulas**
- `extended_price = quantity * unit_price − discount − rebate`
- `cogs = quantity * unit_cost`
- `gross_margin = extended_price − cogs`
- `gm% = gross_margin / extended_price`
- `discount+rebate % = (discount + rebate) / (extended_price + discount + rebate)`
"""
)
