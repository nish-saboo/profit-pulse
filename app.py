import os
import io
import math
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go  # Plotly for PVM & scenario bridges

# =========================
# Config & Constants
# =========================
APP_TITLE = "Profit Pulse — Phase 1 Diagnostic"
PASSWORD = os.getenv("PP_PASSWORD", "PP-PULSE")

THR_DEFAULT = {
    "gm_green": 0.30,
    "disc_reb_green": 0.08,
    "returns_green": 0.02,
    "data_green": 0.90,
}

MAX_FILE_BYTES = 250 * 1024 * 1024  # 250 MB
MAX_ROWS = 1_000_000

st.set_page_config(page_title=APP_TITLE, layout="wide")

COLOR = {
    "green": "#1a9c5d",
    "amber": "#f2a700",
    "red": "#d13c3c",
    "blue": "#1f4b99",
    "slate": "#3c4758",
    "teal": "#008b8b",
}

PII_PATTERNS = ("name", "email", "phone", "mobile", "address", "tax", "ssn")

SYNONYMS = {
    "date": ["invoice_date", "order_date", "posting_date", "transaction_date", "doc_date"],
    "product_id": ["sku", "item_id", "item", "product", "product_code", "material", "material_code", "sku_code", "item_code"],
    "customer_id": ["cust_id", "customer", "account_id", "acct_id"],
    "sales_rep_id": ["rep_id", "sales_rep", "sr_id", "salesperson_id"],
    "quantity": ["qty", "units", "qty_sold", "quantity_sold", "order_qty"],
    "unit_price": ["price", "selling_price", "net_price", "unit_selling_price", "unit_list_price"],
    "unit_cost": ["cost", "unit_cogs", "cogs_per_unit", "unit_cost_local", "std_cost", "standard_cost"],
    "returns_amount": ["returns_value", "returns_val", "returns"],
}
TOTAL_REVENUE_COLS = ["revenue", "net_revenue", "net_amount", "amount", "sales_amount", "line_amount"]
TOTAL_COST_COLS = ["cogs", "cost_amount", "cost_of_goods_sold", "total_cost", "line_cost"]

# =========================
# Header & Access
# =========================
st.title(APP_TITLE)
st.caption("Private-use diagnostic. Operates on anonymized CSV/Excel data only. No PII.")

with st.sidebar:
    st.subheader("Access")
    pw = st.text_input("Password", type="password")
    authed = pw == PASSWORD
    if pw and not authed:
        st.error("That’s not the correct password. I can’t proceed.")

if not authed:
    st.stop()

# Persistent view and input digests
if "view" not in st.session_state:
    st.session_state.view = None
if "last_txn_digest" not in st.session_state:
    st.session_state.last_txn_digest = None

# =========================
# Sidebar: Templates + Uploads (right under password)
# =========================
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
    "freight": [0.0, 0.0],
})
tmpl_prod = pd.DataFrame({
    "product_id": ["P001", "P002"],
    "category_code": ["CAT-A", "CAT-B"],
    "family_code": ["FAM-1", "FAM-2"],
    "brand_code": ["BR-1", "BR-2"],
    "size_tier": ["S", "M"],
})
tmpl_cust = pd.DataFrame({
    "customer_id": ["C001", "C002"],
    "segment_code": ["SMB", "ENT"],
    "region_code": ["NA", "EU"],
    "channel_code": ["DTC", "RESELLER"],
})

with st.sidebar:
    st.markdown("**Templates**")
    c_t1, c_t2, c_t3 = st.columns(3)
    with c_t1:
        st.download_button("Txn CSV", data=df_to_csv_bytes(tmpl_txn), file_name="transactional_template.csv", mime="text/csv")
    with c_t2:
        st.download_button("Product CSV", data=df_to_csv_bytes(tmpl_prod), file_name="product_master_template.csv", mime="text/csv")
    with c_t3:
        st.download_button("Customer CSV", data=df_to_csv_bytes(tmpl_cust), file_name="customer_master_template.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Upload files**")

    txn = st.file_uploader("Transactional (required)", type=["csv", "xlsx", "xls"], key="txn", help="Drag & drop, CSV preferred for large files")

    # Mode & Period (compact)
    c_mode, c_period = st.columns([1,1])
    with c_mode:
        mode_radio = st.radio("Mode", ["Quick", "Full"], horizontal=True, label_visibility="collapsed")
    with c_period:
        period = st.selectbox("Period", ["3 months", "6 months", "12 months", "Full dataset"], index=2, label_visibility="collapsed")
    mode = "Full (Transactional + optional masters)" if mode_radio == "Full" else "Quick (Transactional only)"

    # Optional masters grid when in Full
    price_file = None; cts = None; prod_master = None; cust_master = None; sales_rep_master = None; proj = None
    if mode_radio == "Full":
        g1c1, g1c2 = st.columns(2)
        with g1c1:
            prod_master = st.file_uploader("Product Master", type=["csv", "xlsx", "xls"], key="prod_master")
            cust_master = st.file_uploader("Customer Master", type=["csv", "xlsx", "xls"], key="cust_master")
        with g1c2:
            price_file = st.file_uploader("Price File", type=["csv", "xlsx", "xls"], key="price")
            sales_rep_master = st.file_uploader("Sales Rep Master", type=["csv", "xlsx", "xls"], key="srep_master")

        g2c1, g2c2 = st.columns(2)
        with g2c1:
            cts = st.file_uploader("Cost‑to‑Serve Map", type=["csv", "xlsx", "xls"], key="cts")
        with g2c2:
            proj = st.file_uploader("Projections (preview)", type=["csv", "xlsx", "xls"], key="proj")

# Reset view when new transactional content appears
if txn is not None:
    txn_bytes = txn.getvalue()
    cur_digest = hashlib.md5(txn_bytes).hexdigest()
    if st.session_state.last_txn_digest != cur_digest:
        st.session_state.view = None
        st.session_state.last_txn_digest = cur_digest
else:
    st.info("Upload transactional file to begin.")
    st.stop()

# =========================
# Cacheable Readers
# =========================
@st.cache_data(show_spinner=False)
def read_any_cached(file_bytes: bytes, ext: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    ext = (ext or "").lower()
    if ext.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")
    if ext.endswith(".xls"):
        return pd.read_excel(bio, engine="xlrd")
    return pd.read_csv(bio, low_memory=False)

@st.cache_data(show_spinner=False)
def aggregate_in_chunks_cached(file_bytes: bytes) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    agg = None
    for ch in pd.read_csv(bio, chunksize=200_000, low_memory=False):
        ch.columns = [c.lower().strip() for c in ch.columns]
        for req, alts in SYNONYMS.items():
            if req not in ch.columns:
                for alt in alts:
                    if alt in ch.columns:
                        ch.rename(columns={alt: req}, inplace=True)
                        break
        ch["date"] = pd.to_datetime(ch.get("date"), errors="coerce")
        for col in ["quantity","unit_price","unit_cost","discount","rebate"]:
            if col in ch.columns:
                ch[col] = pd.to_numeric(ch[col], errors="coerce").fillna(0.0)
        ch["extended_price"] = ch.get("quantity",0) * ch.get("unit_price",0) - ch.get("discount",0) - ch.get("rebate",0)
        ch["cogs"] = ch.get("quantity",0) * ch.get("unit_cost",0)
        ch["month"] = ch["date"].dt.to_period("M").dt.to_timestamp()
        keys = ["product_id","month"]
        if "customer_id" in ch.columns:
            keys = ["customer_id"] + keys
        g = ch.groupby(keys, dropna=False).agg(
            quantity=("quantity","sum"),
            revenue=("extended_price","sum"),
            cogs=("cogs","sum"),
            discount=("discount","sum"),
            rebate=("rebate","sum"),
        ).reset_index()
        agg = g if agg is None else pd.concat([agg, g], ignore_index=True)

    if agg is not None:
        by = [c for c in ["customer_id","product_id","month"] if c in agg.columns]
        agg = agg.groupby(by, dropna=False, as_index=False).sum(numeric_only=True)
        agg["gross_margin"] = agg["revenue"] - agg["cogs"]
        agg["gm_pct"] = np.where(agg["revenue"] != 0, agg["gross_margin"]/agg["revenue"], np.nan)
        return agg
    return pd.DataFrame()

# =========================
# Helpers
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

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def auto_map_by_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    df = _norm_cols(df)
    for req, alts in SYNONYMS.items():
        if req not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    df.rename(columns={alt: req}, inplace=True)
                    break
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c:
                df.rename(columns={c: "date"}, inplace=True)
                break
    return df

def fix_excel_serial_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if parsed.isna().mean() > 0.6:
            try:
                if pd.api.types.is_numeric_dtype(df["date"]):
                    df["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["date"].astype(float), unit="D")
                else:
                    df["date"] = parsed
            except Exception:
                df["date"] = parsed
        else:
            df["date"] = parsed
    return df

def derive_missing_price_cost(df: pd.DataFrame, issues: list) -> pd.DataFrame:
    if "unit_price" not in df.columns and "quantity" in df.columns:
        for col in TOTAL_REVENUE_COLS:
            if col in df.columns:
                q = pd.to_numeric(df["quantity"], errors="coerce").replace(0, np.nan)
                amt = pd.to_numeric(df[col], errors="coerce")
                df["unit_price"] = (amt / q).replace([np.inf, -np.inf], np.nan)
                issues.append(f"Derived unit_price from `{col}` / quantity (check accuracy).")
                break
    if "unit_cost" not in df.columns and "quantity" in df.columns:
        for col in TOTAL_COST_COLS:
            if col in df.columns:
                q = pd.to_numeric(df["quantity"], errors="coerce").replace(0, np.nan)
                amt = pd.to_numeric(df[col], errors="coerce")
                df["unit_cost"] = (amt / q).replace([np.inf, -np.inf], np.nan)
                issues.append(f"Derived unit_cost from `{col}` / quantity (check accuracy).")
                break
    return df

def normalize_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    df = auto_map_by_synonyms(df)
    df = fix_excel_serial_dates(df)

    numeric_cols = [
        "quantity", "unit_price", "unit_cost",
        "discount", "rebate", "freight",
        "returns_qty", "returns_amount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    for base_col in ["quantity","unit_price","unit_cost","discount","rebate","freight"]:
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

def read_master_csv_bytes(file_bytes: bytes, filename: str, key_cols, allowed_extra=None, friendly="master"):
    raw = read_any_cached(file_bytes, filename)
    if raw.empty:
        st.warning(f"{friendly} is empty or unreadable.")
        return None, None
    m = _norm_cols(raw)
    drop_cols = [c for c in m.columns if any(pat in c for pat in PII_PATTERNS)]
    m = m.drop(columns=drop_cols, errors="ignore")
    missing_keys = [k for k in key_cols if k not in m.columns]
    if missing_keys:
        st.warning(f"{friendly} is missing key columns: {missing_keys}. Skipping.")
        return None, None
    cols_to_keep = list(key_cols)
    if allowed_extra:
        cols_to_keep += [c for c in allowed_extra if c in m.columns]
    m = m[cols_to_keep].drop_duplicates()
    return m, raw.columns.tolist()

def unmatched_alert_and_coverage(base_df, base_key, master_df, master_key, master_name, weight_col="extended_price"):
    if base_key not in base_df.columns or master_df is None or master_key not in master_df.columns:
        return
    left = base_df[[base_key, weight_col]].dropna(subset=[base_key]).copy()
    left["_w"] = left[weight_col]
    coverage = float(left[base_key].isin(master_df[master_key]).astype(int).mul(left["_w"]).sum()) / max(1e-9, float(left["_w"].sum()))
    st.caption(f"{master_name} coverage by revenue: **{coverage:.1%}**")
    # Unmatched count sample
    missing_ct = (~left[base_key].isin(master_df[master_key])).sum()
    if missing_ct > 0:
        sample = left.loc[~left[base_key].isin(master_df[master_key]), base_key].astype(str).head(10).tolist()
        st.warning(f"Unmatched {base_key} in {master_name}: {missing_ct} distinct values not found. Sample: {sample}")

def download_df(df: pd.DataFrame, label: str, name: str):
    st.download_button(f"Download {label} (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")

# =========================
# Read Transactional (cached) + Basic ETL
# =========================
txn_ext = (txn.name if txn else "").lower()
size_bytes = len(txn_bytes)

if txn_ext.endswith(".csv") and (size_bytes > MAX_FILE_BYTES):
    df = aggregate_in_chunks_cached(txn_bytes)
    aggregated = True
else:
    df = read_any_cached(txn_bytes, txn_ext)
    aggregated = False

if df.empty:
    st.error("The uploaded transactional file appears to be empty or unreadable.")
    st.stop()

df = auto_map_by_synonyms(df)
df = fix_excel_serial_dates(df)

# Manual mapping UI (restore) for keys with safe fallbacks
def manual_map_ui(df: pd.DataFrame, key: str, synonyms: list, descriptor_hints=("name","desc","description","title","label")):
    if key in df.columns:
        return df
    cand = next((c for c in synonyms if c in df.columns), None)
    if cand:
        df.rename(columns={cand: key}, inplace=True)
        return df
    st.warning(f"Missing `{key}`. Map a column manually, or derive a surrogate (limited diagnostic).")
    choice = st.selectbox(f"Select a column to use as `{key}`", options=["— none —"] + list(df.columns), index=0, key=f"map_{key}")
    if choice != "— none —":
        if choice != key:
            df.rename(columns={choice: key}, inplace=True)
        return df
    # surrogate from descriptor
    desc_cols = [c for c in df.columns if any(h in c for h in descriptor_hints)]
    if desc_cols:
        base_col = desc_cols[0]
        st.info(f"Deriving surrogate `{key}` from `{base_col}` (hash). Names will not be echoed.")
        df[key] = df[base_col].fillna("UNK").astype(str).apply(lambda s: key[:1].upper() + hashlib.sha1(s.encode()).hexdigest()[:10])
        return df
    st.warning(f"Using `{key}='UNKNOWN'` (Tier C remediation). Results aggregated across unknown {key.split('_')[0]}.")
    df[key] = "UNKNOWN"
    return df

df = manual_map_ui(df, "product_id", SYNONYMS["product_id"])
if "customer_id" not in df.columns:
    df = manual_map_ui(df, "customer_id", SYNONYMS.get("customer_id", []))
if "sales_rep_id" not in df.columns:
    df = manual_map_ui(df, "sales_rep_id", SYNONYMS.get("sales_rep_id", []))

# Try derivations for price/cost
issues_precheck = []
df = derive_missing_price_cost(df, issues_precheck)
if issues_precheck:
    st.caption("Derivations applied: " + " | ".join(issues_precheck))

required_cols = ["date", "product_id", "quantity", "unit_price", "unit_cost"]
missing_reqs = [c for c in required_cols if c not in df.columns]
if missing_reqs:
    st.error(f"Missing required columns after mapping: {missing_reqs}. Please adjust and re-upload.")
    st.stop()

with st.spinner("Normalizing & deriving metrics..."):
    df = normalize_and_derive(df)
    df = parse_period_filter(df)

# Hard guards
bad_qty = int((df["quantity"] <= 0).sum())
neg_price = int((df["unit_price"] < 0).sum())
neg_cost = int((df["unit_cost"] < 0).sum())
if bad_qty or neg_price or neg_cost:
    msgs = []
    if bad_qty: msgs.append(f"{bad_qty} rows with quantity ≤ 0")
    if neg_price: msgs.append(f"{neg_price} rows with negative unit_price")
    if neg_cost: msgs.append(f"{neg_cost} rows with negative unit_cost")
    st.warning("Data issues detected: " + "; ".join(msgs) + ". These rows can skew metrics.")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

if aggregated:
    st.warning("Large file detected. Aggregation mode in effect (month × product, and customer if present).")

# =========================
# Optional master merges (cached reads)
# =========================
def read_upload_bytes(u):
    return (u.getvalue(), u.name) if u is not None else (None, None)

pm_df = cm_df = pf_df = srm_df = None
prod_bytes, prod_name = read_upload_bytes(prod_master)
cust_bytes, cust_name = read_upload_bytes(cust_master)
price_bytes, price_name = read_upload_bytes(price_file)
srep_bytes, srep_name = read_upload_bytes(sales_rep_master)

if prod_bytes:
    pm_allowed = ["category_code", "family_code", "brand_code", "size_tier", "uom", "launch_year"]
    pm_df, _ = read_master_csv_bytes(prod_bytes, prod_name, key_cols=["product_id"], allowed_extra=pm_allowed, friendly="Product Master")
    if pm_df is not None:
        df = df.merge(pm_df, on="product_id", how="left", validate="m:1")
        st.success("Product Master merged (keys: product_id).")
        unmatched_alert_and_coverage(df, "product_id", pm_df, "product_id", "Product Master")

if cust_bytes:
    cm_allowed = ["segment_code", "region_code", "channel_code", "tier_code", "size_tier"]
    cm_df, _ = read_master_csv_bytes(cust_bytes, cust_name, key_cols=["customer_id"], allowed_extra=cm_allowed, friendly="Customer Master")
    if cm_df is not None:
        if "customer_id" not in df.columns:
            st.warning("Transactional file lacks `customer_id`; Customer Master skipped.")
        else:
            df = df.merge(cm_df, on="customer_id", how="left", validate="m:1")
            st.success("Customer Master merged (keys: customer_id).")
            unmatched_alert_and_coverage(df, "customer_id", cm_df, "customer_id", "Customer Master")

if price_bytes:
    pf_allowed = ["list_price", "target_price", "floor_price", "segment_code", "customer_id"]
    pf_df, _ = read_master_csv_bytes(price_bytes, price_name, key_cols=["product_id"], allowed_extra=pf_allowed, friendly="Price File")
    if pf_df is not None:
        how_keys = ["product_id"]
        if ("customer_id" in pf_df.columns) and ("customer_id" in df.columns):
            how_keys = ["product_id", "customer_id"]
        df = df.merge(pf_df, on=how_keys, how="left", validate="m:m")
        st.success(f"Price File merged (keys: {', '.join(how_keys)}).")

if srep_bytes:
    srm_allowed = ["region_code","team_code"]
    srm_df, _ = read_master_csv_bytes(srep_bytes, srep_name, key_cols=["sales_rep_id"], allowed_extra=srm_allowed, friendly="Sales Rep Master")
    if srm_df is not None:
        if "sales_rep_id" in df.columns:
            df = df.merge(srm_df, on="sales_rep_id", how="left", validate="m:1")
            st.success("Sales Rep Master merged (keys: sales_rep_id).")
        else:
            st.info("Transactional file has no `sales_rep_id`; Sales Rep Master not merged.")

# Optional filters
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
    if "region_code" in df.columns:
        regs = sorted([c for c in df["region_code"].dropna().unique()])
        sel3 = st.multiselect("Region", regs)
        if sel3:
            df = df[df["region_code"].isin(sel3)]

# =========================
# ETL Validation & Remediation
# =========================
st.subheader("Data Remediation Plan")

missing_flags = {
    "date_missing": df["date"].isna().any(),
    "product_id_missing": df["product_id"].isna().any(),
    "qty_missing": df["quantity"].isna().any(),
    "price_missing": df["unit_price"].isna().any(),
    "cost_missing": df["unit_cost"].isna().any(),
    "currency_unhandled": "currency" in df.columns,
}
def remediation_tier(flags):
    score = 0
    for v in flags.values():
        score += 2 if v else 0
    if score >= 6: return "C"
    if score >= 3: return "B"
    return "A"

tier = remediation_tier(missing_flags)
data_score = data_completeness_score(df)

st.write(f"**Tier {tier}** — data completeness: **{data_score:.0%}**")
issues = []
if missing_flags["date_missing"]: issues.append("Some `date` values are missing or invalid.")
if missing_flags["product_id_missing"]: issues.append("`product_id` contains nulls.")
if missing_flags["qty_missing"]: issues.append("Some `quantity` values are missing/invalid.")
if missing_flags["price_missing"]: issues.append("Some `unit_price` values are missing/invalid.")
if missing_flags["cost_missing"]: issues.append("Some `unit_cost` values are missing/invalid.")
if missing_flags["currency_unhandled"]: issues.append("`currency` present — multi-currency not normalized (flagged).")
if issues:
    st.error("• " + "\n• ".join(issues))
else:
    st.success("No critical structural issues detected in required fields.")

if "currency" in df.columns:
    cur_share = df.groupby("currency", as_index=False)["extended_price"].sum()
    total_rev_cur = max(1e-9, cur_share["extended_price"].sum())
    cur_share["share"] = cur_share["extended_price"] / total_rev_cur
    st.info("⚠️ Multi-currency detected (no FX normalization). Revenue share by currency shown below.")
    st.dataframe(cur_share.assign(share=lambda x: (x["share"]*100).round(1)), use_container_width=True)

st.caption("Next steps: supply missing fields; confirm currency handling; provide returns/discount/rebate detail if available.")

# =========================
# Analytics Triggers (AFTER remediation) — persistent view
# =========================
have_secondary_inputs = bool((prod_master is not None) and (cust_master is not None) and (price_file is not None))

c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Run Primary Analytics", type="primary", help="Dashboard, Trends, Pareto, PVM, insights."):
        st.session_state.view = "primary"
with c2:
    if st.button(
        "Generate Secondary Analytics",
        type="secondary",
        help="Segmented profitability, SKU×Region, price policy, scatter (needs Product+Customer+Price).",
        disabled=not have_secondary_inputs
    ):
        st.session_state.view = "secondary"
with c3:
    if st.button("Run What‑If Scenarios", type="secondary", help="Separate hypothetical scenario visuals with explicit assumptions."):
        st.session_state.view = "scenarios"

# Quick nav (optional)
nc1, nc2, nc3, nc4 = st.columns([1,1,1,1])
with nc1:
    if st.button("↺ Reset view"):
        st.session_state.view = None
with nc2:
    if st.button("↤ Primary"):
        st.session_state.view = "primary"
with nc3:
    if st.button("↦ Secondary", disabled=not have_secondary_inputs):
        st.session_state.view = "secondary"
with nc4:
    if st.button("★ Scenarios"):
        st.session_state.view = "scenarios"

view = st.session_state.view
if view is None:
    st.info("Choose an analysis to run using the buttons above.")
    st.stop()

# =========================
# PRIMARY ANALYTICS
# =========================
if view == "primary":
    st.subheader("Dashboard")
    summary = {}
    summary["revenue"] = float(df["extended_price"].sum())
    summary["cogs"] = float((df["cogs"] if "cogs" in df.columns else df["unit_cost"] * df["quantity"]).sum())
    summary["gross_margin"] = summary["revenue"] - summary["cogs"]
    summary["gm_pct"] = (summary["gross_margin"] / summary["revenue"]) if summary["revenue"] else np.nan
    disc_reb_total = float(df.get("discount", 0).sum() + df.get("rebate", 0).sum())
    denom = (summary["revenue"] + disc_reb_total) if (summary["revenue"] + disc_reb_total) else np.nan
    summary["disc_reb_pct"] = (disc_reb_total / denom) if denom and denom != 0 else np.nan

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Revenue", f"${summary['revenue']:,.0f}")
    kpi_cols[1].metric("Gross Margin", f"${summary['gross_margin']:,.0f}")
    kpi_cols[2].metric("GM%", "N/A" if pd.isna(summary["gm_pct"]) else f"{summary['gm_pct']:.1%}")
    kpi_cols[3].metric("Discount+Rebate %", "N/A" if pd.isna(summary["disc_reb_pct"]) else f"{summary['disc_reb_pct']:.1%}")

    # Threshold sliders (horizontal) above traffic lights
    st.markdown("**Thresholds (adjust to your rubric)**")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1:
        gm_thr = st.slider("GM% target", 0.10, 0.60, THR_DEFAULT["gm_green"], 0.01, key="thr_gm")
    with tcol2:
        disc_thr = st.slider("Discount+Rebate % (max)", 0.00, 0.30, THR_DEFAULT["disc_reb_green"], 0.005, key="thr_disc")
    with tcol3:
        ret_thr = st.slider("Returns % (max)", 0.00, 0.10, THR_DEFAULT["returns_green"], 0.005, key="thr_ret")
    with tcol4:
        dat_thr = st.slider("Data completeness (min)", 0.50, 1.00, THR_DEFAULT["data_green"], 0.01, key="thr_dat")
    THR = {"gm_green": gm_thr, "disc_reb_green": disc_thr, "returns_green": ret_thr, "data_green": dat_thr}

    st.markdown("**Traffic-light checks**")
    checks = pd.DataFrame([
        {"Metric": "GM%", "Value": f"{summary['gm_pct']:.1%}" if not pd.isna(summary["gm_pct"]) else "N/A",
         "Status": traffic(summary["gm_pct"], THR["gm_green"])[0]},
        {"Metric": "Discount+Rebate %", "Value": f"{summary['disc_reb_pct']:.1%}" if not pd.isna(summary["disc_reb_pct"]) else "N/A",
         "Status": traffic(summary["disc_reb_pct"], THR["disc_reb_green"], reverse=True)[0]},
        {"Metric": "Returns %",
         "Value": (f"{(df.get('returns_amount', pd.Series([0])).sum() / summary['revenue']):.1%}" if "returns_amount" in df.columns and summary["revenue"] else "N/A"),
         "Status": (traffic((df['returns_amount'].sum() / summary['revenue']), THR["returns_green"], reverse=True)[0] if "returns_amount" in df.columns and summary["revenue"] else "N/A")},
        {"Metric": "Data completeness", "Value": f"{data_score:.0%}", "Status": traffic(data_score, THR["data_green"])[0]},
    ])
    st.dataframe(checks, use_container_width=True)

    # Revenue & GM% Trends (combo dual-axis)
    st.subheader("Revenue & GM% Trends")
    trend = df.groupby("month", as_index=False).agg(
        revenue=("extended_price", "sum"),
        gp=("gross_margin", "sum")
    )
    trend["gm_pct"] = np.where(trend["revenue"] != 0, trend["gp"] / trend["revenue"], np.nan)
    base = alt.Chart(trend).encode(x=alt.X("month:T", title="Month"))
    bars = base.mark_bar(color="#2563eb").encode(y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="$s")))
    line = base.mark_line(color="#10b981", strokeWidth=3).encode(y=alt.Y("gm_pct:Q", title="GM%", axis=alt.Axis(format="%", orient="right")))
    combo = alt.layer(bars, line).resolve_scale(y="independent").properties(height=360)
    st.altair_chart(combo, use_container_width=True)
    download_df(trend, "Trends", "trend.csv")

    # Two Separate Pareto Charts (Product & Customer) with cumulative %
    st.subheader("Pareto (Revenue) — Product & Customer")
    p_cols = st.columns(2)

    if "product_id" in df.columns:
        with p_cols[0]:
            st.write("**Product Pareto**")
            pareto_p = (df.groupby("product_id", as_index=False)["extended_price"]
                        .sum().rename(columns={"extended_price":"revenue"})
                        .sort_values("revenue", ascending=False))
            total_rev_p = max(1e-9, pareto_p["revenue"].sum())
            pareto_p["cum_share"] = pareto_p["revenue"].cumsum() / total_rev_p
            top_n_p = st.slider("Top N (Product)", min_value=20, max_value=min(500, len(pareto_p)), value=min(50, len(pareto_p)), step=10, key="pareto_prod_topn")
            pv_p = pareto_p.head(top_n_p)
            base_p = alt.Chart(pv_p).encode(x=alt.X("product_id:N", sort="-y", title="Product"))
            bars_p = base_p.mark_bar(color="#2563eb").encode(y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="$s")))
            line_p = base_p.mark_line(color="#10b981", strokeWidth=3).encode(y=alt.Y("cum_share:Q", title="Cumulative %", axis=alt.Axis(format="%", orient="right")))
            combo_p = alt.layer(bars_p, line_p).resolve_scale(y="independent").properties(height=320)
            st.altair_chart(combo_p, use_container_width=True)
            eighty_n_p = (pareto_p["cum_share"] <= 0.80).sum()
            st.caption(f"Top 80% revenue reached by ~{eighty_n_p} product(s).")
            download_df(pareto_p, "Product Pareto", "pareto_product.csv")
    else:
        p_cols[0].info("No `product_id` found for Product Pareto.")

    if "customer_id" in df.columns:
        with p_cols[1]:
            st.write("**Customer Pareto**")
            pareto_c = (df.groupby("customer_id", as_index=False)["extended_price"]
                        .sum().rename(columns={"extended_price":"revenue"})
                        .sort_values("revenue", ascending=False))
            total_rev_c = max(1e-9, pareto_c["revenue"].sum())
            pareto_c["cum_share"] = pareto_c["revenue"].cumsum() / total_rev_c
            top_n_c = st.slider("Top N (Customer)", min_value=20, max_value=min(500, len(pareto_c)), value=min(50, len(pareto_c)), step=10, key="pareto_cust_topn")
            pv_c = pareto_c.head(top_n_c)
            base_c = alt.Chart(pv_c).encode(x=alt.X("customer_id:N", sort="-y", title="Customer"))
            bars_c = base_c.mark_bar(color="#2563eb").encode(y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="$s")))
            line_c = base_c.mark_line(color="#10b981", strokeWidth=3).encode(y=alt.Y("cum_share:Q", title="Cumulative %", axis=alt.Axis(format="%", orient="right")))
            combo_c = alt.layer(bars_c, line_c).resolve_scale(y="independent").properties(height=320)
            st.altair_chart(combo_c, use_container_width=True)
            eighty_n_c = (pareto_c["cum_share"] <= 0.80).sum()
            st.caption(f"Top 80% revenue reached by ~{eighty_n_c} customer(s).")
            download_df(pareto_c, "Customer Pareto", "pareto_customer.csv")
    else:
        p_cols[1].info("No `customer_id` found for Customer Pareto.")

    # PVM Waterfall (no scenarios blended)
    def pvm_effects(df):
        m = df.groupby("month", as_index=False).agg(
            revenue=("extended_price", "sum"),
            qty=("quantity", "sum")
        ).sort_values("month")
        if len(m) < 2:
            return None
        cur, prev = m.iloc[-1], m.iloc[-2]
        p_prev = prev["revenue"] / prev["qty"] if prev["qty"] else 0.0
        p_cur  = cur["revenue"] / cur["qty"] if cur["qty"] else 0.0
        volume_eff = (cur["qty"] - prev["qty"]) * (p_prev)
        price_eff  = (p_cur - p_prev) * (cur["qty"])
        mix_eff    = (cur["revenue"] - prev["revenue"]) - volume_eff - price_eff
        total_delta = cur["revenue"] - prev["revenue"]
        return prev["revenue"], price_eff, volume_eff, mix_eff, cur["revenue"], total_delta

    def pvm_plotly(prev_rev, price_eff, volume_eff, mix_eff, cur_rev):
        fig = go.Figure(go.Waterfall(
            name="PVM",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "absolute"],
            x=["Prev Rev", "Price", "Volume", "Mix", "Cur Rev"],
            textposition="outside",
            text=[
                f"${prev_rev:,.0f}",
                f"${price_eff:,.0f}",
                f"${volume_eff:,.0f}",
                f"${mix_eff:,.0f}",
                f"${cur_rev:,.0f}"
            ],
            y=[prev_rev, price_eff, volume_eff, mix_eff, cur_rev],
            connector={"line": {"color":"#2E2E2E"}},
            increasing={"marker":{"color": COLOR["green"]}},
            decreasing={"marker":{"color": COLOR["red"]}},
            totals={"marker":{"color": COLOR["blue"]}}
        ))
        fig.update_layout(
            title="Revenue Bridge (P–V–M)",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="Revenue"
        )
        return fig

    eff = pvm_effects(df)
    if eff is not None:
        prev_rev, price_eff, volume_eff, mix_eff, cur_rev, total_delta = eff
        st.subheader("Revenue Bridge (MoM P–V–M)")
        fig = pvm_plotly(prev_rev, price_eff, volume_eff, mix_eff, cur_rev)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Δ Revenue vs prior month: {total_delta:,.0f}")

    # Automated Insights
    st.subheader("Automated Insights (Local)")
    insights = []
    gm_val = summary["gm_pct"]
    if not pd.isna(gm_val):
        insights.append("GM% on track." if gm_val >= THR["gm_green"] else "GM% below target — review pricing floors and costs.")
    if not pd.isna(summary["disc_reb_pct"]):
        insights.append("Discount/REB load within tolerance." if summary["disc_reb_pct"] <= THR["disc_reb_green"] else "Discount/REB load high — tighten approvals and exceptions.")
    if "returns_amount" in df.columns and summary["revenue"]:
        ret_pct = df["returns_amount"].sum() / summary["revenue"]
        insights.append("Returns manageable." if ret_pct <= THR["returns_green"] else "Returns elevated — audit root causes.")
    st.write("• " + "\n• ".join(insights) if insights else "No notable rule-based findings from current data.")

# =========================
# SECONDARY ANALYTICS
# =========================
if view == "secondary":
    st.subheader("Advanced Segment Analytics")

    # Segment × Category
    if {"segment_code","category_code"}.issubset(df.columns):
        st.markdown("**Profitability by Segment × Category**")
        seg_cat = df.groupby(["segment_code","category_code"], as_index=False).agg(
            revenue=("extended_price", "sum"),
            gross_margin=("gross_margin", "sum")
        )
        seg_cat["gm_pct"] = np.where(seg_cat["revenue"] != 0, seg_cat["gross_margin"] / seg_cat["revenue"], np.nan)
        chart = alt.Chart(seg_cat).mark_bar().encode(
            x=alt.X("segment_code:N", title="Customer Segment"),
            y=alt.Y("gm_pct:Q", title="GM%", axis=alt.Axis(format="%")),
            color=alt.Color("category_code:N", title="Product Category"),
            tooltip=["segment_code", "category_code", alt.Tooltip("gm_pct:Q", format=".1%"), alt.Tooltip("revenue:Q", format="$.2s")]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
        download_df(seg_cat, "Segment x Category", "segment_category.csv")

    # SKU × Region matrix
    if "region_code" in df.columns:
        st.markdown("**SKU × Region Matrix (GM% & Discount Load)**")
        mat = df.groupby(["product_id","region_code"], as_index=False).agg(
            revenue=("extended_price","sum"),
            cogs=("cogs","sum"),
            disc=("discount","sum"),
            reb=("rebate","sum"),
            qty=("quantity","sum"),
        )
        mat["gm_pct"] = np.where(mat["revenue"] != 0, (mat["revenue"] - mat["cogs"]) / mat["revenue"], np.nan)
        mat["disc_load"] = (mat["disc"] + mat["reb"]) / np.maximum(mat["revenue"] + mat["disc"] + mat["reb"], 1e-9)
        chart = alt.Chart(mat).mark_rect().encode(
            x=alt.X("region_code:N", title="Region"),
            y=alt.Y("product_id:N", title="Product"),
            color=alt.Color("gm_pct:Q", title="GM%", scale=alt.Scale(scheme="bluegreen"), legend=alt.Legend(format="%")),
            tooltip=["product_id","region_code", alt.Tooltip("gm_pct:Q", format=".1%"), alt.Tooltip("disc_load:Q", format=".1%"), alt.Tooltip("revenue:Q", format="$.2s")]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        download_df(mat, "SKU x Region", "sku_region.csv")

    # Price band vs policy
    if any(c in df.columns for c in ["list_price","target_price","floor_price"]):
        st.markdown("**Price Band Analysis vs. Policy**")
        pol = df.copy()
        pol["realized_price"] = np.where(pol["quantity"] != 0, pol["extended_price"] / np.maximum(pol["quantity"], 1e-9), pol["unit_price"])
        rows = []
        if "floor_price" in pol.columns:
            mask = pol["realized_price"] < pol["floor_price"]
            rows.append({"Violation": "Below Floor", "Rows": int(mask.sum()), "Share": f"{(mask.sum()/len(pol)):.1%}"})
        if "target_price" in pol.columns:
            mask = pol["realized_price"] < pol["target_price"]
            rows.append({"Violation": "Below Target", "Rows": int(mask.sum()), "Share": f"{(mask.sum()/len(pol)):.1%}"})
        if "list_price" in pol.columns:
            mask = pol["realized_price"] < pol["list_price"]
            rows.append({"Violation": "Below List", "Rows": int(mask.sum()), "Share": f"{(mask.sum()/len(pol)):.1%}"})
        pol_summary = pd.DataFrame(rows)
        if not pol_summary.empty:
            st.dataframe(pol_summary, use_container_width=True)
            download_df(pol_summary, "Policy Violations", "policy_violations.csv")
        chart = alt.Chart(pol).mark_bar().encode(
            x=alt.X("realized_price:Q", bin=alt.Bin(maxbins=30), title="Realized Price"),
            y=alt.Y("count():Q"),
            color=alt.value("#1f4b99")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # Customer GM% distribution
    if "customer_id" in df.columns:
        st.markdown("**Customer GM% Distribution**")
        cust = df.groupby("customer_id", as_index=False).agg(
            revenue=("extended_price","sum"),
            gross_margin=("gross_margin","sum")
        )
        cust["gm_pct"] = np.where(cust["revenue"] != 0, cust["gross_margin"] / cust["revenue"], np.nan)
        st.altair_chart(
            alt.Chart(cust).mark_bar().encode(
                x=alt.X("gm_pct:Q", bin=alt.Bin(maxbins=30), title="GM%"),
                y='count():Q',
            ).properties(height=300),
            use_container_width=True
        )
        download_df(cust, "Customer GM% Distribution", "customer_gm_dist.csv")

    # Contribution waterfall by tier/segment
    tier_dim = "tier_code" if "tier_code" in df.columns else ("segment_code" if "segment_code" in df.columns else None)
    if tier_dim:
        st.markdown(f"**Contribution by {tier_dim}**")
        contr = df.groupby(tier_dim, as_index=False).agg(revenue=("extended_price","sum"), gm=("gross_margin","sum"))
        contr = contr.sort_values("revenue", ascending=False)
        fig = go.Figure(go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["relative"] * len(contr),
            x=contr[tier_dim].astype(str).tolist(),
            y=contr["gm"].tolist(),
            increasing={"marker":{"color": COLOR["green"]}},
            decreasing={"marker":{"color": COLOR["red"]}},
        ))
        fig.update_layout(height=320, title=f"Gross Margin Contribution by {tier_dim}")
        st.plotly_chart(fig, use_container_width=True)
        download_df(contr, f"Contribution by {tier_dim}", "contribution_by_segment.csv")

    # Revenue vs GM% scatter — Customer × Product × Sales Rep (with guards)
    dims = [d for d in ["customer_id","product_id","sales_rep_id"] if d in df.columns]
    if len(dims) == 3:
        st.markdown("**Revenue vs GM% — Customer × Product × Sales Rep**")
        grp = df.groupby(dims, as_index=False).agg(
            revenue=("extended_price","sum"),
            cogs=("cogs","sum"),
            qty=("quantity","sum")
        )
        grp["gm_pct"] = np.where(grp["revenue"] != 0, (grp["revenue"] - grp["cogs"]) / grp["revenue"], np.nan)
        min_rev = st.slider("Minimum revenue to display", min_value=0.0, max_value=float(grp["revenue"].max() or 0), value=0.0, step=max(100.0, float((grp["revenue"].max() or 0)/100)))
        grp = grp[grp["revenue"] >= min_rev]
        topn = st.slider("Limit points (by top revenue)", min_value=50, max_value=int(max(100, len(grp))), value=min(500, len(grp)), step=50)
        grp = grp.sort_values("revenue", ascending=False).head(topn)
        scatter = alt.Chart(grp).mark_circle(size=80, opacity=0.6).encode(
            x=alt.X("revenue:Q", title="Revenue", scale=alt.Scale(type="log", nice=False, zero=False)),
            y=alt.Y("gm_pct:Q", title="GM%", axis=alt.Axis(format="%")),
            color=alt.Color("sales_rep_id:N", title="Sales Rep", legend=None),
            tooltip=[
                alt.Tooltip("customer_id:N", title="Customer"),
                alt.Tooltip("product_id:N", title="Product"),
                alt.Tooltip("sales_rep_id:N", title="Sales Rep"),
                alt.Tooltip("revenue:Q", title="Revenue", format="$.2s"),
                alt.Tooltip("gm_pct:Q", title="GM%", format=".1%")
            ]
        ).properties(height=380)
        st.altair_chart(scatter, use_container_width=True)
        download_df(grp, "Revenue vs GM% (triplet)", "revenue_gmpct_scatter.csv")
    else:
        st.info("Scatter needs customer_id, product_id, and sales_rep_id to be present.")

# =========================
# WHAT‑IF SCENARIOS (SEPARATE VISUALS)
# =========================
if view == "scenarios":
    st.subheader("What‑If Scenarios (Hypothetical) — Separate Visuals")

    # Horizontal controls with 1-line descriptions
    cA, cB, cC, cD = st.columns(4)
    with cA:
        price_delta = st.slider("Price Δ %", -10.0, 10.0, 0.0, 0.5)
        st.caption("Tests a uniform change in realized unit prices.")
    with cB:
        cost_delta = st.slider("Unit Cost Δ %", -10.0, 10.0, 0.0, 0.5)
        st.caption("Tests production/COGS changes per unit.")
    with cC:
        freight_delta = st.slider("Freight Δ %", -20.0, 20.0, 0.0, 1.0)
        st.caption("Tests logistics/handling cost changes (if provided).")
    with cD:
        discount_cap = st.slider("Discount Cap %", 0.0, 50.0, 50.0, 1.0)
        st.caption("Caps discount+rebate vs pre‑discount line value.")

    st.markdown(
        """
**Method (illustrative, not advice):**  
Apply the deltas/cap, recompute revenue/COGS/freight, and bridge **Baseline → Price → Cost → Discount Cap → Freight → Scenario**.
"""
    )

    scenario = df.copy()
    pre_disc_val = scenario["quantity"] * scenario["unit_price"]
    current_discounts = scenario.get("discount", 0).fillna(0) + scenario.get("rebate", 0).fillna(0)
    cap_amount = (discount_cap / 100.0) * pre_disc_val
    capped_discounts = np.minimum(current_discounts, cap_amount)

    scenario_unit_price = scenario["unit_price"] * (1 + price_delta / 100.0)
    scenario_unit_cost = scenario["unit_cost"] * (1 + cost_delta / 100.0)
    scenario_freight = (scenario.get("freight", 0).fillna(0)) * (1 + freight_delta / 100.0)

    scenario_rev = scenario["quantity"] * scenario_unit_price - capped_discounts
    scenario_cogs = scenario["quantity"] * scenario_unit_cost
    scenario_gp_line = scenario_rev - scenario_cogs - scenario_freight

    base_rev = float(df["extended_price"].sum())
    base_cogs = float((df["cogs"] if "cogs" in df.columns else df["unit_cost"] * df["quantity"]).sum())
    base_freight = float(df.get("freight", pd.Series([0]*len(df))).sum())
    base_gp_line = (df["extended_price"] - (df["unit_cost"]*df["quantity"]) - df.get("freight", 0).fillna(0))

    base_gm = float(base_gp_line.sum())
    sc_rev = float(scenario_rev.sum())
    sc_cogs = float(scenario_cogs.sum())
    sc_freight = float(scenario_freight.sum())
    sc_gm = float(scenario_gp_line.sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Revenue (Baseline)", f"${base_rev:,.0f}")
    m2.metric("Revenue (Scenario)", f"${sc_rev:,.0f}", delta=f"${(sc_rev - base_rev):,.0f}")
    m3.metric("Gross Margin (Baseline)", f"${base_gm:,.0f}")
    m4.metric("Gross Margin (Scenario)", f"${sc_gm:,.0f}", delta=f"${(sc_gm - base_gm):,.0f}")

    # Scenario Bridge (Price → Cost → Discount Cap → Freight)
    qty = scenario["quantity"]
    base_price = df["unit_price"]
    base_cost = df["unit_cost"]
    base_disc = (df.get("discount", 0).fillna(0) + df.get("rebate", 0).fillna(0))
    base_pre_disc = qty * base_price
    base_rev_line = base_pre_disc - base_disc
    base_cogs_line = qty * base_cost
    base_freight_line = df.get("freight", 0).fillna(0)
    base_gp_line_seq = base_rev_line - base_cogs_line - base_freight_line

    price_rev_line = qty * scenario_unit_price - base_disc
    price_gp_line = price_rev_line - base_cogs_line - base_freight_line
    price_eff = float(price_gp_line.sum() - base_gp_line_seq.sum())

    cost_cogs_line = qty * scenario_unit_cost
    cost_gp_line = price_rev_line - cost_cogs_line - base_freight_line
    cost_eff = float(cost_gp_line.sum() - price_gp_line.sum())

    cap_rev_line = qty * scenario_unit_price - capped_discounts
    cap_gp_line = cap_rev_line - cost_cogs_line - base_freight_line
    cap_eff = float(cap_gp_line.sum() - cost_gp_line.sum())

    freight_line = scenario_freight
    scen_gp_line_final = cap_rev_line - cost_cogs_line - freight_line
    freight_eff = float(scen_gp_line_final.sum() - cap_gp_line.sum())

    fig = go.Figure(go.Waterfall(
        name="Scenario Bridge",
        orientation="v",
        measure=["absolute","relative","relative","relative","relative","total"],
        x=["Baseline GM", "Price Δ", "Cost Δ", "Discount Cap Δ", "Freight Δ", "Scenario GM"],
        y=[float(base_gp_line_seq.sum()), price_eff, cost_eff, cap_eff, freight_eff, float(scen_gp_line_final.sum())],
        increasing={"marker":{"color": COLOR["green"]}},
        decreasing={"marker":{"color": COLOR["red"]}},
        totals={"marker":{"color": COLOR["blue"]}},
        connector={"line":{"color":"#2E2E2E"}}
    ))
    fig.update_layout(title="Scenario Bridge — Baseline → Scenario", height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Assumes no volume elasticity; order of effects is illustrative. Not financial advice.")

    # Scenario impact by segment (if available)
    if "segment_code" in df.columns:
        st.markdown("**Scenario Impact by Segment (Δ GM $)**")
        seg_view = pd.DataFrame({
            "segment_code": df["segment_code"],
            "base_gp": base_gp_line,
            "sc_gp": scenario_gp_line
        })
        seg_view["delta_gp"] = seg_view["sc_gp"] - seg_view["base_gp"]
        seg_view = seg_view.groupby("segment_code", as_index=False)["delta_gp"].sum().sort_values("delta_gp", ascending=False)
        st.dataframe(seg_view, use_container_width=True)
        download_df(seg_view, "Scenario Impact by Segment", "scenario_impact_segment.csv")

# =========================
# Assumptions & Formulas + Build stamp
# =========================
st.subheader("Assumptions & Formulas")
st.markdown(
    """
**Assumptions**
- Anonymized IDs only; PII is excluded/suppressed.
- Currency not normalized unless explicitly stated.
- Returns and freight only included if provided as `returns_amount` / `freight`.
- Aggregation mode uses `product_id × month` (and `customer_id` if present).

**Formulas**
- `extended_price = quantity * unit_price − discount − rebate`
- `cogs = quantity * unit_cost`
- `gross_margin = extended_price − cogs`
- `gm% = gross_margin / extended_price`
- `discount+rebate % = (discount + rebate) / (extended_price + discount + rebate)`
"""
)

st.caption(f"Build: {datetime.now().strftime('%Y-%m-%d %H:%M')} • Phase 1 Diagnostic • No external benchmarks")
