
import io
import json
import math
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
from difflib import get_close_matches

APP_PW = "PP-PULSE"
st.set_page_config(page_title="Profit Pulse", layout="wide")
st.title("📈 Profit Pulse — Private Price & Profitability Diagnostic")

# ---------------- Password Gate ----------------
pw = st.text_input("Enter password to begin", type="password", help="Phase 1 password required.")
if pw != APP_PW:
    st.info("Please enter the correct password to proceed.")
    st.stop()

with st.expander("About column 'synonyms' (detector)", expanded=False):
    st.write("""
**Synonyms** are alternate header names we recognize for each target field.  
Example: for **invoice_id**, we also try: `invoice`, `invoice_no`, `invoice_number`, `doc_id`, `order_id`, `so_number`, etc.  
If your exports use different labels, saving a **Mapping Profile** (below) trains the app to auto-map your headers next time—no code changes.
""")

# ---------------- Uploads ----------------
st.subheader("Upload Files")
c1, c2 = st.columns(2)
with c1:
    trx_file = st.file_uploader("Transactional (required)", type=["csv", "xlsx"], help="12+ months preferred")
with c2:
    cust_file = st.file_uploader("Customer master (optional)", type=["csv", "xlsx"])

c3, c4, c5 = st.columns(3)
with c3:
    prod_file = st.file_uploader("Product master (optional)", type=["csv", "xlsx"])
with c4:
    price_file = st.file_uploader("Price file (optional)", type=["csv", "xlsx"])
with c5:
    cts_file = st.file_uploader("Cost-to-Serve map (optional)", type=["csv", "xlsx"])

window = st.radio("Time window", ["3 months", "6 months", "12 months (default)", "Full dataset"], index=2, horizontal=True)

# ---------------- Mapping Profile (load) ----------------
st.subheader("Mapping Profiles")
mp_col1, mp_col2 = st.columns([2,1])
with mp_col1:
    mapping_profile_upload = st.file_uploader("Load a Mapping Profile (JSON)", type=["json"], key="mpu")
with mp_col2:
    st.caption("Tip: Save a profile after your first run; reuse it for future uploads from the same system.")

# ---------------- Helpers ----------------
def read_any(upload):
    if upload is None:
        return None
    if upload.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    return pd.read_csv(upload)

def normalize_colname(c):
    return (
        str(c).strip().lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )

SYNONYMS = {
    "invoice_id": {"invoice_id","invoice","invoice_no","invoice_number","inv_id","doc_id","document_id","order_id","so_number","so_id","txn_id"},
    "invoice_date": {"invoice_date","date","invoice_dt","order_date","posting_date","txn_date","document_date"},
    "customer_id": {"customer_id","customer","customer_code","client_id","account","account_id","bill_to_id","buyer_id","ship_to_id"},
    "product_id": {"product_id","sku","item","item_id","material","part_number","product","product_code"},
    "quantity": {"quantity","qty","units","unit_qty","qty_sold","ordered_qty"},
    "net_price": {"net_price","unit_price","price","sell_price","sell_prc","realized_price","net_unit_price"},
    "extended_price": {"extended_price","line_total","line_amount","amount","revenue","sales_amount","net_amount","ext_price","gross_sales"},
    "standard_cost": {"standard_cost","unit_cost","cost","cogs_unit","cogs_per_unit"},
    "variable_cost": {"variable_cost","var_cost","vcost_unit"},
    "discount_amount": {"discount_amount","discount","disc_amt","promo_discount","line_discount"},
    "rebate_amount": {"rebate_amount","rebate","rbt_amt"},
    "freight_cost": {"freight_cost","freight","shipping_cost","logistics_cost","delivery_cost"},
    "returns_flag": {"returns_flag","return","is_return","returns","rma_flag","credit_flag","negative_sale"},
    "currency": {"currency","curr","fx","currency_code","iso_currency"},
    "uom": {"uom","unit","unit_of_measure","measure_unit"},
}

TARGETS_REQUIRED = ["invoice_id","invoice_date","customer_id","product_id","quantity"]
TARGETS_OPTIONAL = ["net_price","extended_price","standard_cost","variable_cost","discount_amount","rebate_amount","freight_cost","returns_flag","currency","uom"]

def auto_map_columns(df_cols):
    cols_norm = {normalize_colname(c): c for c in df_cols}
    result = {}
    # synonym match
    for tgt, syns in SYNONYMS.items():
        match = None
        for n, orig in cols_norm.items():
            if n in syns:
                match = orig
                break
        if match is None:
            # fuzzy fallback
            candidates = list(cols_norm.keys())
            guess = get_close_matches(tgt, candidates, n=1, cutoff=0.82)
            if guess:
                match = cols_norm[guess[0]]
        result[tgt] = match
    return result

def apply_mapping(df, mapping):
    rename_dict = {src: tgt for tgt, src in mapping.items() if src}
    df2 = df.rename(columns=rename_dict).copy()

    # Synthesize IDs if missing but name-like fields exist
    norm_cols = {normalize_colname(c): c for c in df.columns}
    if "customer_id" not in df2.columns:
        for alt in ["customer","customer_name","account","buyer","bill_to_name"]:
            if alt in norm_cols:
                orig = norm_cols[alt]
                codes = df[orig].astype("category").cat.codes + 1
                df2["customer_id"] = codes.map(lambda x: f"CUST{x}")
                break
    if "product_id" not in df2.columns:
        for alt in ["product","product_name","sku","item","material"]:
            if alt in norm_cols:
                orig = norm_cols[alt]
                codes = df[orig].astype("category").cat.codes + 1
                df2["product_id"] = codes.map(lambda x: f"PROD{x}")
                break

    # Derive extended_price if missing
    if "extended_price" not in df2.columns and {"quantity","net_price"}.issubset(df2.columns):
        df2["extended_price"] = pd.to_numeric(df2["quantity"], errors="coerce") * pd.to_numeric(df2["net_price"], errors="coerce")

    return df2

def parse_date(series):
    return pd.to_datetime(series, errors="coerce", utc=False).dt.tz_localize(None)

def traffic_light(metric, value):
    if metric == "GM%":
        if value >= 0.30: return "🟢", "≥30%"
        if value >= 0.20: return "🟡", "20–30%"
        return "🔴", "<20%"
    if metric == "Disc+Rebate % of Rev":
        if value <= 0.08: return "🟢", "≤8%"
        if value <= 0.12: return "🟡", "8–12%"
        return "🔴", ">12%"
    if metric == "Returns %":
        if value <= 0.02: return "🟢", "≤2%"
        if value <= 0.04: return "🟡", "2–4%"
        return "🔴", ">4%"
    if metric == "Data Completeness":
        if value >= 0.90: return "🟢", "≥90%"
        if value >= 0.75: return "🟡", "75–90%"
        return "🔴", "<75%"
    if metric == "Freight Recovery %":
        if value >= 0.80: return "🟢", "≥80%"
        if value >= 0.65: return "🟡", "65–80%"
        return "🔴", "<65%"
    return "⚪", ""

def size_mb(upload):
    return (upload.size / 1_048_576.0) if upload is not None else 0.0

# ---------------- Run ----------------
if st.button("Run diagnostic", type="primary", use_container_width=True):
    if trx_file is None:
        st.error("Transactional file is required.")
        st.stop()

    trx_raw = read_any(trx_file)

    # Optional: load mapping profile JSON
    loaded_profile = None
    if mapping_profile_upload is not None:
        try:
            loaded_profile = json.load(mapping_profile_upload)
            st.success(f"Loaded mapping profile: {loaded_profile.get('profile_name','(unnamed)')}")
        except Exception as e:
            st.warning(f"Could not read mapping profile JSON: {e}")

    st.markdown("### 🔎 Column mapping")
    st.caption("We detected likely matches. Confirm or change each field; choose **(not present)** if the data doesn't exist.")

    # Build mapping UI
    candidate = auto_map_columns(trx_raw.columns)

    # If a mapping profile exists, prefer it (only if the column exists in this file)
    if loaded_profile and "mapping" in loaded_profile:
        for tgt, src in loaded_profile["mapping"].items():
            if src and src in trx_raw.columns:
                candidate[tgt] = src

    ui_mapping = {}
    all_options = ["(not present)"] + list(trx_raw.columns)

    cols_required = st.columns(len(TARGETS_REQUIRED))
    for i, tgt in enumerate(TARGETS_REQUIRED):
        with cols_required[i]:
            default = candidate.get(tgt) if candidate.get(tgt) in all_options else "(not present)"
            ui_mapping[tgt] = st.selectbox(f"Required: {tgt}", all_options, index=all_options.index(default) if default in all_options else 0)

    cols_optional = st.columns(min(5, len(TARGETS_OPTIONAL)))
    for i, tgt in enumerate(TARGETS_OPTIONAL):
        if i % 5 == 0 and i != 0:
            cols_optional = st.columns(5)
        with cols_optional[i % 5]:
            default = candidate.get(tgt) if candidate.get(tgt) in all_options else "(not present)"
            ui_mapping[tgt] = st.selectbox(f"Optional: {tgt}", all_options, index=all_options.index(default) if default in all_options else 0)

    # Save Mapping Profile (download JSON)
    st.markdown("#### 💾 Save this mapping as a profile")
    prof_col1, prof_col2 = st.columns([2,1])
    with prof_col1:
        profile_name = st.text_input("Profile name (e.g., 'Acme NetSuite Export')", value="My Mapping Profile")
    with prof_col2:
        mp = {
            "profile_name": profile_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "mapping": {k:(None if v=='(not present)' else v) for k,v in ui_mapping.items()}
        }
        mp_bytes = json.dumps(mp, indent=2).encode("utf-8")
        st.download_button("Download Mapping Profile (.json)", mp_bytes, file_name=f"{profile_name.replace(' ','_').lower()}_mapping.json", use_container_width=True)

    # Apply mapping
    mapping = {k:(None if v=="(not present)" else v) for k, v in ui_mapping.items()}
    trx = apply_mapping(trx_raw, mapping)

    # Validate minimum required
    missing_after = [c for c in TARGETS_REQUIRED if c not in trx.columns]
    if missing_after:
        st.error("Missing required fields after mapping: " + ", ".join(missing_after))
        st.write("Minimal headers needed: invoice_id, invoice_date, customer_id, product_id, quantity.")
        st.stop()

    # Continue ETL
    etl_issues = []

    # Parse dates
    trx["invoice_date"] = pd.to_datetime(trx["invoice_date"], errors="coerce", utc=False).dt.tz_localize(None)
    if trx["invoice_date"].isna().mean() > 0.05:
        etl_issues.append(">5% invoice_date not parseable")

    # Numeric coercions
    for col in ["quantity","net_price","extended_price","standard_cost","variable_cost","discount_amount","rebate_amount","freight_cost"]:
        if col in trx.columns:
            trx[col] = pd.to_numeric(trx[col], errors="coerce")

    # Dedupe (safe)
    subset_cols = [c for c in ["invoice_id","product_id","invoice_date","customer_id"] if c in trx.columns]
    before = len(trx)
    if subset_cols:
        trx = trx.drop_duplicates(subset=subset_cols, keep="first")
    dedup_rate = (before - len(trx)) / before if before else 0.0

    # Oversize aggregation
    ROW_CAP, MB_CAP = 1_000_000, 250
    oversized = (len(trx) > ROW_CAP) or (trx_file is not None and getattr(trx_file, "size", 0) / 1_048_576.0 > MB_CAP)
    aggregated = False
    if oversized:
        st.warning("Large transactional file detected. Auto-aggregating to month × product_id (and customer_id if present).")
        trx["_ym"] = trx["invoice_date"].dt.to_period("M").astype(str)
        group_cols = ["_ym","product_id"] + (["customer_id"] if "customer_id" in trx.columns else [])
        agg_cols = {c:"sum" for c in ["quantity","extended_price","discount_amount","rebate_amount","freight_cost"] if c in trx.columns}
        cost_col = "standard_cost" if "standard_cost" in trx.columns else ("variable_cost" if "variable_cost" in trx.columns else None)
        if cost_col: agg_cols[cost_col] = "sum"
        trx = trx.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
        aggregated = True

    # Rolling window filter
    if trx["invoice_date"].notna().any() and window != "Full dataset":
        end_date = trx["invoice_date"].max()
        months = {"3 months":3, "6 months":6, "12 months (default)":12}[window]
        start_date = end_date - relativedelta(months=months) + pd.offsets.MonthBegin(0)
        trx = trx[trx["invoice_date"].between(start_date, end_date)]

    # KPIs
    cost_col = "standard_cost" if "standard_cost" in trx.columns else ("variable_cost" if "variable_cost" in trx.columns else None)
    revenue = trx["extended_price"].sum() if "extended_price" in trx.columns else np.nan
    cogs = (trx[cost_col] * trx["quantity"]).sum() if (cost_col in trx.columns and "quantity" in trx.columns) else np.nan
    gm = revenue - cogs if (not pd.isna(revenue) and not pd.isna(cogs)) else np.nan
    gm_pct = gm / revenue if (revenue and not pd.isna(gm)) else np.nan

    disc_pct = ((trx.get("discount_amount", pd.Series([0])).sum() + trx.get("rebate_amount", pd.Series([0])).sum()) / revenue) if revenue else np.nan
    returns_mask = pd.Series(False, index=trx.index)
    if "quantity" in trx.columns:
        returns_mask |= trx["quantity"] < 0
    if "returns_flag" in trx.columns:
        returns_mask |= trx["returns_flag"].fillna(0).astype(int) == 1
    returns_rev = trx.loc[returns_mask, "extended_price"].sum() if ("extended_price" in trx.columns) else 0.0
    returns_pct = (abs(returns_rev) / abs(revenue)) if revenue else np.nan
    freight_cost = trx.get("freight_cost", pd.Series([0])).sum()
    freight_recovery = (0.0 / freight_cost) if freight_cost else np.nan  # proxy if no freight revenue

    # Completeness estimate
    present_mask = pd.Series(True, index=trx.index)
    for key in ["invoice_id","invoice_date","customer_id","product_id"]:
        if key in trx.columns:
            present_mask &= trx[key].notna()
    data_completeness = present_mask.mean() if len(trx) else 0.0

    tier = "Tier A (ready)"
    if data_completeness < 0.90:
        tier = "Tier C (triage)"
    elif data_completeness < 0.99 or len(etl_issues) > 0:
        tier = "Tier B (fixable)"

    # ---------------- Outputs ----------------
    with st.expander("1) ETL & Validation Results", expanded=True):
        st.write({
            "Rows": int(len(trx)),
            "Deduped lines removed %": f"{dedup_rate:.2%}",
            "Derived extended_price": "Yes" if "extended_price" in trx.columns else "No",
            "Aggregated on ingest": aggregated,
        })
        if len(etl_issues) > 0:
            st.warning("ETL Notes:")
            for i in etl_issues:
                st.write("• " + i)
        st.markdown(f"**Data Quality:** {tier}")

    remediation = []
    if cost_col is None:
        remediation.append("Transactional: missing cost field → cannot compute GM% precisely. (High)")
    if data_completeness < 0.90:
        remediation.append("Key presence below 90% → provide missing keys to unlock full diagnostic. (High)")
    if not remediation:
        remediation.append("No critical gaps detected. (Info)")

    st.subheader("3) Dashboard Readout")
    def add_kpi_row(name, val):
        emoji, band = ("⚪","") if pd.isna(val) else (("🟢","≥30%") if (name=="Gross Margin %" and val>=0.30) else ("🟡","") )
        # We will use the shared traffic_light for consistent bands
        from math import isnan
        metric = "Disc+Rebate % of Rev" if name=="Discount+Rebate % of Rev" else name
        em, band = traffic_light(metric, 0.0 if pd.isna(val) else val)
        val_fmt = f"{val:.1%}" if not pd.isna(val) else "n/a"
        return [name, val_fmt, f"{em} {band}"]
    kpi_rows = [
        add_kpi_row("Gross Margin %", gm_pct),
        add_kpi_row("Discount+Rebate % of Rev", disc_pct),
        add_kpi_row("Freight Recovery %", freight_recovery),
        add_kpi_row("Returns %", returns_pct),
        add_kpi_row("Data Completeness", data_completeness),
    ]
    kpis_df = pd.DataFrame(kpi_rows, columns=["Metric","Value","Status / Threshold"])
    st.dataframe(kpis_df, use_container_width=True)

    st.subheader("4) Briefing Deck — Highlights & Questions")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Price–Volume–Mix (indicative)**")
        if "invoice_date" in trx.columns and "quantity" in trx.columns and "extended_price" in trx.columns:
            ym = trx["invoice_date"].dt.to_period("M").astype(str)
            tmp = pd.DataFrame({"ym": ym, "units": trx["quantity"], "rev": trx["extended_price"]})
            pv = tmp.groupby("ym", as_index=False).sum()
            fig = px.line(pv, x="ym", y=["rev","units"], markers=True, title="Revenue & Units by Month")
            st.plotly_chart(fig, use_container_width=True)
        st.write("- Trend lines help spot seasonality and potential mix shifts.")
        st.write("- Price effect approximated via rev/unit over time.")
        st.write("- Validate anomalies against promotions/returns.")
        st.markdown("**Questions:**")
        st.write("- Any policy changes that explain observed dips?")
        st.write("- Seasonality vs. share loss—what’s expected?")

    with colB:
        st.markdown("**Segmentation Pulse**")
        if "product_id" in trx.columns and "extended_price" in trx.columns:
            by_prod = trx.groupby("product_id", dropna=False)["extended_price"].sum().reset_index()
            fig2 = px.bar(by_prod, x="product_id", y="extended_price", title="Revenue by Product")
            st.plotly_chart(fig2, use_container_width=True)
            st.write("- Concentration in a few products can hide margin issues.")
        else:
            st.info("Provide `product_id` and `extended_price` for product revenue view.")

    st.subheader("5) Opportunities & Risks")
    opps = []
    if not pd.isna(disc_pct): opps.append(f"Reduce discount+rebate % from {disc_pct:.1%} toward ≤8%.")
    if not pd.isna(freight_recovery): opps.append("Improve freight recovery toward ≥80%.")
    if tier != "Tier A (ready)": opps.append("Close data quality gaps to Tier A to unlock deeper cuts.")
    if not opps: opps.append("Maintain price discipline; monitor outliers and returns.")
    for i, o in enumerate(opps, 1): st.write(f"{i}) {o}")

    # ---------------- Save Project (bundle) ----------------
    st.subheader("Save / Load Project")
    st.caption("A project includes your mapping profile and key results. No raw transactional rows are included by default.")

    # Build project JSON
    project = {
        "project_name": st.text_input("Project name", value="My Profit Pulse Project"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "window": window,
        "tier": tier,
        "kpis": {r["Metric"]: r["Value"] for _, r in kpis_df.iterrows()},
        "remediation": remediation,
        "mapping_profile": mp,  # from above
        "notes": "No raw data included for privacy."
    }

    # Prepare ZIP in-memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(project, indent=2))
        z.writestr("mapping.json", json.dumps(mp, indent=2))
        z.writestr("kpis.csv", kpis_df.to_csv(index=False))

    st.download_button("💾 Download Project Bundle (.zip)", buf.getvalue(), file_name=f"{project['project_name'].replace(' ','_').lower()}_project.zip", use_container_width=True)

    # -------- Load Project (optional) --------
    proj_upload = st.file_uploader("Load a Project Bundle (.zip)", type=["zip"], key="proj_zip")
    if proj_upload is not None:
        try:
            with zipfile.ZipFile(proj_upload) as z:
                with z.open("project.json") as pj:
                    pj_data = json.loads(pj.read().decode("utf-8"))
                st.success(f"Loaded project: {pj_data.get('project_name','(unnamed)')} — Tier: {pj_data.get('tier')} — Window: {pj_data.get('window')}")
                st.json(pj_data.get("kpis", {}))
        except Exception as e:
            st.error(f"Could not read project zip: {e}")
