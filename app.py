import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta

APP_PW = "PP-PULSE"

# ---------- UI ----------
st.set_page_config(page_title="Profit Pulse", layout="wide")
st.title("📈 Profit Pulse — Private Price & Profitability Diagnostic")

# ---------- Password Gate ----------
pw = st.text_input("Enter password to begin", type="password", help="Phase 1 password required.")
if pw != APP_PW:
    st.info("Please enter the correct password to proceed.")
    st.stop()

# ---------- Uploads ----------
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

# Rolling window selector
window = st.radio("Time window", ["3 months", "6 months", "12 months (default)", "Full dataset"], index=2, horizontal=True)

# ---------- Helpers ----------
def read_any(upload):
    if upload is None:
        return None
    if upload.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    return pd.read_csv(upload)

def parse_date(series):
    # Try best-effort parse
    return pd.to_datetime(series, errors="coerce", utc=False).dt.tz_localize(None)

def traffic_light(metric, value):
    # Defaults from spec; returns emoji + band
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

def month_floor(dt):
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)

def summarize_pct(numer, denom):
    if denom == 0 or pd.isna(denom): return np.nan
    return numer / denom

def detect_mixed_currency(df):
    # Heuristic: presence of 'currency' column OR money-like symbols in price columns
    money_cols = [c for c in df.columns if c.lower() in ("net_price","extended_price","standard_cost","variable_cost","discount_amount","rebate_amount","freight_cost")]
    symbol_found = False
    for c in money_cols:
        if df[c].dtype == object:
            sample = df[c].astype(str).head(200).str.contains(r"[$€£]|[,]", regex=True, na=False).any()
            symbol_found = symbol_found or sample
    has_currency_col = any(c.lower()=="currency" for c in df.columns)
    return (symbol_found and not has_currency_col), has_currency_col

def safe_number(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def kpi_tile(label, value, band_text, emoji):
    st.metric(label=label, value=value, delta=f"{emoji} {band_text}")

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# ---------- Run Diagnostic ----------
if st.button("Run diagnostic", type="primary", use_container_width=True):
    if trx_file is None:
        st.error("Transactional file is required.")
        st.stop()

    trx = read_any(trx_file)
    cust = read_any(cust_file)
    prod = read_any(prod_file)
    price = read_any(price_file)
    cts = read_any(cts_file)

    # -------- Data size guardrail --------
    ROW_CAP, MB_CAP = 1_000_000, 250
    oversized = (len(trx) > ROW_CAP) or (size_mb(trx_file) > MB_CAP)

    # -------- ETL & Validation --------
    etl_issues = []

    # Column presence
    required_keys = ["invoice_id","invoice_date","customer_id","product_id","quantity"]
    missing_required = [c for c in required_keys if c not in trx.columns]
    if missing_required:
        etl_issues.append(f"Transactional: missing required columns {missing_required}")

    # Date parse
    if "invoice_date" in trx.columns:
        trx["invoice_date"] = parse_date(trx["invoice_date"])
        if trx["invoice_date"].isna().mean() > 0.05:
            etl_issues.append(">5% invoice_date not parseable")

    # Numeric coercions
    for col in ["quantity","net_price","extended_price","standard_cost","variable_cost","discount_amount","rebate_amount","freight_cost"]:
        if col in trx.columns:
            trx[col] = pd.to_numeric(trx[col], errors="coerce")

    # Derive extended_price if missing
    if "extended_price" not in trx.columns and "net_price" in trx.columns and "quantity" in trx.columns:
        trx["extended_price"] = trx["quantity"] * trx["net_price"]

    # Derive cost/unit
    cost_col = None
    if "standard_cost" in trx.columns: cost_col = "standard_cost"
    elif "variable_cost" in trx.columns: cost_col = "variable_cost"

    # Deduplicate invoice lines (simple heuristic)
    before = len(trx)
    trx = trx.drop_duplicates(subset=[c for c in ["invoice_id","product_id","invoice_date","customer_id"] if c in trx.columns], keep="first")
    dedup_rate = (before - len(trx)) / before if before else 0.0

    # Orphans after joining
    orphan_customer = orphan_product = np.nan
    if cust is not None and "customer_id" in trx.columns and "customer_id" in cust.columns:
        orphan_customer = 1 - (trx["customer_id"].isin(cust["customer_id"]).mean())

    if prod is not None and "product_id" in trx.columns and "product_id" in prod.columns:
        orphan_product = 1 - (trx["product_id"].isin(prod["product_id"]).mean())

    # Returns detection (negative qty or returns_flag == 1)
    returns_mask = pd.Series(False, index=trx.index)
    if "quantity" in trx.columns:
        returns_mask |= trx["quantity"] < 0
    if "returns_flag" in trx.columns:
        returns_mask |= trx["returns_flag"].fillna(0).astype(int) == 1

    # Mixed currency detection
    mixed_currency_signals, has_currency_col = detect_mixed_currency(trx)
    if mixed_currency_signals and not has_currency_col:
        etl_issues.append("Potential mixed-currency data without a currency column (Tier C). Please add 'currency' or upload FX rates.")

    # Rolling window filter
    if "invoice_date" in trx.columns and trx["invoice_date"].notna().any() and window != "Full dataset":
        end_date = trx["invoice_date"].max()
        months = {"3 months":3, "6 months":6, "12 months (default)":12}[window]
        start_date = end_date - relativedelta(months=months) + pd.offsets.MonthBegin(0)
        trx = trx[trx["invoice_date"].between(start_date, end_date)]

    # Auto-aggregate for over-size
    aggregated = False
    if oversized:
        st.warning("Large transactional file detected. Auto-aggregating to month × product_id (and customer_id if present).")
        if "invoice_date" in trx.columns:
            trx["_ym"] = trx["invoice_date"].dt.to_period("M").astype(str)
        else:
            trx["_ym"] = "Unknown"
        group_cols = ["_ym","product_id"]
        if "customer_id" in trx.columns: group_cols.append("customer_id")
        agg_cols = {c:"sum" for c in ["quantity","extended_price","discount_amount","rebate_amount","freight_cost"] if c in trx.columns}
        if cost_col: agg_cols[cost_col] = "sum"
        trx = trx.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
        aggregated = True

    # Data completeness (keys present for required cols)
    present_mask = pd.Series(True, index=trx.index)
    for key in ["invoice_id","invoice_date","customer_id","product_id"]:
        if key in trx.columns:
            present_mask &= trx[key].notna()
    data_completeness = present_mask.mean() if len(trx) else 0.0

    # GM calc (row-level if possible)
    revenue = trx["extended_price"].sum() if "extended_price" in trx.columns else np.nan
    cogs = (trx[cost_col] * trx["quantity"]).sum() if (cost_col in trx.columns and "quantity" in trx.columns) else np.nan
    gm = revenue - cogs if (not pd.isna(revenue) and not pd.isna(cogs)) else np.nan
    gm_pct = gm / revenue if (revenue and not pd.isna(gm)) else np.nan

    disc_pct = summarize_pct(trx.get("discount_amount", pd.Series([0])).sum() + trx.get("rebate_amount", pd.Series([0])).sum(), revenue)
    returns_rev = trx.loc[returns_mask, "extended_price"].sum() if ("extended_price" in trx.columns) else 0.0
    returns_pct = summarize_pct(abs(returns_rev), abs(revenue)) if revenue else np.nan
    freight_cost = trx.get("freight_cost", pd.Series([0])).sum()
    freight_recovery = summarize_pct(0.0, freight_cost) if freight_cost else np.nan  # proxy

    # Tiering
    tier = "Tier A (ready)"
    if data_completeness < 0.90 or mixed_currency_signals:
        tier = "Tier C (triage)"
    elif data_completeness < 0.99 or len(etl_issues) > 0:
        tier = "Tier B (fixable)"

    # ---------- 1) ETL & Validation ----------
    with st.expander("1) ETL & Validation Results", expanded=True):
        st.write({
            "Rows": len(trx),
            "File size (MB)": round(size_mb(trx_file), 2),
            "Deduped lines removed %": f"{dedup_rate:.2%}",
            "Orphan customer_id %": None if math.isnan(orphan_customer) else f"{orphan_customer:.2%}",
            "Orphan product_id %": None if math.isnan(orphan_product) else f"{orphan_product:.2%}",
            "Derived extended_price": "Yes" if "extended_price" in trx.columns else "No",
            "Aggregated on ingest": aggregated,
        })
        if etl_issues:
            st.warning("ETL Notes:")
            for i in etl_issues:
                st.write("• " + i)
        st.markdown(f"**Data Quality:** {tier}")

    # ---------- 2) Remediation Plan (always) ----------
    remediation = []
    if cost_col is None:
        remediation.append("Transactional: missing cost field → cannot compute GM% precisely. (High)")
    if math.isnan(orphan_customer) or orphan_customer > 0.0:
        remediation.append("Customer master join incomplete → segmentation views limited. (Medium)")
    if math.isnan(orphan_product) or orphan_product > 0.0:
        remediation.append("Product master join incomplete → product family/category views limited. (Medium)")
    if mixed_currency_signals and not has_currency_col:
        remediation.append("Potential mixed currencies without 'currency' column → add currency or FX rates. (High)")
    if data_completeness < 0.90:
        remediation.append("Key presence below 90% → provide missing keys to unlock full diagnostic. (High)")
    if not remediation:
        remediation.append("No critical gaps detected. (Info)")
    with st.expander("2) Data Remediation Plan (ranked checklist)", expanded=True):
        for r in remediation:
            st.write("• " + r)

    # ---------- 3) Dashboard KPIs ----------
    st.subheader("3) Dashboard Readout")
    kpis = []
    # GM%
    emoji, band = traffic_light("GM%", gm_pct if not pd.isna(gm_pct) else -1)
    kpis.append(("Gross Margin %", f"{gm_pct:.1%}" if not pd.isna(gm_pct) else "n/a", f"{emoji} {band}"))
    # Disc+Rebate
    emoji, band = traffic_light("Disc+Rebate % of Rev", disc_pct if not pd.isna(disc_pct) else 0.0)
    kpis.append(("Discount+Rebate % of Rev", f"{disc_pct:.1%}" if not pd.isna(disc_pct) else "n/a", f"{emoji} {band}"))
    # Freight Recovery
    emoji, band = traffic_light("Freight Recovery %", freight_recovery if not pd.isna(freight_recovery) else 0.0)
    kpis.append(("Freight Recovery %", f"{freight_recovery:.1%}" if not pd.isna(freight_recovery) else "n/a", f"{emoji} {band}"))
    # Returns %
    emoji, band = traffic_light("Returns %", returns_pct if not pd.isna(returns_pct) else 0.0)
    kpis.append(("Returns %", f"{returns_pct:.1%}" if not pd.isna(returns_pct) else "n/a", f"{emoji} {band}"))
    # Data completeness
    emoji, band = traffic_light("Data Completeness", data_completeness)
    kpis.append(("Data Completeness", f"{data_completeness:.1%}", f"{emoji} {band}"))

    st.dataframe(pd.DataFrame(kpis, columns=["Metric", "Value", "Status / Threshold"]), use_container_width=True)

    # ---------- 4) Briefing Deck Highlights ----------
    st.subheader("4) Briefing Deck — Highlights & Questions")
    colA, colB = st.columns(2)

    # Price-Volume-Mix (very simple proxy)
    with colA:
        st.markdown("**Price–Volume–Mix (indicative)**")
        if "invoice_date" in trx.columns and "quantity" in trx.columns and "extended_price" in trx.columns:
            ym = trx["invoice_date"].dt.to_period("M").astype(str) if "invoice_date" in trx.columns else "Unknown"
            tmp = pd.DataFrame({"ym": ym, "units": trx["quantity"], "rev": trx["extended_price"]})
            pv = tmp.groupby("ym", as_index=False).sum()
            fig = px.line(pv, x="ym", y=["rev","units"], markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.write("- Revenue trend and units shown by month (indicative).")
        st.write("- Use to spot seasonality and mix-driven shifts.")
        st.write("- Price effect approximated via rev/unit over time.")
        st.write("- Validate anomalies against promotions/returns.")
        st.write("- If multi-currency, confirm FX before interpretation.")
        st.markdown("**Questions:**")
        st.write("- Any policy changes that explain price dips?")
        st.write("- Seasonality vs. share loss—what’s expected?")

    # Segmentation pulse (by product family if product master exists)
    with colB:
        st.markdown("**Segmentation Pulse**")
        if prod is not None and "product_id" in trx.columns and "product_id" in prod.columns and "family" in prod.columns and "extended_price" in trx.columns:
            fam_map = prod[["product_id","family"]].drop_duplicates()
            tmp = trx.merge(fam_map, on="product_id", how="left")
            by_fam = tmp.groupby("family", dropna=False)["extended_price"].sum().reset_index()
            fig2 = px.bar(by_fam, x="family", y="extended_price", title="Revenue by Product Family")
            st.plotly_chart(fig2, use_container_width=True)
            st.write("- Families with outsized revenue may hide low margins.")
            st.write("- Check discount concentration within top families.")
            st.write("- Consider lifecycle (if available) for context.")
            st.markdown("**Questions:**")
            st.write("- Are floors differentiated by family/tier?")
            st.write("- Which families face most competitive pressure?")
        else:
            st.info("Upload a Product master with `product_id` + `family` to unlock family-level segmentation.")

    # ---------- 5) Opportunities, Risks, Questions ----------
    st.subheader("5) Opportunities & Risks")
    opps = []
    if not pd.isna(disc_pct):
        opps.append(f"Reduce discount+rebate % from {disc_pct:.1%} toward ≤8%.")
    if not pd.isna(freight_recovery):
        opps.append("Improve freight recovery toward ≥80%.")
    if tier != "Tier A (ready)":
        opps.append("Close data quality gaps to Tier A to unlock deeper cuts.")
    if not opps:
        opps.append("Maintain price discipline; monitor outliers and returns.")
    for i, o in enumerate(opps, 1):
        st.write(f"{i}) {o}")

    st.subheader("6) Questions for First Client Call")
    st.write("- Is freight under-recovery policy-driven or execution-driven?")
    st.write("- Are rebates targeted to growth or broadly applied?")
    st.write("- How are floors differentiated by segment/tier?")

    st.subheader("7) Assumptions & Formulas")
    st.code(
        "Gross Margin $ = extended_price – (cost * quantity)\\n"
        "Gross Margin % = Gross Margin $ / extended_price\\n"
        "Freight recovery = freight revenue / freight cost  (freight revenue not provided → shown as n/a)\\n"
        "Results may be indicative if inputs are incomplete."
    )

    # ---------- 6) Export XLSX bundle ----------
    st.subheader("Download")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        # ETL sheet
        etl_df = pd.DataFrame([
            ["Rows", len(trx)],
            ["Deduped lines removed %", f"{dedup_rate:.2%}"],
            ["Orphan customer_id %", None if math.isnan(orphan_customer) else f"{orphan_customer:.2%}"],
            ["Orphan product_id %", None if math.isnan(orphan_product) else f"{orphan_product:.2%}"],
            ["Aggregated on ingest", aggregated],
            ["Data Quality", tier],
        ], columns=["Metric","Value"])
        etl_df.to_excel(xw, sheet_name="ETL", index=False)

        # Remediation
        pd.DataFrame({"Remediation": remediation}).to_excel(xw, sheet_name="Remediation", index=False)

        # KPI
        pd.DataFrame(kpis, columns=["Metric","Value","Status/Threshold"]).to_excel(xw, sheet_name="KPI", index=False)

    st.download_button(
        "Download diagnostic bundle (XLSX)",
        data=buf.getvalue(),
        file_name="profit_pulse_diagnostic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
