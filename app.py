# app.py
import os
import io
import json
import re
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from dateutil.parser import parse as parse_date
from rapidfuzz import process, fuzz

# -------------------------------
# Config & Constants
# -------------------------------
APP_TITLE = "Profit Pulse — Phase 1 Diagnostic"
PASSWORD_ENV = "PROFIT_PULSE_PASSWORD"

# Canonical schema the app expects
REQUIRED_COLS = ["transaction_date", "product_id", "quantity", "unit_price", "cost", "currency"]
OPTIONAL_COLS = ["customer_id", "channel", "discount", "rebate", "return_flag"]
ALL_CANONICAL = REQUIRED_COLS + OPTIONAL_COLS

AUTO_AGG_MAX_ROWS = 1_000_000
AUTO_AGG_MAX_BYTES = 250 * 1024 * 1024  # 250 MB

# Traffic-light thresholds
GM_GREEN = 0.30
DISC_REBATE_GREEN = 0.08
RETURNS_GREEN = 0.02
COMPLETENESS_GREEN = 0.90

PALETTE = {
    "bg": "#0f172a",      # slate
    "primary": "#0a4b78", # deep blue
    "accent": "#1e88e5",  # accent blue
    "teal": "#00a3a3",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "muted": "#64748b",
}

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TAXID_REGEX = re.compile(r"\b\d{2,3}-?\d{2,3}-?\d{2,4}\b")
ADDRESS_REGEX = re.compile(r"\d+\s+\w+\s+(St|Ave|Rd|Road|Street|Avenue|Boulevard|Blvd)\b", re.IGNORECASE)

# Synonyms & boolean tokens for schema mapper
SYNONYMS = {
    "transaction_date": ["date","doc_date","invoice_date","posting_date","txn_date","order_date"],
    "product_id": ["sku","item_id","product","material","product_code","item_code"],
    "quantity": ["qty","units","unit_qty","sold_qty","order_qty"],
    "unit_price": ["price","list_price","sell_price","sales_price","unitprice"],
    "cost": ["unit_cost","std_cost","cogs","cost_amt","cost_price"],
    "currency": ["curr","currency_code","iso_currency","fx_currency"],
    "customer_id": ["cust_id","customer","account_id","buyer_id","client_id"],
    "channel": ["sales_channel","channel_name","route_to_market","rtm"],
    "discount": ["disc","promo_discount","markdown","price_discount","allowance"],
    "rebate": ["rebate_amt","retro_rebate","billback","credit"],
    "return_flag": ["is_return","return","rma","returned","is_refund"]
}
BOOL_TRUE_DEFAULT = {"1","true","t","yes","y"}
BOOL_FALSE_DEFAULT = {"0","false","f","no","n"}

# -------------------------------
# Generic Helpers
# -------------------------------
def human_bytes(n):
    if n is None: return "Unknown"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def safe_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def detect_pii(df: pd.DataFrame):
    """Return list of tuples (column, issue) flagged as PII-like. Does not return raw values."""
    flags = []
    for col in df.columns:
        if df[col].dtype == object:
            series = df[col].dropna().astype(str)
            head = series.head(1000)  # sample for speed
            try:
                if head.str.contains(EMAIL_REGEX).any():
                    flags.append((col, "email-like pattern"))
                if head.str.contains(TAXID_REGEX).any():
                    flags.append((col, "tax-id-like pattern"))
                if head.str.contains(ADDRESS_REGEX).any():
                    flags.append((col, "address-like pattern"))
            except Exception:
                # If any weird dtype conversion issues, just skip that column
                pass
    return flags

def completeness_score(df: pd.DataFrame, required_cols=REQUIRED_COLS):
    present = [c for c in required_cols if c in df.columns]
    score = len(present) / len(required_cols) if required_cols else 1.0
    return score, present, [c for c in required_cols if c not in present]

def calc_fields(df: pd.DataFrame):
    df = df.copy()
    # Normalize numeric types
    for c in ["quantity", "unit_price", "cost", "discount", "rebate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics
    df["extended_price"] = df.get("quantity", 0) * df.get("unit_price", 0)
    df["net_price"] = df["extended_price"] - df.get("discount", 0) - df.get("rebate", 0)
    df["gross_margin"] = df["net_price"] - (df.get("quantity", 0) * df.get("cost", 0))
    df["gm_pct"] = np.where(df["net_price"] != 0, df["gross_margin"] / df["net_price"], np.nan)

    # Returns flag
    if "return_flag" in df.columns:
        s = df["return_flag"].astype(str).str.lower().str.strip()
        ret = s.isin(BOOL_TRUE_DEFAULT) | s.isin({"true","t","1","yes","y"})
        df["is_return"] = ret
    else:
        df["is_return"] = False
    return df

def aggregate_bigdata(df: pd.DataFrame, include_customer=False):
    df = df.copy()
    if "transaction_date" in df.columns:
        df["yyyymm"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.to_period("M").astype(str)
    else:
        df["yyyymm"] = "NA"

    # Ensure fields exist for aggregation
    for col, default in [("discount", 0.0), ("rebate", 0.0), ("extended_price", 0.0),
                         ("net_price", 0.0), ("gross_margin", 0.0), ("is_return", False)]:
        if col not in df.columns:
            df[col] = default

    group_keys = ["product_id", "yyyymm"]
    if include_customer and "customer_id" in df.columns:
        group_keys.insert(0, "customer_id")

    agg_dict = {
        "quantity": "sum",
        "unit_price": "mean",
        "cost": "mean",
        "discount": "sum",
        "rebate": "sum",
        "extended_price": "sum",
        "net_price": "sum",
        "gross_margin": "sum",
        "is_return": "sum"
    }
    agg = df.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    agg["gm_pct"] = np.where(agg["net_price"] != 0, agg["gross_margin"]/agg["net_price"], np.nan)
    agg["returns_pct"] = np.where(agg["quantity"] > 0, agg["is_return"]/agg["quantity"], np.nan)
    return agg

def remediation_tier(missing_cols, pii_flags, currency_missing):
    if missing_cols or pii_flags or currency_missing:
        issues = 0
        if missing_cols: issues += len(missing_cols)
        if pii_flags: issues += len(pii_flags)
        if currency_missing: issues += 1
        return "C" if issues >= 3 else "B"
    return "A"

def load_table(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Use CSV, XLSX, or Parquet.")

# -------------------------------
# Schema Mapper (Profiles)
# -------------------------------
def guess_mapping(source_cols, threshold=80):
    """Return dict canonical->source_col using synonyms + fuzzy matching."""
    mapping = {}
    normalized = [c.strip().lower() for c in source_cols]
    for canon in ALL_CANONICAL:
        candidates = [canon] + SYNONYMS.get(canon, [])
        # exact hit first
        for cand in candidates:
            if cand in normalized:
                mapping[canon] = source_cols[normalized.index(cand)]
                break
        if canon not in mapping:
            choice = process.extractOne(canon, normalized, scorer=fuzz.WRatio)
            if choice and choice[1] >= threshold:
                mapping[canon] = source_cols[choice[2]]
    return mapping

def schema_mapper_ui(df, saved_profile=None):
    st.subheader("Schema Mapper")
    source_cols = list(df.columns)
    initial_map = (saved_profile or {}).get("mapping", {}) or guess_mapping(source_cols)

    col_map = {}
    grid = st.columns(2)
    with grid[0]:
        st.caption("Map your source columns → canonical fields")
    with grid[1]:
        st.caption("Select from detected source columns")

    for canon in ALL_CANONICAL:
        options = ["-- none --"] + source_cols
        default = initial_map.get(canon, "-- none --")
        idx = options.index(default) if default in options else 0
        sel = st.selectbox(f"{canon}", options, index=idx, key=f"map_{canon}")
        if sel != "-- none --":
            col_map[canon] = sel

    st.markdown("**Type & Value Hints**")
    c1, c2, c3 = st.columns(3)
    with c1:
        date_fmt = st.text_input(
            "Date format hint (optional)",
            value=(saved_profile or {}).get("date_format", "")
        )
    with c2:
        default_currency = st.text_input(
            "Default currency if missing",
            value=(saved_profile or {}).get("default_currency", "")
        )
    with c3:
        bool_true_vals = st.text_input(
            "True tokens (comma)",
            value=",".join((saved_profile or {}).get("bool_true", list(BOOL_TRUE_DEFAULT)))
        )
    bool_false_vals = st.text_input(
        "False tokens (comma)",
        value=",".join((saved_profile or {}).get("bool_false", list(BOOL_FALSE_DEFAULT)))
    )

    profile = {
        "mapping": col_map,
        "date_format": date_fmt.strip(),
        "default_currency": default_currency.strip(),
        "bool_true": [v.strip().lower() for v in bool_true_vals.split(",") if v.strip()],
        "bool_false": [v.strip().lower() for v in bool_false_vals.split(",") if v.strip()],
    }
    return profile

def apply_profile(df, profile):
    """Return df with canonical columns created from profile mapping + coercions."""
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    mapping = profile.get("mapping", {})
    for canon, src in mapping.items():
        if src in df.columns:
            out[canon] = df[src]

    # Dates
    if "transaction_date" in out.columns:
        fmt = profile.get("date_format")
        if fmt:
            out["transaction_date"] = pd.to_datetime(out["transaction_date"], format=fmt, errors="coerce")
        else:
            out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")

    # Numbers
    for c in ["quantity","unit_price","cost","discount","rebate"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Currency
    if "currency" in out.columns:
        out["currency"] = out["currency"].astype(str).str.upper().str.strip()
    else:
        dc = profile.get("default_currency")
        if dc:
            out["currency"] = dc.upper()

    # Boolean (return flag)
    if "return_flag" in out.columns:
        tset = set(profile.get("bool_true", [])) or BOOL_TRUE_DEFAULT
        fset = set(profile.get("bool_false", [])) or BOOL_FALSE_DEFAULT
        s = out["return_flag"].astype(str).str.lower().str.strip()
        out["return_flag"] = np.where(s.isin(tset), True, np.where(s.isin(fset), False, np.nan))

    return out

def validate_profile_against_pii(profile, pii_flags):
    pii_cols = {col for col,_ in pii_flags}
    bad = [src for src in profile.get("mapping", {}).values() if src in pii_cols]
    if bad:
        raise ValueError(f"Profile attempts to map PII-like columns: {', '.join(bad)}")

def export_profile_button(profile, name="mapping_profile"):
    buf = io.StringIO()
    json.dump(profile, buf, indent=2)
    st.download_button(
        label="Download mapping profile (JSON)",
        file_name=f"{name}.json",
        mime="application/json",
        data=buf.getvalue(),
        use_container_width=True
    )

def import_profile_uploader(key="profile_upload"):
    up = st.file_uploader("Load existing mapping profile (JSON)", type=["json"], key=key)
    if up:
        try:
            return json.load(up)
        except Exception as e:
            st.error(f"Invalid profile: {e}")
    return None

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Private-use diagnostic. Works only with anonymized data. No external benchmarks. Phase 1 only.")

# Password Gate
if "authed" not in st.session_state:
    st.session_state.authed = False

stored_pw = os.getenv(PASSWORD_ENV, "")
inp = st.text_input("Enter password to proceed", type="password")

if not st.session_state.authed:
    if inp:
        if stored_pw and inp == stored_pw:
            st.session_state.authed = True
        else:
            st.error("That’s not the correct password. I can’t proceed.")
            st.stop()
    else:
        st.stop()

# Once authed
st.success("Authenticated. Proceed with Phase 1 intake.")

st.header("Choose Diagnostic Mode")
mode = st.radio(
    "Mode",
    ["Quick Diagnostic (transactional only)", "Full Diagnostic (all available files)"],
    index=0
)

with st.expander("Expected Columns & Rules", expanded=False):
    st.markdown("""
- **Required (minimal)**: `transaction_date`, `product_id`, `quantity`, `unit_price`, `cost`, `currency`  
- **Optional (improves accuracy)**: `customer_id`, `channel`, `discount`, `rebate`, `return_flag`  
- **Auto-aggregation** if file > **1,000,000 rows** or **250 MB** (by `product_id × month` and optionally `customer_id × month`).  
- **No PII**: If detected, the app will flag and block mapping PII columns.
""")

st.header("Upload Files")
st.caption("Provide anonymized files only.")
txn_file = st.file_uploader("Transactional file (CSV, XLSX, or Parquet)", type=["csv","xlsx","xls","parquet"])

product_master = None
price_list = None
cts = None

if mode.startswith("Full"):
    cols = st.columns(3)
    with cols[0]:
        product_master = st.file_uploader("Product master (optional)", type=["csv","xlsx","xls","parquet"], key="pm")
    with cols[1]:
        price_list = st.file_uploader("Price list (optional)", type=["csv","xlsx","xls","parquet"], key="pl")
    with cols[2]:
        cts = st.file_uploader("CTS / Costs (optional)", type=["csv","xlsx","xls","parquet"], key="cts")

period = st.select_slider("Rolling Period", options=["3 months","6 months","12 months","Full dataset"], value="12 months")

if st.button("Run Diagnostic", type="primary", use_container_width=True):
    if not txn_file:
        st.error("Transactional file is required.")
        st.stop()

    file_bytes = getattr(txn_file, "size", None)
    size_label = human_bytes(file_bytes) if file_bytes is not None else "Unknown"
    st.info(f"Transactional file size: {size_label}")

    # Load source table and standardize column names for mapping
    try:
        src_df = load_table(txn_file)
    except Exception as e:
        st.error(f"Could not read transactional file: {e}")
        st.stop()
    src_df = safe_lower_cols(src_df)

    # PII detection on source columns (we won't show values)
    pii_flags = detect_pii(src_df)

    # Profiles: load or create schema mapping
    st.subheader("Profiles")
    loaded_profile = import_profile_uploader()
    if loaded_profile:
        st.success("Loaded mapping profile.")

    st.info("Map your columns to the canonical schema. Save the profile to reuse next time.")
    profile = schema_mapper_ui(src_df, saved_profile=loaded_profile or {})

    # Block mapping to PII-like columns
    try:
        validate_profile_against_pii(profile, pii_flags)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Apply mapping to produce canonical frame
    df = apply_profile(src_df, profile)

    # Quick preview (canonical view)
    st.markdown("**Canonicalized preview**")
    st.dataframe(df.head(10))

    # Offer profile export
    pname = st.text_input("Profile name (for download)", value="client_profile")
    export_profile_button(profile, name=pname)

    # Completeness check on canonical schema
    score, present, missing = completeness_score(df, REQUIRED_COLS)

    # KPI header tiles (initial completeness + PII flags + placeholder for agg decision)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required Columns Present", f"{len(present)}/{len(REQUIRED_COLS)}")
    c2.metric("Data Completeness", f"{score*100:.0f}%")
    c3.metric("PII Flags", str(len(pii_flags)))
    c4.metric("Auto-Aggregation?", "Pending")

    # If required fields missing, limited-scope diagnostic + remediation plan
    if missing:
        st.warning("Running **limited-scope** diagnostic due to missing required columns.")
        st.header("Data Remediation Plan")
        tier = remediation_tier(missing, pii_flags, currency_missing=("currency" not in present))
        st.write(f"**Tier:** {tier}")
        st.write("**Issues & Impact**")
        st.markdown("- Missing required: " + ", ".join(missing))
        if pii_flags:
            st.markdown("- PII-like patterns detected in: " + ", ".join({c for c,_ in pii_flags}))
        st.markdown("""
**Impact**: Limits accuracy of GM%, discount/rebate %, returns %, and time-based aggregation.  
**Next Steps**: Provide the missing required fields. Ensure all IDs are anonymized. Remove/omit PII columns.
""")
        st.stop()

    # Continue with ETL for full diagnostic
    before_rows = len(df)
    df = df.drop_duplicates()
    deduped = before_rows - len(df)

    # Derived fields
    df = calc_fields(df)

    # Rolling period filter
    if period != "Full dataset" and "transaction_date" in df.columns:
        months = int(period.split()[0])
        max_date = pd.to_datetime(df["transaction_date"], errors="coerce").max()
        if pd.isna(max_date):
            st.warning("Invalid or missing dates; using full dataset.")
        else:
            min_cut = (max_date - pd.DateOffset(months=months-1)).normalize()
            df = df[pd.to_datetime(df["transaction_date"], errors="coerce") >= min_cut]

    # Auto-aggregation decision
    need_agg = False
    if file_bytes is not None and file_bytes > AUTO_AGG_MAX_BYTES:
        need_agg = True
    if len(df) > AUTO_AGG_MAX_ROWS:
        need_agg = True
    c4.metric("Auto-Aggregation?", "Yes" if need_agg else "No")

    if need_agg:
        st.info("Dataset is large. Aggregating by product_id × month (and customer_id if present).")
        df = aggregate_bigdata(df, include_customer=("customer_id" in df.columns))

    # Metrics
    total_ext = df["extended_price"].sum(skipna=True) if "extended_price" in df.columns else 0.0
    total_net = df["net_price"].sum(skipna=True) if "net_price" in df.columns else 0.0
    total_gm = df["gross_margin"].sum(skipna=True) if "gross_margin" in df.columns else 0.0
    gm_pct = (total_gm / total_net) if total_net else np.nan
    disc = df["discount"].sum(skipna=True) if "discount" in df.columns else 0.0
    reb = df["rebate"].sum(skipna=True) if "rebate" in df.columns else 0.0
    disc_rebate_pct = ((disc + reb) / (total_ext + 1e-9)) if total_ext else 0.0
    qty_sum = df["quantity"].sum(skipna=True) if "quantity" in df.columns else 0.0
    returns_pct = (df["is_return"].sum(skipna=True) / (qty_sum + 1e-9)) if "is_return" in df.columns else 0.0

    # Dashboard KPI Tiles
    st.header("Dashboard Readout")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("GM%", f"{gm_pct*100:.1f}%" if pd.notna(gm_pct) else "NA", help="Gross Margin / Net Price")
    k2.metric("Discount+Rebate %", f"{disc_rebate_pct*100:.1f}%")
    k3.metric("Returns %", f"{returns_pct*100:.1f}%")
    k4.metric("Data Completeness", f"{score*100:.0f}%")
    k5.metric("Deduplicated Rows", f"{deduped}")

    st.caption(
        f"GM% {'🟢' if (pd.notna(gm_pct) and gm_pct >= GM_GREEN) else '🔴'}  •  "
        f"Disc+Rebate {'🟢' if disc_rebate_pct <= DISC_REBATE_GREEN else '🔴'}  •  "
        f"Returns {'🟢' if returns_pct <= RETURNS_GREEN else '🔴'}  •  "
        f"Completeness {'🟢' if score >= COMPLETENESS_GREEN else '🟠'}"
    )

    # Visuals
    st.subheader("Price-Volume-Mix Bridge (Net → Cost → GM)")
    # Simple waterfall-like bars
    cost_total = (df.get("quantity", 0) * df.get("cost", 0)).sum() if "cost" in df.columns and "quantity" in df.columns else 0.0
    wdf = pd.DataFrame({
        "Component": ["Net Price", "Cost", "Gross Margin"],
        "Value": [total_net, -cost_total, total_gm],
        "Color": ["#1e88e5", "#ef4444", "#22c55e"]
    })
    chart = alt.Chart(wdf).mark_bar().encode(
        x=alt.X("Component:N"),
        y=alt.Y("Value:Q"),
        color=alt.Color("Color:N", scale=None)
    ).properties(width=600, height=300)
    st.altair_chart(chart, use_container_width=True)
    st.write("- Net → GM bridge highlights contribution of cost structure.")
    st.write("- For Full Diagnostic, extend into price/volume/mix decomposition by product-month.")
    st.write("Follow-ups: validate currency normalization; investigate items with negative GM or high discount rates.")

    st.subheader("GM% by Product (Top 25 by Net)")
    if {"product_id","net_price","gross_margin"}.issubset(df.columns):
        by_prod = df.groupby("product_id", dropna=False).agg(
            net=("net_price","sum"),
            gm=("gross_margin","sum")
        ).reset_index()
        by_prod["gm_pct"] = np.where(by_prod["net"] != 0, by_prod["gm"]/by_prod["net"], np.nan)
        top = by_prod.sort_values("net", ascending=False).head(25)
        chart2 = alt.Chart(top).mark_bar().encode(
            x=alt.X("product_id:N", sort="-y"),
            y=alt.Y("gm_pct:Q", axis=alt.Axis(format="%")),
            color=alt.value(PALETTE["teal"]),
            tooltip=["product_id","net","gm","gm_pct"]
        ).properties(height=350)
        st.altair_chart(chart2, use_container_width=True)
        st.write("- Target SKUs with high net but sub-30% GM for pricing/cost actions.")
        st.write("- Check if mix shift is diluting margins.")
    else:
        st.info("Insufficient fields to render GM% by product.")

    st.subheader("Discount+Rebate % by Month")
    if "transaction_date" in df.columns and df["transaction_date"].notna().any():
        m = df.copy()
        m["yyyymm"] = pd.to_datetime(m["transaction_date"], errors="coerce").dt.to_period("M").astype(str)
        ext_sum = m.groupby("yyyymm")["extended_price"].sum() if "extended_price" in m.columns else pd.Series(dtype=float)
        disc_sum = m.groupby("yyyymm")["discount"].sum() if "discount" in m.columns else pd.Series(0.0, index=ext_sum.index)
        reb_sum = m.groupby("yyyymm")["rebate"].sum() if "rebate" in m.columns else pd.Series(0.0, index=ext_sum.index)
        m_disc = pd.DataFrame({"ext": ext_sum}).join(disc_sum.rename("disc"), how="left").join(reb_sum.rename("reb"), how="left").fillna(0.0)
        m_disc["dr_pct"] = (m_disc["disc"] + m_disc["reb"]) / (m_disc["ext"] + 1e-9)
        m_disc = m_disc.reset_index(names=["yyyymm"])
        chart3 = alt.Chart(m_disc).mark_line(point=True).encode(
            x=alt.X("yyyymm:N", title="Month"),
            y=alt.Y("dr_pct:Q", axis=alt.Axis(format="%")),
            color=alt.value(PALETTE["accent"])
        ).properties(height=300)
        st.altair_chart(chart3, use_container_width=True)
        st.write("- Track spikes >8%; review promo controls and leakage.")
        st.write("- Correlate to policy thresholds by channel/customer.")
    else:
        st.info("No valid transaction dates to chart monthly discount/rebate %.")

    # Data Remediation Plan
    st.header("Data Remediation Plan")
    currency_missing = ("currency" not in df.columns) or (df["currency"].isna().all() if "currency" in df.columns else True)
    tier = remediation_tier([], pii_flags, currency_missing)
    st.write(f"**Tier:** {tier}")
    checklist = []
    if pii_flags:
        checklist.append("Remove columns with PII-like patterns or mask before upload.")
    if currency_missing:
        checklist.append("Provide transaction currency or confirm single-currency scope.")
    if deduped > 0:
        checklist.append("Investigate source-system causes of duplicate lines.")
    if "return_flag" not in df.columns:
        checklist.append("Provide a returns indicator to quantify returns % accurately.")
    if "discount" not in df.columns or "rebate" not in df.columns:
        checklist.append("Add discount and rebate fields to separate price effects.")
    if "customer_id" not in df.columns:
        checklist.append("Add anonymized customer_id to enable cohort analysis and aggregation.")

    if checklist:
        st.markdown("- " + "\n- ".join(checklist))
    else:
        st.write("No major data gaps detected for Phase 1.")

    # Briefing Deck
    st.header("Briefing Deck (Summary)")
    st.markdown(f"""
- **GM%**: {f'{gm_pct*100:.1f}%' if pd.notna(gm_pct) else 'NA'} ({'≥30% OK' if (pd.notna(gm_pct) and gm_pct >= GM_GREEN) else '<30% — improvement needed'}).  
- **Discount+Rebate %**: {disc_rebate_pct*100:.1f}% ({'≤8% OK' if disc_rebate_pct <= DISC_REBATE_GREEN else '>8% — potential leakage'}).  
- **Returns %**: {returns_pct*100:.1f}% ({'≤2% OK' if returns_pct <= RETURNS_GREEN else '>2% — quality/fulfillment issue?'}).  
- **Data completeness**: {score*100:.0f}%.
""")

    st.subheader("Opportunities & Risks")
    st.markdown("""
**Opportunities**  
1) Tighten discount/rebate controls to hit ≤8%.  
2) Improve product mix toward higher GM% SKUs.  
3) Validate unit costs & currency normalization.

**Risks**  
1) PII contamination in source files.  
2) Returns underreported without `return_flag`.  
3) Overaggregation can mask outliers in large datasets.
""")

    st.subheader("Questions for First Client Call")
    st.markdown("""
1) What currency(s) are present, and is FX normalization required?  
2) Which discount/rebate mechanisms drive the largest % swings?  
3) Any supply or logistics issues inflating returns?  
4) Do we have customer/channel identifiers for segmentation?  
5) Any upcoming price/cost changes to reflect?
""")

    st.subheader("Assumptions & Formulas")
    st.markdown("""
- **extended_price** = quantity × unit_price  
- **net_price** = extended_price − discount − rebate  
- **gross_margin** = net_price − (quantity × cost)  
- **GM%** = gross_margin ÷ net_price  
- **Returns %** = returns_lines ÷ quantity (approximation without RMA qty)  
Results marked as **indicative** where inputs are weak or missing.
""")

    # PII notice (columns only)
    if pii_flags:
        st.error("PII-like patterns detected (columns only): " + ", ".join({c for c,_ in pii_flags}) + ". Remove these before next run.")

st.caption("Note: No external web browsing or benchmarks are used in Phase 1. Names are never displayed unless explicitly allowed.")
