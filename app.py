# app.py
import os
import io
import json
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from rapidfuzz import process, fuzz

# -------------------------------
# Config & Constants
# -------------------------------
APP_TITLE = "Profit Pulse — Phase 1 Diagnostic"
PASSWORD_ENV = "PROFIT_PULSE_PASSWORD"

# Canonical schema for transactional table
REQUIRED_TXN = ["transaction_date", "product_id", "quantity", "unit_price", "cost", "currency"]
OPTIONAL_TXN = ["customer_id", "channel", "discount", "rebate", "return_flag"]
ALL_TXN = REQUIRED_TXN + OPTIONAL_TXN

# Canonical subsets for other tables
PM_FIELDS  = ["product_id", "channel"]                  # minimal product master fields we may use
PL_FIELDS  = ["product_id", "unit_price", "currency"]   # price list
CTS_FIELDS = ["product_id", "cost", "currency"]         # cost table

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
# Utility functions
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
            series = df[col].dropna().astype(str).head(1000)
            try:
                if series.str.contains(EMAIL_REGEX).any():
                    flags.append((col, "email-like pattern"))
                if series.str.contains(TAXID_REGEX).any():
                    flags.append((col, "tax-id-like pattern"))
                if series.str.contains(ADDRESS_REGEX).any():
                    flags.append((col, "address-like pattern"))
            except Exception:
                pass
    return flags

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

def completeness_score(df: pd.DataFrame, required_cols):
    present = [c for c in required_cols if c in df.columns]
    score = len(present) / len(required_cols) if required_cols else 1.0
    return score, present, [c for c in required_cols if c not in present]

def calc_fields(df: pd.DataFrame):
    df = df.copy()
    for c in ["quantity", "unit_price", "cost", "discount", "rebate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["extended_price"] = df.get("quantity", 0) * df.get("unit_price", 0)
    df["net_price"] = df["extended_price"] - df.get("discount", 0) - df.get("rebate", 0)
    df["gross_margin"] = df["net_price"] - (df.get("quantity", 0) * df.get("cost", 0))
    df["gm_pct"] = np.where(df["net_price"] != 0, df["gross_margin"] / df["net_price"], np.nan)
    if "return_flag" in df.columns:
        s = df["return_flag"].astype(str).str.lower().str.strip()
        df["is_return"] = s.isin(BOOL_TRUE_DEFAULT) | s.isin({"true","t","1","yes","y"})
    else:
        df["is_return"] = False
    return df

def aggregate_bigdata(df: pd.DataFrame, include_customer=False):
    df = df.copy()
    if "transaction_date" in df.columns:
        df["yyyymm"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.to_period("M").astype(str)
    else:
        df["yyyymm"] = "NA"
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

# -------------------------------
# Schema Mapping & Profiles
# -------------------------------
def guess_mapping(source_cols, canon_fields, threshold=80):
    mapping = {}
    normalized = [c.strip().lower() for c in source_cols]
    for canon in canon_fields:
        candidates = [canon] + SYNONYMS.get(canon, [])
        for cand in candidates:
            if cand in normalized:
                mapping[canon] = source_cols[normalized.index(cand)]
                break
        if canon not in mapping:
            choice = process.extractOne(canon, normalized, scorer=fuzz.WRatio)
            if choice and choice[1] >= threshold:
                mapping[canon] = source_cols[choice[2]]
    return mapping

def schema_mapper_ui(df, allowed_fields, saved_profile=None, key_prefix=""):
    st.subheader(f"Schema Mapper — {key_prefix or 'table'}")
    source_cols = list(df.columns)
    initial_map = (saved_profile or {}).get("mapping", {}) or guess_mapping(source_cols, allowed_fields)

    col_map = {}
    grid = st.columns(2)
    with grid[0]:
        st.caption("Map your source columns → canonical fields")
    with grid[1]:
        st.caption("Select from detected source columns")

    for canon in allowed_fields:
        options = ["-- none --"] + source_cols
        default = initial_map.get(canon, "-- none --")
        idx = options.index(default) if default in options else 0
        sel = st.selectbox(f"{canon}", options, index=idx, key=f"{key_prefix}_map_{canon}")
        if sel != "-- none --":
            col_map[canon] = sel

    st.markdown("**Type & Value Hints**")
    c1, c2, c3 = st.columns(3)
    with c1:
        date_fmt = st.text_input(
            "Date format hint (optional)",
            value=(saved_profile or {}).get("date_format", ""),
            key=f"{key_prefix}_datefmt"
        )
    with c2:
        default_currency = st.text_input(
            "Default currency if missing",
            value=(saved_profile or {}).get("default_currency", ""),
            key=f"{key_prefix}_defcur"
        )
    with c3:
        bool_true_vals = st.text_input(
            "True tokens (comma)",
            value=",".join((saved_profile or {}).get("bool_true", list(BOOL_TRUE_DEFAULT))),
            key=f"{key_prefix}_btrue"
        )
    bool_false_vals = st.text_input(
        "False tokens (comma)",
        value=",".join((saved_profile or {}).get("bool_false", list(BOOL_FALSE_DEFAULT))),
        key=f"{key_prefix}_bfalse"
    )

    profile = {
        "mapping": col_map,
        "date_format": date_fmt.strip(),
        "default_currency": default_currency.strip(),
        "bool_true": [v.strip().lower() for v in bool_true_vals.split(",") if v.strip()],
        "bool_false": [v.strip().lower() for v in bool_false_vals.split(",") if v.strip()],
    }
    return profile

def apply_profile(df, profile, allowed_fields):
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    mapping = profile.get("mapping", {})
    for canon, src in mapping.items():
        if canon in allowed_fields and src in df.columns:
            out[canon] = df[src]
    # Dates
    if "transaction_date" in out.columns:
        fmt = profile.get("date_format")
        if fmt:
            out["transaction_date"] = pd.to_datetime(out["transaction_date"], format=fmt, errors="coerce")
        else:
            out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    # Numbers
    for c in set(allowed_fields).intersection({"quantity","unit_price","cost","discount","rebate"}):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Currency
    if "currency" in out.columns:
        out["currency"] = out["currency"].astype(str).str.upper().str.strip()
    else:
        dc = profile.get("default_currency")
        if dc:
            out["currency"] = dc.upper()
    # Booleans
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

def export_combined_profile_button(profile_txn, profile_pm, profile_pl, profile_cts, name="mapping_profile_multi"):
    combined = {
        "transactional": profile_txn or {},
        "product_master": profile_pm or {},
        "price_list": profile_pl or {},
        "cts": profile_cts or {},
        "meta": {
            "version": "1.0",
            "notes": "Multi-table mapping profile for Profit Pulse Phase 1"
        }
    }
    buf = io.StringIO()
    json.dump(combined, buf, indent=2)
    st.download_button(
        label="Download combined mapping profile (JSON)",
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
- **Required (transactional)**: `transaction_date`, `product_id`, `quantity`, `unit_price`, `cost`, `currency`  
- **Optional (transactional)**: `customer_id`, `channel`, `discount`, `rebate`, `return_flag`  
- **Masters (optional)**:  
  - Product Master: `product_id`, (optional) `channel`  
  - Price List: `product_id`, `unit_price`, `currency`  
  - CTS/Costs: `product_id`, `cost`, `currency`  
- **Auto-aggregation** if transactional > **1,000,000 rows** or **250 MB**.  
- **No PII**: PII-like columns are flagged and blocked from mapping.
""")

st.header("Upload Files")
st.caption("Provide anonymized files only.")

txn_file = st.file_uploader("Transactional file (CSV, XLSX, or Parquet)", type=["csv","xlsx","xls","parquet"], key="txn")
pm_file = None
pl_file = None
cts_file = None

if mode.startswith("Full"):
    cols = st.columns(3)
    with cols[0]:
        pm_file = st.file_uploader("Product master (optional)", type=["csv","xlsx","xls","parquet"], key="pm")
    with cols[1]:
        pl_file = st.file_uploader("Price list (optional)", type=["csv","xlsx","xls","parquet"], key="pl")
    with cols[2]:
        cts_file = st.file_uploader("CTS / Costs (optional)", type=["csv","xlsx","xls","parquet"], key="cts")

period = st.select_slider("Rolling Period", options=["3 months","6 months","12 months","Full dataset"], value="12 months")

if st.button("Run Diagnostic", type="primary", use_container_width=True):
    if not txn_file:
        st.error("Transactional file is required.")
        st.stop()

    # ---------- Load all uploaded tables ----------
    def try_load(file):
        if not file: return None
        try:
            return safe_lower_cols(load_table(file))
        except Exception as e:
            st.error(f"Could not read file {file.name}: {e}")
            st.stop()

    src_txn = try_load(txn_file)
    src_pm  = try_load(pm_file)
    src_pl  = try_load(pl_file)
    src_cts = try_load(cts_file)

    # file size note
    file_bytes = getattr(txn_file, "size", None)
    st.info(f"Transactional file size: {human_bytes(file_bytes)}")

    # ---------- PII detection per table ----------
    pii_txn = detect_pii(src_txn)
    pii_pm  = detect_pii(src_pm)  if src_pm  is not None else []
    pii_pl  = detect_pii(src_pl)  if src_pl  is not None else []
    pii_cts = detect_pii(src_cts) if src_cts is not None else []

    # ---------- Profile load ----------
    st.subheader("Profiles")
    loaded_profile = import_profile_uploader()
    if loaded_profile:
        st.success("Loaded multi-table mapping profile.")

    # ---------- Mapping UIs ----------
    st.info("Map your columns to canonical fields. Save the combined profile to reuse later.")

    txn_profile = schema_mapper_ui(
        src_txn,
        allowed_fields=ALL_TXN,
        saved_profile=(loaded_profile or {}).get("transactional", {}),
        key_prefix="txn"
    )
    try:
        validate_profile_against_pii(txn_profile, pii_txn)
    except ValueError as e:
        st.error(str(e)); st.stop()
    df_txn = apply_profile(src_txn, txn_profile, allowed_fields=ALL_TXN)
    st.markdown("**Transactional — canonicalized preview**")
    st.dataframe(df_txn.head(10))

    df_pm = None; pm_profile = None
    if src_pm is not None:
        pm_profile = schema_mapper_ui(
            src_pm,
            allowed_fields=PM_FIELDS,
            saved_profile=(loaded_profile or {}).get("product_master", {}),
            key_prefix="pm"
        )
        try:
            validate_profile_against_pii(pm_profile, pii_pm)
        except ValueError as e:
            st.error(str(e)); st.stop()
        df_pm = apply_profile(src_pm, pm_profile, allowed_fields=PM_FIELDS)
        st.markdown("**Product Master — canonicalized preview**")
        st.dataframe(df_pm.head(10))

    df_pl = None; pl_profile = None
    if src_pl is not None:
        pl_profile = schema_mapper_ui(
            src_pl,
            allowed_fields=PL_FIELDS,
            saved_profile=(loaded_profile or {}).get("price_list", {}),
            key_prefix="pl"
        )
        try:
            validate_profile_against_pii(pl_profile, pii_pl)
        except ValueError as e:
            st.error(str(e)); st.stop()
        df_pl = apply_profile(src_pl, pl_profile, allowed_fields=PL_FIELDS)
        st.markdown("**Price List — canonicalized preview**")
        st.dataframe(df_pl.head(10))

    df_cts = None; cts_profile = None
    if src_cts is not None:
        cts_profile = schema_mapper_ui(
            src_cts,
            allowed_fields=CTS_FIELDS,
            saved_profile=(loaded_profile or {}).get("cts", {}),
            key_prefix="cts"
        )
        try:
            validate_profile_against_pii(cts_profile, pii_cts)
        except ValueError as e:
            st.error(str(e)); st.stop()
        df_cts = apply_profile(src_cts, cts_profile, allowed_fields=CTS_FIELDS)
        st.markdown("**CTS / Costs — canonicalized preview**")
        st.dataframe(df_cts.head(10))

    # ---------- Export combined profile ----------
    pname = st.text_input("Combined profile name (for download)", value="client_profile_multi")
    export_combined_profile_button(txn_profile, pm_profile, pl_profile, cts_profile, name=pname)

    # ---------- Enrichment from masters BEFORE completeness ----------
    # Fill unit_price from price_list where missing
    if df_pl is not None and "product_id" in df_pl.columns:
        # Join on product_id; prefer transactional currency if present
        jcols = ["product_id"]
        enrich = df_pl[["product_id"] + [c for c in ["unit_price","currency"] if c in df_pl.columns]].copy()
        df_txn = df_txn.merge(enrich, on=jcols, how="left", suffixes=("", "_pl"))
        if "unit_price" not in df_txn.columns and "unit_price_pl" in df_txn.columns:
            df_txn["unit_price"] = df_txn["unit_price_pl"]
        elif "unit_price_pl" in df_txn.columns:
            df_txn["unit_price"] = df_txn["unit_price"].fillna(df_txn["unit_price_pl"])
        if "currency" not in df_txn.columns and "currency_pl" in df_txn.columns:
            df_txn["currency"] = df_txn["currency_pl"]
        df_txn.drop(columns=[c for c in ["unit_price_pl","currency_pl"] if c in df_txn.columns], inplace=True)

    # Fill cost from CTS where missing
    if df_cts is not None and "product_id" in df_cts.columns:
        enrich = df_cts[["product_id"] + [c for c in ["cost","currency"] if c in df_cts.columns]].copy()
        df_txn = df_txn.merge(enrich, on=["product_id"], how="left", suffixes=("", "_cts"))
        if "cost" not in df_txn.columns and "cost_cts" in df_txn.columns:
            df_txn["cost"] = df_txn["cost_cts"]
        elif "cost_cts" in df_txn.columns:
            df_txn["cost"] = df_txn["cost"].fillna(df_txn["cost_cts"])
        df_txn.drop(columns=[c for c in ["cost_cts","currency_cts"] if c in df_txn.columns], inplace=True)

    # Fill channel from Product Master if missing in txn
    if df_pm is not None and "product_id" in df_pm.columns and "channel" in df_pm.columns:
        df_txn = df_txn.merge(df_pm[["product_id","channel"]], on="product_id", how="left", suffixes=("", "_pm"))
        if "channel" not in df_txn.columns and "channel_pm" in df_txn.columns:
            df_txn["channel"] = df_txn["channel_pm"]
        elif "channel_pm" in df_txn.columns:
            df_txn["channel"] = df_txn["channel"].fillna(df_txn["channel_pm"])
        if "channel_pm" in df_txn.columns:
            df_txn.drop(columns=["channel_pm"], inplace=True)

    # ---------- Completeness check (after enrichment) ----------
    score, present, missing = completeness_score(df_txn, REQUIRED_TXN)

    # KPI header tiles (initial completeness + PII flags + placeholder for agg decision)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required Columns Present", f"{len(present)}/{len(REQUIRED_TXN)}")
    c2.metric("Data Completeness", f"{score*100:.0f}%")
    total_pii = len(pii_txn) + len(pii_pm) + len(pii_pl) + len(pii_cts)
    c3.metric("PII Flags (any table)", str(total_pii))
    c4.metric("Auto-Aggregation?", "Pending")

    if missing:
        st.warning("Running **limited-scope** diagnostic due to missing required transactional columns.")
        st.header("Data Remediation Plan")
        tier = remediation_tier(missing, pii_txn, currency_missing=("currency" not in present))
        st.write(f"**Tier:** {tier}")
        st.write("**Issues & Impact**")
        st.markdown("- Missing required (transactional): " + ", ".join(missing))
        for table_name, flags in [("product_master", pii_pm), ("price_list", pii_pl), ("cts", pii_cts)]:
            if flags:
                st.markdown(f"- PII-like patterns detected in **{table_name}**: " + ", ".join({c for c,_ in flags}))
        st.markdown("""
**Impact**: Limits accuracy of GM%, discount/rebate %, returns %, and time-based aggregation.  
**Next Steps**: Provide the missing required fields in the transactional file; keep all IDs anonymized; remove/omit PII columns in any table.
""")
        st.stop()

    # ---------- ETL & Derived fields ----------
    before_rows = len(df_txn)
    df_txn = df_txn.drop_duplicates()
    deduped = before_rows - len(df_txn)
    df_txn = calc_fields(df_txn)

    # ---------- Rolling period filter ----------
    period_choice = period
    if period_choice != "Full dataset" and "transaction_date" in df_txn.columns:
        months = int(period_choice.split()[0])
        max_date = pd.to_datetime(df_txn["transaction_date"], errors="coerce").max()
        if pd.isna(max_date):
            st.warning("Invalid or missing dates; using full dataset.")
        else:
            min_cut = (max_date - pd.DateOffset(months=months-1)).normalize()
            df_txn = df_txn[pd.to_datetime(df_txn["transaction_date"], errors="coerce") >= min_cut]

    # ---------- Auto-aggregation ----------
    need_agg = False
    file_bytes = getattr(txn_file, "size", None)
    if file_bytes is not None and file_bytes > AUTO_AGG_MAX_BYTES:
        need_agg = True
    if len(df_txn) > AUTO_AGG_MAX_ROWS:
        need_agg = True
    c4.metric("Auto-Aggregation?", "Yes" if need_agg else "No")
    if need_agg:
        st.info("Dataset is large. Aggregating by product_id × month (and customer_id if present).")
        df_txn = aggregate_bigdata(df_txn, include_customer=("customer_id" in df_txn.columns))

    # ---------- Metrics ----------
    total_ext = df_txn["extended_price"].sum(skipna=True) if "extended_price" in df_txn.columns else 0.0
    total_net = df_txn["net_price"].sum(skipna=True) if "net_price" in df_txn.columns else 0.0
    total_gm  = df_txn["gross_margin"].sum(skipna=True) if "gross_margin" in df_txn.columns else 0.0
    gm_pct = (total_gm / total_net) if total_net else np.nan
    disc = df_txn["discount"].sum(skipna=True) if "discount" in df_txn.columns else 0.0
    reb  = df_txn["rebate"].sum(skipna=True) if "rebate" in df_txn.columns else 0.0
    disc_rebate_pct = ((disc + reb) / (total_ext + 1e-9)) if total_ext else 0.0
    qty_sum = df_txn["quantity"].sum(skipna=True) if "quantity" in df_txn.columns else 0.0
    returns_pct = (df_txn["is_return"].sum(skipna=True) / (qty_sum + 1e-9)) if "is_return" in df_txn.columns else 0.0

    # ---------- Dashboard ----------
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
    cost_total = (df_txn.get("quantity", 0) * df_txn.get("cost", 0)).sum() if "cost" in df_txn.columns and "quantity" in df_txn.columns else 0.0
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
    st.write("- Net → GM bridge shows cost contribution to margin.")
    st.write("- Extend to full PVM decomposition in later phases.")

    st.subheader("GM% by Product (Top 25 by Net)")
    if {"product_id","net_price","gross_margin"}.issubset(df_txn.columns):
        by_prod = df_txn.groupby("product_id", dropna=False).agg(
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
    else:
        st.info("Insufficient fields to render GM% by product.")

    st.subheader("Discount+Rebate % by Month")
    if "transaction_date" in df_txn.columns and df_txn["transaction_date"].notna().any():
        m = df_txn.copy()
        m["yyyymm"] = pd.to_datetime(m["transaction_date"], errors="coerce").dt.to_period("M").astype(str)
        ext_sum = m.groupby("yyyymm")["extended_price"].sum() if "extended_price" in m.columns else pd.Series(dtype=float)
        disc_sum = m.groupby("yyyymm")["discount"].sum() if "discount" in m.columns else pd.Series(0.0, index=ext_sum.index)
        reb_sum  = m.groupby("yyyymm")["rebate"].sum() if "rebate" in m.columns else pd.Series(0.0, index=ext_sum.index)
        m_disc = pd.DataFrame({"ext": ext_sum}).join(disc_sum.rename("disc"), how="left").join(reb_sum.rename("reb"), how="left").fillna(0.0)
        m_disc["dr_pct"] = (m_disc["disc"] + m_disc["reb"]) / (m_disc["ext"] + 1e-9)
        m_disc = m_disc.reset_index(names=["yyyymm"])
        chart3 = alt.Chart(m_disc).mark_line(point=True).encode(
            x=alt.X("yyyymm:N", title="Month"),
            y=alt.Y("dr_pct:Q", axis=alt.Axis(format="%")),
            color=alt.value(PALETTE["accent"])
        ).properties(height=300)
        st.altair_chart(chart3, use_container_width=True)
        st.write("- Watch for spikes >8% and investigate promo/leakage.")
    else:
        st.info("No valid transaction dates to chart monthly discount/rebate %.")

    # ---------- Data Remediation Plan ----------
    st.header("Data Remediation Plan")
    currency_missing = ("currency" not in df_txn.columns) or (df_txn["currency"].isna().all() if "currency" in df_txn.columns else True)
    tier = remediation_tier([], pii_txn, currency_missing)
    st.write(f"**Tier:** {tier}")
    checklist = []
    if total_pii:
        checklist.append("Remove columns with PII-like patterns in any table before next run.")
    if currency_missing:
        checklist.append("Provide transaction currency or confirm single-currency scope.")
    if "return_flag" not in df_txn.columns:
        checklist.append("Provide a returns indicator to quantify returns % accurately.")
    if "discount" not in df_txn.columns or "rebate" not in df_txn.columns:
        checklist.append("Add discount and rebate fields to separate price effects.")
    if "customer_id" not in df_txn.columns:
        checklist.append("Add anonymized customer_id to enable cohort analysis.")
    if deduped > 0:
        checklist.append("Investigate source-system causes of duplicate lines.")
    if checklist:
        st.markdown("- " + "\n- ".join(checklist))
    else:
        st.write("No major data gaps detected for Phase 1.")

    # ---------- Briefing Deck ----------
    st.header("Briefing Deck (Summary)")
    gm_text = f"{gm_pct*100:.1f}%" if pd.notna(gm_pct) else "NA"
    st.markdown(f"""
- **GM%**: {gm_text} ({'≥30% OK' if (pd.notna(gm_pct) and gm_pct >= GM_GREEN) else '<30% — improvement needed'}).  
- **Discount+Rebate %**: {disc_rebate_pct*100:.1f}% ({'≤8% OK' if disc_rebate_pct <= DISC_REBATE_GREEN else '>8% — potential leakage'}).  
- **Returns %**: {returns_pct*100:.1f}% ({'≤2% OK' if returns_pct <= RETURNS_GREEN else '>2% — quality/fulfillment issue?'}).  
- **Data completeness**: {score*100:.0f}%.
""")

    st.subheader("Opportunities & Risks")
    st.markdown("""
**Opportunities**  
1) Tighten discount/rebate controls to hit ≤8%.  
2) Improve product mix toward higher GM% SKUs.  
3) Validate unit costs & currency normalization (consider masters as source of truth).

**Risks**  
1) PII contamination in any table.  
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
- **Returns %** = returns_lines ÷ quantity (approx. without explicit RMA qty)  
Results marked as **indicative** where inputs are weak or missing.
""")

    # PII notice (columns only)
    if total_pii:
        pills = []
        if pii_txn: pills.append("transactional: " + ", ".join({c for c,_ in pii_txn}))
        if pii_pm:  pills.append("product_master: " + ", ".join({c for c,_ in pii_pm}))
        if pii_pl:  pills.append("price_list: " + ", ".join({c for c,_ in pii_pl}))
        if pii_cts: pills.append("cts: " + ", ".join({c for c,_ in pii_cts}))
        st.error("PII-like patterns detected (columns only) → " + " | ".join(pills) + ". Remove before next run.")

st.caption("Note: Phase 1 avoids external benchmarks. Names are never displayed unless explicitly allowed.")
