# Profit Pulse — Phase 1 Diagnostic

Confidential, Streamlit-based tool for rapid profitability diagnostics using **anonymized CSVs** only.

## Features
- Password gate (`PP-PULSE`)
- Quick (transactional-only) and Full modes
- Auto-aggregation for very large files (by month × product, + customer if present)
- Data Remediation Plan with Tier (A/B/C)
- Traffic-light dashboard (GM%, discounts/rebates, returns, data completeness)
- Briefing deck, opportunities/risks, call questions, assumptions/formulas

## Expected Columns
**Required (Quick):**
- `date` (YYYY-MM-DD)
- `product_id`
- `quantity` (numeric)
- `unit_price` (numeric)
- `unit_cost` (numeric)

**Optional (improves accuracy):**
- `customer_id`
- `discount` (absolute)
- `rebate` (absolute)
- `currency` (e.g., USD/EUR) — flagged unless normalized
- `returns_qty`, `returns_amount`

> PII is not allowed. Use anonymized IDs only.

## Install & Run (Local)
```bash
python -m pip install -U pip setuptools wheel
PIP_PREFER_BINARY=1 pip install --only-binary=:all: -r requirements.txt
streamlit run app.py
