# Profit Pulse — Streamlit Phase 1 Diagnostic

Private-use diagnostic app that ingests anonymized transactional data and produces a Phase 1 profitability readout. It mirrors the Profit Pulse guardrails: password gate, PII flagging, schema mapping with reusable profiles, auto-aggregation for large datasets, remediation plan, and concise dashboards.

## Features
- 🔐 Password gate (env var) before any data is read
- 🧭 Schema Mapper UI with fuzzy matching + **downloadable/uploadable JSON profiles**
- 🙈 PII pattern flagging (emails, tax-ids, address-like); profile mapping blocks PII columns
- 🧮 ETL: normalization, derived fields (extended_price, net_price, gross_margin, GM%)
- 📊 KPIs & visuals: GM%, discount+rebate %, returns %, completeness, PVM bridge, GM% by product, monthly discount trend
- 🧱 Auto-aggregation if file >1M rows or >250MB (product_id × month; optionally by customer_id)
- 🧰 Data Remediation Plan + Briefing summary
- 🚫 No external web browsing or benchmarks; anonymized outputs only

## Quickstart

```bash
git clone <your-repo>
cd profit-pulse-app
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PROFIT_PULSE_PASSWORD="PP-PULSE"   # set your password
streamlit run app.py
