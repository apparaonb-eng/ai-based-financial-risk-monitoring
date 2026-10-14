# AI-Based Financial Risk Monitoring

Detects risky financial transactions using an Isolation Forest (AI) combined with rule-based scoring.

## Setup
    pip install -r requirements.txt

## Run
    python financial_risk_monitor.py

## Output
- risk_results.csv      - all transactions with risk scores
- high_risk_alerts.csv  - HIGH-risk transactions only
- risk_report.png       - summary charts

## Use your own data
Replace `generate_data()` in the main block with `pd.read_csv("transactions.csv")`.
Required columns: txn_id, amount, hour, txns_last_24h, is_foreign, balance_ratio
