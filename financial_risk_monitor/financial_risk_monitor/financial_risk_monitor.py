"""
AI-Based Financial Risk Monitoring (simple base project)
--------------------------------------------------------
Pipeline:
  1. Generate synthetic transaction data (replace with your own CSV later)
  2. Train an Isolation Forest (unsupervised AI model) to detect anomalies
  3. Combine the AI score with simple business rules into a 0-100 risk score
  4. Classify each transaction as LOW / MEDIUM / HIGH risk and raise alerts

Install:  pip install pandas numpy scikit-learn matplotlib
Run:      python financial_risk_monitor.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to file; remove this line to show windows
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- 1. DATA
def generate_data(n_normal=2000, n_risky=60) -> pd.DataFrame:
    """Create synthetic transactions: mostly normal, a few suspicious."""
    normal = pd.DataFrame({
        "amount": RNG.lognormal(mean=4.0, sigma=0.8, size=n_normal),   # ~ $50
        "hour": RNG.integers(8, 22, n_normal),                         # daytime
        "txns_last_24h": RNG.poisson(2, n_normal),
        "is_foreign": RNG.choice([0, 1], n_normal, p=[0.95, 0.05]),
        "balance_ratio": RNG.uniform(0.01, 0.15, n_normal),            # amount / balance
    })
    risky = pd.DataFrame({
        "amount": RNG.lognormal(mean=7.0, sigma=0.7, size=n_risky),    # ~ $1000+
        "hour": RNG.choice([0, 1, 2, 3, 4, 23], n_risky),              # night
        "txns_last_24h": RNG.poisson(12, n_risky),                     # burst
        "is_foreign": RNG.choice([0, 1], n_risky, p=[0.2, 0.8]),
        "balance_ratio": RNG.uniform(0.5, 0.95, n_risky),
    })
    df = pd.concat([normal, risky], ignore_index=True).sample(frac=1, random_state=1)
    df.insert(0, "txn_id", [f"TXN{i:05d}" for i in range(len(df))])
    df["amount"] = df["amount"].round(2)
    return df.reset_index(drop=True)


# ------------------------------------------------------------ 2. AI MODEL
FEATURES = ["amount", "hour", "txns_last_24h", "is_foreign", "balance_ratio"]


def train_model(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    model.fit(X)
    return model, scaler


def ai_risk_score(df, model, scaler) -> np.ndarray:
    """Convert Isolation Forest output to a 0-100 score (higher = riskier)."""
    raw = -model.score_samples(scaler.transform(df[FEATURES]))  # higher = more anomalous
    return 100 * (raw - raw.min()) / (raw.max() - raw.min())


# ------------------------------------------------------------- 3. RULES
def rule_risk_score(row) -> int:
    """Simple explainable business rules (0-100)."""
    score = 0
    if row["amount"] > 1000:
        score += 30
    if row["hour"] < 5 or row["hour"] >= 23:
        score += 20
    if row["txns_last_24h"] > 8:
        score += 20
    if row["is_foreign"]:
        score += 10
    if row["balance_ratio"] > 0.5:
        score += 20
    return score


def explain(row) -> str:
    reasons = []
    if row["amount"] > 1000: reasons.append("high amount")
    if row["hour"] < 5 or row["hour"] >= 23: reasons.append("odd hour")
    if row["txns_last_24h"] > 8: reasons.append("burst of transactions")
    if row["is_foreign"]: reasons.append("foreign transaction")
    if row["balance_ratio"] > 0.5: reasons.append("large share of balance")
    return ", ".join(reasons) or "model-detected anomaly"


# ------------------------------------------------------------- 4. MONITOR
def risk_level(score: float) -> str:
    if score >= 60: return "HIGH"
    if score >= 35: return "MEDIUM"
    return "LOW"


def monitor(df: pd.DataFrame) -> pd.DataFrame:
    model, scaler = train_model(df)
    df = df.copy()
    df["ai_score"] = ai_risk_score(df, model, scaler)
    df["rule_score"] = df.apply(rule_risk_score, axis=1)
    df["risk_score"] = (0.6 * df["ai_score"] + 0.4 * df["rule_score"]).round(1)
    df["risk_level"] = df["risk_score"].apply(risk_level)
    df["reason"] = df.apply(explain, axis=1)
    return df


def plot_results(df: pd.DataFrame, path="risk_report.png"):
    colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(df["amount"], df["risk_score"], c=df["risk_level"].map(colors), s=12, alpha=0.7)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Transaction amount (log scale)")
    ax[0].set_ylabel("Risk score")
    ax[0].set_title("Risk score vs amount")
    df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).plot(
        kind="bar", ax=ax[1], color=["green", "orange", "red"])
    ax[1].set_title("Transactions by risk level")
    ax[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# ------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    # To use real data:  df = pd.read_csv("transactions.csv")  (same columns as FEATURES + txn_id)
    data = generate_data()
    result = monitor(data)

    print("=== Risk summary ===")
    print(result["risk_level"].value_counts(), "\n")

    alerts = result[result["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)
    print(f"=== Top 10 HIGH-risk alerts ({len(alerts)} total) ===")
    print(alerts[["txn_id", "amount", "hour", "risk_score", "reason"]].head(10).to_string(index=False))

    result.to_csv("risk_results.csv", index=False)
    alerts.to_csv("high_risk_alerts.csv", index=False)
    print("\nSaved: risk_results.csv, high_risk_alerts.csv")
    plot_results(result)
