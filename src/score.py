"""Step 5: score every active customer as of the day after the data ends, and assign an action segment."""
import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, REPORT_DIR, get_engine
from features import build_dataset, model_matrix
from train import load_transactions

ACTIONS = {
    ("High value", "High risk"): "Win back now: personal call or account manager",
    ("High value", "Medium risk"): "Protect: loyalty perk or early access",
    ("High value", "Low risk"): "Nurture: thank-you, cross-sell, no discount",
    ("Mid value", "High risk"): "Targeted discount on usual products",
    ("Mid value", "Medium risk"): "Reminder email with recommendations",
    ("Mid value", "Low risk"): "Upsell bundles",
    ("Low value", "High risk"): "Low-cost automated email only",
    ("Low value", "Medium risk"): "Automated newsletter",
    ("Low value", "Low risk"): "Automated newsletter",
}


def assign_segments(df):
    rank = df["net_revenue"].rank(pct=True, method="first")
    df["value_tier"] = np.select([rank > 0.8, rank > 0.5], ["High value", "Mid value"], "Low value")
    df["risk_tier"] = np.select([df["churn_prob"] >= 0.6, df["churn_prob"] >= 0.4], ["High risk", "Medium risk"], "Low risk")
    df["value_at_risk"] = df["churn_prob"] * df["net_revenue"] * 90 / 365
    df["recommended_action"] = [ACTIONS[(v, r)] for v, r in zip(df["value_tier"], df["risk_tier"])]
    return df


def main():
    tx = load_transactions()
    cutoff = (tx["invoice_date"].max() + pd.Timedelta(days=1)).normalize()
    bundle = joblib.load(MODEL_DIR / "models.joblib")
    feats = build_dataset(tx, cutoff, with_labels=False)
    X = model_matrix(feats)
    feats["churn_prob"] = bundle["churn"].predict_proba(X)[:, 1]
    feats["pred_value_90d"] = bundle["value"].predict(X)
    scored = assign_segments(feats)
    scored["customer_id"] = scored["customer_id"].astype(int)
    scored.to_sql("customer_scores", get_engine(), if_exists="replace", index=False)
    scored.to_csv(REPORT_DIR / "customer_scores.csv", index=False)
    
    tx["month"] = tx["invoice_date"].dt.to_period("M").dt.to_timestamp()
    sales = tx[~tx["is_return"]]
    monthly = pd.DataFrame({
        "net_revenue": tx.groupby("month")["revenue"].sum(),
        "orders": sales.groupby("month")["invoice"].nunique(),
        "active_customers": sales.groupby("month")["customer_id"].nunique(),
    }).reset_index()
    monthly.to_csv(REPORT_DIR / "monthly_trend.csv", index=False)
    print(f"Scored {len(scored):,} customers as of {cutoff.date()}")
    print(scored.groupby(["value_tier", "risk_tier"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
