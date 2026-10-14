import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import REPORT_DIR, get_engine

st.set_page_config(page_title="Customer Value & Churn Engine", layout="wide")

TIER_ORDER = ["High value", "Mid value", "Low value"]
RISK_ORDER = ["High risk", "Medium risk", "Low risk"]


@st.cache_data
def load_scores():
    return pd.read_sql("SELECT * FROM customer_scores", get_engine())


@st.cache_data
def load_monthly():
    tx = pd.read_sql("SELECT invoice_date, revenue, invoice, customer_id, is_return FROM clean_transactions",
                     get_engine(), parse_dates=["invoice_date"])
    tx["month"] = tx["invoice_date"].dt.to_period("M").dt.to_timestamp()
    sales = tx[tx["is_return"] == 0]
    return pd.DataFrame({
        "net_revenue": tx.groupby("month")["revenue"].sum(),
        "orders": sales.groupby("month")["invoice"].nunique(),
        "active_customers": sales.groupby("month")["customer_id"].nunique(),
    }).reset_index()


@st.cache_data
def load_reports():
    with open(REPORT_DIR / "metrics.json") as fh:
        metrics = json.load(fh)
    return (metrics,
            pd.read_csv(REPORT_DIR / "validation_predictions.csv"),
            pd.read_csv(REPORT_DIR / "churn_feature_importance.csv"),
            pd.read_csv(REPORT_DIR / "value_feature_importance.csv"),
            pd.read_csv(REPORT_DIR / "data_quality_log.csv"))


def money(x):
    return f"£{x:,.0f}"


scores = load_scores()
metrics, valid, churn_imp, value_imp, dq = load_reports()

st.sidebar.title("Customer Value & Churn Engine")
page = st.sidebar.radio("Page", ["Overview", "Retention action list", "Customer lookup",
                                 "Model performance", "Data quality"])
st.sidebar.caption(f"Customers scored: {len(scores):,}")

if page == "Overview":
    st.title("Which customers are worth saving next quarter?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active customers (last 12 months)", f"{len(scores):,}")
    c2.metric("Predicted revenue, next 90 days", money(scores["pred_value_90d"].sum()))
    c3.metric("Customers at high churn risk", f"{(scores['risk_tier'] == 'High risk').sum():,}")
    c4.metric("Quarterly revenue at risk", money(scores["value_at_risk"].sum()))

    monthly = load_monthly()
    st.subheader("Business trend")
    st.plotly_chart(px.line(monthly, x="month", y=["net_revenue"], markers=True,
                            labels={"value": "Net revenue (£)", "month": ""}), width="stretch")

    st.subheader("Value × risk matrix")
    grid = scores.pivot_table(index="value_tier", columns="risk_tier", values="customer_id",
                              aggfunc="count", fill_value=0).reindex(index=TIER_ORDER, columns=RISK_ORDER, fill_value=0)
    var = scores.pivot_table(index="value_tier", columns="risk_tier", values="value_at_risk",
                             aggfunc="sum", fill_value=0).reindex(index=TIER_ORDER, columns=RISK_ORDER, fill_value=0)
    left, right = st.columns(2)
    left.plotly_chart(px.imshow(grid, text_auto=True, color_continuous_scale="Blues",
                                title="Number of customers"), width="stretch")
    right.plotly_chart(px.imshow(var.round(0), text_auto=",.0f", color_continuous_scale="Reds",
                                 title="Revenue at risk (£)"), width="stretch")

    sorted_value = np.sort(scores["pred_value_90d"].values)[::-1]
    top20 = sorted_value[: int(len(sorted_value) * 0.2)].sum() / max(sorted_value.sum(), 1)
    st.info(f"The top 20% of customers by predicted value account for {top20:.0%} of next-quarter predicted revenue.")

elif page == "Retention action list":
    st.title("Retention action list")
    tiers = st.multiselect("Value tier", TIER_ORDER, default=["High value"])
    risks = st.multiselect("Risk tier", RISK_ORDER, default=["High risk", "Medium risk"])
    budget = st.slider("How many customers can the team contact?", 10, 500, 50, step=10)
    view = scores[scores["value_tier"].isin(tiers) & scores["risk_tier"].isin(risks)] \
        .sort_values("value_at_risk", ascending=False).head(budget)
    st.metric("Revenue at risk covered by this list", money(view["value_at_risk"].sum()),
              f"{view['value_at_risk'].sum() / max(scores['value_at_risk'].sum(), 1):.0%} of total")
    cols = ["customer_id", "churn_prob", "pred_value_90d", "value_at_risk", "recency_days",
            "frequency", "net_revenue", "value_tier", "risk_tier", "recommended_action"]
    st.dataframe(view[cols].round(2), width="stretch", hide_index=True)
    st.download_button("Download list as CSV", view[cols].to_csv(index=False), "retention_list.csv")

elif page == "Customer lookup":
    st.title("Customer lookup")
    cid = st.selectbox("Customer ID", scores.sort_values("value_at_risk", ascending=False)["customer_id"])
    row = scores[scores["customer_id"] == cid].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability (next 90 days)", f"{row['churn_prob']:.0%}")
    c2.metric("Predicted spend (next 90 days)", money(row["pred_value_90d"]))
    c3.metric("Segment", f"{row['value_tier']} · {row['risk_tier']}")
    st.success(f"Recommended action: {row['recommended_action']}")
    st.subheader("Why the model sees this customer this way")
    profile = pd.DataFrame({
        "This customer": [row["recency_days"], row["frequency"], row["avg_gap_days"], row["overdue_ratio"],
                          row["net_revenue"], row["return_rate"], row["revenue_trend"]],
        "Median customer": [scores["recency_days"].median(), scores["frequency"].median(),
                            scores["avg_gap_days"].median(), scores["overdue_ratio"].median(),
                            scores["net_revenue"].median(), scores["return_rate"].median(),
                            scores["revenue_trend"].median()],
    }, index=["Days since last order", "Orders (12 months)", "Average days between orders",
              "Overdue ratio (recency ÷ usual gap)", "Net revenue (12 months)", "Return rate",
              "Revenue trend (last 90 ÷ previous 90)"]).round(2)
    st.dataframe(profile, width="stretch")

elif page == "Model performance":
    st.title("How good are the models?")
    d = metrics["data"]
    st.caption(f"Trained on {d['train_rows']:,} customer-snapshots from cutoffs {', '.join(d['train_cutoffs'])}. "
               f"Validated on {d['valid_rows']:,} customers at {d['valid_cutoff']}, a later period the models never saw.")
    st.subheader("Churn model vs baseline")
    st.dataframe(pd.DataFrame(metrics["churn"]).T, width="stretch")
    st.subheader("Value model vs baseline")
    st.dataframe(pd.DataFrame(metrics["value"]).T, width="stretch")
    st.write(f"Selected: churn = **{metrics['selected']['churn']}**, value = **{metrics['selected']['value']}**")

    valid["decile"] = pd.qcut(valid["churn_prob"].rank(method="first"), 10, labels=range(1, 11))
    lift = valid.groupby("decile", observed=True)["churned"].mean().reset_index()
    st.plotly_chart(px.bar(lift, x="decile", y="churned", title="Actual churn rate by predicted-risk decile (10 = riskiest)",
                           labels={"churned": "Actual churn rate"}), width="stretch")

    valid["value_decile"] = pd.qcut(valid["pred_value"].rank(method="first"), 10, labels=range(1, 11))
    cal = valid.groupby("value_decile", observed=True)[["pred_value", "future_revenue"]].mean().reset_index()
    st.plotly_chart(px.bar(cal, x="value_decile", y=["pred_value", "future_revenue"], barmode="group",
                           title="Predicted vs actual 90-day revenue by predicted-value decile"),
                    width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(px.bar(churn_imp.head(10).iloc[::-1], x="importance", y="feature", orientation="h",
                             title="Top churn drivers (permutation importance)"), width="stretch")
    right.plotly_chart(px.bar(value_imp.head(10).iloc[::-1], x="importance", y="feature", orientation="h",
                              title="Top value drivers (permutation importance)"), width="stretch")

else:
    st.title("Data quality: what was removed and why")
    st.dataframe(dq, width="stretch", hide_index=True)
    steps = dq[dq["step"] != "note_missing_customer"]
    st.plotly_chart(px.bar(steps, x="step", y="rows_after", title="Rows remaining after each cleaning rule"),
                    width="stretch")
    codes = REPORT_DIR / "non_product_codes.csv"
    if codes.exists():
        st.subheader("Most common non-product codes removed")
        st.dataframe(pd.read_csv(codes), width="stretch", hide_index=True)
