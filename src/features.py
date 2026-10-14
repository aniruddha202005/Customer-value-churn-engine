"""Step 3: build one row per customer as of a cutoff date.

Features only use transactions BEFORE the cutoff. Labels only use the PREDICTION_DAYS AFTER it.
This split is what prevents leakage.
"""
import numpy as np
import pandas as pd

from config import OBSERVATION_DAYS, PREDICTION_DAYS


def build_features(tx, cutoff):
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=OBSERVATION_DAYS)
    tx = tx[tx["has_customer"]]
    history = tx[tx["invoice_date"] < cutoff]
    window = history[history["invoice_date"] >= start]
    sales = window[~window["is_return"]]
    returns = window[window["is_return"]]

    if sales.empty:
        return pd.DataFrame()

    orders = (sales.groupby(["customer_id", "invoice"])
                   .agg(order_date=("invoice_date", "min"), order_value=("revenue", "sum"),
                        lines=("stock_code", "size"))
                   .reset_index()
                   .sort_values(["customer_id", "order_date"]))
    orders["gap_days"] = orders.groupby("customer_id")["order_date"].diff().dt.days

    f = orders.groupby("customer_id").agg(
        last_order=("order_date", "max"),
        frequency=("invoice", "nunique"),
        gross_revenue=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        max_order_value=("order_value", "max"),
        avg_basket_lines=("lines", "mean"),
        avg_gap_days=("gap_days", "mean"),
        std_gap_days=("gap_days", "std"),
    )
    f["recency_days"] = (cutoff - f["last_order"]).dt.days
    f = f.drop(columns="last_order")

    first_ever = history[~history["is_return"]].groupby("customer_id")["invoice_date"].min()
    f["tenure_days"] = (cutoff - first_ever.reindex(f.index)).dt.days

    f["distinct_products"] = sales.groupby("customer_id")["stock_code"].nunique()
    f["active_months"] = sales.groupby("customer_id")["invoice_date"].apply(lambda s: s.dt.to_period("M").nunique())

    returned = returns.groupby("customer_id")["revenue"].sum().abs()
    f["returned_revenue"] = returned.reindex(f.index).fillna(0)
    f["return_rate"] = f["returned_revenue"] / f["gross_revenue"]
    f["net_revenue"] = f["gross_revenue"] - f["returned_revenue"]

    for label, lo, hi in [("last_90", 0, 90), ("prev_90", 90, 180)]:
        part = sales[(sales["invoice_date"] >= cutoff - pd.Timedelta(days=hi)) &
                     (sales["invoice_date"] < cutoff - pd.Timedelta(days=lo))]
        f[f"revenue_{label}"] = part.groupby("customer_id")["revenue"].sum().reindex(f.index).fillna(0)
        f[f"orders_{label}"] = part.groupby("customer_id")["invoice"].nunique().reindex(f.index).fillna(0)

    f["revenue_trend"] = (f["revenue_last_90"] + 1) / (f["revenue_prev_90"] + 1)
    f["one_time_buyer"] = (f["frequency"] == 1).astype(int)
    f["avg_gap_days"] = f["avg_gap_days"].fillna(OBSERVATION_DAYS)
    f["std_gap_days"] = f["std_gap_days"].fillna(0)
    f["overdue_ratio"] = f["recency_days"] / f["avg_gap_days"].clip(lower=1)

    country = sales.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])
    f["is_uk"] = (country.reindex(f.index) == "United Kingdom").astype(int)

    f["cutoff"] = cutoff
    return f.reset_index()


def build_labels(tx, cutoff):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)
    future = tx[tx["has_customer"] & (tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end)]
    bought = future[~future["is_return"]].groupby("customer_id")["invoice"].nunique()
    value = future.groupby("customer_id")["revenue"].sum().clip(lower=0)
    labels = pd.DataFrame({"future_orders": bought, "future_revenue": value}).fillna(0)
    return labels


def build_dataset(tx, cutoff, with_labels=True):
    feats = build_features(tx, cutoff)
    if feats.empty or not with_labels:
        return feats
    labels = build_labels(tx, cutoff)
    data = feats.merge(labels, left_on="customer_id", right_index=True, how="left")
    data[["future_orders", "future_revenue"]] = data[["future_orders", "future_revenue"]].fillna(0)
    data["churned"] = (data["future_orders"] == 0).astype(int)
    return data


FEATURES = [
    "recency_days", "frequency", "gross_revenue", "net_revenue", "avg_order_value",
    "max_order_value", "avg_basket_lines", "avg_gap_days", "std_gap_days", "tenure_days",
    "distinct_products", "active_months", "return_rate", "revenue_last_90", "orders_last_90",
    "revenue_prev_90", "orders_prev_90", "revenue_trend", "one_time_buyer", "overdue_ratio", "is_uk",
]

LOG_FEATURES = ["gross_revenue", "net_revenue", "avg_order_value", "max_order_value",
                "revenue_last_90", "revenue_prev_90"]


def model_matrix(df):
    X = df[FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col].clip(lower=0))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X
