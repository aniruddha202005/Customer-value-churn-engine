"""Step 4: compare baselines and models on a time-based split, then save the winners."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, RANDOM_STATE, REPORT_DIR, TRAIN_CUTOFFS, VALID_CUTOFF, get_engine
from features import FEATURES, build_dataset, model_matrix
from value_models import HurdleValueModel, LogTargetRegressor


def load_transactions():
    tx = pd.read_sql("SELECT * FROM clean_transactions", get_engine(), parse_dates=["invoice_date"])
    tx["is_return"] = tx["is_return"].astype(bool)
    tx["has_customer"] = tx["has_customer"].astype(bool)
    return tx


def precision_at_top(y_true, score, share=0.2):
    k = max(1, int(len(score) * share))
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y_true)[top].mean())


def revenue_capture(y_true, score, share=0.2):
    k = max(1, int(len(score) * share))
    top = np.argsort(-np.asarray(score))[:k]
    total = np.asarray(y_true).sum()
    return float(np.asarray(y_true)[top].sum() / total) if total > 0 else 0.0


def churn_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                            min_samples_leaf=40, l2_regularization=1.0,
                                                            random_state=RANDOM_STATE),
    }


def gbm_regressor():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=RANDOM_STATE)


def value_models(churn_model):
    return {
        "ridge_log": LogTargetRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "gradient_boosting_log": LogTargetRegressor(gbm_regressor()),
        "hurdle_churn_x_spend": HurdleValueModel(churn_model, gbm_regressor()),
    }


def fit_value(model, X, y, churned):
    if isinstance(model, HurdleValueModel):
        return model.fit(X, y, churned)
    return model.fit(X, y)


def spearman_scorer(model, X, y):
    return spearmanr(y, model.predict(X)).statistic


def evaluate_churn(y, score):
    return {"roc_auc": round(roc_auc_score(y, score), 4),
            "pr_auc": round(average_precision_score(y, score), 4),
            "precision_top20": round(precision_at_top(y, score), 4)}


def evaluate_value(y, pred):
    return {"mae": round(mean_absolute_error(y, pred), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            "spearman": round(float(spearmanr(y, pred).statistic), 4),
            "top20_revenue_capture": round(revenue_capture(y, pred), 4),
            "total_pred_vs_actual": round(float(np.sum(pred) / np.sum(y)), 3)}


def main():
    tx = load_transactions()
    train = pd.concat([build_dataset(tx, c) for c in TRAIN_CUTOFFS], ignore_index=True)
    valid = build_dataset(tx, VALID_CUTOFF)
    print(f"Train rows: {len(train):,} across {len(TRAIN_CUTOFFS)} cutoffs | churn rate {train['churned'].mean():.1%}")
    print(f"Valid rows: {len(valid):,} at {VALID_CUTOFF} | churn rate {valid['churned'].mean():.1%}")

    X_tr, X_va = model_matrix(train), model_matrix(valid)
    yc_tr, yc_va = train["churned"], valid["churned"]
    yv_tr, yv_va = train["future_revenue"], valid["future_revenue"]

    results = {"churn": {}, "value": {}, "data": {
        "train_rows": len(train), "valid_rows": len(valid),
        "train_churn_rate": round(float(yc_tr.mean()), 4), "valid_churn_rate": round(float(yc_va.mean()), 4),
        "train_cutoffs": TRAIN_CUTOFFS, "valid_cutoff": VALID_CUTOFF}}

    results["churn"]["baseline_recency_rule"] = evaluate_churn(yc_va, valid["recency_days"])
    fitted_churn = {}
    for name, model in churn_models().items():
        model.fit(X_tr, yc_tr)
        fitted_churn[name] = model
        results["churn"][name] = evaluate_churn(yc_va, model.predict_proba(X_va)[:, 1])

    best_churn = max(fitted_churn, key=lambda n: results["churn"][n]["pr_auc"])

    results["value"]["baseline_last_90_days"] = evaluate_value(yv_va, valid["revenue_last_90"])
    fitted_value = {}
    for name, model in value_models(churn_models()[best_churn]).items():
        fit_value(model, X_tr, yv_tr, yc_tr)
        fitted_value[name] = model
        results["value"][name] = evaluate_value(yv_va, model.predict(X_va))

    best_value = min(fitted_value, key=lambda n: results["value"][n]["mae"])
    results["selected"] = {"churn": best_churn, "value": best_value}
    print(json.dumps(results, indent=2))

    imp = permutation_importance(fitted_churn[best_churn], X_va, yc_va, scoring="average_precision",
                                 n_repeats=5, random_state=RANDOM_STATE)
    pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False).to_csv(REPORT_DIR / "churn_feature_importance.csv", index=False)

    imp_v = permutation_importance(fitted_value[best_value], X_va, yv_va, scoring=spearman_scorer,
                                   n_repeats=5, random_state=RANDOM_STATE)
    pd.DataFrame({"feature": FEATURES, "importance": imp_v.importances_mean}) \
        .sort_values("importance", ascending=False).to_csv(REPORT_DIR / "value_feature_importance.csv", index=False)

    valid_out = valid[["customer_id", "churned", "future_revenue", "revenue_last_90"]].copy()
    valid_out["churn_prob"] = fitted_churn[best_churn].predict_proba(X_va)[:, 1]
    valid_out["pred_value"] = fitted_value[best_value].predict(X_va)
    valid_out.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    full = pd.concat([train, valid], ignore_index=True)
    X_full = model_matrix(full)
    final_churn = churn_models()[best_churn].fit(X_full, full["churned"])
    final_value = fit_value(value_models(churn_models()[best_churn])[best_value], X_full,
                            full["future_revenue"], full["churned"])
    joblib.dump({"churn": final_churn, "value": final_value, "features": FEATURES}, MODEL_DIR / "models.joblib")

    with open(REPORT_DIR / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved models (churn={best_churn}, value={best_value}) retrained on train + valid")


if __name__ == "__main__":
    main()
