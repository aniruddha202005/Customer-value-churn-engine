# Customer Value & Churn Engine

Predicts, for every active customer of a UK online gift retailer, **how likely they are to stop buying in the next 90 days** and **how much they are likely to spend in that period**, then turns both into a prioritised retention action list.

**Business question:** the retention team can only contact a limited number of customers each quarter. Which ones are worth saving first?

## Dataset
UCI Machine Learning Repository, *Online Retail II* (Chen, 2019): about 1.07 million invoice lines from a UK online retailer, December 2009 to December 2011, across two Excel sheets.

Known problems handled in this project:
- About a fifth of the lines have no Customer ID.
- Cancellations are invoices starting with "C" and carry negative quantities.
- Non-product stock codes such as postage, bank charges, Amazon fees, manual adjustments and test codes.
- Zero-price lines, bad-debt adjustment invoices ("A"), and exact duplicate rows.

## Pipeline
| Step | File | What it does |
|---|---|---|
| 1 | `src/load_data.py` | Loads both Excel sheets into MySQL as `raw_transactions`, untouched |
| 2 | `src/clean.py` | Applies cleaning rules, logs rows removed by each rule, writes `clean_transactions` |
| 3 | `src/features.py` | Builds one row per customer as of a cutoff date: 21 behavioural features from the 365 days before, labels from the 90 days after |
| 4 | `src/train.py` | Time-based validation: trains on 4 past cutoffs, validates on a later one, compares baselines with models |
| 5 | `src/score.py` | Scores all active customers as of the day after the data ends and assigns a value × risk segment and action |
| — | `app.py` | Streamlit dashboard: overview, action list, customer lookup, model performance, data quality |
| — | `sql/analysis_queries.sql` | MySQL data-quality checks, cohort retention, RFM scoring, Pareto check, action lists |

## Key design decisions
- **Churn definition:** a customer who made at least one purchase in the last 365 days but makes no purchase in the next 90 days.
- **Leakage control:** features use only data before the cutoff, and labels use only data after it.
- **Time-based validation:** the models are tested on a later period they never saw, not on a random split.
- **Baselines first:** the churn model must beat a "days since last order" rule. The value model must beat a "next quarter = last quarter" rule.
- **Skewed revenue:** the value model is trained on log(1 + revenue), with Duan's smearing correction so that totals are not underestimated. A hurdle model (P(buy) × spend-if-buy) is also compared.

## Results
Validated on 4,324 customers at the 2011-09-01 cutoff, a later period the models never saw during training (16,122 customer-snapshots from 4 earlier cutoffs).

### Churn: will the customer place no order in the next 90 days?
Validation churn rate: 50.5%.

| Model | ROC-AUC | PR-AUC | Precision in top 20% |
|---|---|---|---|
| Recency rule (baseline) | 0.708 | 0.679 | 71.6% |
| Logistic regression | 0.766 | 0.724 | 75.8% |
| **Gradient boosting (selected)** | **0.767** | **0.727** | **75.7%** |

Both models beat the recency baseline by about 6 points of ROC-AUC. Logistic regression and gradient boosting are effectively tied.

### Value: how much will the customer spend in the next 90 days?
| Model | MAE (£) | Spearman | Top-20% revenue capture | Predicted ÷ actual total |
|---|---|---|---|---|
| Last 90 days (baseline) | 454 | 0.45 | 66.6% | 0.71 |
| Ridge (log target) | 1,464 | 0.54 | 67.6% | 2.86 |
| Gradient boosting (log target) | 1,669 | 0.55 | 69.4% | 3.24 |
| **Hurdle: P(buy) × spend-if-buy (selected)** | **429** | **0.54** | **70.5%** | **0.58** |

The single-stage log models over-predicted total revenue about three times over: half the customers spend £0, which breaks the smearing correction's assumption of similar errors for every customer. The hurdle model separates "will they buy?" from "how much?" and has the lowest error.

### Retention segments (4,261 active customers)
| | High risk | Medium risk | Low risk |
|---|---|---|---|
| High value | 23 | 50 | 780 |
| Mid value | 257 | 423 | 598 |
| Low value | 1,801 | 285 | 44 |

73 high-value customers are at high or medium risk. They form the priority retention list.

### What I changed along the way
1. **Segmentation:** the first version ranked value by predicted spend, which already discounts churn, so no high-value customer could ever be at risk. Value tiers now use the last 12 months' spend.
2. **Model selection:** choosing the value model by rank correlation alone picked a model that over-predicted revenue 3×. Selection now uses MAE in pounds.

### Open issue
All value models under-predict the validation period (the hurdle model predicts 58% of actual revenue). The validation window, September to November, is the pre-Christmas peak, and churn was also lower then (50%) than in training (60%). The next experiment is a seasonality feature.
