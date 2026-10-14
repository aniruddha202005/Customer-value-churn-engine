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
Fill these in from `reports/metrics.json` after running on the real data.

| Model | ROC-AUC | PR-AUC | Precision in top 20% |
|---|---|---|---|
| Recency rule (baseline) | | | |
| Logistic regression | | | |
| Gradient boosting | | | |

| Model | MAE (£) | Spearman | Top-20% revenue capture | Predicted ÷ actual total |
|---|---|---|---|---|
| Last 90 days (baseline) | | | | |
| Ridge (log target) | | | | |
| Gradient boosting (log target) | | | | |
| Hurdle | | | | |

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
1. Download `online_retail_II.xlsx` from https://archive.ics.uci.edu/dataset/502/online+retail+ii and place it in `data/`.
2. In MySQL, run `CREATE DATABASE retail_clv;`
3. Copy `.env.example` to `.env` and enter your MySQL password.
4. Run `python run_pipeline.py`. The first run spends a few minutes reading the Excel file. Use `--skip-load` afterwards.
5. Run `streamlit run app.py`.

## Limitations
- The data covers only two years, so the 90-day windows and the four training cutoffs overlap in history.
- The customers are mostly wholesalers, so purchase patterns are lumpier than in pure retail.
- There is no marketing-contact data, so the project predicts who is at risk, not who would respond to an offer. Measuring that would need an uplift model and an A/B test.
