-- Run once before the pipeline
CREATE DATABASE IF NOT EXISTS retail_clv;
USE retail_clv;

-- ============================================================
-- A. DATA QUALITY CHECKS on raw_transactions (run after load_data.py)
-- ============================================================

-- A1. How many rows have no customer, and how much revenue do they carry?
SELECT
    COUNT(*)                                             AS total_rows,
    SUM(customer_id IS NULL)                             AS rows_without_customer,
    ROUND(100 * SUM(customer_id IS NULL) / COUNT(*), 1)  AS pct_rows_without_customer,
    ROUND(SUM(CASE WHEN customer_id IS NULL THEN quantity * price ELSE 0 END), 0) AS revenue_without_customer
FROM raw_transactions;

-- A2. Cancellations vs sales
SELECT
    CASE WHEN invoice LIKE 'C%' THEN 'cancellation'
         WHEN invoice LIKE 'A%' THEN 'bad debt adjustment'
         ELSE 'sale' END                        AS invoice_type,
    COUNT(*)                                    AS lines,
    ROUND(SUM(quantity * price), 0)             AS revenue
FROM raw_transactions
GROUP BY invoice_type;

-- A3. Non-product stock codes (postage, fees, manual entries)
SELECT stock_code, MIN(description) AS example_description, COUNT(*) AS lines
FROM raw_transactions
WHERE stock_code NOT REGEXP '^[0-9]{5}[A-Za-z]{0,2}$'
GROUP BY stock_code
ORDER BY lines DESC
LIMIT 20;

-- A4. Exact duplicate rows
SELECT COUNT(*) - COUNT(DISTINCT invoice, stock_code, description, quantity, invoice_date, price, customer_id, country)
       AS duplicate_rows
FROM raw_transactions;

-- ============================================================
-- B. BUSINESS ANALYSIS on clean_transactions (run after clean.py)
-- ============================================================

-- B1. Monthly revenue, orders and active customers
SELECT
    DATE_FORMAT(invoice_date, '%Y-%m')       AS month,
    ROUND(SUM(revenue), 0)                   AS net_revenue,
    COUNT(DISTINCT CASE WHEN is_return = 0 THEN invoice END) AS orders,
    COUNT(DISTINCT customer_id)              AS active_customers
FROM clean_transactions
GROUP BY month
ORDER BY month;

-- B2. Monthly acquisition cohorts and retention (share of cohort buying again in month N)
WITH first_purchase AS (
    SELECT customer_id, DATE_FORMAT(MIN(invoice_date), '%Y-%m-01') AS cohort_month
    FROM clean_transactions
    WHERE has_customer = 1 AND is_return = 0
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT t.customer_id, f.cohort_month,
           TIMESTAMPDIFF(MONTH, f.cohort_month, DATE_FORMAT(t.invoice_date, '%Y-%m-01')) AS months_since_first
    FROM clean_transactions t
    JOIN first_purchase f ON f.customer_id = t.customer_id
    WHERE t.is_return = 0
)
SELECT cohort_month,
       COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END) AS cohort_size,
       ROUND(100 * COUNT(DISTINCT CASE WHEN months_since_first = 1 THEN customer_id END)
             / COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END), 1) AS m1_retention_pct,
       ROUND(100 * COUNT(DISTINCT CASE WHEN months_since_first = 3 THEN customer_id END)
             / COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END), 1) AS m3_retention_pct,
       ROUND(100 * COUNT(DISTINCT CASE WHEN months_since_first = 6 THEN customer_id END)
             / COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END), 1) AS m6_retention_pct
FROM activity
GROUP BY cohort_month
ORDER BY cohort_month;

-- B3. Classic RFM quintile scoring as of the last date in the data
WITH snapshot AS (SELECT DATE_ADD(MAX(invoice_date), INTERVAL 1 DAY) AS as_of FROM clean_transactions),
rfm AS (
    SELECT t.customer_id,
           DATEDIFF((SELECT as_of FROM snapshot), MAX(t.invoice_date)) AS recency_days,
           COUNT(DISTINCT t.invoice)                                   AS frequency,
           ROUND(SUM(t.revenue), 2)                                    AS monetary
    FROM clean_transactions t
    WHERE t.has_customer = 1 AND t.is_return = 0
    GROUP BY t.customer_id
)
SELECT customer_id, recency_days, frequency, monetary,
       6 - NTILE(5) OVER (ORDER BY recency_days) AS r_score,
       NTILE(5) OVER (ORDER BY frequency)        AS f_score,
       NTILE(5) OVER (ORDER BY monetary)         AS m_score
FROM rfm
ORDER BY monetary DESC;

-- B4. Pareto check: what share of revenue comes from the top 20% of customers?
WITH spend AS (
    SELECT customer_id, SUM(revenue) AS total
    FROM clean_transactions WHERE has_customer = 1
    GROUP BY customer_id
),
ranked AS (
    SELECT total, PERCENT_RANK() OVER (ORDER BY total DESC) AS pct_rank FROM spend
)
SELECT ROUND(100 * SUM(CASE WHEN pct_rank < 0.2 THEN total ELSE 0 END) / SUM(total), 1) AS top20_revenue_share_pct
FROM ranked;

-- ============================================================
-- C. ACTION LISTS on customer_scores (run after score.py)
-- ============================================================

-- C1. Segment summary for the retention team
SELECT value_tier, risk_tier,
       COUNT(*)                         AS customers,
       ROUND(SUM(pred_value_90d), 0)    AS predicted_90d_value,
       ROUND(SUM(value_at_risk), 0)     AS value_at_risk
FROM customer_scores
GROUP BY value_tier, risk_tier
ORDER BY value_at_risk DESC;

-- C2. Top 50 customers to call this week
SELECT customer_id, ROUND(churn_prob, 2) AS churn_prob, ROUND(pred_value_90d, 0) AS pred_value_90d,
       ROUND(value_at_risk, 0) AS value_at_risk, recency_days, frequency, recommended_action
FROM customer_scores
WHERE value_tier = 'High value' AND risk_tier IN ('High risk', 'Medium risk')
ORDER BY value_at_risk DESC
LIMIT 50;
