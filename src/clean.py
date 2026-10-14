"""Step 2: turn raw_transactions into clean_transactions and record what every rule removed."""
import pandas as pd

from config import REPORT_DIR, get_engine

PRODUCT_CODE_PATTERN = r"^\d{5}[A-Z]{0,2}$"


def log_step(log, df, step, reason):
    log.append({"step": step, "rows_after": len(df), "reason": reason})
    print(f"  {step:<28}{len(df):>10,} rows")


def clean(raw):
    log = []
    df = raw.copy()
    log_step(log, df, "raw", "all rows from both sheets")

    df = df.drop_duplicates(subset=[c for c in df.columns if c != "source_sheet"])
    log_step(log, df, "drop_exact_duplicates", "identical rows, incl. any sheet overlap")

    df["invoice"] = df["invoice"].astype(str).str.strip().str.upper()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype(str).str.strip()

    df = df[~df["invoice"].str.startswith("A")]
    log_step(log, df, "drop_bad_debt_invoices", "invoices starting with A are accounting adjustments")

    is_product = df["stock_code"].str.match(PRODUCT_CODE_PATTERN)
    non_product = df.loc[~is_product, "stock_code"].value_counts().head(25)
    non_product.to_csv(REPORT_DIR / "non_product_codes.csv", header=["rows"])
    df = df[is_product]
    log_step(log, df, "drop_non_product_codes", "postage, fees, manual and test codes")

    df = df[df["price"] > 0]
    log_step(log, df, "drop_zero_price", "free items and price adjustments")

    df["is_return"] = df["invoice"].str.startswith("C")
    consistent = (df["is_return"] & (df["quantity"] < 0)) | (~df["is_return"] & (df["quantity"] > 0))
    df = df[consistent]
    log_step(log, df, "drop_sign_mismatch", "sales must be positive, cancellations negative")

    df["revenue"] = df["quantity"] * df["price"]
    df["has_customer"] = df["customer_id"].notna()

    missing_share = df.loc[~df["has_customer"] & ~df["is_return"], "revenue"].sum() / df.loc[~df["is_return"], "revenue"].sum()
    log.append({"step": "note_missing_customer", "rows_after": int((~df["has_customer"]).sum()),
                "reason": f"rows without customer_id kept for revenue totals but excluded from modelling ({missing_share:.1%} of sales revenue)"})
    print(f"  rows without customer_id: {(~df['has_customer']).sum():,} ({missing_share:.1%} of sales revenue)")

    cols = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "revenue", "customer_id", "country", "is_return", "has_customer"]
    return df[cols].reset_index(drop=True), pd.DataFrame(log)


def main():
    engine = get_engine()
    raw = pd.read_sql("SELECT * FROM raw_transactions", engine, parse_dates=["invoice_date"])
    print("Cleaning ...")
    clean_df, log = clean(raw)
    clean_df.to_sql("clean_transactions", engine, if_exists="replace", index=False, chunksize=20000)
    log.to_sql("data_quality_log", engine, if_exists="replace", index=False)
    log.to_csv(REPORT_DIR / "data_quality_log.csv", index=False)
    print(f"Saved {len(clean_df):,} rows to clean_transactions")


if __name__ == "__main__":
    main()
