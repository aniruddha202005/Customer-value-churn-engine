"""Step 1: load both sheets of the UCI Online Retail II workbook into a raw MySQL table, untouched."""
import sys

import pandas as pd

from config import RAW_FILE, get_engine

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def read_workbook(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    frames = []
    for name, df in sheets.items():
        df = df.rename(columns=COLUMN_MAP)
        df["source_sheet"] = name
        frames.append(df)
        print(f"  sheet '{name}': {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def main(path=RAW_FILE):
    print(f"Reading {path} ...")
    raw = read_workbook(path)
    raw["invoice_date"] = pd.to_datetime(raw["invoice_date"])
    raw["customer_id"] = raw["customer_id"].astype("Int64")
    engine = get_engine()
    raw.to_sql("raw_transactions", engine, if_exists="replace", index=False, chunksize=20000)
    print(f"Loaded {len(raw):,} rows into raw_transactions")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RAW_FILE)
