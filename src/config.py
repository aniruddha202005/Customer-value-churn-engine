import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RAW_FILE = DATA_DIR / "online_retail_II.xlsx"

OBSERVATION_DAYS = 365
PREDICTION_DAYS = 90
TRAIN_CUTOFFS = ["2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]
VALID_CUTOFF = "2011-09-01"
RANDOM_STATE = 42

NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "AMAZONFEE",
    "CRUK", "PADS", "ADJUST", "ADJUST2", "TEST001", "TEST002", "GIFT",
}


def get_engine():
    url = os.getenv("DB_URL")
    if not url:
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        db = os.getenv("MYSQL_DB", "retail_clv")
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)
