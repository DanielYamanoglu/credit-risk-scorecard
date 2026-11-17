"""Download and load the UCI 'Default of Credit Card Clients' dataset (Taiwan, 2005)."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "default"
TEST_SIZE = 0.3
SEED = 42

# Delay codes per month: -2 no consumption, -1 paid in full, 0 revolving credit, 1..9 months late.
PAY_COLS = ["pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAID_COLS = [f"pay_amt{i}" for i in range(1, 7)]


def download(path: Path = RAW) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.write_bytes(archive.read(next(n for n in archive.namelist() if n.endswith(".xls"))))
    return path


def load(path: Path = RAW) -> pd.DataFrame:
    frame = pd.read_excel(path, header=1).rename(columns=str.lower)
    # The source names the most recent month PAY_0 but the amounts BILL_AMT1; align them.
    frame = frame.rename(columns={"pay_0": "pay_1", "default payment next month": TARGET})
    return frame.drop(columns=["id"])


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split; the hold-out is touched once, for final scoring."""
    train, test = train_test_split(frame, test_size=TEST_SIZE, stratify=frame[TARGET], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)
