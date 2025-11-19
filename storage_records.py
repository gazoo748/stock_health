# storage_records.py

import os
import sqlite3
from typing import Optional

import pandas as pd

from models import StockCheckup

DB_PATH = "stock_health_records.sqlite"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            rsi REAL,
            pe REAL,
            wk52_pct REAL,
            spy_trend_ok INTEGER,
            ttm_net_income REAL,
            debt_assets_ratio REAL
        )
        """
    )
    return conn


def save_checkup(checkup: StockCheckup) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO records (
                ticker, rsi, pe, wk52_pct, spy_trend_ok,
                ttm_net_income, debt_assets_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkup.identity.symbol,
                checkup.technical.rsi_14,
                checkup.valuation.pe_ratio,
                checkup.technical.wk52_position_pct,
                1 if checkup.market.spy_trend_ok else 0,
                checkup.fundamentals.ttm_net_income,
                checkup.fundamentals.debt_asset_ratio,
            ),
        )
    conn.close()


def load_history(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            ts,
            rsi,
            pe,
            wk52_pct,
            spy_trend_ok,
            ttm_net_income,
            debt_assets_ratio
        FROM records
        WHERE ticker = ?
        ORDER BY ts ASC
        """,
        conn,
        params=(ticker.upper(),),
        parse_dates=["ts"],
    )
    conn.close()
    return df

