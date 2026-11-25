"""Load OHLCV from local CSV files: <dir>/<SYMBOL>_<interval>.csv with a
timestamp column (ISO or epoch ms) plus open/high/low/close/volume."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .base import DataProvider, validate_ohlcv


class CsvProvider(DataProvider):
    name = "csv"

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 1000,
                    start: datetime | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.dir / f"{symbol}_{interval}.csv")
        ts_col = next(c for c in df.columns if c.lower() in ("timestamp", "time", "date", "t", "open_time"))
        ts = df[ts_col]
        df.index = pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts) \
            else pd.to_datetime(ts, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = validate_ohlcv(df)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        return df.tail(limit)
