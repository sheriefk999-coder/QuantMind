"""Provider interface. Providers only *deliver data*; no analysis lives here.

All frames returned are indexed by tz-aware UTC timestamps of bar OPEN time and
contain only bars that have fully closed (no partial candle leaks into analysis).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class OrderBookSnapshot:
    ts: datetime
    bids: list[tuple[float, float]]  # (price, qty) best first
    asks: list[tuple[float, float]]

    @property
    def mid(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self) -> float:
        return (self.asks[0][0] - self.bids[0][0]) / self.mid * 1e4

    def imbalance(self, depth_bps: float = 25.0) -> float:
        """(bid_qty - ask_qty) / total within depth_bps of mid. Range [-1, 1]."""
        lo, hi = self.mid * (1 - depth_bps / 1e4), self.mid * (1 + depth_bps / 1e4)
        b = sum(q for p, q in self.bids if p >= lo)
        a = sum(q for p, q in self.asks if p <= hi)
        return 0.0 if a + b == 0 else (b - a) / (a + b)


@dataclass
class MarketSnapshot:
    """Everything known about one symbol at decision time. Missing data stays None
    so agents can abstain instead of inventing inputs."""
    symbol: str
    ohlcv: pd.DataFrame
    funding: pd.Series | None = None
    open_interest: pd.Series | None = None
    orderbook: OrderBookSnapshot | None = None
    related: dict[str, pd.DataFrame] = field(default_factory=dict)
    sentiment: pd.Series | None = None
    calendar: list[dict] = field(default_factory=list)  # [{"ts":..., "title":..., "impact": "high"}]


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 1000,
                    start: datetime | None = None) -> pd.DataFrame: ...

    def fetch_funding(self, symbol: str, limit: int = 500) -> pd.Series | None:
        return None

    def fetch_open_interest(self, symbol: str, interval: str, limit: int = 500) -> pd.Series | None:
        return None

    def fetch_orderbook(self, symbol: str, depth: int = 100) -> OrderBookSnapshot | None:
        return None


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Reject corrupt data rather than silently analysing it."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    df = df[OHLCV_COLUMNS].astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    bad |= (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        df = df[~bad]
    return df
