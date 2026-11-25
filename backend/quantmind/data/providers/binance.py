"""Binance public market-data adapter (no API key needed).

Used purely as a data pipe: candles, funding, open interest, order book.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

from .base import DataProvider, OrderBookSnapshot, validate_ohlcv

# data-api.binance.vision serves public market data and, unlike api.binance.com,
# is reachable from GitHub Actions runners (US IPs). Override with QM_BINANCE_SPOT_URL.
import os
SPOT = os.getenv("QM_BINANCE_SPOT_URL", "https://data-api.binance.vision")
FUTURES = "https://fapi.binance.com"

_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
                "4h": 14_400_000, "1d": 86_400_000}


class BinanceProvider(DataProvider):
    name = "binance"

    def __init__(self, timeout: float = 15.0):
        self.client = httpx.Client(timeout=timeout)

    def _get(self, url: str, params: dict) -> list | dict:
        r = self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 1000,
                    start: datetime | None = None) -> pd.DataFrame:
        step = _INTERVAL_MS[interval]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000) if start else now_ms - step * (limit + 1)
        rows: list = []
        while len(rows) < limit:
            batch = self._get(f"{SPOT}/api/v3/klines", {
                "symbol": symbol, "interval": interval, "startTime": start_ms,
                "limit": min(1000, limit - len(rows))})
            if not batch:
                break
            rows.extend(batch)
            start_ms = batch[-1][0] + step
            if start_ms >= now_ms:
                break
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "close_t",
                                         "qv", "n", "tb", "tq", "_"])
        if df.empty:
            return df
        # Drop the still-forming candle: its close time is in the future.
        df = df[df["close_t"] < now_ms]
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return validate_ohlcv(df)

    def fetch_funding(self, symbol: str, limit: int = 500) -> pd.Series | None:
        try:
            data = self._get(f"{FUTURES}/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
        except httpx.HTTPError:
            from .alt import hl_funding   # Binance futures is geo-blocked from US runners
            return hl_funding(symbol, client=self.client)
        s = pd.Series({pd.to_datetime(d["fundingTime"], unit="ms", utc=True): float(d["fundingRate"])
                       for d in data}, name="funding")
        return s.sort_index()

    def fetch_open_interest(self, symbol: str, interval: str = "1h", limit: int = 500) -> pd.Series | None:
        try:
            data = self._get(f"{FUTURES}/futures/data/openInterestHist",
                             {"symbol": symbol, "period": interval, "limit": min(limit, 500)})
        except httpx.HTTPError:
            return None
        s = pd.Series({pd.to_datetime(d["timestamp"], unit="ms", utc=True): float(d["sumOpenInterestValue"])
                       for d in data}, name="open_interest")
        return s.sort_index()

    def fetch_orderbook(self, symbol: str, depth: int = 100) -> OrderBookSnapshot | None:
        try:
            d = self._get(f"{SPOT}/api/v3/depth", {"symbol": symbol, "limit": depth})
        except httpx.HTTPError:
            return None
        return OrderBookSnapshot(
            ts=datetime.now(timezone.utc),
            bids=[(float(p), float(q)) for p, q in d["bids"]],
            asks=[(float(p), float(q)) for p, q in d["asks"]],
        )
