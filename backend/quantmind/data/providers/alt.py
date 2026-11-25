"""Alternative free data sources that are reachable from GitHub Actions (US IPs),
where Binance futures endpoints are geo-blocked.

* Crypto Fear & Greed Index (alternative.me) - daily market sentiment, 0..100
* Hyperliquid public API - perpetual funding history + current open interest
  (no key). Hyperliquid funds hourly; values are converted to an 8h-equivalent
  rate so they are comparable with Binance funding.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

HL = "https://api.hyperliquid.xyz/info"


def fear_greed(limit: int = 400, client: httpx.Client | None = None) -> pd.Series | None:
    c = client or httpx.Client(timeout=15)
    try:
        r = c.get("https://api.alternative.me/fng/", params={"limit": limit, "format": "json"})
        r.raise_for_status()
        data = r.json()["data"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    s = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s", utc=True): float(d["value"]) for d in data},
                  name="fear_greed").sort_index()
    # the daily value is published for the whole day; make it visible from the
    # END of that day so intraday bars never see a value before it existed
    s.index = s.index + pd.Timedelta(days=1)
    return s


def _coin(symbol: str) -> str:
    return symbol.replace("USDT", "").replace("USDC", "")


def hl_funding(symbol: str, days: int = 30, client: httpx.Client | None = None) -> pd.Series | None:
    c = client or httpx.Client(timeout=15)
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    try:
        r = c.post(HL, json={"type": "fundingHistory", "coin": _coin(symbol), "startTime": start})
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not data:
        return None
    s = pd.Series({pd.to_datetime(d["time"], unit="ms", utc=True): float(d["fundingRate"]) * 8 for d in data},
                  name="funding")
    return s.sort_index()


def hl_open_interest_now(client: httpx.Client | None = None) -> dict[str, float]:
    """Current OI (in USD) for every Hyperliquid perp. History is built up by
    storing one snapshot per cycle (see Orchestrator)."""
    c = client or httpx.Client(timeout=15)
    try:
        r = c.post(HL, json={"type": "metaAndAssetCtxs"})
        r.raise_for_status()
        meta, ctxs = r.json()
    except (httpx.HTTPError, ValueError):
        return {}
    out = {}
    for u, ctx in zip(meta["universe"], ctxs):
        try:
            out[u["name"]] = float(ctx["openInterest"]) * float(ctx["markPx"])
        except (KeyError, TypeError, ValueError):
            pass
    return out
