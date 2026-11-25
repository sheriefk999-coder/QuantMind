"""SYNTHETIC data for tests and offline demos ONLY.

Anything produced here is tagged source="synthetic" and the API refuses to
report its results as market performance. Two generators:
  - random_walk: a pure null (no edge exists). The lab must find nothing here.
  - regime_switching: trends/ranges/vol bursts, for testing regime detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(close: np.ndarray, rng: np.random.Generator, start: str, freq: str) -> pd.DataFrame:
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(10, 0.4, n) * (1 + 20 * np.abs(np.diff(np.log(np.r_[close[0], close]))))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    df.attrs["source"] = "synthetic"
    return df


def random_walk(n: int = 3000, sigma: float = 0.01, seed: int = 0,
                start: str = "2022-01-01", freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    return _frame(close, rng, start, freq)


def regime_switching(n: int = 4000, seed: int = 0, start: str = "2022-01-01",
                     freq: str = "1h") -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (ohlcv, true_regime_labels). Labels: 0 range, 1 uptrend, 2 downtrend, 3 high-vol."""
    rng = np.random.default_rng(seed)
    params = {0: (0.0, 0.006), 1: (0.0015, 0.008), 2: (-0.0015, 0.008), 3: (0.0, 0.025)}
    labels = np.empty(n, dtype=int)
    rets = np.empty(n)
    state, i = 0, 0
    while i < n:
        length = int(rng.integers(150, 500))
        mu, sd = params[state]
        seg = slice(i, min(n, i + length))
        m = seg.stop - seg.start
        if state == 0:  # mean-reverting range
            x = np.zeros(m)
            for k in range(1, m):
                x[k] = 0.95 * x[k - 1] + rng.normal(0, sd)
            rets[seg] = np.diff(np.r_[0.0, x])
        else:
            rets[seg] = rng.normal(mu, sd, m)
        labels[seg] = state
        i += length
        state = int(rng.choice([s for s in params if s != state]))
    close = 100 * np.exp(np.cumsum(rets))
    return _frame(close, rng, start, freq), labels
