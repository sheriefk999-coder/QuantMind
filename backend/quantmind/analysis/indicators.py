"""QuantMind's own indicator library. Every function is CAUSAL: the value at bar t
uses only data up to and including bar t. tests/test_causality.py enforces this."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    m, sd = sma(close, n), close.rolling(n, min_periods=n).std()
    return (close - (m - k * sd)) / (2 * k * sd)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, sig)


def zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n, min_periods=n).mean()) / s.rolling(n, min_periods=n).std()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(close: pd.Series, n: int = 24) -> pd.Series:
    return log_returns(close).rolling(n, min_periods=n).std()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / tr
    mdi = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman: |net move| / path length. ~1 = clean trend, ~0 = noise."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0, np.nan)


def donchian_position(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Where close sits inside the PRIOR n-bar channel. >1 = breakout up, <0 = down."""
    hi = df["high"].shift(1).rolling(n, min_periods=n).max()
    lo = df["low"].shift(1).rolling(n, min_periods=n).min()
    return (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def rolling_vwap_dev(df: pd.DataFrame, n: int = 24) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()
    return df["close"] / vwap - 1


def hurst_rs(close: pd.Series, n: int = 128) -> pd.Series:
    """Rolling Hurst exponent via variance-of-lagged-differences.
    H>0.5 trending/persistent, H<0.5 mean-reverting, ~0.5 random walk."""
    lp = np.log(close.values)
    lags = np.array([2, 4, 8, 16])
    out = np.full(len(lp), np.nan)
    for t in range(n, len(lp)):
        w = lp[t - n + 1: t + 1]
        tau = [np.std(w[lag:] - w[:-lag]) for lag in lags]
        if min(tau) > 0:
            out[t] = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return pd.Series(out, index=close.index)


def swing_structure(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """Price-action structure score in [-1,1]: higher-highs & higher-lows -> +1.
    A swing high at bar i is only CONFIRMED at bar i+k, so we shift by k."""
    h, l = df["high"], df["low"]
    w = 2 * k + 1
    sh = (h == h.rolling(w, center=True).max()).shift(k).fillna(False).astype(bool)
    sl = (l == l.rolling(w, center=True).min()).shift(k).fillna(False).astype(bool)
    hp = h.shift(k).where(sh).dropna()
    lp = l.shift(k).where(sl).dropna()
    last_hi, prev_hi = hp.reindex(df.index).ffill(), hp.shift(1).reindex(df.index).ffill()
    last_lo, prev_lo = lp.reindex(df.index).ffill(), lp.shift(1).reindex(df.index).ffill()
    score = np.sign(last_hi - prev_hi).fillna(0) * 0.5 + np.sign(last_lo - prev_lo).fillna(0) * 0.5
    return score
