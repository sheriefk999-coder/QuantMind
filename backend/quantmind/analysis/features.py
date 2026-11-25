"""Feature engine: turns raw market data into a normalized, causal feature matrix.

The feature names here are the vocabulary the Strategy Discovery Lab composes
rules from, so adding a feature here automatically expands what the lab can
invent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

# name -> (description, typical range used for threshold sampling)
FEATURE_SPACE: dict[str, tuple[str, tuple[float, float]]] = {
    "rsi_14": ("RSI(14)", (10, 90)),
    "rsi_3": ("RSI(3), very short-term exhaustion", (5, 95)),
    "z_close_50": ("z-score of close vs 50-bar mean", (-3, 3)),
    "z_close_200": ("z-score of close vs 200-bar mean", (-3, 3)),
    "mom_24": ("24-bar log return / vol", (-3, 3)),
    "mom_168": ("168-bar log return / vol", (-3, 3)),
    "er_20": ("Kaufman efficiency ratio 20", (0, 1)),
    "adx_14": ("ADX(14) trend strength", (5, 60)),
    "pctb_20": ("Bollinger %B", (-0.5, 1.5)),
    "macdh_norm": ("MACD histogram / ATR", (-1, 1)),
    "don_20": ("position in prior 20-bar Donchian channel", (-0.3, 1.3)),
    "don_100": ("position in prior 100-bar Donchian channel", (-0.3, 1.3)),
    "vol_ratio": ("short vol / long vol", (0.3, 3)),
    "vol_pct_500": ("realized-vol percentile over 500 bars", (0, 1)),
    "volume_z": ("volume z-score (50)", (-2, 4)),
    "vwap_dev_24": ("deviation from 24-bar VWAP", (-0.05, 0.05)),
    "structure": ("swing structure (HH/HL = +1)", (-1, 1)),
    "hour_sin": ("time-of-day (sin)", (-1, 1)),
    "ret_1": ("last bar return / vol", (-3, 3)),
    # ---- multi-timeframe (only COMPLETED higher-timeframe bars are used)
    "h4_mom_42": ("4h: 42-bar (7-day) momentum / vol", (-3, 3)),
    "h4_rsi_14": ("4h: RSI(14)", (10, 90)),
    "h4_adx_14": ("4h: ADX(14)", (5, 60)),
    "d1_mom_20": ("1d: 20-day momentum / vol", (-3, 3)),
    "d1_z_20": ("1d: z-score vs 20-day mean", (-3, 3)),
    "d1_er_10": ("1d: 10-day efficiency ratio", (0, 1)),
}

# Feature pairs that are meaningfully comparable (same units) for
# feature-vs-feature rules, e.g. "1h momentum > 4h momentum".
COMPARABLE_GROUPS = [
    ["mom_24", "mom_168", "h4_mom_42", "d1_mom_20"],
    ["rsi_14", "rsi_3", "h4_rsi_14"],
    ["z_close_50", "z_close_200", "d1_z_20"],
    ["er_20", "d1_er_10"],
    ["adx_14", "h4_adx_14"],
    ["don_20", "don_100"],
]

_HTF = {"4h": "h4", "1D": "d1"}


def higher_timeframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Causal higher-timeframe features aligned to the base index.

    HTF bars are labelled by their open time; shifting by one HTF bar means a
    value only becomes visible once that HTF bar has fully closed (conservative:
    it appears on the first base bar that OPENS after the HTF close)."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 50:
        return pd.DataFrame(index=df.index)
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    c = agg["close"]
    vol = ind.log_returns(c).rolling(30, min_periods=20).std()
    out = pd.DataFrame(index=agg.index)
    if rule == "4h":
        out["h4_mom_42"] = np.log(c / c.shift(42)) / (vol * np.sqrt(42))
        out["h4_rsi_14"] = ind.rsi(c, 14)
        out["h4_adx_14"] = ind.adx(agg, 14)
    else:
        out["d1_mom_20"] = np.log(c / c.shift(20)) / (vol * np.sqrt(20))
        out["d1_z_20"] = ind.zscore(c, 20)
        out["d1_er_10"] = ind.efficiency_ratio(c, 10)
    out = out.shift(1)                                   # only completed HTF bars
    out.index = out.index + pd.tseries.frequencies.to_offset(rule)  # visible from next HTF open
    return out.reindex(df.index, method="ffill")


def build_features(df: pd.DataFrame, extra: dict[str, pd.Series] | None = None,
                   with_hurst: bool = False) -> pd.DataFrame:
    c = df["close"]
    lr = ind.log_returns(c)
    vol = lr.rolling(48, min_periods=48).std()
    a = ind.atr(df, 14)
    f = pd.DataFrame(index=df.index)
    f["rsi_14"] = ind.rsi(c, 14)
    f["rsi_3"] = ind.rsi(c, 3)
    f["z_close_50"] = ind.zscore(c, 50)
    f["z_close_200"] = ind.zscore(c, 200)
    f["mom_24"] = np.log(c / c.shift(24)) / (vol * np.sqrt(24))
    f["mom_168"] = np.log(c / c.shift(168)) / (vol * np.sqrt(168))
    f["er_20"] = ind.efficiency_ratio(c, 20)
    f["adx_14"] = ind.adx(df, 14)
    f["pctb_20"] = ind.bollinger_pctb(c, 20)
    f["macdh_norm"] = ind.macd_hist(c) / a
    f["don_20"] = ind.donchian_position(df, 20)
    f["don_100"] = ind.donchian_position(df, 100)
    rv_s, rv_l = ind.realized_vol(c, 12), ind.realized_vol(c, 96)
    f["vol_ratio"] = rv_s / rv_l
    f["vol_pct_500"] = rv_l.rolling(500, min_periods=100).rank(pct=True)
    f["volume_z"] = ind.zscore(np.log1p(df["volume"]), 50)
    f["vwap_dev_24"] = ind.rolling_vwap_dev(df, 24)
    f["structure"] = ind.swing_structure(df, 5)
    hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else np.zeros(len(df))
    f["hour_sin"] = np.sin(2 * np.pi * np.asarray(hours) / 24)
    f["ret_1"] = lr / vol
    f["atr"] = a
    f["vol_48"] = vol
    for rule in ("4h", "1D"):
        f = f.join(higher_timeframe(df, rule))
    from ..regime.classifier import label_codes
    f["regime_id"] = label_codes(f)
    if with_hurst:
        f["hurst_128"] = ind.hurst_rs(c, 128)
    for name, s in (extra or {}).items():
        # External series (funding, OI) arrive at their own timestamps; align by
        # as-of (last known value), never forward-looking.
        f[name] = s.reindex(df.index, method="ffill")
    return f.replace([np.inf, -np.inf], np.nan)
