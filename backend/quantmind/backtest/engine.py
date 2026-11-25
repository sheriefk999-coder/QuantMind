"""Backtesting engine.

Timing contract (the most common source of fake backtest profits):
  * a strategy decides at the CLOSE of bar t using data <= t
  * the position is held over the return from close t to close t+1
  * the fill price includes fees + slippage, where slippage grows with
    volatility and with order size relative to bar volume
Implemented by shifting positions one bar before multiplying by returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.config import settings
from .metrics import compute_metrics

BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "1h": 8_760, "4h": 2_190, "1d": 365}


@dataclass
class CostModel:
    fee_bps: float = settings.fee_bps
    slippage_bps: float = settings.slippage_bps
    vol_slippage_mult: float = 0.1      # extra slippage = mult * bar volatility
    funding_per_bar: pd.Series | None = None  # perp funding charged to longs, paid to shorts
    notional: float = 10_000.0          # used for market-impact estimate
    impact_coef: float = 0.1            # bps per 1% of bar dollar volume

    def per_side_cost(self, df: pd.DataFrame) -> pd.Series:
        lr = np.log(df["close"]).diff().abs()
        vol_part = self.vol_slippage_mult * lr.rolling(24, min_periods=1).mean() * 1e4
        dollar_vol = (df["close"] * df["volume"]).replace(0, np.nan)
        impact = self.impact_coef * (self.notional / dollar_vol * 100)
        return ((self.fee_bps + self.slippage_bps + vol_part.fillna(0) + impact.fillna(0)) / 1e4).clip(upper=0.05)


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict
    meta: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, target: pd.Series, cost: CostModel | None = None,
                 interval: str = "1h", stops: pd.Series | None = None,
                 cost_multiplier: float = 1.0) -> BacktestResult:
    """stops: stop level decided at close t-1 for the position carried through bar t.
    If bar t trades through it, the bar's return is computed at the stop price
    (or at the open if the market gapped past it), plus extra stop slippage."""
    cost = cost or CostModel()
    target = target.reindex(df.index).fillna(0.0).clip(-1, 1)
    held = target.shift(1).fillna(0.0)                    # <- the anti-look-ahead shift
    mkt = df["close"].pct_change().fillna(0.0)
    if stops is not None:
        st = stops.reindex(df.index).shift(1)
        prev_c = df["close"].shift(1)
        hit_long = (held > 0) & (df["low"] <= st)
        hit_short = (held < 0) & (df["high"] >= st)
        fill_long = np.minimum(df["open"], st)            # gap below the stop fills at the open
        fill_short = np.maximum(df["open"], st)
        mkt = mkt.where(~hit_long, fill_long / prev_c - 1).where(~hit_short, fill_short / prev_c - 1)
        stop_slip = (hit_long | hit_short).astype(float) * cost.slippage_bps * 2 / 1e4
    else:
        stop_slip = 0.0
    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * cost.per_side_cost(df) * cost_multiplier + stop_slip * cost_multiplier
    funding = 0.0
    if cost.funding_per_bar is not None:
        funding = held * cost.funding_per_bar.reindex(df.index).fillna(0.0)
    ret = held * mkt - costs - funding
    equity = (1 + ret).cumprod()
    trades = extract_trades(held, df["close"], ret)
    m = compute_metrics(ret, trades, BARS_PER_YEAR.get(interval, 8760))
    m["exposure"] = float((held != 0).mean())
    m["turnover_per_year"] = float(turnover.sum() / max(len(df), 1) * BARS_PER_YEAR.get(interval, 8760))
    m["benchmark_return"] = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    return BacktestResult(ret, equity, held, trades, m,
                          meta={"source": df.attrs.get("source", "market"), "bars": len(df)})


def extract_trades(held: pd.Series, close: pd.Series, ret: pd.Series) -> pd.DataFrame:
    rows = []
    h = held.to_numpy()
    r = ret.to_numpy()
    idx = held.index
    start, side, acc = None, 0.0, 0.0
    for t in range(len(h)):
        if side != 0 and h[t] == side:
            acc = (1 + acc) * (1 + r[t]) - 1
        if h[t] != side:
            if side != 0:
                if h[t] == 0:  # bar t carries only the exit cost; book it to this trade
                    acc = (1 + acc) * (1 + r[t]) - 1
                rows.append({"entry": idx[start], "exit": idx[t], "side": int(side), "bars": t - start,
                             "return": acc})
            side, start, acc = h[t], t, (r[t] if h[t] != 0 else 0.0)
    if side != 0:
        rows.append({"entry": idx[start], "exit": idx[-1], "side": int(side), "bars": len(h) - start,
                     "return": acc, "open": True})
    return pd.DataFrame(rows)
