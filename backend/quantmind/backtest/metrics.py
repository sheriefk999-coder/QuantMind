from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe(ret: pd.Series, bars_per_year: int) -> float:
    sd = ret.std()
    return 0.0 if not sd or np.isnan(sd) else float(ret.mean() / sd * np.sqrt(bars_per_year))


def max_drawdown(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def compute_metrics(ret: pd.Series, trades: pd.DataFrame, bars_per_year: int) -> dict:
    n = len(ret)
    total = float((1 + ret).prod() - 1)
    years = n / bars_per_year
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0
    downside = ret[ret < 0].std()
    mdd = max_drawdown(ret)
    tr = trades["return"] if len(trades) else pd.Series(dtype=float)
    gains, losses = tr[tr > 0].sum(), -tr[tr < 0].sum()
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe(ret, bars_per_year),
        "sortino": float(ret.mean() / downside * np.sqrt(bars_per_year)) if downside and downside > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "n_trades": int(len(trades)),
        "win_rate": float((tr > 0).mean()) if len(tr) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "avg_trade": float(tr.mean()) if len(tr) else 0.0,
        "skew": float(stats.skew(ret)) if n > 3 else 0.0,
        "kurtosis": float(stats.kurtosis(ret, fisher=False)) if n > 3 else 3.0,
        "n_bars": n,
    }
