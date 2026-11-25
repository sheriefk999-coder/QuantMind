"""Statistical safeguards against overfitting and data mining.

* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012)
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): corrects the best
  Sharpe found for the NUMBER OF TRIALS run to find it. The lab passes the
  true count of every strategy it evaluated - that is what makes this honest.
* Permutation test: re-run the strategy on price paths whose returns were
  shuffled (destroying any temporal structure). If it does as well there, the
  "edge" is noise.
* Causality check: recompute signals on truncated data; any difference means
  the strategy peeked at the future.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """sr is the NON-annualized (per-bar) Sharpe."""
    if n < 3:
        return 0.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of n_trials Sharpe estimates drawn under the null (true SR = 0)."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_variance: float | None = None) -> float:
    """Probability that the true per-bar Sharpe > 0 after accounting for selection
    among n_trials. trial_sr_variance = variance of per-bar Sharpes across trials;
    if unknown, use the null variance 1/T."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    var = trial_sr_variance if trial_sr_variance and trial_sr_variance > 0 else 1.0 / n
    sr0 = expected_max_sharpe(n_trials, var)
    return probabilistic_sharpe(sr, n, stats.skew(r), stats.kurtosis(r, fisher=False), sr0)


def permute_prices(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle bar-level log returns (keeping each bar's internal OHLC shape and
    volume together) and rebuild a price path. Destroys serial structure,
    preserves the return distribution."""
    c = df["close"].to_numpy()
    lr = np.diff(np.log(c))
    perm = rng.permutation(len(lr))
    new_c = c[0] * np.exp(np.r_[0.0, np.cumsum(lr[perm])])
    shape = df[["open", "high", "low"]].to_numpy() / c[:, None]
    idx = np.r_[0, perm + 1]
    out = pd.DataFrame(shape[idx] * new_c[:, None], columns=["open", "high", "low"], index=df.index)
    out["close"] = new_c
    out["high"] = out[["high", "close", "open"]].max(axis=1)
    out["low"] = out[["low", "close", "open"]].min(axis=1)
    out["volume"] = df["volume"].to_numpy()[idx]
    return out[["open", "high", "low", "close", "volume"]]


def permutation_pvalue(df: pd.DataFrame, score_fn: Callable[[pd.DataFrame], float],
                       n_perm: int = 50, seed: int = 0) -> tuple[float, float, list[float]]:
    """p = share of shuffled-market runs scoring >= the real run (with +1 smoothing)."""
    rng = np.random.default_rng(seed)
    real = score_fn(df)
    null = [score_fn(permute_prices(df, rng)) for _ in range(n_perm)]
    p = (1 + sum(s >= real for s in null)) / (n_perm + 1)
    return float(p), float(real), null


def causality_violations(signal_fn: Callable[[pd.DataFrame], pd.Series], df: pd.DataFrame,
                         n_checks: int = 8, seed: int = 0, warmup: int = 300) -> list[str]:
    """Signals computed on df[:t] must equal signals computed on full df, at t."""
    rng = np.random.default_rng(seed)
    full = signal_fn(df)
    errors = []
    if len(df) <= warmup + 2:
        return errors
    for t in rng.integers(warmup, len(df) - 1, n_checks):
        part = signal_fn(df.iloc[: t + 1])
        a, b = full.iloc[t], part.iloc[-1]
        if not (pd.isna(a) and pd.isna(b)) and not np.isclose(a, b, equal_nan=True):
            errors.append(f"bar {df.index[t]}: full={a} truncated={b}")
    return errors


def timing_pvalue(positions: pd.Series, df: pd.DataFrame, score_fn: Callable[[pd.Series], float],
                  n_perm: int = 100, seed: int = 0, min_shift: int = 200) -> tuple[float, float]:
    """Does the strategy's TIMING carry information? Circularly shift its position
    series against the market (keeps the exact trade pattern, destroys its
    alignment with prices). Cheap enough for ML strategies, where re-training on
    permuted prices would be too expensive."""
    rng = np.random.default_rng(seed)
    real = score_fn(positions)
    n = len(positions)
    if n < 2 * min_shift:
        return 1.0, real
    null = [score_fn(pd.Series(np.roll(positions.to_numpy(), int(rng.integers(min_shift, n - min_shift))),
                               index=positions.index)) for _ in range(n_perm)]
    return float((1 + sum(s >= real for s in null)) / (n_perm + 1)), float(real)
