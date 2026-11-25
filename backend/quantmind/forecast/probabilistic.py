"""Predictive intelligence with honest probabilities.

* Scenarios come from the HISTORICAL distribution of forward returns observed
  in the same regime (empirical frequencies, labelled as such).
* A model probability (logistic regression, walk-forward) is published ONLY if
  its out-of-sample calibration error is small and it beats the base rate on
  Brier score. Otherwise the output says "uncalibrated" instead of a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..regime.classifier import RegimeEngine

MODEL_FEATURES = ["mom_24", "mom_168", "z_close_50", "rsi_14", "er_20", "vol_ratio", "volume_z", "don_20", "structure"]


def regime_scenarios(df: pd.DataFrame, f: pd.DataFrame, regime_label: str, horizon: int = 24) -> dict:
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    labels = RegimeEngine().label_series(f.iloc[:-horizon])
    sample = fwd.iloc[:-horizon][labels == regime_label].dropna()
    if len(sample) < 100:
        sample, basis = fwd.dropna(), "all history (too few bars in this regime)"
    else:
        basis = f"{len(sample)} past bars in regime '{regime_label}'"
    vol = float(sample.std())
    q = lambda p: float(np.percentile(sample, p))  # noqa: E731
    return {
        "horizon_bars": horizon, "basis": basis, "kind": "historical frequency, not a guarantee",
        "bullish": {"move": q(80), "freq": float((sample > 0.5 * vol).mean())},
        "neutral": {"range": [q(35), q(65)], "freq": float((sample.abs() <= 0.5 * vol).mean())},
        "bearish": {"move": q(20), "freq": float((sample < -0.5 * vol).mean())},
        "unexpected_event": {"down_tail": q(1), "up_tail": q(99), "freq": 0.02},
    }


def calibrated_up_probability(f: pd.DataFrame, close: pd.Series, horizon: int = 24,
                              folds: int = 5, max_ece: float = 0.03) -> dict:
    y = (close.shift(-horizon) > close).astype(float)
    X = f[MODEL_FEATURES]
    ok = X.notna().all(axis=1) & close.shift(-horizon).notna()
    Xv, yv = X[ok].to_numpy(), y[ok].to_numpy()
    n = len(yv)
    if n < 2000:
        return {"status": "insufficient_data"}
    preds, truth = [], []
    step = n // (folds + 1)
    for k in range(1, folds + 1):
        tr_end, te_end = k * step, min(n, (k + 1) * step)
        tr = slice(0, tr_end - horizon)                       # purge overlapping labels
        m = LogisticRegression(C=0.1, max_iter=500).fit(Xv[tr], yv[tr])
        preds.append(m.predict_proba(Xv[tr_end:te_end])[:, 1])
        truth.append(yv[tr_end:te_end])
    p, t = np.concatenate(preds), np.concatenate(truth)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - t[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    brier, base = float(((p - t) ** 2).mean()), float(((t.mean() - t) ** 2).mean())
    final = LogisticRegression(C=0.1, max_iter=500).fit(Xv[:-horizon], yv[:-horizon])
    x_now = X.iloc[[-1]]
    prob = None if x_now.isna().any(axis=None) else float(final.predict_proba(x_now.to_numpy())[0, 1])
    usable = ece <= max_ece and brier < base
    return {"status": "calibrated" if usable else "uncalibrated",
            "p_up": round(prob, 3) if usable and prob is not None else None,
            "oos_ece": round(float(ece), 4), "oos_brier": round(brier, 4), "base_brier": round(base, 4),
            "horizon_bars": horizon,
            "note": None if usable else "model is not reliably better than a coin flip out-of-sample; no probability shown"}
