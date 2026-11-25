"""Strategy ensemble: combines every validated strategy for a symbol into one
target position.

weight_i  ∝  regime_fit_i  x  shrunk holdout Sharpe_i  /  recent volatility_i
          /  sqrt(redundancy_i)     (strategies that trade the same thing
                                     don't get counted twice)
target    =  Σ weight_i x position_i        in [-1, 1]

A strategy that has gone flat or is out of its regime contributes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..strategies.spec import spec_from_dict


@dataclass
class EnsembleView:
    target: float
    agreement: float                   # share of weight on the target's side
    components: list[dict] = field(default_factory=list)
    stop: float | None = None
    recent_sharpe: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def ensemble(entries: list[dict], f: pd.DataFrame, ohlcv: pd.DataFrame, family_fit: dict[str, float],
             lookback: int = 2000, spot_only: bool = False) -> EnsembleView | None:
    if not entries:
        return None
    comps, rets = [], {}
    for e in entries:
        spec = spec_from_dict(e["definition"])
        pos, stops = spec.simulate(f, ohlcv)
        r = run_backtest(ohlcv.tail(lookback), pos.tail(lookback), stops=stops.tail(lookback)).returns
        hs = (e["results"].get("holdout") or {}).get("sharpe") or 0.0
        n_h = (e["results"].get("holdout") or {}).get("n_bars") or 1000
        shrunk = max(0.0, hs) * n_h / (n_h + 2000)              # shrink short track records toward 0
        vol = float(r.std()) or 1e-4
        p_now = float(pos.iloc[-1])
        if spot_only:
            p_now = max(0.0, p_now)
        comps.append({"id": e["id"], "name": e["name"], "family": e["family"], "position": p_now,
                      "raw_weight": family_fit.get(e["family"], 0.5) * shrunk / vol,
                      "stop": None if np.isnan(stops.iloc[-1]) else float(stops.iloc[-1])})
        rets[e["id"]] = r
    R = pd.DataFrame(rets)
    corr = R.corr().fillna(0).clip(lower=0) if len(R.columns) > 1 else pd.DataFrame([[1.0]], R.columns, R.columns)
    for c in comps:
        redundancy = float(corr.loc[c["id"]].sum()) if c["id"] in corr else 1.0
        c["raw_weight"] /= np.sqrt(max(redundancy, 1.0))
    tot = sum(c["raw_weight"] for c in comps)
    if tot <= 0:
        return EnsembleView(0.0, 0.0, comps)
    for c in comps:
        c["weight"] = round(c["raw_weight"] / tot, 4)
    target = float(np.clip(sum(c["weight"] * c["position"] for c in comps), -1, 1))
    side = np.sign(target)
    agree = sum(c["weight"] for c in comps if side != 0 and np.sign(c["position"]) == side)
    lead = max((c for c in comps if side != 0 and np.sign(c["position"]) == side and c["stop"]),
               key=lambda c: c["weight"], default=None)
    w = pd.Series({c["id"]: c["weight"] for c in comps})
    ens_r = (R * w.reindex(R.columns).fillna(0)).sum(axis=1)
    sr = float(ens_r.mean() / ens_r.std() * np.sqrt(8760)) if ens_r.std() > 0 else None
    return EnsembleView(round(target, 4), round(float(agree), 3), comps, lead["stop"] if lead else None,
                        round(sr, 2) if sr is not None else None)
