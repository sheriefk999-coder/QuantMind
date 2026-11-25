"""Autonomous Researcher + Unconventional Thinking Mode.

Generates hypotheses FROM MEASUREMENTS (never invented), turns the promising
ones into experiments for the lab, and keeps a prioritized backlog:
    priority = expected_value x feasibility / validation_cost

Questions it asks itself every cycle, each answered with a computation:
  * Can two weak signals combine into a strong one?   -> pairwise feature-interaction IC scan
  * Is there an opportunity between markets?          -> lead-lag cross-correlation scan
  * Is this regime different from history?            -> novelty score
  * Is a deployed strategy decaying?                  -> live vs backtest Sharpe (decay monitor)
  * Am I seeing a real inefficiency or noise?         -> every idea goes through the lab's gates
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from ..analysis.features import FEATURE_SPACE
from ..core.store import Store, now_iso
from ..strategies.spec import Rule, StrategySpec

SCAN_FEATURES = [k for k in FEATURE_SPACE if k not in ("hour_sin",)]


def _ic(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    ok = x.notna() & y.notna()
    if ok.sum() < 300:
        return 0.0, 1.0
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p)


class Researcher:
    def __init__(self, store: Store):
        self.store = store
        self.data = store.load("research.json", {"backlog": [], "done": [], "insights": []})

    def _add(self, kind: str, title: str, detail: dict, value: float, feasibility: float, cost: float,
             spec: StrategySpec | None = None) -> None:
        key = hashlib.sha1((kind + title).encode()).hexdigest()[:10]
        if any(b["key"] == key for b in self.data["backlog"] + self.data["done"]):
            return
        self.data["backlog"].append({
            "key": key, "kind": kind, "title": title, "detail": detail, "created": now_iso(),
            "priority": round(value * feasibility / max(cost, 0.1), 4),
            "spec": spec.to_dict() if spec else None, "status": "queued"})
        self.data["backlog"].sort(key=lambda b: -b["priority"])

    # ---- weak + weak = strong?
    def interaction_scan(self, f: pd.DataFrame, close: pd.Series, symbol: str, horizon: int = 24,
                         train_frac: float = 0.6) -> int:
        n = int(len(f) * train_frac)                        # research uses the discovery segment only
        fwd = (close.shift(-horizon) / close - 1).iloc[:n]
        X = f.iloc[:n]
        singles = {c: _ic(X[c], fwd)[0] for c in SCAN_FEATURES if c in X}
        added = 0
        for a, b in itertools.combinations(singles, 2):
            za = (X[a] - X[a].mean()) / X[a].std()
            zb = (X[b] - X[b].mean()) / X[b].std()
            ic, p = _ic(za * zb, fwd)
            gain = abs(ic) - max(abs(singles[a]), abs(singles[b]))
            if p < 0.001 and gain > 0.02:
                sgn = ">" if ic > 0 else "<"
                spec = StrategySpec(f"H:{a}x{b}", "hybrid",
                                    long_entry=[Rule(a, sgn if singles[a] >= 0 else "<", float(X[a].median())),
                                                Rule(b, ">" if singles[b] >= 0 else "<", float(X[b].median()))],
                                    max_hold=horizon, stop_atr=3.0, origin="hypothesis")
                self._add("interaction", f"{symbol}: {a} x {b} predicts {horizon}-bar return",
                          {"ic": round(ic, 4), "p": p, "single_ics": [round(singles[a], 4), round(singles[b], 4)],
                           "symbol": symbol}, value=abs(ic) * 10, feasibility=0.9, cost=1.0, spec=spec)
                added += 1
        return added

    # ---- relationships between markets
    def lead_lag_scan(self, closes: dict[str, pd.Series], max_lag: int = 6) -> int:
        rets = {s: np.log(c).diff() for s, c in closes.items()}
        added = 0
        for a, b in itertools.permutations(rets, 2):
            ra, rb = rets[a], rets[b].reindex(rets[a].index)
            for lag in range(1, max_lag + 1):
                ic, p = _ic(ra.shift(lag), rb)
                if p < 0.001 and abs(ic) > 0.05:
                    self._add("lead_lag", f"{a} returns lead {b} by {lag} bars",
                              {"ic": round(ic, 4), "p": p}, value=abs(ic) * 8, feasibility=0.6, cost=2.0)
                    added += 1
        return added

    def correlation_breaks(self, closes: dict[str, pd.Series], window: int = 168) -> list[str]:
        out = []
        for a, b in itertools.combinations(closes, 2):
            ra, rb = np.log(closes[a]).diff(), np.log(closes[b]).diff().reindex(closes[a].index)
            roll = ra.rolling(window).corr(rb)
            if roll.notna().sum() < 500:
                continue
            now, mu, sd = roll.iloc[-1], roll.mean(), roll.std()
            if sd and abs(now - mu) / sd > 2.5:
                msg = f"correlation {a}/{b} = {now:.2f} vs normal {mu:.2f}: relationship breaking down"
                out.append(msg)
                self._add("correlation_break", msg, {"z": float((now - mu) / sd)}, 3, 0.7, 1.5)
        return out

    def note(self, text: str) -> None:
        self.data["insights"].append({"ts": now_iso(), "text": text})
        self.data["insights"] = self.data["insights"][-300:]

    def pop_experiments(self, n: int = 5, symbol: str | None = None) -> list[StrategySpec]:
        out = []
        for b in list(self.data["backlog"]):
            if len(out) >= n:
                break
            if b["spec"] and (symbol is None or b["detail"].get("symbol") == symbol):
                out.append(StrategySpec.from_dict(b["spec"]))
                b["status"] = "sent_to_lab"
                self.data["backlog"].remove(b)
                self.data["done"].append(b)
        self.data["done"] = self.data["done"][-1000:]
        return out

    def save(self) -> None:
        self.data["backlog"] = self.data["backlog"][:300]
        self.store.save("research.json", self.data)


def decay_check(live_returns: pd.Series, expected_sharpe_per_bar: float, min_bars: int = 200) -> dict:
    """Is the strategy's live performance consistent with its backtest?
    One-sided test of live mean against the backtest mean."""
    r = live_returns.dropna()
    if len(r) < min_bars or r.std() == 0:
        return {"status": "insufficient", "bars": len(r)}
    t = (r.mean() - expected_sharpe_per_bar * r.std()) / (r.std() / np.sqrt(len(r)))
    p = float(stats.norm.cdf(t))
    return {"status": "decayed" if p < 0.01 else "ok", "p": p, "bars": len(r),
            "live_sharpe_per_bar": float(r.mean() / r.std())}
