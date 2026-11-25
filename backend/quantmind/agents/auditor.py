"""Performance Auditor: scores every agent's past calls against what actually
happened, so trust is earned from out-of-sample evidence, not assumed."""
from __future__ import annotations

from ..core.store import Store


class PerformanceAuditor:
    def __init__(self, store: Store, horizon_bars: int = 24):
        self.store = store
        self.h = horizon_bars
        self.data = store.load("agents.json", {"pending": [], "stats": {}})

    def record(self, symbol: str, bar_ts: str, price: float, opinions: list[dict]) -> None:
        for o in opinions:
            if not o["abstain"] and abs(o["score"]) >= 0.1:
                self.data["pending"].append({"agent": o["agent"], "symbol": symbol, "ts": bar_ts,
                                             "price": price, "score": o["score"],
                                             "conf": o.get("adjusted_confidence") or o["confidence"]})

    def resolve(self, symbol: str, closes) -> int:
        """closes: pd.Series of closes indexed by timestamp. Resolves calls older than horizon."""
        import pandas as pd

        keep, done = [], 0
        idx = closes.index
        for p in self.data["pending"]:
            if p["symbol"] != symbol:
                keep.append(p)
                continue
            t = pd.Timestamp(p["ts"])
            pos = idx.searchsorted(t)
            if pos + self.h >= len(idx):
                keep.append(p)
                continue
            ret = closes.iloc[pos + self.h] / p["price"] - 1
            hit = (ret > 0) == (p["score"] > 0)
            st = self.data["stats"].setdefault(p["agent"], {"n": 0, "hits": 0, "brier_sum": 0.0})
            st["n"] += 1
            st["hits"] += int(hit)
            prob = 0.5 + 0.5 * abs(p["score"]) * p["conf"]      # implied probability of being right
            st["brier_sum"] += (prob - float(hit)) ** 2
            done += 1
        self.data["pending"] = keep[-20000:]
        return done

    def reliabilities(self) -> dict[str, float]:
        # Beta(10,10) prior: an agent needs a real track record to move far from 0.5
        return {a: (s["hits"] + 10) / (s["n"] + 20) for a, s in self.data["stats"].items()}

    def report(self) -> list[dict]:
        rel = self.reliabilities()
        return [{"agent": a, "calls": s["n"], "hit_rate": s["hits"] / s["n"] if s["n"] else None,
                 "brier": s["brier_sum"] / s["n"] if s["n"] else None, "reliability": round(rel[a], 3)}
                for a, s in self.data["stats"].items()]

    def save(self) -> None:
        self.store.save("agents.json", self.data)
