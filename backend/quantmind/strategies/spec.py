"""Strategy representation: a small, serializable rule language (v2).

A strategy is data, not code. That lets the lab mutate, combine and version
strategies, store their exact mathematical definition, and re-run any version
later bit-for-bit.

    Rule      := feature OP (threshold | other_feature)   OP in { >, <, xa (cross above), xb (cross below) }
    Entry     := AND(rules)  AND  regime(t) in allowed_regimes        evaluated at the close of bar t
    Exit      := OR(exit rules) | bars_held >= max_hold | stop touched intrabar
    Stop      := entry -/+ stop_atr * ATR14, optionally trailing at close
    Position  := +1 / -1 / 0, held from bar t+1 (enforced by the backtester)

Two strategy kinds share one interface (`simulate(f, ohlcv)`):
    StrategySpec  - rule-based (this file)
    ModelSpec     - machine-learning, walk-forward trained (strategies/model.py)
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .kernel import simulate


class Status(str, Enum):
    CANDIDATE = "candidate"       # generated, untested
    BACKTESTED = "backtested"     # passed in-sample screening
    VALIDATED = "validated"       # passed every out-of-sample + robustness gate
    PAPER = "paper"               # trading on paper
    LIVE = "live"                 # human-authorized live
    INACTIVE = "inactive"         # decayed / disabled by monitor
    REJECTED = "rejected"         # failed a gate (kept, never hidden)
    ARCHIVED = "archived"


@dataclass
class Rule:
    feature: str
    op: str
    threshold: float = 0.0
    ref: str | None = None        # compare against another feature instead of a constant

    def evaluate(self, f: pd.DataFrame) -> np.ndarray:
        x = f[self.feature].to_numpy(dtype=float)
        y = f[self.ref].to_numpy(dtype=float) if self.ref else np.full(len(x), self.threshold)
        px, py = np.r_[np.nan, x[:-1]], np.r_[np.nan, y[:-1]]
        with np.errstate(invalid="ignore"):
            if self.op == ">":
                r = x > y
            elif self.op == "<":
                r = x < y
            elif self.op == "xa":
                r = (x > y) & (px <= py)
            elif self.op == "xb":
                r = (x < y) & (px >= py)
            else:
                raise ValueError(self.op)
        return r & ~np.isnan(x) & ~np.isnan(y)

    def text(self) -> str:
        sym = {">": ">", "<": "<", "xa": "crosses above", "xb": "crosses below"}[self.op]
        return f"{self.feature} {sym} {self.ref if self.ref else format(self.threshold, '.4g')}"


@dataclass
class StrategySpec:
    name: str
    family: str
    long_entry: list[Rule] = field(default_factory=list)
    short_entry: list[Rule] = field(default_factory=list)
    exit_rules: list[Rule] = field(default_factory=list)
    max_hold: int = 48
    stop_atr: float = 3.0
    trail: bool = False
    regimes: list[str] | None = None   # None = trade in every regime
    version: int = 1
    parent_ids: list[str] = field(default_factory=list)
    origin: str = "handcrafted"        # handcrafted | genetic | bayesian | hypothesis
    kind: str = "rules"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ---------- identity ----------
    def fingerprint(self) -> str:
        core = {k: v for k, v in self.to_dict().items()
                if k not in ("id", "name", "version", "parent_ids", "origin", "family")}
        return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def complexity(self) -> int:
        return len(self.long_entry) + len(self.short_entry) + len(self.exit_rules) + (1 if self.regimes else 0)

    def required_features(self) -> set[str]:
        rs = self.long_entry + self.short_entry + self.exit_rules
        return {r.feature for r in rs} | {r.ref for r in rs if r.ref}

    def math_definition(self) -> str:
        def conj(rs):
            return " AND ".join(r.text() for r in rs) or "never"
        reg = f"  AND regime in {{{', '.join(self.regimes)}}}" if self.regimes else ""
        return "\n".join([
            f"LONG  at close t  if  {conj(self.long_entry)}{reg if self.long_entry else ''}",
            f"SHORT at close t  if  {conj(self.short_entry)}{reg if self.short_entry else ''}",
            "EXIT  if  " + (" OR ".join(r.text() for r in self.exit_rules) or "no rule")
            + f"  OR bars_held >= {self.max_hold}"
            + f"  OR price touches {'trailing ' if self.trail else ''}stop {self.stop_atr:.2f}*ATR14 (intrabar)",
            "position applied from bar t+1; costs charged on every change in position",
        ])

    # ---------- signal generation ----------
    def _conj(self, rules: list[Rule], f: pd.DataFrame) -> np.ndarray:
        if not rules:
            return np.zeros(len(f), dtype=bool)
        m = np.ones(len(f), dtype=bool)
        for r in rules:
            m &= r.evaluate(f)
        return m

    def simulate(self, f: pd.DataFrame, data: pd.DataFrame | pd.Series) -> tuple[pd.Series, pd.Series]:
        """Returns (target position series, stop level carried into the next bar)."""
        if isinstance(data, pd.Series):     # close only: no intrabar information
            close = high = low = data.to_numpy(dtype=float)
        else:
            close, high, low = (data[c].to_numpy(dtype=float) for c in ("close", "high", "low"))
        ex = np.zeros(len(f), dtype=bool)
        for r in self.exit_rules:
            ex |= r.evaluate(f)
        if self.regimes and "regime_id" in f:
            from ..regime.classifier import REGIME_LIST
            codes = [REGIME_LIST.index(r) for r in self.regimes if r in REGIME_LIST]
            allowed = np.isin(f["regime_id"].to_numpy(), codes)
        else:
            allowed = np.ones(len(f), dtype=bool)
        pos, stop_next, _ = simulate(self._conj(self.long_entry, f), self._conj(self.short_entry, f), ex, allowed,
                                     close, high, low, f["atr"].to_numpy(dtype=float),
                                     int(min(self.max_hold, 2**31 - 1)), float(self.stop_atr), bool(self.trail))
        return pd.Series(pos, index=f.index, name=self.id), pd.Series(stop_next, index=f.index)

    def positions(self, f: pd.DataFrame, data: pd.DataFrame | pd.Series) -> pd.Series:
        return self.simulate(f, data)[0]

    # ---------- serialization ----------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        d = dict(d)
        for k in ("long_entry", "short_entry", "exit_rules"):
            d[k] = [Rule(**r) for r in d.get(k, [])]
        return cls(**d)


def spec_from_dict(d: dict):
    """Dispatch on strategy kind (rules / model)."""
    if d.get("kind") == "model":
        from .model import ModelSpec
        return ModelSpec.from_dict(d)
    return StrategySpec.from_dict(d)
