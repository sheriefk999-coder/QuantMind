"""Machine-learning strategies, trained walk-forward.

Every prediction is made by a model that never saw the future:
  * retrain every `retrain_every` bars on an expanding window
  * a training row i needs the close at i+horizon, so only rows with
    i + horizon < retrain point are used (purging - no label leakage)
  * noise filter: rows whose forward move is tiny vs volatility are not used
    as training examples (the model learns from real moves only)
Predictions before `min_train` bars are flat. The spec stores the recipe (not
the weights), so any version can be re-trained bit-for-bit.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .kernel import simulate

FEATURE_SETS = {
    "all": ["rsi_14", "rsi_3", "z_close_50", "z_close_200", "mom_24", "mom_168", "er_20", "adx_14", "pctb_20",
            "macdh_norm", "don_20", "don_100", "vol_ratio", "vol_pct_500", "volume_z", "vwap_dev_24", "structure",
            "ret_1", "h4_mom_42", "h4_rsi_14", "h4_adx_14", "d1_mom_20", "d1_z_20", "d1_er_10", "regime_id"],
    "trend": ["mom_24", "mom_168", "er_20", "adx_14", "macdh_norm", "don_20", "don_100", "structure",
              "h4_mom_42", "h4_adx_14", "d1_mom_20", "d1_er_10"],
    "reversion": ["rsi_14", "rsi_3", "z_close_50", "z_close_200", "pctb_20", "vwap_dev_24", "ret_1",
                  "er_20", "d1_z_20", "h4_rsi_14", "vol_ratio"],
    "flow": ["volume_z", "vol_ratio", "vol_pct_500", "ret_1", "mom_24", "vwap_dev_24", "structure", "funding"],
}

_PROBA_CACHE: dict[tuple, np.ndarray] = {}


@dataclass
class ModelSpec:
    name: str
    feature_set: str = "all"
    horizon: int = 24
    margin: float = 0.06          # trade only when P(up) is this far from 0.5
    retrain_every: int = 720      # ~30 days of 1h bars
    min_train: int = 3000
    max_depth: int = 3
    learning_rate: float = 0.05
    max_iter: int = 150
    stop_atr: float = 3.0
    family: str = "ml"
    kind: str = "model"
    regimes: list[str] | None = None
    version: int = 1
    parent_ids: list[str] = field(default_factory=list)
    origin: str = "ml"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def max_hold(self) -> int:
        return self.horizon * 2

    def features(self, f: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_SETS[self.feature_set] if c in f]

    def _core_key(self) -> dict:
        return {k: getattr(self, k) for k in ("feature_set", "horizon", "retrain_every", "min_train",
                                              "max_depth", "learning_rate", "max_iter")}

    def fingerprint(self) -> str:
        core = {**self._core_key(), "margin": self.margin, "stop_atr": self.stop_atr, "regimes": self.regimes}
        return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]

    def complexity(self) -> int:
        return 3

    def required_features(self) -> set[str]:
        return set(FEATURE_SETS[self.feature_set])

    def math_definition(self) -> str:
        return (f"P_t = GradientBoosting(depth={self.max_depth}, lr={self.learning_rate}, trees={self.max_iter})"
                f"(features[{self.feature_set}] at close t)\n"
                f"  trained walk-forward: refit every {self.retrain_every} bars on rows i with i+{self.horizon} < refit bar\n"
                f"  target: sign of {self.horizon}-bar forward return (tiny moves excluded)\n"
                f"LONG if P_t > {0.5 + self.margin:.3f}; SHORT if P_t < {0.5 - self.margin:.3f}; "
                f"EXIT when |P_t-0.5| < {self.margin / 4:.3f} OR bars_held >= {self.max_hold} "
                f"OR stop {self.stop_atr:.1f}*ATR14 (intrabar)")

    # ------------------------------------------------------------ predict
    def probabilities(self, f: pd.DataFrame, close: pd.Series) -> np.ndarray:
        key = (json.dumps(self._core_key(), sort_keys=True), len(f), str(f.index[0]), str(f.index[-1]))
        if key in _PROBA_CACHE:
            return _PROBA_CACHE[key]
        from sklearn.ensemble import HistGradientBoostingClassifier

        cols = self.features(f)
        X = f[cols].to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        n, h = len(c), self.horizon
        fwd = np.full(n, np.nan)
        fwd[:-h] = np.log(c[h:] / c[:-h])
        vol = f["vol_48"].to_numpy(dtype=float) * np.sqrt(h)
        y = (fwd > 0).astype(int)
        informative = np.abs(fwd) > 0.25 * vol
        proba = np.full(n, 0.5)
        for r in range(self.min_train, n, self.retrain_every):
            train_end = r - h                                   # purge: labels must be known at r
            rows = np.arange(train_end)
            rows = rows[~np.isnan(fwd[rows]) & informative[rows]]
            if len(rows) < 500 or len(np.unique(y[rows])) < 2:
                continue
            # use only columns that actually carry information in THIS training
            # window (e.g. a data source that only started recently is skipped
            # until it has enough history) - decided from past rows only
            Xt = X[rows]
            finite = np.isfinite(Xt)
            usable = [j for j in range(Xt.shape[1])
                      if finite[:, j].mean() >= 0.5 and len(np.unique(Xt[finite[:, j], j])) >= 2]
            if len(usable) < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                               max_iter=self.max_iter, l2_regularization=1.0,
                                               early_stopping=False, random_state=0)
            m.fit(Xt[:, usable], y[rows])
            seg = slice(r, min(n, r + self.retrain_every))
            proba[seg] = m.predict_proba(X[seg][:, usable])[:, 1]
        if len(_PROBA_CACHE) > 64:
            _PROBA_CACHE.clear()
        _PROBA_CACHE[key] = proba
        return proba

    def simulate(self, f: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        p = self.probabilities(f, data["close"])
        le, se = p > 0.5 + self.margin, p < 0.5 - self.margin
        ex = np.abs(p - 0.5) < self.margin / 4
        if self.regimes and "regime_id" in f:
            from ..regime.classifier import REGIME_LIST
            allowed = np.isin(f["regime_id"].to_numpy(), [REGIME_LIST.index(r) for r in self.regimes])
        else:
            allowed = np.ones(len(f), dtype=bool)
        pos, stop_next, _ = simulate(le, se, ex, allowed, *(data[c].to_numpy(dtype=float) for c in ("close", "high", "low")),
                                     f["atr"].to_numpy(dtype=float), self.max_hold, float(self.stop_atr), False)
        return pd.Series(pos, index=f.index, name=self.id), pd.Series(stop_next, index=f.index)

    def positions(self, f, data):
        return self.simulate(f, data)[0]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
