"""Market Regime Intelligence.

Two complementary views:
1. Interpretable labels from measured market properties (trend strength,
   volatility percentile, volume shocks, drawdown speed).
2. Unsupervised clustering (Gaussian mixture) fitted only on PAST data, which
   can surface states nobody named, plus a novelty score: how far today is from
   anything seen before. High novelty means "investigate, don't trust the rules".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Which strategy families each regime favours (prior; the learning loop
# overwrites these with measured per-regime performance).
REGIME_PREFERENCES: dict[str, dict[str, float]] = {
    "strong_trend_up": {"momentum": 1.0, "breakout": 0.8, "mean_reversion": 0.2, "volatility": 0.5},
    "strong_trend_down": {"momentum": 1.0, "breakout": 0.8, "mean_reversion": 0.2, "volatility": 0.5},
    "weak_trend": {"momentum": 0.6, "breakout": 0.5, "mean_reversion": 0.5, "volatility": 0.5},
    "sideways": {"momentum": 0.2, "breakout": 0.4, "mean_reversion": 1.0, "volatility": 0.7},
    "high_volatility": {"momentum": 0.4, "breakout": 0.4, "mean_reversion": 0.3, "volatility": 0.4},
    "low_volatility": {"momentum": 0.5, "breakout": 0.6, "mean_reversion": 0.8, "volatility": 1.0},
    "panic_selling": {"momentum": 0.1, "breakout": 0.1, "mean_reversion": 0.1, "volatility": 0.1},
    "short_squeeze": {"momentum": 0.2, "breakout": 0.2, "mean_reversion": 0.1, "volatility": 0.1},
    "accumulation": {"momentum": 0.4, "breakout": 0.8, "mean_reversion": 0.7, "volatility": 0.8},
    "distribution": {"momentum": 0.4, "breakout": 0.8, "mean_reversion": 0.7, "volatility": 0.8},
}

REGIME_AR = {
    "strong_trend_up": "صعود قوي", "strong_trend_down": "هبوط قوي", "weak_trend": "اتجاه ضعيف",
    "sideways": "حركة عرضية", "high_volatility": "تذبذب عالي", "low_volatility": "هدوء",
    "panic_selling": "بيع بذعر", "short_squeeze": "ارتفاع مفاجئ (سكويز)",
    "accumulation": "تجميع", "distribution": "تصريف",
}


@dataclass
class RegimeState:
    label: str
    label_ar: str
    confidence: float
    cluster: int | None
    novelty: float                 # 0..1, share of history that looked less unusual
    evidence: dict = field(default_factory=dict)
    family_weights: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


REGIME_LIST = list(REGIME_PREFERENCES)          # stable integer codes for strategies' regime genes


def label_codes(f: pd.DataFrame) -> np.ndarray:
    """Vectorized twin of _label_row (same rules, same order) -> integer codes."""
    g = lambda c: f[c].fillna(0.0).to_numpy() if c in f else np.zeros(len(f))  # noqa: E731
    er, adx, mom, volp = g("er_20"), g("adx_14"), g("mom_168"), g("vol_pct_500")
    ret1, vz, z200 = g("ret_1"), g("volume_z"), g("z_close_200")
    side = (adx < 20) & (er < 0.25)
    conds = [
        (ret1 < -3) & (vz > 2) & (volp > 0.8),
        (ret1 > 3) & (vz > 2) & (volp > 0.8),
        volp > 0.9,
        (adx > 30) & (er > 0.35) & (np.abs(mom) > 1) & (mom > 0),
        (adx > 30) & (er > 0.35) & (np.abs(mom) > 1),
        volp < 0.15,
        side & (vz > 0.5) & (z200 < -1),
        side & (vz > 0.5) & (z200 > 1),
        side,
    ]
    names = ["panic_selling", "short_squeeze", "high_volatility", "strong_trend_up", "strong_trend_down",
             "low_volatility", "accumulation", "distribution", "sideways"]
    codes = np.select(conds, [REGIME_LIST.index(n) for n in names], REGIME_LIST.index("weak_trend"))
    return codes.astype(float)


def _label_row(f: pd.Series) -> tuple[str, float, dict]:
    er, adx, mom, volp = f.get("er_20"), f.get("adx_14"), f.get("mom_168"), f.get("vol_pct_500")
    ret1, vz, z200 = f.get("ret_1"), f.get("volume_z"), f.get("z_close_200")
    ev = {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in
          dict(er=er, adx=adx, mom_168=mom, vol_pct=volp, ret_1=ret1, volume_z=vz, z200=z200).items()}
    nz = lambda v, d=0.0: d if v is None or pd.isna(v) else float(v)  # noqa: E731
    er, adx, mom, volp, ret1, vz, z200 = map(nz, (er, adx, mom, volp, ret1, vz, z200))
    if ret1 < -3 and vz > 2 and volp > 0.8:
        return "panic_selling", min(1.0, abs(ret1) / 5), ev
    if ret1 > 3 and vz > 2 and volp > 0.8:
        return "short_squeeze", min(1.0, ret1 / 5), ev
    if volp > 0.9:
        return "high_volatility", volp, ev
    if adx > 30 and er > 0.35 and abs(mom) > 1:
        return ("strong_trend_up" if mom > 0 else "strong_trend_down"), min(1.0, adx / 50), ev
    if volp < 0.15:
        return "low_volatility", 1 - volp, ev
    if adx < 20 and er < 0.25:
        if vz > 0.5 and z200 < -1:
            return "accumulation", 0.5, ev
        if vz > 0.5 and z200 > 1:
            return "distribution", 0.5, ev
        return "sideways", 1 - er, ev
    return "weak_trend", 0.5, ev


class RegimeEngine:
    CLUSTER_FEATURES = ["er_20", "adx_14", "mom_168", "vol_pct_500", "volume_z", "vol_ratio", "z_close_200"]

    def __init__(self, n_clusters: int = 6, seed: int = 0):
        self.n_clusters, self.seed = n_clusters, seed

    def label_series(self, f: pd.DataFrame) -> pd.Series:
        return pd.Series([REGIME_LIST[int(c)] for c in label_codes(f)], index=f.index)

    def current(self, f: pd.DataFrame, fit_window: int = 3000,
                learned_prefs: dict | None = None) -> RegimeState:
        last = f.iloc[-1]
        label, conf, ev = _label_row(last)
        cluster, novelty = None, 0.0
        hist = f[self.CLUSTER_FEATURES].iloc[-fit_window:-1].dropna()
        x_now = f[self.CLUSTER_FEATURES].iloc[[-1]]
        if len(hist) > 200 and not x_now.isna().any(axis=None):
            sc = StandardScaler().fit(hist)            # fitted on the past only
            H, X = sc.transform(hist), sc.transform(x_now)
            gm = GaussianMixture(self.n_clusters, random_state=self.seed, covariance_type="full").fit(H)
            cluster = int(gm.predict(X)[0])
            iso = IsolationForest(random_state=self.seed, n_estimators=150).fit(H)
            s_hist, s_now = iso.score_samples(H), iso.score_samples(X)[0]
            novelty = float((s_hist > s_now).mean())   # 0.99 => stranger than 99% of history
        prefs = (learned_prefs or {}).get(label) or REGIME_PREFERENCES.get(label, {})
        return RegimeState(label, REGIME_AR.get(label, label), round(conf, 3), cluster, round(novelty, 3), ev, prefs)
