"""Dynamic Risk Engine.

Recomputed for every coin on every cycle. Ten measured factors, each scaled
0..1 and weighted, give a risk score 0-100:

  factor              what it measures                                   weight
  volatility          today's volatility vs the last ~3 weeks            0.16
  regime              how dangerous the current market state is          0.16
  novelty             how unusual the market is vs all history           0.08
  disagreement        how much the agents disagree                       0.10
  uncertainty         how weak the consensus signal is                   0.10
  stop_distance       how far the stop must be (bigger = bigger swings)  0.10
  crowding            Fear & Greed extreme on the same side as the trade 0.06
  portfolio_heat      drawdown, open trades, losing streak               0.10
  events              high-impact news soon                              0.04
  track_record        how this kind of signal has actually done          0.10

The score is not decoration: it scales the position size, sets the minimum
stop quality, and above the block level the coach says "don't".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

WEIGHTS = {"volatility": 0.16, "regime": 0.16, "novelty": 0.08, "disagreement": 0.10, "uncertainty": 0.10,
           "stop_distance": 0.10, "crowding": 0.06, "portfolio_heat": 0.10, "events": 0.04, "track_record": 0.10}

REGIME_DANGER = {"panic_selling": 1.0, "short_squeeze": 0.9, "high_volatility": 0.75, "strong_trend_down": 0.6,
                 "distribution": 0.55, "weak_trend": 0.4, "sideways": 0.35, "accumulation": 0.3,
                 "strong_trend_up": 0.25, "low_volatility": 0.2}

FACTOR_AR = {"volatility": "التذبذب", "regime": "حالة السوق", "novelty": "غرابة الوضع",
             "disagreement": "اختلاف الوكلاء", "uncertainty": "ضعف الإشارة", "stop_distance": "بُعد وقف الخسارة",
             "crowding": "الطمع/الخوف", "portfolio_heat": "حالة محفظتك", "events": "أخبار قريبة",
             "track_record": "سجل الإشارات"}

PROFILES = {"conservative": 0.5, "balanced": 1.0, "bold": 1.5}
PROFILE_AR = {"conservative": "حذِر", "balanced": "متوازن", "bold": "جريء"}

BLOCK_AT = 70        # >= this: do not open new trades
CAUTION_AT = 50


def level_ar(score: float) -> str:
    return "منخفض 🟢" if score < 30 else "متوسط 🟡" if score < CAUTION_AT else "عالي 🟠" if score < BLOCK_AT \
        else "عالي جدًا 🔴"


@dataclass
class RiskAssessment:
    score: float
    level_ar: str
    factors: dict[str, float]
    top_reasons_ar: list[str]
    size_multiplier: float
    blocked: bool
    loss_if_stopped_pct: float | None = None      # of total capital
    loss_probability: float | None = None         # historical, from validated strategies
    loss_probability_basis: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _clip01(x) -> float:
    return float(np.clip(0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else x, 0.0, 1.0))


def assess(*, features: pd.DataFrame, regime: dict, debate: dict, portfolio: dict, direction: int,
           price: float, stop: float | None, sentiment: float | None, calendar: list[dict],
           track_record: dict, strategy_stats: dict | None, profile: str = "balanced") -> RiskAssessment:
    last = features.iloc[-1]
    atr = float(last.get("atr", np.nan))
    f: dict[str, float] = {}

    vr = last.get("vol_ratio")
    vp = last.get("vol_pct_500")
    f["volatility"] = _clip01(0.6 * _clip01(vp) + 0.4 * _clip01(((vr or 1) - 0.8) / 1.5))
    f["regime"] = REGIME_DANGER.get(regime.get("label"), 0.5)
    f["novelty"] = _clip01((regime.get("novelty", 0) - 0.6) / 0.4)
    f["disagreement"] = _clip01(debate.get("disagreement", 0.5) / 0.6)
    f["uncertainty"] = _clip01(1 - debate.get("conviction", 0) / 0.4)
    if stop and price:
        stop_pct = abs(price - stop) / price
    else:
        stop_pct = 3 * atr / price if price and not np.isnan(atr) else 0.05
    f["stop_distance"] = _clip01((stop_pct - 0.01) / 0.09)        # 1% -> 0, 10% -> 1
    if sentiment is not None and direction != 0:
        same_side = (sentiment - 50) * direction                  # buying into greed / selling into fear
        f["crowding"] = _clip01((same_side - 15) / 30)
    else:
        f["crowding"] = 0.3
    eq, peak = portfolio.get("equity", 1), portfolio.get("peak_equity", 1)
    dd = max(0.0, 1 - eq / peak) if peak else 0.0
    heat = 0.5 * _clip01(dd / 0.08) + 0.3 * _clip01(len(portfolio.get("positions", {})) / 4) \
        + 0.2 * _clip01(portfolio.get("consecutive_losses", 0) / 4)
    f["portfolio_heat"] = _clip01(heat)
    now = datetime.now(timezone.utc)
    soon = [e for e in calendar if e.get("impact") == "high"
            and 0 <= (pd.Timestamp(e["ts"]).to_pydatetime() - now).total_seconds() < 24 * 3600]
    f["events"] = 1.0 if soon else 0.0
    n = track_record.get("signals", 0)
    if n >= 10:
        f["track_record"] = _clip01((0.55 - track_record.get("win_rate", 0.5)) / 0.25 + 0.3)
    else:
        f["track_record"] = 0.5          # unknown record = moderate risk, not zero

    score = round(100 * sum(WEIGHTS[k] * v for k, v in f.items()), 1)
    contrib = sorted(f, key=lambda k: -WEIGHTS[k] * f[k])
    reasons = [f"{FACTOR_AR[k]} ({f[k]:.0%})" for k in contrib[:3] if f[k] > 0.3]
    mult = float(np.clip(1.3 - score / 70, 0.2, 1.2)) * PROFILES.get(profile, 1.0)
    blocked = score >= BLOCK_AT or (profile == "conservative" and score >= BLOCK_AT - 10)

    ra = RiskAssessment(score, level_ar(score), {k: round(v, 3) for k, v in f.items()}, reasons,
                        round(mult, 3), blocked, extra={"stop_pct": round(stop_pct * 100, 2)})
    if strategy_stats:
        ra.extra["proof"] = strategy_stats.get("proof")
    if strategy_stats and strategy_stats.get("trades", 0) >= 10:
        ra.loss_probability = round(1 - strategy_stats["win_rate"], 3)
        ra.loss_probability_basis = f"{strategy_stats['trades']} صفقة تاريخية للاستراتيجيات دي على بيانات ما شافتهاش"
    return ra


def strategy_stats_from(components: list[dict], registry_entries: dict[str, dict], direction: int) -> dict | None:
    """Weighted historical win rate / trade count of the strategies backing this trade."""
    wr, n, w, pf, at = 0.0, 0, 0.0, 0.0, 0.0
    for c in components:
        if np.sign(c.get("position", 0)) != direction or c.get("weight", 0) <= 0:
            continue
        h = (registry_entries.get(c["id"], {}).get("results") or {}).get("holdout") or {}
        if h.get("n_trades"):
            wr += c["weight"] * h.get("win_rate", 0.5)
            pff = h.get("profit_factor")
            pf += c["weight"] * (min(pff, 10.0) if isinstance(pff, (int, float)) and np.isfinite(pff) else 1.0)
            at += c["weight"] * (h.get("avg_trade") or 0.0)
            n += h["n_trades"]
            w += c["weight"]
    if w <= 0:
        return None
    from .proof import edge_proof
    st = {"win_rate": wr / w, "trades": n, "profit_factor": pf / w, "avg_trade": at / w}
    st["proof"] = edge_proof(st["win_rate"], n, st["profit_factor"], st["avg_trade"])
    return st
