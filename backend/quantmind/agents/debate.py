"""Debate engine: not a majority vote.

Round 1  Independent analysis: every specialist forms an opinion alone.
Round 2  Cross-examination:
           - the Contrarian attacks whichever side is winning (crowding,
             over-extension, novelty, thin evidence)
           - every opinion is tested against the current regime: a trend argument
             in a sideways market, or a fade-the-move argument in a strong trend,
             loses credibility
           - each side's strongest argument is put to the other side; opinions
             whose invalidation condition is already met are discounted
Round 3  Weighted synthesis: weight = measured reliability (Performance Auditor,
         out-of-sample hit rate with a Bayesian prior) x confidence after
         challenges. Disagreement is measured; high disagreement => stand aside.
Gate     A trade may only be ORIGINATED by validated strategies (quant_research).
         Discretionary agents can strengthen, weaken or block it, never start one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import AgentContext, Opinion

REGIME_STYLE_FIT = {
    "strong_trend_up": {"trend": 1.0, "reversion": 0.4, "flow": 0.9, "relative": 0.7},
    "strong_trend_down": {"trend": 1.0, "reversion": 0.4, "flow": 0.9, "relative": 0.7},
    "sideways": {"trend": 0.4, "reversion": 1.0, "flow": 0.8, "relative": 1.0},
    "high_volatility": {"trend": 0.6, "reversion": 0.5, "flow": 0.7, "relative": 0.6},
    "panic_selling": {"trend": 0.5, "reversion": 0.3, "flow": 0.8, "relative": 0.4},
    "short_squeeze": {"trend": 0.5, "reversion": 0.3, "flow": 0.8, "relative": 0.4},
}


@dataclass
class DebateResult:
    direction: int                 # +1 long, -1 short, 0 stand aside
    edge: float                    # weighted consensus score in [-1, 1]
    conviction: float              # 0..1 after disagreement penalty
    disagreement: float
    opinions: list[dict]
    challenges: list[str]
    rationale: list[str]
    stand_aside_reason: str | None = None
    transcript: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ContrarianAgent:
    name, name_ar = "contrarian", "المحلل المعاكس"

    def challenge(self, ctx: AgentContext, leading_side: int) -> list[tuple[str, float]]:
        """Return (argument, severity 0..1) against the leading side."""
        if leading_side == 0:
            return []
        f, out = ctx.features.iloc[-1], []
        rsi, z, er = f.get("rsi_14", 50), f.get("z_close_50", 0), f.get("er_20", 0.5)
        if leading_side > 0 and rsi > 75:
            out.append((f"RSI {rsi:.0f}: buyers may be exhausted, late entry risk", 0.3))
        if leading_side < 0 and rsi < 25:
            out.append((f"RSI {rsi:.0f}: sellers may be exhausted, late entry risk", 0.3))
        if abs(z) > 2.5 and np.sign(z) == leading_side:
            out.append((f"price is {z:+.1f} std from its mean: stretched", 0.25))
        fund = ctx.snapshot.funding
        if fund is not None and len(fund) and np.sign(fund.iloc[-1]) == leading_side and abs(fund.iloc[-1]) > 0.0003:
            out.append(("funding shows the crowd already positioned this way", 0.3))
        if ctx.regime.novelty > 0.9:
            out.append((f"market state is stranger than {ctx.regime.novelty:.0%} of history: past patterns may not hold", 0.4))
        if er < 0.15:
            out.append(("price path is mostly noise (efficiency ratio low)", 0.2))
        if ctx.regime.label in ("panic_selling", "short_squeeze", "high_volatility"):
            out.append((f"{ctx.regime.label}: slippage and gap risk are elevated", 0.3))
        return out


class DebateEngine:
    def __init__(self, reliabilities: dict[str, float] | None = None, min_edge: float = 0.15,
                 max_disagreement: float = 0.6, max_novelty: float = 0.98):
        self.rel = reliabilities or {}
        self.min_edge, self.max_dis, self.max_nov = min_edge, max_disagreement, max_novelty
        self.contrarian = ContrarianAgent()

    def run(self, ctx: AgentContext, opinions: list[Opinion]) -> DebateResult:
        tr: list[str] = ["ROUND 1 - independent opinions"]
        active = [o for o in opinions if not o.abstain]
        for o in opinions:
            tr.append(f"  {o.agent}: " + ("ABSTAINS - " + o.evidence[0] if o.abstain
                                          else f"score {o.score:+.2f}, conf {o.confidence:.2f} | " + "; ".join(o.evidence[:2])))

        # ---- round 2: cross-examination
        tr.append("ROUND 2 - cross-examination")
        fit = REGIME_STYLE_FIT.get(ctx.regime.label, {})
        for o in active:
            o.adjusted_confidence = o.confidence
            m = fit.get(o.style, 1.0)
            if m < 1.0:
                o.adjusted_confidence *= m
                o.challenges.append(f"{o.style} reasoning is a weaker fit for {ctx.regime.label} (x{m})")
        bulls = sorted([o for o in active if o.score > 0.1], key=lambda o: -o.score * o.adjusted_confidence)
        bears = sorted([o for o in active if o.score < -0.1], key=lambda o: o.score * o.adjusted_confidence)
        if bulls and bears:
            b, s = bulls[0], bears[0]
            tr.append(f"  strongest bull ({b.agent}): {b.evidence[0]}")
            tr.append(f"  strongest bear ({s.agent}): {s.evidence[0]}")
            # the weaker champion concedes more
            sb, ss = abs(b.score) * b.adjusted_confidence, abs(s.score) * s.adjusted_confidence
            loser = s if sb > ss else b
            loser.adjusted_confidence *= 0.85
            loser.challenges.append("its camp's best argument was weaker than the other side's")
        prelim = self._aggregate(active)[0]
        side = int(np.sign(prelim)) if abs(prelim) > 0.05 else 0
        challenges = self.contrarian.challenge(ctx, side)
        severity = 1.0
        for arg, sev in challenges:
            tr.append(f"  contrarian: {arg} (severity {sev})")
            severity *= (1 - sev)
        for o in active:
            if np.sign(o.score) == side and side != 0:
                o.adjusted_confidence *= severity
                if challenges:
                    o.challenges.append(f"contrarian challenges cut confidence x{severity:.2f}")

        # ---- round 3: synthesis
        edge, disagreement = self._aggregate(active)
        conviction = float(abs(edge) * (1 - disagreement))
        tr.append(f"ROUND 3 - synthesis: edge {edge:+.3f}, disagreement {disagreement:.2f}, conviction {conviction:.3f}")
        direction, reason = int(np.sign(edge)), None
        qr = next((o for o in active if o.agent == "quant_research"), None)
        if ctx.regime.novelty > self.max_nov:
            reason = "market state unlike anything in history - investigate, don't trade"
        elif qr is None:
            reason = "no validated strategy has an opinion - discretionary views alone may not open trades"
        elif np.sign(qr.score) != np.sign(edge) or abs(qr.score) < 0.1:
            reason = "validated strategies do not support the consensus direction"
        elif abs(edge) < self.min_edge:
            reason = f"edge {abs(edge):.2f} below threshold {self.min_edge}"
        elif disagreement > self.max_dis:
            reason = f"agents disagree too much ({disagreement:.2f})"
        if reason:
            direction = 0
            tr.append("DECISION: stand aside - " + reason)
        else:
            tr.append(f"DECISION: {'LONG' if direction > 0 else 'SHORT'} candidate -> risk manager")
        rationale = [f"{o.agent}: {o.score:+.2f} x conf {o.adjusted_confidence:.2f} x reliability "
                     f"{self.rel.get(o.agent, 0.5):.2f}" for o in active]
        return DebateResult(direction, round(edge, 4), round(conviction, 4), round(disagreement, 4),
                            [o.to_dict() for o in opinions], [c for c, _ in challenges], rationale, reason, tr)

    def _aggregate(self, ops: list[Opinion]) -> tuple[float, float]:
        if not ops:
            return 0.0, 1.0
        # reliability 0.5 = coin flip -> weight floor; proven agents count more
        w = np.array([max(0.05, (self.rel.get(o.agent, 0.5) - 0.4) * 2) * (o.adjusted_confidence or o.confidence)
                      for o in ops])
        s = np.array([o.score for o in ops])
        if w.sum() == 0:
            return 0.0, 1.0
        edge = float((w * s).sum() / w.sum())
        dis = float(np.sqrt((w * (s - edge) ** 2).sum() / w.sum()))
        return edge, min(1.0, dis)
