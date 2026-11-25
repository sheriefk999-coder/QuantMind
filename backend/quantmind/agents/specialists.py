"""The specialist agents. Each looks at the market through a different lens and
must cite its evidence and state what would invalidate its view. An agent with
no data for its lens abstains; it never invents inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Agent, AgentContext, Opinion, clip


def _v(f: pd.DataFrame, col: str, default: float = 0.0) -> float:
    x = f[col].iloc[-1] if col in f else np.nan
    return default if pd.isna(x) else float(x)


class MarketIntelligenceAgent(Agent):
    name, name_ar, style = "market_intelligence", "استخبارات السوق", "context"

    def analyze(self, ctx: AgentContext) -> Opinion:
        r = ctx.regime
        bias = {"strong_trend_up": 0.5, "strong_trend_down": -0.5, "panic_selling": -0.3,
                "short_squeeze": 0.3}.get(r.label, 0.0)
        ev = [f"regime={r.label} (conf {r.confidence:.2f})", f"novelty={r.novelty:.2f}",
              f"cluster={r.cluster}"]
        if r.novelty > 0.95:
            ev.append("market state is unusual vs history: rules are less reliable")
        return Opinion(self.name, bias, r.confidence * (1 - r.novelty / 2), self.style, ev,
                       invalidation="regime label flips on next bars")


class QuantResearchAgent(Agent):
    """Speaks only for strategies that passed every validation gate, combined
    through the ensemble (regime-fit, risk-parity, redundancy-aware)."""
    name, name_ar, style = "quant_research", "البحث الكمي", "evidence"

    def analyze(self, ctx: AgentContext) -> Opinion:
        from ..core.config import settings
        from ..portfolio.ensemble import ensemble

        sym = ctx.snapshot.symbol
        strats = [s for s in ctx.validated_strategies if s["results"].get("symbol") == sym]
        if not strats:
            return self.abstain("no validated strategy for this symbol yet")
        view = ensemble(strats, ctx.features, ctx.snapshot.ohlcv, ctx.regime.family_weights,
                        spot_only=settings.market_type == "spot")
        ctx.extras["ensemble"] = view
        ev = [f"ensemble of {len(strats)} validated strategies -> target {view.target:+.2f} "
              f"(agreement {view.agreement:.0%}, recent Sharpe {view.recent_sharpe})"]
        ev += [f"{c['name']} [{c['family']}] pos {c['position']:+.0f} weight {c.get('weight', 0):.2f}"
               for c in sorted(view.components, key=lambda c: -c.get("weight", 0))[:5]]
        return Opinion(self.name, clip(view.target), min(1.0, 0.3 + 0.6 * view.agreement), self.style, ev,
                       invalidation="ensemble target returns to flat, or decay monitor disables its strategies")


class TechnicalAgent(Agent):
    name, name_ar, style = "technical", "التحليل الفني", "trend"

    def analyze(self, ctx: AgentContext) -> Opinion:
        f = ctx.features
        mom, macd, rsi, adx = _v(f, "mom_168"), _v(f, "macdh_norm"), _v(f, "rsi_14", 50), _v(f, "adx_14", 20)
        score = 0.5 * np.tanh(mom) + 0.3 * np.tanh(2 * macd) + 0.2 * (rsi - 50) / 50
        return Opinion(self.name, clip(score), min(1.0, adx / 40), self.style,
                       [f"momentum(168)={mom:.2f}", f"MACD hist/ATR={macd:.2f}", f"RSI={rsi:.0f}", f"ADX={adx:.0f}"],
                       invalidation="momentum turns negative / ADX collapses below 15")


class PriceActionAgent(Agent):
    name, name_ar, style = "price_action", "حركة السعر", "trend"

    def analyze(self, ctx: AgentContext) -> Opinion:
        f = ctx.features
        st, don = _v(f, "structure"), _v(f, "don_100", 0.5)
        score = 0.6 * st + 0.4 * clip((don - 0.5) * 2)
        return Opinion(self.name, clip(score), 0.3 + 0.4 * abs(st), self.style,
                       [f"swing structure={st:+.1f} (HH/HL=+1, LH/LL=-1)", f"position in 100-bar range={don:.2f}"],
                       invalidation="break of the last confirmed swing low/high")


class MeanReversionAgent(Agent):
    name, name_ar, style = "mean_reversion", "الارتداد للمتوسط", "reversion"

    def analyze(self, ctx: AgentContext) -> Opinion:
        f = ctx.features
        z, rsi3, er = _v(f, "z_close_50"), _v(f, "rsi_3", 50), _v(f, "er_20", 0.5)
        score = -0.6 * np.tanh(z / 2) - 0.4 * (rsi3 - 50) / 50
        return Opinion(self.name, clip(score), clip(1 - er, 0, 1) * min(1, abs(z) / 2), self.style,
                       [f"z-score vs 50-bar mean={z:.2f}", f"RSI(3)={rsi3:.0f}", f"efficiency ratio={er:.2f}"],
                       invalidation="price keeps extending with rising efficiency ratio (a real trend)")


class LiquidityOrderFlowAgent(Agent):
    name, name_ar, style = "liquidity_flow", "السيولة وتدفق الأوامر", "flow"

    def analyze(self, ctx: AgentContext) -> Opinion:
        s, ev, parts = ctx.snapshot, [], []
        if s.orderbook is not None:
            imb = s.orderbook.imbalance()
            parts.append(0.5 * imb)
            ev += [f"order-book imbalance (±25bps)={imb:+.2f}", f"spread={s.orderbook.spread_bps:.1f}bps"]
        if s.funding is not None and len(s.funding):
            fr = float(s.funding.iloc[-1])
            parts.append(-clip(fr / 0.0005) * 0.3)   # crowded longs pay funding -> fade
            ev.append(f"funding={fr * 100:.4f}% (positive = longs crowded)")
        if s.open_interest is not None and len(s.open_interest) > 24:
            oi_chg = float(s.open_interest.iloc[-1] / s.open_interest.iloc[-24] - 1)
            px_chg = float(s.ohlcv["close"].iloc[-1] / s.ohlcv["close"].iloc[-24] - 1)
            parts.append(0.2 * np.sign(px_chg) * clip(oi_chg * 10))  # OI confirms price move
            ev.append(f"OI 24-bar change={oi_chg:+.1%} with price {px_chg:+.1%}")
        vz = _v(ctx.features, "volume_z")
        ev.append(f"volume z-score={vz:.2f}")
        if not parts:
            return Opinion(self.name, 0.0, 0.1, self.style, ev + ["no order-book/derivatives data"],
                           invalidation="n/a")
        return Opinion(self.name, clip(sum(parts)), min(1.0, 0.3 + 0.1 * len(parts)), self.style, ev,
                       invalidation="imbalance flips / funding normalises")


class StatArbAgent(Agent):
    """Relative value vs the most correlated related asset (e.g. ETH vs BTC)."""
    name, name_ar, style = "stat_arb", "المراجحة الإحصائية", "relative"

    def analyze(self, ctx: AgentContext) -> Opinion:
        if not ctx.snapshot.related:
            return self.abstain("no related asset loaded")
        a = np.log(ctx.snapshot.ohlcv["close"])
        best = None
        for sym, df in ctx.snapshot.related.items():
            b = np.log(df["close"]).reindex(a.index).ffill()
            ra, rb = a.diff().tail(500), b.diff().tail(500)
            c = ra.corr(rb)
            if best is None or (c is not None and c > best[1]):
                best = (sym, c, b)
        sym, corr, b = best
        if corr is None or np.isnan(corr) or corr < 0.6:
            return self.abstain(f"no strongly correlated pair (best {sym} corr={corr:.2f})")
        j = pd.concat([a.diff(), b.diff()], axis=1).tail(500).dropna()
        beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
        spread = (a - beta * b).tail(500)
        z = float((spread.iloc[-1] - spread.mean()) / spread.std())
        return Opinion(self.name, clip(-z / 2.5), min(1.0, abs(z) / 3) * corr, self.style,
                       [f"pair vs {sym}: corr={corr:.2f}, beta={beta:.2f}, spread z={z:+.2f}"],
                       invalidation="correlation breaks down (corr < 0.5) or spread keeps widening")


class SentimentAgent(Agent):
    name, name_ar, style = "sentiment", "المشاعر", "flow"

    def analyze(self, ctx: AgentContext) -> Opinion:
        s = ctx.snapshot.sentiment
        if s is None or not len(s):
            return self.abstain("no sentiment source available (will not guess)")
        v = float(s.iloc[-1])
        chg = float(v - s.iloc[-8]) if len(s) > 8 else 0.0
        # Contrarian reading of the Fear & Greed index: extreme fear has
        # historically been a better time to buy than extreme greed. Only the
        # extremes carry weight; the middle of the range is treated as noise.
        extreme = max(0.0, abs(v - 50) - 20) / 30            # 0 inside 30..70, 1 at 0 / 100
        score = -np.sign(v - 50) * extreme
        label = "extreme fear" if v <= 25 else "fear" if v < 45 else "extreme greed" if v >= 75 else \
            "greed" if v > 55 else "neutral"
        return Opinion(self.name, clip(score), 0.2 + 0.4 * extreme, self.style,
                       [f"Fear & Greed = {v:.0f} ({label}), 7-day change {chg:+.0f}"],
                       invalidation="sentiment keeps getting more extreme in the same direction")


class FundamentalAgent(Agent):
    name, name_ar, style = "fundamental", "التحليل الأساسي", "context"

    def analyze(self, ctx: AgentContext) -> Opinion:
        high = [e for e in ctx.snapshot.calendar if e.get("impact") == "high"]
        if not high:
            return self.abstain("no macro calendar / fundamental feed configured")
        return Opinion(self.name, 0.0, 0.5, self.style, [f"upcoming: {e['title']} @ {e['ts']}" for e in high[:3]],
                       invalidation="event passes without surprise")


SPECIALISTS: list[type[Agent]] = [MarketIntelligenceAgent, QuantResearchAgent, TechnicalAgent, PriceActionAgent,
                                  MeanReversionAgent, LiquidityOrderFlowAgent, StatArbAgent, SentimentAgent,
                                  FundamentalAgent]
