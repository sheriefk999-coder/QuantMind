"""Risk Management Agent - independent, with absolute veto power.

No other component can override it. It checks the proposed trade against hard
limits and current portfolio state, and can also SHRINK a trade rather than
block it. The emergency shutdown (kill switch) halts all new risk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..core.config import settings


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20        # of equity, per symbol
    max_gross_exposure: float = 0.60      # sum |positions| / equity
    risk_per_trade_pct: float = 0.005     # equity lost if stop is hit
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10        # from equity peak -> emergency shutdown
    max_consecutive_losses: int = 5
    loss_cooldown_hours: int = 24
    max_corr_exposure: float = 0.40       # exposure in same direction across assets with corr > 0.7
    max_spread_bps: float = 15.0
    max_participation: float = 0.01       # order <= 1% of recent bar dollar volume
    news_blackout_minutes: int = 60
    min_conviction: float = 0.10


@dataclass
class TradeIntent:
    symbol: str
    direction: int
    conviction: float
    price: float
    atr: float
    stop_atr: float = 3.0


@dataclass
class RiskVerdict:
    approved: bool
    size_units: float = 0.0
    notional: float = 0.0
    stop_price: float | None = None
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.L = limits or RiskLimits()

    def kill_switch_active(self, portfolio: dict) -> str | None:
        if os.path.exists(settings.kill_switch_file) or os.getenv("QM_KILL_SWITCH") == "1":
            return "manual kill switch engaged"
        if portfolio.get("halted"):
            return f"trading halted: {portfolio.get('halt_reason')}"
        return None

    def portfolio_breach(self, portfolio: dict) -> str | None:
        eq, peak = portfolio["equity"], portfolio.get("peak_equity", portfolio["equity"])
        if peak > 0 and eq / peak - 1 <= -self.L.max_drawdown_pct:
            return f"max drawdown {eq / peak - 1:.1%} breached (limit -{self.L.max_drawdown_pct:.0%})"
        return None

    def evaluate(self, t: TradeIntent, portfolio: dict, ohlcv: pd.DataFrame,
                 corr: pd.DataFrame | None = None, spread_bps: float | None = None,
                 calendar: list[dict] | None = None) -> RiskVerdict:
        v = RiskVerdict(False)
        L, eq = self.L, portfolio["equity"]
        now = datetime.now(timezone.utc)

        def fail(msg):
            v.reasons.append(msg)
            return v

        if (k := self.kill_switch_active(portfolio)):
            return fail(k)
        if (b := self.portfolio_breach(portfolio)):
            return fail(b)
        if t.direction == 0:
            return fail("no direction")
        if t.conviction < L.min_conviction:
            return fail(f"conviction {t.conviction:.2f} < {L.min_conviction}")
        day_pnl = portfolio.get("day_pnl", 0.0)
        v.checks["day_pnl_pct"] = day_pnl / eq
        if day_pnl / eq <= -L.max_daily_loss_pct:
            return fail(f"daily loss limit hit ({day_pnl / eq:.2%})")
        cl = portfolio.get("consecutive_losses", 0)
        last_loss = portfolio.get("last_loss_ts")
        if cl >= L.max_consecutive_losses and last_loss and \
                now - datetime.fromisoformat(last_loss) < timedelta(hours=L.loss_cooldown_hours):
            return fail(f"{cl} consecutive losses: cooling down {L.loss_cooldown_hours}h")
        if spread_bps is not None and spread_bps > L.max_spread_bps:
            return fail(f"spread {spread_bps:.1f}bps too wide")
        for e in calendar or []:
            ts = pd.Timestamp(e["ts"]).to_pydatetime()
            if e.get("impact") == "high" and abs((ts - now).total_seconds()) < L.news_blackout_minutes * 60:
                return fail(f"news blackout: {e.get('title')}")
        if not t.atr or np.isnan(t.atr) or t.atr <= 0:
            return fail("no volatility estimate")

        # ---- sizing: risk-based, then capped by every exposure limit
        stop_dist = t.stop_atr * t.atr
        units = (eq * L.risk_per_trade_pct * min(1.0, 0.5 + t.conviction)) / stop_dist
        caps = {"per_trade_risk": units * t.price}
        caps["max_position"] = eq * L.max_position_pct
        gross = sum(abs(p["qty"] * p["last_price"]) for p in portfolio.get("positions", {}).values())
        caps["gross_exposure"] = max(0.0, eq * L.max_gross_exposure - gross)
        dv = float((ohlcv["close"] * ohlcv["volume"]).tail(24).mean())
        caps["liquidity"] = dv * L.max_participation
        if corr is not None and t.symbol in corr:
            same_dir = 0.0
            for sym, p in portfolio.get("positions", {}).items():
                if sym != t.symbol and sym in corr and corr.loc[t.symbol, sym] > 0.7 and np.sign(p["qty"]) == t.direction:
                    same_dir += abs(p["qty"] * p["last_price"])
            caps["correlation"] = max(0.0, eq * L.max_corr_exposure - same_dir)
        notional = min(caps.values())
        v.checks["caps"] = {k: round(c, 2) for k, c in caps.items()}
        binding = min(caps, key=caps.get)
        if notional < eq * 0.002:
            return fail(f"position too small after limits (binding: {binding})")
        v.approved = True
        v.notional = round(notional, 2)
        v.size_units = notional / t.price
        v.stop_price = t.price - t.direction * stop_dist
        v.reasons.append(f"approved; size bound by {binding}")
        return v
