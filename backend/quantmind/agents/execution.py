"""Execution Optimization Agent: HOW to enter, not whether.

Chooses market vs limit order and the limit price, from spread, speed of the
move and regime. Paying a little more to get in is right when price is
running away (breakouts, strong trends); otherwise a patient limit order slightly
below the price saves the spread and slippage, and if it doesn't fill, the
trade is simply skipped.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionPlan:
    order_type: str          # "market" | "limit"
    limit_price: float | None
    expected_slippage_bps: float
    tip_ar: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def plan_entry(direction: int, price: float, atr: float, regime: str, don_20: float | None,
               spread_bps: float | None, vol_ratio: float | None) -> ExecutionPlan:
    fast = regime in ("strong_trend_up", "strong_trend_down", "short_squeeze") or \
        (don_20 is not None and (don_20 > 1.0 or don_20 < 0.0))
    spread = spread_bps if spread_bps is not None else 2.0
    vr = vol_ratio or 1.0
    slip = spread / 2 + 1.0 * vr
    if fast and spread < 10:
        return ExecutionPlan("market", None, round(slip + 2, 1),
                             "السعر بيتحرك بسرعة: اشتري بأمر Market على طول عشان متفوتكش الحركة." if direction > 0 else
                             "السعر بيتحرك بسرعة: نفّذ بأمر Market على طول.")
    off = min(0.25 * atr, price * 0.004)
    lp = price - direction * off
    return ExecutionPlan("limit", lp, round(max(0.0, slip - spread / 2), 1),
                         f"حط أمر Limit عند {lp:,.4g} (أرخص شوية من السعر الحالي). "
                         f"لو ما اتنفذش في خلال مدة الإشارة، سيبه ومتطاردش السعر.")
