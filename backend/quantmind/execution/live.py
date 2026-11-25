"""Live execution gateway - LOCKED by default.

Every order must pass ALL of:
  1. autonomy level == LIVE
  2. QM_LIVE_TRADING_ENABLED=true AND a non-empty QM_LIVE_AUTHORIZATION_TOKEN
     (both set by a human, outside the code)
  3. an approved RiskVerdict for this exact order
  4. order notional within QM_LIVE_MAX_CAPITAL_USD minus capital already deployed
  5. no kill switch
Only strategies with status `live` (promoted by a human after paper trading)
may reach this class. The order-sending code itself is intentionally left as a
narrow, reviewed stub: wire it up only after months of paper results.
"""
from __future__ import annotations

from ..core.config import AutonomyLevel, settings
from ..risk.manager import RiskVerdict


class LiveTradingLocked(Exception):
    pass


class LiveGateway:
    def __init__(self, deployed_usd: float = 0.0):
        self.deployed = deployed_usd

    def preflight(self, verdict: RiskVerdict, level: AutonomyLevel) -> None:
        if level != AutonomyLevel.LIVE:
            raise LiveTradingLocked("autonomy level is not LIVE")
        if not settings.live_trading_enabled or not settings.live_authorization_token:
            raise LiveTradingLocked("live trading not authorized by a human")
        if not (settings.binance_api_key and settings.binance_api_secret):
            raise LiveTradingLocked("no exchange credentials configured")
        if not verdict.approved:
            raise LiveTradingLocked("risk manager did not approve: " + "; ".join(verdict.reasons))
        if self.deployed + verdict.notional > settings.live_max_capital_usd:
            raise LiveTradingLocked(f"capital cap {settings.live_max_capital_usd} USD would be exceeded")

    def submit(self, symbol: str, side: str, verdict: RiskVerdict, level: AutonomyLevel) -> dict:
        self.preflight(verdict, level)
        raise LiveTradingLocked("order routing not enabled in this build; complete paper-trading review first")
