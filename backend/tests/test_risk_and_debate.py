import os
import tempfile

import pytest

from quantmind.agents.base import AgentContext, Opinion
from quantmind.agents.debate import DebateEngine
from quantmind.analysis.features import build_features
from quantmind.core.config import AutonomyLevel, settings
from quantmind.core.store import Store
from quantmind.data.providers.base import MarketSnapshot
from quantmind.data.synthetic import random_walk
from quantmind.execution.live import LiveGateway, LiveTradingLocked
from quantmind.execution.paper import PaperBroker
from quantmind.regime.classifier import RegimeEngine
from quantmind.risk.manager import RiskManager, RiskVerdict, TradeIntent

DF = random_walk(3000, seed=5)
F = build_features(DF)
PORT = {"equity": 10000, "peak_equity": 10000, "positions": {}, "day_pnl": 0, "consecutive_losses": 0}


def intent():
    return TradeIntent("X", 1, 0.5, float(DF.close.iloc[-1]), float(F.atr.iloc[-1]))


def test_risk_approves_normal_trade_with_capped_size():
    v = RiskManager().evaluate(intent(), dict(PORT), DF)
    assert v.approved and 0 < v.notional <= 2000


@pytest.mark.parametrize("patch,expect", [
    ({"peak_equity": 12000}, "drawdown"),
    ({"day_pnl": -300}, "daily loss"),
    ({"halted": True, "halt_reason": "x"}, "halted"),
])
def test_risk_vetoes(patch, expect):
    v = RiskManager().evaluate(intent(), {**PORT, **patch}, DF)
    assert not v.approved and expect in " ".join(v.reasons)


def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("QM_KILL_SWITCH", "1")
    assert not RiskManager().evaluate(intent(), dict(PORT), DF).approved


def test_live_gateway_locked_by_default():
    with pytest.raises(LiveTradingLocked):
        LiveGateway().submit("X", "BUY", RiskVerdict(True, 1, 100), AutonomyLevel.LIVE)


def test_no_trade_without_validated_strategy():
    ctx = AgentContext(MarketSnapshot("X", DF), F, RegimeEngine().current(F), [])
    ops = [Opinion("technical", 0.9, 0.9, "trend", ["strong"]), Opinion("price_action", 0.8, 0.9, "trend", ["hh"])]
    res = DebateEngine().run(ctx, ops)
    assert res.direction == 0 and "validated" in res.stand_aside_reason


def test_paper_broker_round_trip_short():
    b = PaperBroker(Store(tempfile.mkdtemp()))
    b.open("X", -1, 1.0, 100.0, 110.0, {})
    b.mark({"X": 90.0})
    assert b.p["equity"] > 10000
    rec = b.close("X", 90.0, "test")
    assert rec["pnl"] > 9 and b.p["positions"] == {}
