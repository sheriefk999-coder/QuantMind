import httpx
import numpy as np
import pandas as pd

from quantmind.agents.base import AgentContext
from quantmind.agents.execution import plan_entry
from quantmind.agents.specialists import SentimentAgent
from quantmind.analysis.features import build_features
from quantmind.data.providers import alt
from quantmind.data.providers.base import MarketSnapshot
from quantmind.data.synthetic import random_walk
from quantmind.regime.classifier import RegimeEngine


def _client(payload):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))


def test_fear_greed_parsing_is_lagged_one_day():
    s = alt.fear_greed(client=_client({"data": [{"value": "20", "timestamp": "1735689600"}]}))
    assert s.iloc[0] == 20 and s.index[0] == pd.Timestamp("2025-01-02", tz="UTC")


def test_hyperliquid_funding_scaled_to_8h():
    s = alt.hl_funding("BTCUSDT", client=_client([{"time": 1735689600000, "fundingRate": "0.0000125"}]))
    assert np.isclose(s.iloc[0], 0.0001)


def test_sentiment_agent_is_contrarian_at_extremes_only():
    df = random_walk(600)
    f = build_features(df)
    reg = RegimeEngine().current(f)
    def op(v):
        snap = MarketSnapshot("X", df, sentiment=pd.Series([v] * 10, index=pd.date_range("2024", periods=10, tz="UTC")))
        return SentimentAgent().analyze(AgentContext(snap, f, reg, []))
    assert op(10).score > 0 and op(90).score < 0 and op(50).score == 0


def test_execution_plan():
    assert plan_entry(1, 100, 2, "strong_trend_up", 1.2, 2, 1).order_type == "market"
    p = plan_entry(1, 100, 2, "sideways", 0.5, 2, 1)
    assert p.order_type == "limit" and p.limit_price < 100


def test_proof_math():
    from quantmind.coach.proof import edge_proof, losing_streak_prob, wilson
    assert abs(losing_streak_prob(0.5, 5, 20) - 0.2499) < 0.001          # known value for a fair coin
    lo, hi = wilson(0.5, 100)
    assert abs(lo - 0.4038) < 0.001 and abs(hi - 0.5962) < 0.001       # textbook Wilson interval
    trap = edge_proof(0.9, 200, 0.6, -0.001)                           # 90% win rate, still a loser
    assert trap["expectancy_R"] < 0 and trap["kelly"] < 0
    good = edge_proof(0.45, 300, 1.6, 0.004)
    assert good["proven"] and good["quarter_kelly"] > 0


def test_risk_engine_rises_with_danger():
    from quantmind.coach.risk import assess
    df = random_walk(800)
    f = build_features(df)
    base = dict(features=f, debate={"disagreement": 0.1, "conviction": 0.4},
                portfolio={"equity": 1, "peak_equity": 1, "positions": {}}, direction=1,
                price=float(df.close.iloc[-1]), stop=None, sentiment=50, calendar=[], track_record={},
                strategy_stats=None)
    calm = assess(regime={"label": "strong_trend_up", "novelty": 0.1}, **base)
    panic = assess(regime={"label": "panic_selling", "novelty": 0.99},
                   **{**base, "debate": {"disagreement": 0.6, "conviction": 0.05}, "sentiment": 95})
    assert panic.score > calm.score + 20 and panic.size_multiplier < calm.size_multiplier
    cons = assess(regime={"label": "strong_trend_up", "novelty": 0.1}, profile="conservative", **base)
    assert cons.size_multiplier < calm.size_multiplier
