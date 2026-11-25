"""Core v2: compiled engine, intrabar stops, ML causality, ensemble, advisor."""
import tempfile

import numpy as np
import pandas as pd

from quantmind.analysis.features import build_features
from quantmind.backtest.engine import CostModel, run_backtest
from quantmind.core.store import Store
from quantmind.data.synthetic import random_walk, regime_switching
from quantmind.portfolio.ensemble import ensemble
from quantmind.regime.classifier import REGIME_LIST, _label_row
from quantmind.signals.advisor import Advisor, message_ar
from quantmind.strategies.model import ModelSpec
from quantmind.strategies.spec import Rule, StrategySpec
from quantmind.validation.overfitting import causality_violations

DF, _ = regime_switching(7000, seed=2)
F = build_features(DF)
ZERO = CostModel(fee_bps=0, slippage_bps=0, vol_slippage_mult=0, impact_coef=0)


def test_vectorized_regimes_match_rowwise():
    for i in range(300, len(F), 97):
        assert REGIME_LIST[int(F.regime_id.iloc[i])] == _label_row(F.iloc[i])[0]


def test_feature_vs_feature_and_regime_gene_are_causal():
    s = StrategySpec("x", "hybrid", long_entry=[Rule("mom_24", "xa", ref="h4_mom_42")],
                     short_entry=[Rule("d1_z_20", ">", 1.5)], regimes=["sideways", "weak_trend"], trail=True)
    assert not causality_violations(lambda d: s.positions(build_features(d), d), DF.iloc[-2500:], n_checks=4)
    pos = s.positions(F, DF)
    allowed = np.isin(F.regime_id, [REGIME_LIST.index("sideways"), REGIME_LIST.index("weak_trend")])
    entries = (pos != 0) & (pos.shift(1) == 0)
    assert allowed[entries.to_numpy()].all()


def test_ml_strategy_is_causal():
    m = ModelSpec("ml", feature_set="trend", min_train=2000, retrain_every=500)
    assert not causality_violations(lambda d: m.positions(build_features(d), d), DF.iloc[-3200:],
                                    n_checks=3, warmup=2800)


def test_intrabar_stop_fills_at_stop_or_gap():
    df = pd.DataFrame({"open": [100, 100, 100, 90.0], "high": [101, 101, 101, 91],
                       "low": [99, 99, 99, 85], "close": [100, 100, 100, 88], "volume": [1e9] * 4},
                      index=pd.date_range("2024", periods=4, freq="h", tz="UTC"))
    target = pd.Series([0, 1, 1, 0.0], index=df.index)
    stops = pd.Series([np.nan, 95, 95, np.nan], index=df.index)
    r = run_backtest(df, target, ZERO, stops=stops)
    assert np.isclose(r.returns.iloc[3], 90 / 100 - 1)       # gapped below the stop -> filled at the open


def test_ensemble_and_spot_mode():
    s = StrategySpec("mom", "momentum", long_entry=[Rule("mom_168", ">", 0)], short_entry=[Rule("mom_168", "<", 0)])
    e = {"id": "a", "name": "mom", "family": "momentum", "definition": s.to_dict(),
         "results": {"holdout": {"sharpe": 1.5, "n_bars": 1500}}}
    v = ensemble([e], F, DF, {"momentum": 1.0})
    assert v.target in (-1.0, 0.0, 1.0) and v.components[0]["weight"] == 1.0
    vs = ensemble([e], F, DF, {"momentum": 1.0}, spot_only=True)
    assert vs.target >= 0


def test_advisor_lifecycle_and_honest_record():
    a = Advisor(Store(tempfile.mkdtemp()))
    a.open_signal("BTCUSDT", 1, 100.0, 2.0, 94.0, 5.0, 0.4, ["r"], "صعود قوي", ["s1"])
    assert "اشتري" in message_ar(a.book["outbox"][-1])
    a.on_bar("BTCUSDT", 109.5, 99.0)                         # target1 = 109 -> stop to break-even
    assert a.book["open"]["BTCUSDT"]["target1_hit"] and a.book["open"]["BTCUSDT"]["stop"] == 100.0
    closed = a.on_bar("BTCUSDT", 101, 99.5)                  # touches break-even stop
    assert closed["kind"] == "stop" and closed["return_pct"] < 0  # fees make break-even a small loss
    assert a.track_record()["signals"] == 1


def test_ml_survives_a_data_source_that_starts_late():
    """Real-world case: funding history only covers the last few weeks."""
    f = F.copy()
    f["funding"] = np.nan
    f.iloc[-500:, f.columns.get_loc("funding")] = 0.0001
    m = ModelSpec("flow", feature_set="flow", min_train=2000, retrain_every=1000)
    pos = m.positions(f, DF)
    assert len(pos) == len(DF)
