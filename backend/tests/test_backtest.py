import numpy as np
import pandas as pd

from quantmind.backtest.engine import CostModel, run_backtest
from quantmind.data.synthetic import random_walk

DF = random_walk(2000, seed=3)


def test_signal_is_applied_next_bar():
    # position decided at close t only earns return t->t+1
    target = pd.Series(0.0, index=DF.index)
    target.iloc[100] = 1.0
    r = run_backtest(DF, target, CostModel(fee_bps=0, slippage_bps=0, vol_slippage_mult=0, impact_coef=0))
    exp = DF["close"].iloc[101] / DF["close"].iloc[100] - 1
    assert np.isclose(r.returns.iloc[101], exp)
    assert r.returns.iloc[100] == 0


def test_costs_make_churning_lose():
    churn = pd.Series(np.where(np.arange(len(DF)) % 2, 1.0, -1.0), index=DF.index)
    assert run_backtest(DF, churn).metrics["total_return"] < -0.5


def test_flat_is_zero():
    r = run_backtest(DF, pd.Series(0.0, index=DF.index))
    assert r.metrics["total_return"] == 0 and r.metrics["n_trades"] == 0
