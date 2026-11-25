"""No look-ahead anywhere: features and strategy signals must not change when future bars are removed."""
import numpy as np
import pandas as pd

from quantmind.analysis.features import build_features
from quantmind.data.synthetic import random_walk
from quantmind.strategies.library import seeds
from quantmind.validation.overfitting import causality_violations

DF = random_walk(1500, seed=7)


def test_every_feature_is_causal():
    full = build_features(DF)
    for t in (400, 777, 1200):
        part = build_features(DF.iloc[: t + 1])
        a, b = full.iloc[t], part.iloc[-1]
        bad = [c for c in full.columns if not np.isclose(a[c], b[c], equal_nan=True, rtol=1e-6, atol=1e-9)]
        assert not bad, f"look-ahead in features {bad} at bar {t}"


def test_seed_strategies_are_causal():
    for s in seeds():
        assert not causality_violations(lambda d: s.positions(build_features(d), d["close"]), DF, n_checks=3)


def test_causality_check_catches_a_cheater():
    cheat = lambda d: np.sign(d["close"].shift(-1) - d["close"]).fillna(0)  # peeks one bar ahead
    assert causality_violations(cheat, DF, n_checks=5)
