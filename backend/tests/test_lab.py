"""The lab must find nothing in pure noise and something in data with a real edge."""
import tempfile

import numpy as np

from quantmind.analysis.features import build_features
from quantmind.core.store import Store
from quantmind.data.synthetic import _frame, random_walk
from quantmind.lab.discovery import DiscoveryLab, LabConfig
from quantmind.lab.registry import Registry

CFG = dict(population=30, generations=6, n_perm=30)


def _run(df, seed=0):
    reg = Registry(Store(tempfile.mkdtemp()))
    return DiscoveryLab(df, build_features(df), reg, LabConfig(seed=seed, **CFG)).run(), reg


def test_rejects_everything_on_random_walk():
    for seed in (0, 1):
        summary, reg = _run(random_walk(12000, seed=seed), seed)
        assert summary["validated"] == [], summary["validated"]
        assert reg.trial_count > 100            # trials are counted, never hidden
        assert all(e["status_reason"] for e in reg.data["strategies"].values())


def test_finds_planted_edge():
    rng = np.random.default_rng(3)
    n = 12000
    drift = np.zeros(n)
    for t in range(1, n):
        drift[t] = 0.995 * drift[t - 1] + rng.normal(0, 0.00025)
    df = _frame(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n))), rng, "2021-01-01", "1h")
    summary, _ = _run(df)
    assert summary["validated"], "lab failed to find a genuine persistent-trend edge"
