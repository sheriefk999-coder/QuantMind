"""Monte Carlo risk analysis: what range of outcomes is consistent with the
observed returns? Block bootstrap keeps short-range dependence (vol clustering)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap(ret: pd.Series, n_paths: int = 1000, block: int = 24, horizon: int | None = None,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = ret.dropna().to_numpy()
    horizon = horizon or len(r)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, max(1, len(r) - block), size=(n_paths, n_blocks))
    paths = np.stack([np.concatenate([r[s: s + block] for s in row])[:horizon] for row in starts])
    return paths


def drawdown_distribution(ret: pd.Series, **kw) -> dict:
    paths = block_bootstrap(ret, **kw)
    eq = np.cumprod(1 + paths, axis=1)
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    final = eq[:, -1] - 1
    return {
        "max_dd_p50": float(np.percentile(dd, 50)),
        "max_dd_p95": float(np.percentile(dd, 5)),   # 95th-worst
        "max_dd_p99": float(np.percentile(dd, 1)),
        "return_p05": float(np.percentile(final, 5)),
        "return_p50": float(np.percentile(final, 50)),
        "return_p95": float(np.percentile(final, 95)),
        "prob_loss": float((final < 0).mean()),
    }
