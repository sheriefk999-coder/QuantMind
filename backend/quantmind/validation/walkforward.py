"""Data splitting that respects time.

    |------ discovery (in-sample) ------|gap|-- validation --|gap|-- HOLDOUT --|

The holdout is touched exactly once per strategy, at the end. A gap (embargo)
between segments prevents indicator windows straddling the boundary from
leaking information.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: slice
    valid: slice
    holdout: slice


def three_way_split(n: int, train: float = 0.6, valid: float = 0.2, gap: int = 200) -> Split:
    a = int(n * train)
    b = int(n * (train + valid))
    return Split(slice(0, a), slice(a + gap, b), slice(b + gap, n))


def walk_forward_windows(n: int, n_folds: int = 5, min_train: int = 1000, gap: int = 200,
                         anchored: bool = True) -> list[tuple[slice, slice]]:
    """Expanding (anchored) or rolling train windows, each followed by an unseen test window."""
    test_len = (n - min_train - gap) // n_folds
    out = []
    for k in range(n_folds):
        tr_end = min_train + k * test_len
        tr_start = 0 if anchored else max(0, tr_end - min_train)
        te_start = tr_end + gap
        te_end = min(n, te_start + test_len)
        if te_end - te_start < 50:
            break
        out.append((slice(tr_start, tr_end), slice(te_start, te_end)))
    return out


def slice_with_warmup(df: pd.DataFrame, s: slice, warmup: int = 300) -> tuple[pd.DataFrame, int]:
    """Return df from (start - warmup) so indicators are primed, plus the offset
    where the evaluated segment actually begins. Warm-up bars are never scored."""
    start = max(0, (s.start or 0) - warmup)
    return df.iloc[start: s.stop], (s.start or 0) - start
