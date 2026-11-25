"""Compiled position state machine (numba). ~100x faster than pure Python, which
lets the lab search far more strategies in the same hourly budget. Falls back
to plain Python if numba is unavailable (same results, slower)."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def simulate(le, se, ex, allowed, close, high, low, atr, max_hold, stop_atr, trail):
    """Returns (position at close t, stop level active for the NEXT bar, stopped-at-bar flag).

    At bar t:
      1. if a position was carried into bar t, check whether bar t's low/high
         touched the stop -> flat (backtester fills at the stop price, with gap
         handling)
      2. otherwise apply exit rules / max-hold at the close
      3. if flat (and not stopped this bar), check entries at the close
         (entries only when the regime gene allows)
    """
    n = close.shape[0]
    pos = np.zeros(n)
    stop_next = np.full(n, np.nan)
    stopped = np.zeros(n, dtype=np.bool_)
    cur = 0
    stop = np.nan
    held = 0
    for t in range(n):
        just_stopped = False
        if cur != 0:
            held += 1
            if cur > 0 and low[t] <= stop:
                cur = 0
                just_stopped = True
            elif cur < 0 and high[t] >= stop:
                cur = 0
                just_stopped = True
            elif ex[t] or held >= max_hold:
                cur = 0
            elif trail and not np.isnan(atr[t]):
                if cur > 0:
                    stop = max(stop, close[t] - stop_atr * atr[t])
                else:
                    stop = min(stop, close[t] + stop_atr * atr[t])
        if cur == 0 and not just_stopped and allowed[t] and not np.isnan(atr[t]):
            if le[t] and not se[t]:
                cur, held, stop = 1, 0, close[t] - stop_atr * atr[t]
            elif se[t] and not le[t]:
                cur, held, stop = -1, 0, close[t] + stop_atr * atr[t]
        pos[t] = cur
        stopped[t] = just_stopped
        if cur != 0:
            stop_next[t] = stop
    return pos, stop_next, stopped
