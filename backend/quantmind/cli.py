"""Command line entry points.

  python -m quantmind.cli cycle                 one full intelligence cycle (used hourly by GitHub Actions)
  python -m quantmind.cli cycle --demo          same, on offline synthetic data (clearly labelled)
  python -m quantmind.cli control --level 3     set autonomy level (1-3; 4 needs the human live gate)
  python -m quantmind.cli control --pause "reason" / --resume
"""
from __future__ import annotations

import argparse
import json
import sys

from .core.config import AutonomyLevel, settings
from .core.store import Store
from .data.providers.base import DataProvider


class SyntheticProvider(DataProvider):
    """Offline demo only. Output is tagged source='synthetic'."""
    name = "synthetic"

    def fetch_ohlcv(self, symbol, interval, limit=1000, start=None):
        from .data.synthetic import regime_switching
        df, _ = regime_switching(min(limit, 8000), seed=sum(map(ord, symbol)))
        return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("quantmind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cycle")
    c.add_argument("--demo", action="store_true")
    c.add_argument("--quick-lab", action="store_true", help="smaller search (CI / demo)")
    k = sub.add_parser("control")
    k.add_argument("--level", type=int, choices=[1, 2, 3])
    k.add_argument("--pause")
    k.add_argument("--resume", action="store_true")
    k.add_argument("--research-now", action="store_true")
    k.add_argument("--taken", metavar="SYMBOL", help="user entered the recommended trade")
    k.add_argument("--skipped", metavar="SYMBOL", help="user did not enter the recommended trade")
    a = ap.parse_args(argv)
    store = Store()

    if a.cmd == "control":
        ctl = store.load("control.json", {})
        p = store.load("portfolio.json", None)
        if a.level:
            ctl["level"] = a.level
        if a.pause and p:
            p["halted"], p["halt_reason"] = True, f"paused by user: {a.pause}"
        if a.resume and p:
            p["halted"], p["halt_reason"] = False, None
            p["peak_equity"] = p["equity"]   # a deliberate human reset of the drawdown baseline
        if a.research_now:
            meta = store.load("meta.json", {})
            meta["last_research"] = {}
            store.save("meta.json", meta)
        if a.taken or a.skipped:
            from .signals.advisor import Advisor
            adv = Advisor(store)
            adv.mark_user((a.taken or a.skipped).upper(), bool(a.taken))
            adv.save()
        store.save("control.json", ctl)
        if p:
            store.save("portfolio.json", p)
        print(json.dumps({"control": ctl, "halted": p and p["halted"]}, ensure_ascii=False))
        return 0

    from .lab.discovery import LabConfig
    from .notify import telegram_daily_report, telegram_signals, telegram_summary
    from .orchestrator.engine import Orchestrator

    if a.demo:
        provider = SyntheticProvider()
    else:
        from .data.providers.binance import BinanceProvider
        provider = BinanceProvider()
    cfg = LabConfig(population=24, generations=4, n_perm=25, model_trials=1) if a.quick_lab else LabConfig()
    level = AutonomyLevel(min(int(settings.autonomy_level), 3)) if settings.autonomy_level else None
    orch = Orchestrator(provider, store, level=level, lab_cfg=cfg)
    snap = orch.run_cycle()
    telegram_signals(snap)
    telegram_summary(snap)
    telegram_daily_report(snap)
    print(json.dumps({"cycle": snap["cycle"], "level": snap["level"], "equity": snap["portfolio"]["equity"],
                      "actions": {s: m["action"] for s, m in snap["markets"].items()},
                      "new_signals": len(snap.get("new_signals", [])),
                      "validated": snap["lab"]["validated"], "alerts": snap["alerts"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
