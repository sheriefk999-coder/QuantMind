"""The intelligence cycle:

OBSERVE -> HYPOTHESIZE -> RESEARCH -> EXPERIMENT -> BACKTEST -> VALIDATE
        -> DEPLOY (paper) -> MONITOR -> LEARN -> IMPROVE

One call to `run_cycle()` performs one pass for every symbol. On GitHub
Actions it runs hourly; the full state is written to the state directory and a
compact `snapshot.json` is published for the Android app.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..agents.auditor import PerformanceAuditor
from ..agents.base import AgentContext
from ..agents.debate import DebateEngine
from ..agents.specialists import SPECIALISTS
from ..analysis.features import build_features
from ..backtest.engine import run_backtest
from ..core.config import AutonomyLevel, settings
from ..core.events import bus
from ..core.logging import get_logger
from ..core.store import Store, now_iso
from ..data.providers.base import DataProvider, MarketSnapshot
from ..forecast.probabilistic import calibrated_up_probability, regime_scenarios
from ..lab.discovery import DiscoveryLab, LabConfig
from ..lab.registry import Registry
from ..regime.classifier import RegimeEngine
from ..research.researcher import Researcher, decay_check
from ..risk.manager import RiskManager, TradeIntent
from ..strategies.spec import Status, spec_from_dict
from .narrator import explain_ar

log = get_logger("orchestrator")
LEVEL_AR = {1: "وضع البحث فقط", 2: "وضع الإشارات", 3: "تداول وهمي (Paper)", 4: "تداول حقيقي"}


class Orchestrator:
    def __init__(self, provider: DataProvider, store: Store | None = None,
                 level: AutonomyLevel | None = None, symbols: list[str] | None = None,
                 lab_cfg: LabConfig | None = None, research_every_hours: int = 24):
        self.provider = provider
        self.store = store or Store()
        self.control = self.store.load("control.json", {})
        self.level = AutonomyLevel(self.control.get("level") or level or settings.autonomy_level)
        if self.level == AutonomyLevel.LIVE and not settings.live_trading_enabled:
            self.level = AutonomyLevel.PAPER   # cannot escalate to LIVE without the human gate
        self.symbols = symbols or settings.symbols
        self.registry = Registry(self.store)
        self.auditor = PerformanceAuditor(self.store)
        self.researcher = Researcher(self.store)
        self.risk = RiskManager()
        self.regimes = RegimeEngine()
        self.lab_cfg = lab_cfg or LabConfig()
        self.research_every = timedelta(hours=research_every_hours)
        from ..execution.paper import PaperBroker
        from ..signals.advisor import Advisor
        self.broker = PaperBroker(self.store)
        self.advisor = Advisor(self.store)
        self._sentiment, self._oi_now = None, {}
        self.meta = self.store.load("meta.json", {"last_research": {}, "learned_prefs": {}, "cycles": 0})

    # ------------------------------------------------------------------ data
    def observe(self, symbol: str) -> MarketSnapshot:
        df = self.provider.fetch_ohlcv(symbol, settings.interval, limit=settings.history_bars)
        snap = MarketSnapshot(symbol, df)
        snap.funding = self.provider.fetch_funding(symbol)
        snap.open_interest = self.provider.fetch_open_interest(symbol, settings.interval)
        if snap.open_interest is None and self._oi_now:
            snap.open_interest = self._own_oi_history(symbol)
        snap.orderbook = self.provider.fetch_orderbook(symbol)
        snap.calendar = self.store.load("calendar.json", [])
        snap.sentiment = self._sentiment
        return snap

    def _load_alt_data(self) -> None:
        """Sentiment + OI snapshot, once per cycle (skipped for offline demo data)."""
        self._sentiment, self._oi_now = None, {}
        if self.provider.name == "synthetic":
            return
        from ..data.providers.alt import fear_greed, hl_open_interest_now
        self._sentiment = fear_greed()
        self._oi_now = hl_open_interest_now()

    def _own_oi_history(self, symbol: str) -> pd.Series | None:
        """Open-interest history built by QuantMind itself: one snapshot per cycle."""
        coin = symbol.replace("USDT", "")
        if coin not in self._oi_now:
            return None
        hist = self.store.load("oi_history.json", {})
        hist.setdefault(coin, []).append([now_iso(), self._oi_now[coin]])
        hist[coin] = hist[coin][-2000:]
        self.store.save("oi_history.json", hist)
        s = pd.Series({pd.Timestamp(t): v for t, v in hist[coin]}, name="open_interest")
        return s if len(s) > 24 else None

    # ------------------------------------------------------------- research
    def research(self, symbol: str, snap: MarketSnapshot, f: pd.DataFrame) -> dict | None:
        last = self.meta["last_research"].get(symbol)
        due = last is None or datetime.fromisoformat(last) + self.research_every <= datetime.now(timezone.utc)
        if not due:
            return None
        self.researcher.interaction_scan(f, snap.ohlcv["close"], symbol)
        hypotheses = self.researcher.pop_experiments(5, symbol)
        lab = DiscoveryLab(snap.ohlcv, f, self.registry, self.lab_cfg, settings.interval, symbol)
        summary = lab.run(extra_seeds=hypotheses)
        self.meta["last_research"][symbol] = now_iso()
        self.researcher.note(f"{symbol}: tested {summary['screened']} strategy variants; "
                             f"{len(summary['validated'])} passed every gate, {len(summary['rejected'])} finalists rejected.")
        return summary

    def monitor_decay(self, symbol: str, snap: MarketSnapshot, f: pd.DataFrame) -> None:
        for e in self.registry.by_status(Status.VALIDATED, Status.PAPER):
            if e["results"].get("symbol") != symbol:
                continue
            since = pd.Timestamp(e["history"][-1]["ts"]) if e["status"] == "paper" else pd.Timestamp(e["updated"])
            spec = spec_from_dict(e["definition"])
            pos, stops = spec.simulate(f, snap.ohlcv)
            seg = snap.ohlcv.index >= since
            if seg.sum() < 200:
                continue
            r = run_backtest(snap.ohlcv[seg], pos[seg], stops=stops[seg]).returns
            h = e["results"].get("holdout", {})
            exp = (h.get("sharpe") or 0) / np.sqrt(8760)
            d = decay_check(r, exp * 0.5)   # tolerate live Sharpe at half of backtest
            if d["status"] == "decayed":
                self.registry.set_status(e["id"], Status.INACTIVE, f"performance decay since deployment (p={d['p']:.4f})")
                self.researcher.note(f"{e['name']} stopped working on {symbol}; disabled and queued for re-research.")

    # ------------------------------------------------------------- decision
    def decide(self, symbol: str, snap: MarketSnapshot, f: pd.DataFrame, corr: pd.DataFrame | None) -> dict:
        regime = self.regimes.current(f, learned_prefs=self.meta.get("learned_prefs"))
        validated = self.registry.by_status(Status.VALIDATED, Status.PAPER, Status.LIVE)
        ctx = AgentContext(snap, f, regime, validated)
        opinions = []
        for cls in SPECIALISTS:
            try:
                opinions.append(cls().analyze(ctx))
            except Exception as ex:  # one broken agent must not stop the system
                a = cls()
                opinions.append(a.abstain(f"agent error: {ex}"))
        debate = DebateEngine(self.auditor.reliabilities()).run(ctx, opinions)
        last_bar = snap.ohlcv.iloc[-1]
        price = float(last_bar["close"])
        self.auditor.record(symbol, str(snap.ohlcv.index[-1]), price, debate.opinions)

        action, verdict = "لا شيء", None
        spot = settings.market_type == "spot"
        atr = float(f["atr"].iloc[-1])
        ens = ctx.extras.get("ensemble")
        open_sig = self.advisor.book["open"].get(symbol)
        # ---- 1. manage an open recommendation (and the mirrored paper position)
        if open_sig:
            side = 1 if open_sig["side"] == "buy" else -1
            qr = next((o for o in debate.opinions if o["agent"] == "quant_research" and not o["abstain"]), None)
            why = None
            if qr is None or np.sign(qr["score"]) != side:
                why = "الاستراتيجيات المُختبرة خرجت من الصفقة"
            elif debate.direction == -side:
                why = "الوكلاء اتفقوا إن الاتجاه انعكس"
            elif self.risk.kill_switch_active(self.broker.p):
                why = "إيقاف طوارئ"
            if why and self.level >= AutonomyLevel.SIGNAL:
                self.advisor.close_signal(symbol, price, why)
                action = f"إشارة بيع/إغلاق: {why}"
                if self.level >= AutonomyLevel.PAPER and symbol in self.broker.p["positions"]:
                    rec = self.broker.close(symbol, price, why)
                    self._learn_from_trade(rec)
            else:
                action = "مستمر في الصفقة"
        # ---- 2. new recommendation
        wants = debate.direction != 0 and not (spot and debate.direction < 0)
        if wants and symbol not in self.advisor.book["open"] and self.level >= AutonomyLevel.SIGNAL:
            intent = TradeIntent(symbol, debate.direction, debate.conviction, price, atr)
            verdict = self.risk.evaluate(intent, self.broker.p, snap.ohlcv, corr,
                                         snap.orderbook.spread_bps if snap.orderbook else None, snap.calendar)
            if verdict.approved:
                stop = verdict.stop_price
                if ens and ens.stop and 0.5 * atr <= debate.direction * (price - ens.stop) <= 6 * atr:
                    stop = ens.stop                      # the strategies' own stop, if sane
                size_pct = verdict.notional / self.broker.p["equity"] * 100
                strat_names = [c["name"] for c in (ens.components if ens else []) if c["position"] == debate.direction]
                from ..agents.execution import plan_entry
                last = f.iloc[-1]
                plan = plan_entry(debate.direction, price, atr, regime.label,
                                  None if pd.isna(last.get("don_20")) else float(last["don_20"]),
                                  snap.orderbook.spread_bps if snap.orderbook else None,
                                  None if pd.isna(last.get("vol_ratio")) else float(last["vol_ratio"]))
                self.advisor.open_signal(symbol, debate.direction, price, atr, stop, size_pct, debate.conviction,
                                         debate.rationale, regime.label_ar, strat_names, plan.to_dict())
                action = f"إشارة {'شراء' if debate.direction > 0 else 'بيع على المكشوف'} بحجم {size_pct:.1f}% من رأس المال"
                if self.level >= AutonomyLevel.PAPER:
                    self.broker.open(symbol, debate.direction, verdict.size_units, price, stop,
                                     {"regime": regime.label, "families": self._supporting_families(validated, symbol)})
                    for e in validated:
                        if e["results"].get("symbol") == symbol and e["status"] == "validated":
                            self.registry.set_status(e["id"], Status.PAPER, "first paper deployment")
            else:
                action = "المخاطر رفضت الصفقة"
        elif debate.direction < 0 and spot and not open_sig:
            action = "السوق ضعيف: متشتريش دلوقتي"

        decision = {
            "symbol": symbol, "bar": str(snap.ohlcv.index[-1]), "price": price, "level": int(self.level),
            "regime": regime.to_dict(), "debate": debate.to_dict(),
            "risk": verdict.to_dict() if verdict else None, "action": action,
            "explanation_ar": explain_ar(symbol, regime.to_dict(), debate.to_dict(),
                                         verdict.to_dict() if verdict else None, action),
        }
        self.store.append("decisions.jsonl", decision, keep_last=3000)
        bus.publish("decision", {"symbol": symbol, "direction": debate.direction, "action": action})
        return decision

    def _supporting_families(self, validated, symbol) -> list[str]:
        return sorted({e["family"] for e in validated if e["results"].get("symbol") == symbol})

    def _learn_from_trade(self, rec: dict | None) -> None:
        """Update per-regime strategy-family preferences from real (paper) outcomes."""
        if not rec or not rec.get("context"):
            return
        reg = rec["context"].get("regime")
        prefs = self.meta["learned_prefs"].setdefault(reg, {})
        win = 1.0 if rec["pnl"] > 0 else 0.0
        for fam in rec["context"].get("families", []):
            prefs[fam] = round(0.9 * prefs.get(fam, 0.5) + 0.1 * win, 4)

    # ----------------------------------------------------------------- cycle
    def run_cycle(self) -> dict:
        started = now_iso()
        self.meta["cycles"] += 1
        decisions, snaps, errors, research = {}, {}, [], {}
        self._load_alt_data()
        for sym in self.symbols:
            try:
                snaps[sym] = self.observe(sym)
            except Exception as ex:
                errors.append(f"{sym}: data error {ex}")
        closes = {s: sn.ohlcv["close"] for s, sn in snaps.items()}
        for s, sn in snaps.items():
            sn.related = {o: snaps[o].ohlcv for o in snaps if o != s}
        corr = pd.DataFrame({s: np.log(c).diff() for s, c in closes.items()}).tail(720).corr() if closes else None
        # mark-to-market + stops first: protect capital before thinking about new trades
        prices = {s: float(sn.ohlcv["close"].iloc[-1]) for s, sn in snaps.items()}
        stopped = self.broker.check_stops(prices, {s: float(sn.ohlcv["low"].iloc[-1]) for s, sn in snaps.items()},
                                          {s: float(sn.ohlcv["high"].iloc[-1]) for s, sn in snaps.items()})
        for rec in stopped:
            self._learn_from_trade(rec)
        self.broker.mark(prices)
        if (b := self.risk.portfolio_breach(self.broker.p)) and not self.broker.p["halted"]:
            for s in list(self.broker.p["positions"]):
                self.broker.close(s, prices[s], "emergency shutdown")
            self.broker.halt(b)
            self.researcher.note("🚨 Emergency shutdown: " + b)
        forecasts = {}
        for sym, snap in snaps.items():
            try:
                f = build_features(snap.ohlcv, extra={"funding": snap.funding} if snap.funding is not None else None)
                self.auditor.resolve(sym, snap.ohlcv["close"])
                sig = self.advisor.book["open"].get(sym)
                if sig:
                    since = pd.Timestamp(sig.get("checked_until") or sig["opened"])
                    new_bars = snap.ohlcv[snap.ohlcv.index >= since.floor("h")]
                    if len(new_bars):
                        sig["checked_until"] = str(new_bars.index[-1] + pd.Timedelta(hours=1))
                        self.advisor.on_bar(sym, float(new_bars["high"].max()), float(new_bars["low"].min()))
                if (r := self.research(sym, snap, f)):
                    research[sym] = r
                self.monitor_decay(sym, snap, f)
                decisions[sym] = self.decide(sym, snap, f, corr)
                reg = decisions[sym]["regime"]["label"]
                forecasts[sym] = {"scenarios": regime_scenarios(snap.ohlcv, f, reg),
                                  "model": calibrated_up_probability(f, snap.ohlcv["close"])}
            except Exception as ex:
                errors.append(f"{sym}: {ex}")
                log.error("cycle error", extra={"data": {"symbol": sym, "tb": traceback.format_exc()}})
        if len(closes) > 1:
            breaks = self.researcher.correlation_breaks(closes)
            for b in breaks:
                self.researcher.note(b)
            if self.meta["cycles"] % 24 == 1:
                self.researcher.lead_lag_scan(closes)
        self.broker.mark(prices)
        snapshot = self.snapshot(decisions, forecasts, research, errors, started, snaps)
        snapshot["new_signals"] = self.advisor.drain_outbox()
        for x in (self.registry, self.auditor, self.researcher, self.broker, self.advisor):
            x.save()
        self.store.save("meta.json", self.meta)
        self.store.save("snapshot.json", snapshot)
        return snapshot

    # -------------------------------------------------------------- publish
    def snapshot(self, decisions, forecasts, research, errors, started, snaps) -> dict:
        p = self.broker.p
        source = next((sn.ohlcv.attrs.get("source", "market") for sn in snaps.values()), "market")
        alerts = []
        if p["halted"]:
            alerts.append({"level": "critical", "text": f"التداول متوقف: {p['halt_reason']}"})
        dd = p["equity"] / p["peak_equity"] - 1
        if dd < -0.05:
            alerts.append({"level": "warning", "text": f"تراجع المحفظة {dd:.1%} من أعلى قيمة"})
        for e in errors:
            alerts.append({"level": "warning", "text": e})
        validated = self.registry.by_status(Status.VALIDATED, Status.PAPER, Status.LIVE)
        all_s = self.registry.data["strategies"].values()
        return {
            "generated": now_iso(), "cycle_started": started, "cycle": self.meta["cycles"],
            "data_source": source,
            "level": int(self.level), "level_ar": LEVEL_AR[int(self.level)],
            "disclaimer_ar": "كل الأرقام هنا من تداول وهمي أو اختبارات تاريخية، ومفيش أي ضمان إنها تتكرر بفلوس حقيقية.",
            "portfolio": {
                "equity": round(p["equity"], 2), "start_equity": p["start_equity"], "cash": round(p["cash"], 2),
                "return_pct": round((p["equity"] / p["start_equity"] - 1) * 100, 2),
                "drawdown_pct": round(dd * 100, 2), "day_pnl": round(p["day_pnl"], 2),
                "n_trades": p["n_trades"], "win_rate": round(p["wins"] / p["n_trades"], 3) if p["n_trades"] else None,
                "positions": {s: {"side": "شراء" if v["qty"] > 0 else "بيع", "entry": round(v["entry"], 4),
                                  "last": v["last_price"], "stop": round(v["stop"], 4),
                                  "pnl": round(v["qty"] * (v["last_price"] - v["entry"]), 2)}
                              for s, v in p["positions"].items()},
                "equity_curve": p["equity_curve"][-500:], "halted": p["halted"],
            },
            "markets": {s: {"price": d["price"], "regime": d["regime"], "direction": d["debate"]["direction"],
                            "conviction": d["debate"]["conviction"], "action": d["action"],
                            "explanation_ar": d["explanation_ar"], "opinions": d["debate"]["opinions"],
                            "transcript": d["debate"]["transcript"], "risk": d["risk"],
                            "forecast": forecasts.get(s)} for s, d in decisions.items()},
            "lab": {
                "strategies_total": len(self.registry.data["strategies"]),
                "variants_tested": self.registry.trial_count,
                "validated": len(validated),
                "rejected": sum(1 for e in all_s if e["status"] == "rejected"),
                "inactive": sum(1 for e in all_s if e["status"] == "inactive"),
                "leaderboard": self.registry.leaderboard(15),
                "this_cycle": research,
                "active_strategies": [{"name": e["name"], "family": e["family"], "status": e["status"],
                                       "symbol": e["results"].get("symbol"), "math": e["math"],
                                       "holdout_sharpe": (e["results"].get("holdout") or {}).get("sharpe"),
                                       "stability": e["results"].get("stability"),
                                       "monte_carlo": e["results"].get("monte_carlo")} for e in validated],
            },
            "signals": {"open": list(self.advisor.book["open"].values()),
                        "history": self.advisor.book["history"][-50:][::-1],
                        "track_record": self.advisor.track_record(),
                        "market_type": settings.market_type},
            "sentiment": None if self._sentiment is None else float(self._sentiment.iloc[-1]),
            "agents": self.auditor.report(),
            "research": {"backlog": [{k: b[k] for k in ("title", "kind", "priority")}
                                     for b in self.researcher.data["backlog"][:15]],
                         "insights": self.researcher.data["insights"][-30:]},
            "learning_history": self.store.tail("model_changes.jsonl", 40),
            "recent_trades": self.store.tail("trades.jsonl", 30),
            "alerts": alerts,
        }
