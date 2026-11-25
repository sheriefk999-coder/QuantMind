"""Paper broker: simulated fills on REAL market prices, with fees and slippage
modelled pessimistically. State is persisted so the paper account survives
between hourly runs."""
from __future__ import annotations

from ..core.config import settings
from ..core.store import Store, now_iso


class PaperBroker:
    def __init__(self, store: Store):
        self.store = store
        self.p = store.load("portfolio.json", {
            "cash": settings.paper_starting_equity, "equity": settings.paper_starting_equity,
            "peak_equity": settings.paper_starting_equity, "start_equity": settings.paper_starting_equity,
            "positions": {}, "day": now_iso()[:10], "day_start_equity": settings.paper_starting_equity,
            "day_pnl": 0.0, "consecutive_losses": 0, "last_loss_ts": None, "halted": False,
            "halt_reason": None, "equity_curve": [], "realized_pnl": 0.0, "n_trades": 0, "wins": 0,
        })

    def _cost(self, notional: float) -> float:
        return abs(notional) * (settings.fee_bps + settings.slippage_bps * 2) / 1e4

    def mark(self, prices: dict[str, float]) -> None:
        p = self.p
        for sym, pos in p["positions"].items():
            if sym in prices:
                pos["last_price"] = prices[sym]
        mv = sum(pos["qty"] * pos["last_price"] for pos in p["positions"].values())
        p["equity"] = p["cash"] + mv
        p["peak_equity"] = max(p["peak_equity"], p["equity"])
        today = now_iso()[:10]
        if today != p["day"]:
            p["day"], p["day_start_equity"] = today, p["equity"]
        p["day_pnl"] = p["equity"] - p["day_start_equity"]
        p["equity_curve"].append({"ts": now_iso(), "equity": round(p["equity"], 2)})
        p["equity_curve"] = p["equity_curve"][-3000:]

    def open(self, symbol: str, direction: int, units: float, price: float, stop: float,
             strategy_ctx: dict) -> dict:
        slip = price * settings.slippage_bps / 1e4 * direction
        fill = price + slip
        qty = units * direction
        self.p["cash"] -= qty * fill + self._cost(qty * fill)
        self.p["positions"][symbol] = {"qty": qty, "entry": fill, "last_price": price, "stop": stop,
                                       "opened": now_iso(), "context": strategy_ctx}
        fill_rec = {"action": "open", "symbol": symbol, "side": "long" if direction > 0 else "short",
                    "qty": qty, "price": fill, "stop": stop}
        self.store.append("trades.jsonl", fill_rec)
        return fill_rec

    def close(self, symbol: str, price: float, reason: str) -> dict | None:
        pos = self.p["positions"].pop(symbol, None)
        if not pos:
            return None
        direction = 1 if pos["qty"] > 0 else -1
        fill = price - price * settings.slippage_bps / 1e4 * direction
        self.p["cash"] += pos["qty"] * fill - self._cost(pos["qty"] * fill)
        pnl = pos["qty"] * (fill - pos["entry"]) - self._cost(pos["qty"] * fill) - self._cost(pos["qty"] * pos["entry"])
        self.p["realized_pnl"] += pnl
        self.p["n_trades"] += 1
        if pnl > 0:
            self.p["wins"] += 1
            self.p["consecutive_losses"] = 0
        else:
            self.p["consecutive_losses"] += 1
            self.p["last_loss_ts"] = now_iso()
        rec = {"action": "close", "symbol": symbol, "price": fill, "pnl": round(pnl, 2), "reason": reason,
               "entry": pos["entry"], "opened": pos["opened"], "context": pos.get("context")}
        self.store.append("trades.jsonl", rec)
        return rec

    def check_stops(self, prices: dict[str, float], lows: dict[str, float], highs: dict[str, float]) -> list[dict]:
        out = []
        for sym, pos in list(self.p["positions"].items()):
            if sym not in prices:
                continue
            if pos["qty"] > 0 and lows.get(sym, prices[sym]) <= pos["stop"]:
                out.append(self.close(sym, min(pos["stop"], prices[sym]), "stop-loss"))
            elif pos["qty"] < 0 and highs.get(sym, prices[sym]) >= pos["stop"]:
                out.append(self.close(sym, max(pos["stop"], prices[sym]), "stop-loss"))
        return [o for o in out if o]

    def halt(self, reason: str) -> None:
        self.p["halted"], self.p["halt_reason"] = True, reason

    def save(self) -> None:
        self.store.save("portfolio.json", self.p)
