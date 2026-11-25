"""Trading Assistant: turns the brain's decisions into clear, manual
instructions the user can follow in ANY trading app.

  🟢 BUY   - entry zone, size as % of capital, stop-loss, first target, expiry
  🔴 SELL  - close the position now, with the reason and the result
  ⚠️ STOP  - price hit the stop-loss: sell now if still holding

Every signal is recorded and resolved (win / loss) as if it had been followed
at the stated prices, net of fees, so the track record is honest and visible
before the user trusts it with real money.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from ..core.config import settings
from ..core.store import Store, now_iso


def _fmt(p: float) -> str:
    return f"{p:,.2f}" if p >= 100 else f"{p:,.4f}" if p >= 1 else f"{p:.6f}"


class Advisor:
    def __init__(self, store: Store):
        self.store = store
        self.book = store.load("signals.json", {"open": {}, "history": [], "outbox": []})

    # ------------------------------------------------------------------ open
    def open_signal(self, symbol: str, direction: int, price: float, atr: float, stop: float, size_pct: float,
                    conviction: float, reasons: list[str], regime_ar: str, strategies: list[str],
                    execution: dict | None = None) -> dict:
        risk = abs(price - stop)
        valid_until = datetime.now(timezone.utc) + timedelta(hours=settings.signal_valid_hours)
        sig = {
            "id": uuid.uuid4().hex[:8], "symbol": symbol, "side": "buy" if direction > 0 else "short",
            "price": price, "entry_max": price + 0.3 * atr if direction > 0 else price - 0.3 * atr,
            "stop": stop, "target1": price + direction * 1.5 * risk, "target2": price + direction * 3 * risk,
            "size_pct": round(size_pct, 1), "conviction": round(conviction, 2), "regime_ar": regime_ar,
            "reasons": reasons[:3], "strategies": strategies[:3], "opened": now_iso(),
            "valid_until": valid_until.isoformat(timespec="minutes"), "user_in": None,
            "target1_hit": False, "status": "open", "execution": execution,
        }
        self.book["open"][symbol] = sig
        self.book["outbox"].append({"type": "open", **sig})
        return sig

    # ----------------------------------------------------------------- close
    def close_signal(self, symbol: str, price: float, reason_ar: str, kind: str = "close") -> dict | None:
        sig = self.book["open"].pop(symbol, None)
        if not sig:
            return None
        d = 1 if sig["side"] == "buy" else -1
        fees = 2 * (settings.fee_bps + settings.slippage_bps) / 1e4
        ret = d * (price / sig["price"] - 1) - fees
        sig.update({"status": "closed", "closed": now_iso(), "exit_price": price, "exit_reason": reason_ar,
                    "return_pct": round(ret * 100, 2), "win": ret > 0, "kind": kind})
        self.book["history"].append(sig)
        self.book["history"] = self.book["history"][-500:]
        self.book["outbox"].append({"type": kind, **sig})
        return sig

    def on_bar(self, symbol: str, bar_high: float, bar_low: float) -> dict | None:
        """Intrabar events since the last cycle: stop-loss or first target touched."""
        sig = self.book["open"].get(symbol)
        if not sig:
            return None
        d = 1 if sig["side"] == "buy" else -1
        if (d > 0 and bar_low <= sig["stop"]) or (d < 0 and bar_high >= sig["stop"]):
            return self.close_signal(symbol, sig["stop"], "السعر وصل لوقف الخسارة", kind="stop")
        if not sig["target1_hit"] and ((d > 0 and bar_high >= sig["target1"]) or (d < 0 and bar_low <= sig["target1"])):
            sig["target1_hit"] = True
            # protect the trade: move the stop to break-even
            sig["stop"] = sig["price"]
            self.book["outbox"].append({"type": "target1", **sig})
        return None

    def mark_user(self, symbol: str, entered: bool) -> None:
        if symbol in self.book["open"]:
            self.book["open"][symbol]["user_in"] = entered

    # ----------------------------------------------------------------- stats
    def track_record(self) -> dict:
        h = self.book["history"]
        if not h:
            return {"signals": 0}
        r = np.array([x["return_pct"] for x in h]) / 100
        return {"signals": len(h), "wins": int((r > 0).sum()), "win_rate": round(float((r > 0).mean()), 3),
                "avg_return_pct": round(float(r.mean() * 100), 2),
                "best_pct": round(float(r.max() * 100), 2), "worst_pct": round(float(r.min() * 100), 2),
                "compounded_pct_if_all_taken_at_size": round(
                    float((np.prod(1 + r * np.array([x["size_pct"] / 100 for x in h])) - 1) * 100), 2)}

    def drain_outbox(self) -> list[dict]:
        out, self.book["outbox"] = self.book["outbox"], []
        return out

    def save(self) -> None:
        self.store.save("signals.json", self.book)


# ------------------------------------------------------------------ messages
def _n_strats(n: int) -> str:
    return {0: "الوكلاء متفقين", 1: "استراتيجية مُختبرة بتقول اشتري", 2: "استراتيجيتين مُختبرتين متفقين"}.get(
        n, f"{n} استراتيجيات مُختبرة متفقة")


def message_ar(ev: dict) -> str:
    s, t = ev["symbol"], ev["type"]
    coin = s.replace("USDT", "")
    if t == "open":
        buy = ev["side"] == "buy"
        return "\n".join([
            f"{'🟢 إشارة شراء' if buy else '🟠 إشارة بيع على المكشوف (فيوتشرز)'}: {coin}",
            f"💵 السعر دلوقتي: {_fmt(ev['price'])}",
            f"✅ {'اشتري لو السعر أقل من' if buy else 'ادخل لو السعر أعلى من'}: {_fmt(ev['entry_max'])}",
            f"📦 حجم الصفقة: {ev['size_pct']}% من رأس مالك",
            *([f"🧾 طريقة الدخول: {ev['execution']['tip_ar']}"] if ev.get("execution") else []),
            f"🛑 وقف الخسارة: {_fmt(ev['stop'])}  (حطه أمر Stop-Loss في التطبيق فورًا)",
            f"🎯 الهدف الأول: {_fmt(ev['target1'])}  •  الهدف التاني: {_fmt(ev['target2'])}",
            f"🧠 ليه؟ السوق في حالة «{ev['regime_ar']}»، و{_n_strats(len(ev['strategies']))} "
            f"(قوة الإشارة {ev['conviction']:.0%}).",
            f"⏳ الإشارة صالحة لحد {ev['valid_until'][11:16]} UTC. بعد كده متدخلش.",
            "📣 هبعتلك أول ما يبقى وقت البيع.",
            "⚠️ مفيش ضمان. ادخل بفلوس تقدر تستغنى عنها بس.",
        ])
    res = f"{ev['return_pct']:+.2f}%"
    if t == "stop":
        return f"⚠️ {coin}: السعر وصل وقف الخسارة ({_fmt(ev['exit_price'])}).\n" \
               f"لو الأمر ما اتنفذش تلقائي، بيع دلوقتي.\nالنتيجة: {res}. الخسارة الصغيرة جزء طبيعي من الخطة."
    if t == "target1":
        return f"🎯 {coin} وصل الهدف الأول ({_fmt(ev['target1'])})!\n" \
               f"ممكن تبيع نص الكمية وتحرك وقف الخسارة لسعر دخولك ({_fmt(ev['price'])}) عشان الصفقة متخسرش."
    return f"🔴 بيع {coin} دلوقتي (اقفل الصفقة)\n💵 السعر: {_fmt(ev['exit_price'])}\n" \
           f"📊 النتيجة لو كنت دخلت: {res}\n🧠 السبب: {ev['exit_reason']}"
