"""Telegram reports (QM_TELEGRAM_TOKEN + QM_TELEGRAM_CHAT_ID)."""
from __future__ import annotations

import os

import httpx


def _send(text: str) -> bool:
    token, chat = os.getenv("QM_TELEGRAM_TOKEN"), os.getenv("QM_TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": chat, "text": text}, timeout=15).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def telegram_signals(snapshot: dict) -> int:
    """Each trading instruction goes out as its own message, immediately - a
    phone notification that appears over whatever trading app is open."""
    from .signals.advisor import message_ar
    return sum(_send(message_ar(ev)) for ev in snapshot.get("new_signals", []))


def telegram_daily_report(snapshot: dict, hour_utc: int = 18) -> bool:
    """One rich report per day: what the brain did, learned and how its signals did."""
    from datetime import datetime, timezone
    if datetime.now(timezone.utc).hour != hour_utc:
        return False
    sig = snapshot.get("signals", {})
    tr = sig.get("track_record", {})
    lab = snapshot["lab"]
    lines = ["📅 التقرير اليومي - QuantMind",
             f"🔔 إشارات مفتوحة: {len(sig.get('open', []))}"]
    if tr.get("signals"):
        lines.append(f"📊 سجل الإشارات: {tr['signals']} إشارة، نجاح {tr['win_rate']:.0%}، "
                     f"متوسط {tr['avg_return_pct']:+.2f}% للصفقة")
    lines.append(f"🧪 المختبر: جرّب {lab['variants_tested']:,} استراتيجية لحد دلوقتي، "
                 f"{lab['validated']} شغالة، {lab['rejected']} اترفضت، {lab['inactive']} اتوقفت")
    if snapshot.get("sentiment") is not None:
        lines.append(f"😨😀 مؤشر الخوف والطمع: {snapshot['sentiment']:.0f}/100")
    for s, m in snapshot["markets"].items():
        lines.append(f"• {s.replace('USDT', '')}: {m['regime']['label_ar']}")
    for i in snapshot["research"]["insights"][-3:]:
        lines.append("💡 " + i["text"][:150])
    lines.append("ℹ️ " + snapshot["disclaimer_ar"])
    return _send("\n".join(lines))


def telegram_summary(snapshot: dict, only_if_changed: bool = True) -> bool:
    token, chat = os.getenv("QM_TELEGRAM_TOKEN"), os.getenv("QM_TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    p = snapshot["portfolio"]
    acted = []  # trade instructions are sent separately by telegram_signals
    lab_new = {s: r for s, r in snapshot["lab"].get("this_cycle", {}).items()}
    if only_if_changed and not acted and not lab_new and not snapshot["alerts"]:
        return False
    lines = [f"🤖 QuantMind | {snapshot['level_ar']}",
             f"💼 المحفظة الوهمية: {p['equity']:,.0f}$ ({p['return_pct']:+.2f}%) | تراجع {p['drawdown_pct']:.1f}%"]
    for s, m in snapshot["markets"].items():
        lines.append(f"• {s}: {m['regime']['label_ar']} → {m['action']}")
    for s, r in lab_new.items():
        lines.append(f"🧪 {s}: جرّب {r['screened']} استراتيجية، نجح منها {len(r['validated'])}")
    for a in snapshot["alerts"][:3]:
        lines.append(("🚨 " if a["level"] == "critical" else "⚠️ ") + a["text"])
    lines.append("ℹ️ " + snapshot["disclaimer_ar"])
    return _send("\n".join(lines))
