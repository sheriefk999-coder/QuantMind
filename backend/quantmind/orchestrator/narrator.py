"""Turns a structured decision into plain Arabic for a non-trader.

Template-based by default (free, deterministic). If QM_ANTHROPIC_API_KEY is
set, an LLM rewrites the same facts more naturally. The LLM only EXPLAINS: it
receives the finished decision and cannot change it.
"""
from __future__ import annotations

import json

import httpx

from ..core.config import settings

DIR_AR = {1: "شراء (توقع صعود)", -1: "بيع (توقع هبوط)", 0: "انتظار بدون صفقة"}


def explain_ar(symbol: str, regime: dict, debate: dict, risk: dict | None, action: str) -> str:
    lines = [f"🔎 {symbol}: السوق دلوقتي في حالة «{regime['label_ar']}»."]
    if regime.get("novelty", 0) > 0.9:
        lines.append("⚠️ الوضع الحالي غريب مقارنة بالتاريخ، فالبرنامج بيدرسه بحذر.")
    d = debate["direction"]
    lines.append(f"🧠 نتيجة نقاش الوكلاء: {DIR_AR[d]}، قوة الإشارة {debate['conviction']:.0%}، "
                 f"والاختلاف بينهم {debate['disagreement']:.0%}.")
    if debate.get("stand_aside_reason"):
        lines.append(f"✋ السبب: {_why_ar(debate['stand_aside_reason'])}")
    if debate.get("challenges"):
        lines.append(f"🥊 المحلل المعاكس اعترض: {len(debate['challenges'])} اعتراض.")
    if risk is not None:
        lines.append("🛡️ مدير المخاطر: " + ("وافق" if risk["approved"] else "رفض") +
                     (f" ({risk['reasons'][0]})" if risk["reasons"] else ""))
    lines.append(f"✅ الإجراء: {action}")
    text = "\n".join(lines)
    if settings.anthropic_api_key:
        text = _llm_polish(text, debate) or text
    return text


def _why_ar(reason: str) -> str:
    table = {
        "no validated strategy": "لسه مفيش استراتيجية نجحت في كل الاختبارات على العملة دي.",
        "do not support": "الاستراتيجيات المثبتة مش موافقة على الاتجاه ده.",
        "below threshold": "الإشارة ضعيفة ومش تستاهل المخاطرة.",
        "disagree too much": "الوكلاء مختلفين جدًا مع بعض.",
        "unlike anything": "حالة السوق مختلفة عن أي حاجة شافها قبل كده.",
    }
    for k, v in table.items():
        if k in reason:
            return v
    return reason


def _llm_polish(text: str, debate: dict) -> str | None:
    try:
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=30, headers={
            "x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"}, json={
            "model": settings.llm_model, "max_tokens": 400,
            "messages": [{"role": "user", "content":
                          "أعد صياغة هذا الملخص بالعامية المصرية لشخص لا يفهم في التداول، في 4 أسطر كحد أقصى. "
                          "لا تضف أي معلومة أو توقع غير موجود، ولا تغيّر القرار.\n\n" + text +
                          "\n\nتفاصيل النقاش (للسياق فقط):\n" + json.dumps(debate.get("transcript", [])[:20],
                                                                       ensure_ascii=False)}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception:
        return None
