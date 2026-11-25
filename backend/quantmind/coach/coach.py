"""The Coach: one clear instruction per coin, every hour.

  ✅ BUY        "اعمل: اشتري"            opportunity + risk acceptable
  ⛔ DONT       "متعملش"                 opportunity exists but risk too high
  ⏸️ WAIT       "استنى"                  no proven edge right now
  🚫 AVOID      "متشتريش"                market is weak / dangerous
  🤝 HOLD       "خليك في الصفقة"          open trade still valid
  ⬆️ RAISE_STOP "ارفع وقف الخسارة"        lock in profit
  ✂️ REDUCE     "قلّل الصفقة"             risk jumped while in a trade
  🔴 CLOSE      "اقفل الصفقة"
Plus a general do / don't list for the user's whole situation.
"""
from __future__ import annotations

from .proof import proof_lines_ar
from .risk import BLOCK_AT, CAUTION_AT, PROFILE_AR

ICON = {"BUY": "✅", "DONT": "⛔", "WAIT": "⏸️", "AVOID": "🚫", "HOLD": "🤝", "RAISE_STOP": "⬆️",
        "REDUCE": "✂️", "CLOSE": "🔴"}


def _money(pct_of_capital: float, profile: dict) -> str:
    cap = profile.get("capital")
    if not cap:
        return f"{pct_of_capital:.1f}% من فلوسك"
    return f"{pct_of_capital:.1f}% من فلوسك = {cap * pct_of_capital / 100:,.0f} {profile.get('currency', 'USD')}"


def verdict(*, symbol: str, kind: str, risk: dict | None, profile: dict, reason_ar: str = "",
            size_pct: float | None = None, stop: float | None = None, new_stop: float | None = None) -> dict:
    coin = symbol.replace("USDT", "")
    titles = {
        "BUY": f"اعمل: اشتري {coin}", "DONT": f"متعملش: متدخلش {coin} دلوقتي", "WAIT": f"استنى في {coin}",
        "AVOID": f"متشتريش {coin}", "HOLD": f"خليك في صفقة {coin}", "RAISE_STOP": f"ارفع وقف الخسارة في {coin}",
        "REDUCE": f"قلّل صفقة {coin}", "CLOSE": f"اقفل صفقة {coin}",
    }
    lines = []
    if reason_ar:
        lines.append(reason_ar)
    if risk:
        lines.append(f"نسبة الخطر: {risk['score']:.0f}% ({risk['level_ar']})"
                     + (f". أهم الأسباب: {', '.join(risk['top_reasons_ar'])}" if risk["top_reasons_ar"] else ""))
    if kind == "BUY" and size_pct is not None and risk:
        loss = size_pct * risk["extra"]["stop_pct"] / 100
        risk["loss_if_stopped_pct"] = round(loss, 2)
        lines.append(f"ادخل بـ {_money(size_pct, profile)}.")
        lines.append(f"لو وقف الخسارة اتضرب هتخسر حوالي {_money(loss, profile)} بس.")
        pr = (risk.get("extra") or {}).get("proof")
        if pr:
            lines += proof_lines_ar(pr)
        if risk.get("loss_probability") is not None:
            lines.append(f"تاريخيًا الصفقات دي خسرت {risk['loss_probability']:.0%} من المرات ({risk['loss_probability_basis']}).")
    if kind == "RAISE_STOP" and new_stop:
        lines.append(f"غيّر أمر وقف الخسارة من {stop:,.4g} لـ {new_stop:,.4g}. كده لو السعر رجع، مكسبك محفوظ.")
    if kind == "REDUCE":
        lines.append("بيع نص الكمية دلوقتي وسيب النص التاني بوقف الخسارة.")
    return {"symbol": symbol, "kind": kind, "icon": ICON[kind], "title_ar": titles[kind], "lines_ar": lines,
            "risk": risk}


def general_guidance(portfolio: dict, risks: dict[str, dict], profile: dict, open_signals: int) -> dict:
    """Do / don't list for the user's whole situation right now."""
    do, dont = [], []
    scores = [r["score"] for r in risks.values() if r]
    avg = sum(scores) / len(scores) if scores else 50
    if avg >= BLOCK_AT:
        dont.append("السوق كله خطر النهارده. متفتحش أي صفقة جديدة.")
    elif avg >= CAUTION_AT:
        do.append("السوق متوتر: لو دخلت، ادخل بنص الحجم المقترح.")
    else:
        do.append("السوق هادي نسبيًا. نفّذ الإشارات بالحجم المكتوب بس.")
    if portfolio.get("consecutive_losses", 0) >= 3:
        dont.append(f"خسرت {portfolio['consecutive_losses']} صفقات ورا بعض. متكبّرش الصفقة عشان تعوّض، "
                    "وخد راحة يوم.")
    if open_signals >= 3:
        dont.append("عندك 3 صفقات مفتوحة أو أكتر. متفتحش صفقات جديدة لحد ما واحدة تتقفل.")
    if not profile.get("capital"):
        do.append("اكتب رأس مالك في الإعدادات عشان أقولك المبالغ بالظبط بدل النسب.")
    do += ["حط أمر وقف الخسارة أول ما تدخل أي صفقة.",
           "نفّذ إشارة «اقفل» أول ما توصلك، من غير تردد."]
    dont += ["متدخلش صفقة من غير إشارة من البرنامج أو بعد ما وقتها يخلص.",
             "متستخدمش رافعة مالية (Futures) ولا فلوس مستلفة."]
    avg_ar = "منخفض" if avg < 30 else "متوسط" if avg < CAUTION_AT else "عالي" if avg < BLOCK_AT else "عالي جدًا"
    return {"market_risk": round(avg, 1), "market_risk_ar": avg_ar, "do_ar": do, "dont_ar": dont,
            "profile_ar": PROFILE_AR.get(profile.get("risk", "balanced"), "متوازن")}
