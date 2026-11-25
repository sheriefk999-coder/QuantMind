"""Edge proof: the mathematics behind every recommendation.

Inputs come from the out-of-sample (holdout) trades of the validated
strategies backing a signal - never from the data they were fitted on.

  p      win rate                         n  number of trades
  R      payoff ratio  W / L  =  PF (1-p) / p        (PF = profit factor = Σwins / Σlosses)
         shrunk (pessimistically) for small samples:  R' = min(R, 1 + (R - 1) · n / (n + 30))
  p*     break-even win rate              p* = 1 / (1 + R)
  E      expectancy per trade, in units of the average loss:  E = p R - (1 - p)
  CI     Wilson 95% interval for p        (p̂ + z²/2n ± z √(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
  proof  the edge is "statistically proven" iff  n >= 30  and  CI_low > p*
         (even the pessimistic end of the win-rate interval beats break-even)
  Kelly  optimal fraction of capital to RISK:  f* = p - (1 - p) / R
         QuantMind uses 1/4 Kelly (full Kelly is far too volatile in practice)
  streak probability of >= k losses in a row within the next N trades (exact DP)
"""
from __future__ import annotations

import math


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def losing_streak_prob(p_loss: float, k: int = 5, n: int = 20) -> float:
    """P(at least one run of >= k consecutive losses in n independent trades)."""
    # state = current run length (0..k-1); absorbing when it reaches k
    probs = [1.0] + [0.0] * (k - 1)
    hit = 0.0
    for _ in range(n):
        nxt = [0.0] * k
        for run, pr in enumerate(probs):
            nxt[0] += pr * (1 - p_loss)
            if run + 1 >= k:
                hit += pr * p_loss
            else:
                nxt[run + 1] += pr * p_loss
        probs = nxt
    return hit


def edge_proof(p: float, n: int, profit_factor: float | None, avg_trade: float | None) -> dict | None:
    if n < 5 or p <= 0 or p >= 1 or not profit_factor or not math.isfinite(profit_factor):
        return None
    R_raw = profit_factor * (1 - p) / p
    # R is itself estimated from few trades: shrink it toward 1 (no edge in
    # payoff) so a handful of lucky big wins can't fake a "proof"
    R = min(R_raw, 1 + (R_raw - 1) * n / (n + 30))    # only ever more pessimistic
    p_star = 1 / (1 + R)
    lo, hi = wilson(p, n)
    E = p * R - (1 - p)
    kelly = p - (1 - p) / R if R > 0 else -1.0
    proven = n >= 30 and lo > p_star
    return {
        "win_rate": round(p, 3), "trades": n, "ci95": [round(lo, 3), round(hi, 3)],
        "payoff_ratio": round(R, 2), "payoff_ratio_raw": round(R_raw, 2), "breakeven_win_rate": round(p_star, 3),
        "expectancy_R": round(E, 3), "avg_trade_pct": None if avg_trade is None else round(avg_trade * 100, 3),
        "kelly": round(kelly, 4), "quarter_kelly": round(max(0.0, kelly) / 4, 4),
        "proven": proven, "streak5_in_20": round(losing_streak_prob(1 - p, 5, 20), 3),
        "verdict_ar": ("✅ الميزة مُثبتة إحصائيًا: حتى أسوأ تقدير لنسبة النجاح أعلى من نقطة التعادل"
                       if proven else
                       "⚠️ الميزة إيجابية بس لسه مش مُثبتة إحصائيًا (محتاج صفقات أكتر)، فالحجم اتنصّ" if E > 0 else
                       "❌ التوقع الرياضي سالب: الصفقة دي مش في صالحك"),
    }


def proof_lines_ar(pr: dict) -> list[str]:
    return [
        f"📐 نسبة النجاح p = {pr['win_rate']:.0%} من {pr['trades']} صفقة (مدى الثقة 95%: "
        f"{pr['ci95'][0]:.0%} – {pr['ci95'][1]:.0%})",
        f"📐 المكسب/الخسارة R = {pr['payoff_ratio']:.2f} ← نقطة التعادل p* = 1/(1+R) = {pr['breakeven_win_rate']:.0%}",
        f"📐 التوقع E = p·R − (1−p) = {pr['expectancy_R']:+.2f} من متوسط الخسارة لكل صفقة",
        f"📐 Kelly f* = p − (1−p)/R = {pr['kelly']:.1%} ← البرنامج بيستخدم الربع ({pr['quarter_kelly']:.1%}) كحد أقصى للمخاطرة",
        f"🎲 احتمال تخسر 5 صفقات ورا بعض خلال 20 صفقة: {pr['streak5_in_20']:.0%}",
        pr["verdict_ar"],
    ]
