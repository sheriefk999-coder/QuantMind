from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from ..data.providers.base import MarketSnapshot
from ..regime.classifier import RegimeState


@dataclass
class Opinion:
    agent: str
    score: float                 # -1 strong bearish .. +1 strong bullish
    confidence: float            # 0..1, the agent's own certainty
    style: str                   # trend | reversion | flow | relative | context | evidence
    evidence: list[str] = field(default_factory=list)
    invalidation: str = ""       # what would prove this opinion wrong
    abstain: bool = False        # no usable data -> says so instead of guessing
    adjusted_confidence: float | None = None
    challenges: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentContext:
    snapshot: MarketSnapshot
    features: pd.DataFrame
    regime: RegimeState
    validated_strategies: list[dict]   # registry entries with status validated/paper/live
    extras: dict = field(default_factory=dict)  # shared scratch (e.g. ensemble view for the advisor)


class Agent:
    name = "agent"
    name_ar = "وكيل"
    style = "context"

    def analyze(self, ctx: AgentContext) -> Opinion:  # pragma: no cover
        raise NotImplementedError

    def abstain(self, why: str) -> Opinion:
        return Opinion(self.name, 0.0, 0.0, self.style, [why], abstain=True)


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))
