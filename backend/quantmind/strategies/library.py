"""Seed strategies + benchmarks. These are starting points for the lab to mutate
and the yardsticks every discovered strategy must beat - not trusted alphas."""
from __future__ import annotations

from .spec import Rule, StrategySpec


def seeds() -> list[StrategySpec]:
    return [
        StrategySpec("TS momentum", "momentum",
                     long_entry=[Rule("mom_168", ">", 0.5)], short_entry=[Rule("mom_168", "<", -0.5)],
                     exit_rules=[Rule("mom_24", "xb", -1.0), Rule("mom_24", "xa", 1.0)], max_hold=168, stop_atr=4),
        StrategySpec("Z mean reversion", "mean_reversion",
                     long_entry=[Rule("z_close_50", "<", -2.0), Rule("er_20", "<", 0.3)],
                     short_entry=[Rule("z_close_50", ">", 2.0), Rule("er_20", "<", 0.3)],
                     exit_rules=[Rule("z_close_50", "xa", 0.0), Rule("z_close_50", "xb", 0.0)],
                     max_hold=48, stop_atr=3),
        StrategySpec("Donchian breakout", "breakout",
                     long_entry=[Rule("don_20", ">", 1.0), Rule("volume_z", ">", 1.0)],
                     short_entry=[Rule("don_20", "<", 0.0), Rule("volume_z", ">", 1.0)],
                     exit_rules=[Rule("don_20", "xb", 0.5), Rule("don_20", "xa", 0.5)], max_hold=72, stop_atr=2.5),
        StrategySpec("Vol-squeeze expansion", "volatility",
                     long_entry=[Rule("vol_pct_500", "<", 0.2), Rule("pctb_20", ">", 1.0)],
                     short_entry=[Rule("vol_pct_500", "<", 0.2), Rule("pctb_20", "<", 0.0)],
                     exit_rules=[Rule("vol_ratio", ">", 2.0)], max_hold=48, stop_atr=2),
    ]


def buy_and_hold() -> StrategySpec:
    """Benchmark. Discovered strategies are compared against this AND against zero."""
    return StrategySpec("Buy & hold", "benchmark", long_entry=[Rule("atr", ">", 0.0)],
                        max_hold=10**9, stop_atr=1e9, id="benchmark_bh")
