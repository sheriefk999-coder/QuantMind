"""Strategy Discovery Lab (v2).

Search space
  * rule strategies: thresholds, feature-vs-feature comparisons (e.g. "1h
    momentum crosses above daily momentum"), multi-timeframe features, regime
    genes (trade only in chosen regimes), fixed or trailing stops
  * ML strategies: walk-forward gradient boosting on different feature sets /
    horizons
  * hypotheses queued by the Researcher, plus seed strategies

Search engines: genetic algorithm (mutation, crossover, immigrants, niche
de-duplication) + Gaussian-process Bayesian optimisation of the finalists.

Every finalist must pass ALL gates, in order (cheapest first):
  1. validation segment: Sharpe > 0, enough trades           (search never saw it)
  2. causality: identical signals on truncated data            (no look-ahead)
  3. stability: positive in >= 4/6 time slices of train+valid  (not one lucky period)
  4. robustness: neighbouring parameters still work           (not a knife-edge fit)
  5. cost stress: still profitable on validation at 2x costs   (survives worse fills)
  6. holdout (touched once): Sharpe > 0, beats buy & hold, enough trades
  7. Deflated Sharpe >= 0.95 given every holdout test ever run
  8. permutation (rules: shuffled prices / ML: shifted timing) p <= 0.05
On a random walk this lab must validate nothing; tests/test_lab.py checks it.
"""
from __future__ import annotations

import copy
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..analysis.features import COMPARABLE_GROUPS, FEATURE_SPACE, build_features
from ..backtest.engine import CostModel, run_backtest
from ..core.events import bus
from ..core.logging import get_logger
from ..regime.classifier import REGIME_LIST
from ..strategies.library import buy_and_hold, seeds
from ..strategies.model import FEATURE_SETS, ModelSpec
from ..strategies.spec import Rule, Status, StrategySpec
from ..validation.overfitting import (causality_violations, deflated_sharpe, permutation_pvalue,
                                      timing_pvalue)
from ..validation.walkforward import three_way_split
from .registry import Registry

log = get_logger("lab")
FEATURES = [k for k in FEATURE_SPACE if k != "hour_sin"]
OPS = [">", "<", "xa", "xb"]
FAMILY_HINT = {"mom": "momentum", "h4_mom": "momentum", "d1_mom": "momentum", "don": "breakout",
               "z_": "mean_reversion", "d1_z": "mean_reversion", "rsi": "mean_reversion", "h4_rsi": "mean_reversion",
               "pctb": "mean_reversion", "vol": "volatility", "er": "momentum", "d1_er": "momentum",
               "adx": "momentum", "h4_adx": "momentum", "struct": "momentum"}
REGIME_GROUPS = [["strong_trend_up", "strong_trend_down", "weak_trend"], ["sideways", "low_volatility"],
                 ["strong_trend_up", "strong_trend_down"], ["high_volatility", "panic_selling", "short_squeeze"],
                 ["sideways", "low_volatility", "accumulation", "distribution"]]


@dataclass
class LabConfig:
    population: int = 60
    generations: int = 12
    elite: int = 8
    top_k_validate: int = 10
    tune_top: int = 3
    model_trials: int = 3              # ML configs tried per run
    min_trades_train: int = 25
    min_trades_holdout: int = 10
    stability_slices: int = 6
    stability_min_positive: int = 4
    robustness_neighbours: int = 8
    robustness_ratio: float = 0.5
    dsr_threshold: float = 0.95
    perm_alpha: float = 0.05
    n_perm: int = 40
    complexity_penalty: float = 0.08
    seed: int = 0


def _family(spec) -> str:
    if getattr(spec, "kind", "rules") == "model":
        return "ml"
    for r in spec.long_entry + spec.short_entry:
        for k in sorted(FAMILY_HINT, key=len, reverse=True):
            if r.feature.startswith(k):
                return FAMILY_HINT[k]
    return "hybrid"


class DiscoveryLab:
    def __init__(self, df: pd.DataFrame, features: pd.DataFrame, registry: Registry,
                 cfg: LabConfig | None = None, interval: str = "1h", symbol: str = "?"):
        self.df, self.f, self.reg = df, features, registry
        self.cfg = cfg or LabConfig()
        self.interval, self.symbol = interval, symbol
        self.rng = random.Random(self.cfg.seed)
        self.split = three_way_split(len(df))
        self.cost = CostModel()
        self.trials_this_run = 0
        self._cache: dict[str, tuple[pd.Series, pd.Series]] = {}
        self._feats = [c for c in FEATURES if c in features]
        self._groups = [[c for c in g if c in features] for g in COMPARABLE_GROUPS]
        self._groups = [g for g in self._groups if len(g) > 1]

    # ------------------------------------------------------------------ eval
    def _sim(self, spec) -> tuple[pd.Series, pd.Series]:
        fp = spec.fingerprint()
        if fp not in self._cache:
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[fp] = spec.simulate(self.f, self.df)
        return self._cache[fp]

    def _metrics(self, spec, seg: slice, cost_mult: float = 1.0) -> tuple[dict, pd.Series]:
        pos, stops = self._sim(spec)
        r = run_backtest(self.df.iloc[seg], pos.iloc[seg], self.cost, self.interval, stops.iloc[seg], cost_mult)
        return r.metrics, r.returns

    def screen(self, spec) -> float:
        self.trials_this_run += 1
        m, _ = self._metrics(spec, self.split.train)
        if m["n_trades"] < self.cfg.min_trades_train:
            return -10.0
        return m["sharpe"] - self.cfg.complexity_penalty * spec.complexity()

    # ------------------------------------------------------------- generate
    def random_rule(self) -> Rule:
        if self._groups and self.rng.random() < 0.25:                   # feature-vs-feature
            a, b = self.rng.sample(self.rng.choice(self._groups), 2)
            return Rule(a, self.rng.choice(OPS), 0.0, ref=b)
        feat = self.rng.choice(self._feats)
        lo, hi = FEATURE_SPACE[feat][1]
        return Rule(feat, self.rng.choice(OPS), round(self.rng.uniform(lo, hi), 3))

    def random_spec(self) -> StrategySpec:
        mode = self.rng.choice(["long", "short", "both", "both"])
        le = [self.random_rule() for _ in range(self.rng.randint(1, 3))] if mode in ("long", "both") else []
        se = [self.random_rule() for _ in range(self.rng.randint(1, 3))] if mode in ("short", "both") else []
        ex = [self.random_rule() for _ in range(self.rng.randint(0, 2))]
        s = StrategySpec(f"GA-{self.rng.randrange(16**6):06x}", "hybrid", le, se, ex,
                         max_hold=self.rng.choice([12, 24, 48, 96, 168]),
                         stop_atr=round(self.rng.uniform(1.5, 5), 2), trail=self.rng.random() < 0.3,
                         regimes=self.rng.choice(REGIME_GROUPS) if self.rng.random() < 0.3 else None,
                         origin="genetic")
        s.family = _family(s)
        return s

    def _child(self, parent_ids: list[str], base: StrategySpec, origin: str = "genetic") -> StrategySpec:
        c = copy.deepcopy(base)
        c.id, c.parent_ids, c.origin, c.version = StrategySpec("", "").id, parent_ids, origin, 1
        c.name = f"GA-{self.rng.randrange(16**6):06x}"
        return c

    def mutate(self, s: StrategySpec) -> StrategySpec:
        c = self._child([s.id], s)
        groups = [g for g in (c.long_entry, c.short_entry, c.exit_rules) if g]
        a = self.rng.random()
        if a < 0.40 and groups:
            r = self.rng.choice(self.rng.choice(groups))
            if r.ref:
                r.ref = self.rng.choice(next((g for g in self._groups if r.feature in g), [r.ref]))
            else:
                lo, hi = FEATURE_SPACE.get(r.feature, ("", (-1, 1)))[1]
                r.threshold = round(float(np.clip(r.threshold + self.rng.gauss(0, (hi - lo) * 0.1), lo, hi)), 3)
        elif a < 0.52 and groups:
            self.rng.choice(self.rng.choice(groups)).op = self.rng.choice(OPS)
        elif a < 0.66:
            self.rng.choice([c.long_entry, c.short_entry, c.exit_rules]).append(self.random_rule())
        elif a < 0.76 and groups:
            g = self.rng.choice(groups)
            if len(g) > 1 or g is c.exit_rules:
                g.pop(self.rng.randrange(len(g)))
        elif a < 0.84:
            c.regimes = None if c.regimes else self.rng.choice(REGIME_GROUPS)
        elif a < 0.90:
            c.trail = not c.trail
        elif a < 0.95:
            c.max_hold = max(4, int(c.max_hold * self.rng.choice([0.5, 0.75, 1.5, 2])))
        else:
            c.stop_atr = round(float(np.clip(c.stop_atr * self.rng.uniform(0.7, 1.4), 1, 8)), 2)
        c.family = _family(c)
        return c

    def crossover(self, a: StrategySpec, b: StrategySpec) -> StrategySpec:
        c = self._child([a.id, b.id], a)
        if self.rng.random() < 0.5:
            c.short_entry = copy.deepcopy(b.short_entry)
        if self.rng.random() < 0.5:
            c.exit_rules = copy.deepcopy(b.exit_rules)
        if self.rng.random() < 0.3 and b.long_entry:          # splice one entry condition
            c.long_entry = (c.long_entry[:1] + [copy.deepcopy(self.rng.choice(b.long_entry))])[:3]
        c.regimes = self.rng.choice([a.regimes, b.regimes])
        c.max_hold = self.rng.choice([a.max_hold, b.max_hold])
        c.family = _family(c)
        return c

    # --------------------------------------------------------------- search
    def genetic_search(self, extra_seeds=None) -> list[tuple[float, StrategySpec]]:
        pop = seeds() + list(extra_seeds or [])
        pop += [self.random_spec() for _ in range(max(0, self.cfg.population - len(pop)))]
        scored: dict[str, tuple[float, StrategySpec]] = {}
        for gen in range(self.cfg.generations):
            for s in pop:
                fp = s.fingerprint()
                if fp not in scored:
                    scored[fp] = (self.screen(s), s)
            ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
            bus.publish("lab.generation", {"symbol": self.symbol, "gen": gen, "best": round(ranked[0][0], 3),
                                           "evaluated": len(scored)})
            # niche protection: at most 2 elites per family so the search keeps diversity
            elite, per_fam = [], {}
            for _, s in ranked:
                if per_fam.get(s.family, 0) < 2:
                    elite.append(s)
                    per_fam[s.family] = per_fam.get(s.family, 0) + 1
                if len(elite) >= self.cfg.elite:
                    break
            nxt = list(elite)
            while len(nxt) < self.cfg.population:
                r = self.rng.random()
                if r < 0.5:
                    nxt.append(self.mutate(self._tournament(ranked)))
                elif r < 0.8:
                    nxt.append(self.crossover(self._tournament(ranked), self._tournament(ranked)))
                else:
                    nxt.append(self.random_spec())
            pop = nxt
        return sorted(scored.values(), key=lambda x: x[0], reverse=True)

    def _tournament(self, ranked, k: int = 3) -> StrategySpec:
        return max(self.rng.sample(ranked, min(k, len(ranked))), key=lambda x: x[0])[1]

    def bayes_tune(self, spec: StrategySpec, n_iter: int = 20) -> StrategySpec:
        """Gaussian-process Bayesian optimisation of thresholds + stop (train only)."""
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        warnings.filterwarnings("ignore", module="sklearn.gaussian_process")
        rules = [r for r in spec.long_entry + spec.short_entry + spec.exit_rules if not r.ref]
        bounds = np.array([FEATURE_SPACE.get(r.feature, ("", (-1, 1)))[1] for r in rules] + [(1.0, 6.0)], float)

        def make(x):
            c = copy.deepcopy(spec)
            for r, v in zip([r for r in c.long_entry + c.short_entry + c.exit_rules if not r.ref], x[:-1]):
                r.threshold = round(float(v), 4)
            c.stop_atr = round(float(x[-1]), 2)
            return c

        X = [np.array([r.threshold for r in rules] + [spec.stop_atr])]
        Y = [self.screen(spec)]
        nprng = np.random.default_rng(self.cfg.seed)
        for _ in range(n_iter):
            gp = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True, alpha=1e-3).fit(np.array(X), Y)
            cand = nprng.uniform(bounds[:, 0], bounds[:, 1], size=(256, len(bounds)))
            mu, sd = gp.predict(cand, return_std=True)
            z = (mu - max(Y)) / np.maximum(sd, 1e-9)
            x = cand[int(np.argmax((mu - max(Y)) * norm.cdf(z) + sd * norm.pdf(z)))]
            X.append(x)
            Y.append(self.screen(make(x)))
        tuned = make(X[int(np.argmax(Y))])
        tuned.id, tuned.parent_ids, tuned.origin = StrategySpec("", "").id, [spec.id], "bayesian"
        tuned.version, tuned.name = spec.version + 1, spec.name + " (BO)"
        return tuned

    def model_candidates(self) -> list[tuple[float, ModelSpec]]:
        if len(self.df) < 6000 or self.cfg.model_trials <= 0:
            return []
        grid = [(fs, h, m) for fs in FEATURE_SETS for h in (12, 24, 48) for m in (0.04, 0.08)]
        self.rng.shuffle(grid)
        out = []
        for fs, h, m in grid[: self.cfg.model_trials]:
            spec = ModelSpec(f"ML-{fs}-{h}h", feature_set=fs, horizon=h, margin=m,
                             min_train=min(3000, self.split.train.stop // 2))
            out.append((self.screen(spec), spec))
        return out

    # ---------------------------------------------------------------- gates
    def _neighbours(self, spec) -> list:
        out = []
        for _ in range(self.cfg.robustness_neighbours):
            c = copy.deepcopy(spec)
            if isinstance(c, ModelSpec):
                c.margin = round(c.margin * self.rng.choice([0.6, 0.8, 1.25, 1.5]), 4)
                c.stop_atr *= self.rng.choice([0.8, 1.2])
            else:
                for r in c.long_entry + c.short_entry + c.exit_rules:
                    if not r.ref:
                        lo, hi = FEATURE_SPACE.get(r.feature, ("", (-1, 1)))[1]
                        r.threshold += self.rng.uniform(-0.1, 0.1) * (hi - lo)
                c.max_hold = max(2, int(c.max_hold * self.rng.choice([0.75, 1, 1.25])))
                c.stop_atr *= self.rng.choice([0.8, 1.0, 1.2])
            out.append(c)
        return out

    def gate(self, spec) -> tuple[Status, dict, str]:
        cfg, res = self.cfg, {}
        is_model = isinstance(spec, ModelSpec)
        v, _ = self._metrics(spec, self.split.valid)
        res["valid"] = _clean(v)
        if v["sharpe"] <= 0 or v["n_trades"] < cfg.min_trades_holdout:
            return Status.REJECTED, res, f"validation failed (Sharpe {v['sharpe']:.2f}, {v['n_trades']} trades)"

        window = max(1500, getattr(spec, "min_train", 0) + 1200)
        sig = lambda d: spec.simulate(build_features(d), d)[0]  # noqa: E731
        bad = causality_violations(sig, self.df.iloc[-window:], n_checks=3 if is_model else 4,
                                   warmup=window - 400 if is_model else 300)
        if bad:
            return Status.REJECTED, res, "LOOK-AHEAD detected: " + bad[0]

        start = getattr(spec, "min_train", 0) if is_model else 0
        edges = np.linspace(start, self.split.valid.stop, cfg.stability_slices + 1).astype(int)
        slice_sr = [self._metrics(spec, slice(a, b))[0]["sharpe"] for a, b in zip(edges[:-1], edges[1:])]
        res["stability"] = [round(x, 2) for x in slice_sr]
        if sum(x > 0 for x in slice_sr) < cfg.stability_min_positive:
            return Status.REJECTED, res, f"unstable: profitable in only {sum(x > 0 for x in slice_sr)}" \
                                         f"/{cfg.stability_slices} time periods"

        nb = [self._metrics(n, self.split.valid)[0]["sharpe"] for n in self._neighbours(spec)]
        self.trials_this_run += len(nb)
        res["neighbour_sharpe_median"] = round(float(np.median(nb)), 3)
        if np.median(nb) <= 0 or np.median(nb) < cfg.robustness_ratio * v["sharpe"]:
            return Status.REJECTED, res, f"fragile: nearby parameters give Sharpe {np.median(nb):.2f} " \
                                         f"vs {v['sharpe']:.2f} (knife-edge fit)"

        stress, _ = self._metrics(spec, self.split.valid, cost_mult=2.0)
        res["valid_sharpe_2x_costs"] = round(stress["sharpe"], 3)
        if stress["sharpe"] <= 0:
            return Status.REJECTED, res, "edge disappears when trading costs double"

        h, h_ret = self._metrics(spec, self.split.holdout)
        res["holdout"] = _clean(h)
        bh_pos = buy_and_hold().positions(self.f, self.df)
        bh = run_backtest(self.df.iloc[self.split.holdout], bh_pos.iloc[self.split.holdout], self.cost,
                          self.interval).metrics
        res["benchmark_holdout_sharpe"] = round(bh["sharpe"], 3)
        # The holdout was never seen by the search, so the selection bias to
        # correct for is the cumulative number of strategies ever tested ON IT.
        self.reg.data["holdout_tests"] = self.reg.data.get("holdout_tests", 0) + 1
        n_trials = self.reg.data["holdout_tests"]
        res["dsr"] = round(deflated_sharpe(h_ret, n_trials), 4)
        res["n_trials_accounted"] = n_trials
        res["search_trials_total"] = self.reg.trial_count + self.trials_this_run
        if h["n_trades"] < cfg.min_trades_holdout:
            return Status.REJECTED, res, f"too few holdout trades ({h['n_trades']})"
        if h["sharpe"] <= 0:
            return Status.REJECTED, res, f"holdout Sharpe {h['sharpe']:.2f} <= 0"
        if h["sharpe"] <= bh["sharpe"]:
            return Status.REJECTED, res, f"does not beat buy & hold on holdout ({h['sharpe']:.2f} vs {bh['sharpe']:.2f})"
        if res["dsr"] < cfg.dsr_threshold:
            return Status.REJECTED, res, f"deflated Sharpe {res['dsr']:.2f} < {cfg.dsr_threshold} " \
                                         f"after {n_trials} holdout tests (could be luck)"

        seg = self.split.holdout
        if is_model:
            pos, stops = self._sim(spec)
            d = self.df.iloc[seg]
            score = lambda p: run_backtest(d, p, self.cost, self.interval).metrics["sharpe"]  # noqa: E731
            p, _ = timing_pvalue(pos.iloc[seg], d, score, n_perm=max(60, cfg.n_perm), seed=cfg.seed)
            res["perm_test"] = "timing-shift"
        else:
            hold_df = self.df.iloc[max(0, seg.start - 300):]
            off = seg.start - max(0, seg.start - 300)

            def score(d):
                ps, st = spec.simulate(build_features(d), d)
                return run_backtest(d.iloc[off:], ps.iloc[off:], self.cost, self.interval, st.iloc[off:]).metrics["sharpe"]
            p, _, _ = permutation_pvalue(hold_df, score, n_perm=cfg.n_perm, seed=cfg.seed)
            res["perm_test"] = "price-permutation"
        res["perm_p"] = round(p, 4)
        from ..validation.montecarlo import drawdown_distribution
        _, vr = self._metrics(spec, slice(self.split.valid.start, self.split.holdout.stop))
        res["monte_carlo"] = {k: round(v, 4) for k, v in
                              drawdown_distribution(vr, n_paths=500, horizon=2000, seed=cfg.seed).items()}
        if p > cfg.perm_alpha:
            return Status.REJECTED, res, f"permutation test p={p:.3f}: indistinguishable from chance"
        return Status.VALIDATED, res, "passed all 8 gates (validation, causality, stability, robustness, " \
                                      "cost stress, holdout, deflated Sharpe, permutation)"

    # ------------------------------------------------------------------ run
    def run(self, extra_seeds=None) -> dict:
        source = self.df.attrs.get("source", "market")
        ranked = self.genetic_search(extra_seeds)
        finalists = [s for sc, s in ranked[: self.cfg.top_k_validate] if sc > -10]
        finalists += [self.bayes_tune(s) for s in finalists[: self.cfg.tune_top]]
        try:
            models = self.model_candidates()
        except Exception as ex:   # an ML failure must not cost the rule-based results
            log.error("model candidates failed", extra={"data": {"symbol": self.symbol, "err": str(ex)}})
            models = []
        finalists += [m for sc, m in sorted(models, key=lambda x: -x[0])[:1] if sc > -10]
        summary = {"symbol": self.symbol, "source": source, "screened": 0, "validated": [], "rejected": []}
        for s in finalists:
            if self.reg.seen(s.fingerprint()):
                continue
            status, res, reason = self.gate(s)
            res["train_score"] = round(self.screen(s), 3)
            res["symbol"], res["data_source"] = self.symbol, source
            self.reg.upsert(s, status, res, reason)
            (summary["validated"] if status == Status.VALIDATED else summary["rejected"]).append(
                {"id": s.id, "name": s.name, "reason": reason})
            bus.publish("lab.result", {"id": s.id, "status": status.value, "reason": reason})
        summary["screened"] = self.trials_this_run
        self.reg.add_trials(self.trials_this_run)
        self.reg.save()
        log.info("lab run complete", extra={"data": {"symbol": self.symbol, "screened": self.trials_this_run,
                                                       "validated": len(summary["validated"])}})
        return summary


def _clean(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in m.items()
            if not k.startswith("_")}
