"""Strategy & experiment registry. Nothing is ever deleted: rejected strategies
and failed experiments stay visible, with the reason they failed."""
from __future__ import annotations

from ..core.store import Store, now_iso
from ..strategies.spec import Status, StrategySpec, spec_from_dict


class Registry:
    def __init__(self, store: Store):
        self.store = store
        self.data = store.load("registry.json", {"strategies": {}, "trial_count": 0, "fingerprints": []})

    # trial_count is the TOTAL number of strategy variants ever evaluated. The
    # deflated-Sharpe gate uses it, so it must never be reset to look better.
    @property
    def trial_count(self) -> int:
        return self.data["trial_count"]

    def add_trials(self, n: int) -> None:
        self.data["trial_count"] += n

    def seen(self, fp: str) -> bool:
        return fp in self.data["fingerprints"]

    def upsert(self, spec, status: Status, results: dict, reason: str = "") -> None:
        fp = spec.fingerprint()
        if fp not in self.data["fingerprints"]:
            self.data["fingerprints"].append(fp)
        prev = self.data["strategies"].get(spec.id)
        entry = {
            "id": spec.id, "name": spec.name, "family": spec.family, "version": spec.version,
            "origin": spec.origin, "parents": spec.parent_ids, "fingerprint": fp,
            "definition": spec.to_dict(), "math": spec.math_definition(),
            "required_data": sorted(spec.required_features()),
            "status": status.value, "status_reason": reason, "results": results,
            "updated": now_iso(), "history": (prev or {}).get("history", []),
        }
        entry["history"].append({"ts": now_iso(), "status": status.value, "reason": reason})
        self.data["strategies"][spec.id] = entry
        self.store.append("model_changes.jsonl", {"strategy": spec.id, "name": spec.name,
                                                   "status": status.value, "reason": reason})

    def set_status(self, sid: str, status: Status, reason: str) -> None:
        e = self.data["strategies"][sid]
        e["status"], e["status_reason"], e["updated"] = status.value, reason, now_iso()
        e["history"].append({"ts": now_iso(), "status": status.value, "reason": reason})
        self.store.append("model_changes.jsonl", {"strategy": sid, "status": status.value, "reason": reason})

    def by_status(self, *statuses: Status) -> list[dict]:
        vals = {s.value for s in statuses}
        return [e for e in self.data["strategies"].values() if e["status"] in vals]

    def spec(self, sid: str):
        return spec_from_dict(self.data["strategies"][sid]["definition"])

    def leaderboard(self, n: int = 20) -> list[dict]:
        rows = []
        for e in self.data["strategies"].values():
            h = e["results"].get("holdout") or {}
            v = e["results"].get("valid") or {}
            rows.append({"id": e["id"], "name": e["name"], "family": e["family"], "status": e["status"],
                         "valid_sharpe": v.get("sharpe"), "holdout_sharpe": h.get("sharpe"),
                         "dsr": e["results"].get("dsr"), "p_value": e["results"].get("perm_p")})
        rows.sort(key=lambda r: (r["holdout_sharpe"] is not None, r["holdout_sharpe"] or r["valid_sharpe"] or -9),
                  reverse=True)
        return rows[:n]

    def save(self) -> None:
        self.data["fingerprints"] = self.data["fingerprints"][-20000:]
        self.store.save("registry.json", self.data)
