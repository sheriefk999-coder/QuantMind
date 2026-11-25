"""Persistent state as plain JSON files in one directory.

Why files and not Postgres by default: the system runs free on GitHub Actions,
where the state directory is committed to a `state` branch after each cycle.
That gives a full, diffable history of every change for free, and the Android
app reads the published snapshot directly. (A Postgres backend can implement
the same small interface later.)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("QM_STATE_DIR", "./state"))
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def load(self, name: str, default: Any) -> Any:
        p = self.path(name)
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt state must never crash the risk engine; keep a copy and start clean.
            p.rename(p.with_suffix(f".corrupt.{int(datetime.now().timestamp())}"))
            return default

    def save(self, name: str, obj: Any) -> None:
        p = self.path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)  # atomic

    def append(self, name: str, record: dict, keep_last: int = 5000) -> None:
        """JSON-lines log, trimmed to keep_last records."""
        p = self.path(name)
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        lines.append(json.dumps({"ts": now_iso(), **record}, default=str, ensure_ascii=False))
        p.write_text("\n".join(lines[-keep_last:]) + "\n", encoding="utf-8")

    def tail(self, name: str, n: int = 50) -> list[dict]:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]
