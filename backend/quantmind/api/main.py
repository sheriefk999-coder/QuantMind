"""Optional server mode (VPS / home PC): the same brain behind a FastAPI API,
running a cycle every hour in the background. The Android app can point at it
instead of GitHub.

    uvicorn quantmind.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..cli import main as cli_main
from ..core.events import bus
from ..core.store import Store

app = FastAPI(title="QuantMind", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store = Store()
API_KEY = os.getenv("QM_API_KEY")  # set this if the server is reachable from the internet


def _auth(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "bad api key")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/snapshot")
def snapshot():
    s = store.load("snapshot.json", None)
    if s is None:
        raise HTTPException(404, "no cycle has run yet")
    return s


@app.get("/decisions")
def decisions(n: int = 50):
    return store.tail("decisions.jsonl", n)


@app.get("/strategies")
def strategies():
    return store.load("registry.json", {}).get("strategies", {})


class Control(BaseModel):
    action: str
    level: int | str | None = None
    symbol: str | None = None
    capital: float | None = None
    currency: str | None = None
    risk_profile: str | None = None


@app.post("/control")
def control(c: Control, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    args = {"pause": ["control", "--pause", "from app"], "resume": ["control", "--resume"],
            "research_now": ["control", "--research-now"]}.get(c.action)
    if c.action in ("taken", "skipped") and c.symbol and c.symbol.isalnum():
        args = ["control", f"--{c.action}", c.symbol]
    if c.action == "profile":
        args = ["control"]
        if c.capital:
            args += ["--capital", str(float(c.capital))]
        if c.currency in ("USD", "EGP", "SAR", "EUR"):
            args += ["--currency", c.currency]
        if c.risk_profile in ("conservative", "balanced", "bold"):
            args += ["--risk-profile", c.risk_profile]
    if c.action == "level" and str(c.level) in ("1", "2", "3"):
        args = ["control", "--level", str(c.level)]
    if not args:
        raise HTTPException(400, "unknown action (live trading cannot be enabled from the API)")
    cli_main(args)
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q = bus.subscribe()
    try:
        while True:
            await sock.send_json(await q.get())
    except WebSocketDisconnect:
        bus.unsubscribe(q)


@app.on_event("startup")
async def scheduler():
    async def loop():
        while True:
            try:
                await asyncio.to_thread(cli_main, ["cycle"])
            except Exception as ex:  # keep the scheduler alive
                bus.publish("error", {"msg": str(ex)})
            await asyncio.sleep(3600)
    if os.getenv("QM_DISABLE_SCHEDULER") != "1":
        asyncio.create_task(loop())
