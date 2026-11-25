"""Central configuration. Every value can be overridden with an env var prefixed QM_."""
from __future__ import annotations

from enum import IntEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class AutonomyLevel(IntEnum):
    RESEARCH = 1  # research + experiments only, no signals acted on
    SIGNAL = 2    # emits explained signals, executes nothing
    PAPER = 3     # executes simulated trades on real market data
    LIVE = 4      # real orders - requires explicit human authorization


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QM_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./quantmind.db"
    redis_url: str | None = None

    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    interval: str = "1h"
    history_bars: int = 17520  # 2 years of 1h bars

    autonomy_level: AutonomyLevel = AutonomyLevel.RESEARCH

    # "spot": buy / sell only, no leverage, no shorting - the safe default for
    # someone following signals by hand. "futures" also allows short signals.
    market_type: str = "spot"
    signal_valid_hours: int = 2

    # Live trading gate. Both must be set by a human; the API also requires a
    # confirmation phrase at switch time. Never set these in committed files.
    live_trading_enabled: bool = False
    live_authorization_token: str | None = None
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    live_max_capital_usd: float = 0.0

    paper_starting_equity: float = 10_000.0
    fee_bps: float = 10.0        # taker fee per side, basis points
    slippage_bps: float = 2.0    # base slippage per side

    # Optional LLM narrator: explains decisions, never makes them.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"

    kill_switch_file: str = "./KILL_SWITCH"


settings = Settings()
