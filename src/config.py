from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class HardwareConfig(BaseModel):
    enable_mlx: bool = Field(False, description="Enable MLX backend on Apple Silicon")
    enable_coreml: bool = Field(False, description="Enable CoreML backend")


class AppSettings(BaseSettings):
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    mexc_api_key: str = Field("", env="MEXC_API_KEY")
    mexc_api_secret: str = Field("", env="MEXC_API_SECRET")
    exchange_id: str = Field("mexc", env="EXCHANGE_ID")
    exchange_testnet: bool = Field(False, env="EXCHANGE_TESTNET")
    paper_trading: bool = Field(True, env="PAPER_TRADING")
    risk_max_trade_usd: float = Field(20.0, env="RISK_MAX_TRADE_USD")
    risk_max_trade_pct_equity: float = Field(0.05, env="RISK_MAX_TRADE_PCT_EQUITY")
    risk_min_liquidity_usd: float = Field(50_000.0, env="RISK_MIN_LIQUIDITY_USD")
    risk_low_liquidity_cap_usd: float = Field(5.0, env="RISK_LOW_LIQUIDITY_CAP_USD")
    risk_max_daily_loss_usd: float = Field(50.0, env="RISK_MAX_DAILY_LOSS_USD")
    risk_max_open_positions: int = Field(5, env="RISK_MAX_OPEN_POSITIONS")
    risk_max_spread_pct: float = Field(2.0, env="RISK_MAX_SPREAD_PCT")
    watchlist_symbols: list[str] = Field(default_factory=list, env="WATCHLIST_SYMBOLS")
    sqlite_db_path: Path = Field(Path("./data/bot.sqlite3"), env="SQLITE_DB_PATH")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("exchange_id")
    def normalize_exchange(cls, value: str) -> str:
        return value.lower()

    @validator("watchlist_symbols", pre=True)
    def parse_watchlist(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            return [sym.strip() for sym in value.split(",") if sym.strip()]
        return []


@lru_cache()
def get_settings(env_file: Optional[str] = None) -> AppSettings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    return AppSettings()


__all__ = ["AppSettings", "get_settings"]
