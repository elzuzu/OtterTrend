from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, BaseSettings, Field, validator


class HardwareConfig(BaseModel):
    enable_mlx: bool = Field(False, description="Enable MLX backend on Apple Silicon")
    enable_coreml: bool = Field(False, description="Enable CoreML backend")


class AppSettings(BaseSettings):
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    mexc_api_key: str = Field("", env="MEXC_API_KEY")
    mexc_api_secret: str = Field("", env="MEXC_API_SECRET")
    exchange_id: str = Field("mexc", env="EXCHANGE_ID")
    paper_trading: bool = Field(True, env="PAPER_TRADING")
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


@lru_cache()
def get_settings(env_file: Optional[str] = None) -> AppSettings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    return AppSettings()


__all__ = ["AppSettings", "get_settings"]
