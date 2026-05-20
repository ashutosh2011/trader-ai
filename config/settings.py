"""Application settings loaded from YAML and environment variables."""

from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class ExpiryDayRules(BaseModel):
    """Options expiry-day restrictions."""

    block_new_options_entries_after: str = "14:30"


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_loss_per_trade_pct: float = Field(default=0.5, gt=0)
    daily_loss_cap_pct: float = Field(default=2.0, gt=0)
    max_open_positions: int = Field(default=3, ge=1)
    position_sizing: Literal["atr_based", "fixed_pct"] = "atr_based"
    fixed_position_pct: float = Field(default=1.0, gt=0)
    no_trade_windows: list[str] = Field(default_factory=list)
    expiry_day_rules: ExpiryDayRules = Field(default_factory=ExpiryDayRules)


class PaperConfig(BaseModel):
    """Paper broker simulation settings."""

    slippage_bps: float = Field(default=5.0, ge=0)
    account_equity: float = Field(default=100_000.0, gt=0)


class KiteConfig(BaseModel):
    """Zerodha Kite Connect credentials (prefer env over YAML)."""

    api_key: str | None = None
    api_secret: str | None = None
    access_token: str | None = None
    request_token: str | None = None


class DataConfig(BaseModel):
    """Data layer paths and defaults."""

    duckdb_path: Path = Path("data/candles.duckdb")
    default_timeframe: str = "5minute"


class SchedulerConfig(BaseModel):
    """Market calendar and session rules (IST)."""

    market_open: str = "09:15"
    market_close: str = "15:30"
    pre_open_buffer_minutes: int = Field(default=0, ge=0)
    holiday_skip_dates: list[str] = Field(default_factory=list)
    expiry_weekday: int = Field(default=3, ge=0, le=6)
    block_fno_on_expiry_after: str = "14:30"


class AnalystProviderConfig(BaseModel):
    """LLM provider credentials (from env)."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    default_provider: Literal["anthropic", "openai", "google", "mock"] = "mock"
    model_anthropic: str = "claude-3-5-haiku-latest"
    model_openai: str = "gpt-4o-mini"
    model_google: str = "gemini-2.0-flash"


class AppSettings(BaseSettings):
    """Top-level settings with YAML overlay and env overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    kite: KiteConfig = Field(default_factory=KiteConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    analyst: AnalystProviderConfig = Field(default_factory=AnalystProviderConfig)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    kite_api_key: str | None = None
    kite_api_secret: str | None = None
    kite_access_token: str | None = None
    kill_switch_env: str = "KILL_SWITCH"
    kill_switch_file: Path = Path("KILL")
    live_default_qty: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def overlay_secrets(self) -> "AppSettings":
        """Overlay top-level env vars onto nested config models."""
        analyst_updates: dict[str, str] = {}
        if self.anthropic_api_key:
            analyst_updates["anthropic_api_key"] = self.anthropic_api_key
        if self.openai_api_key:
            analyst_updates["openai_api_key"] = self.openai_api_key
        if self.google_api_key:
            analyst_updates["google_api_key"] = self.google_api_key
        if analyst_updates:
            object.__setattr__(
                self, "analyst", self.analyst.model_copy(update=analyst_updates)
            )

        kite_updates: dict[str, str] = {}
        if self.kite_api_key:
            kite_updates["api_key"] = self.kite_api_key
        if self.kite_api_secret:
            kite_updates["api_secret"] = self.kite_api_secret
        if self.kite_access_token:
            kite_updates["access_token"] = self.kite_access_token
        if kite_updates:
            object.__setattr__(self, "kite", self.kite.model_copy(update=kite_updates))
        return self

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "AppSettings":
        """Load settings from YAML merged with environment variables."""
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if config_path.is_file():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = cast(dict[str, Any], raw)
        return cls.model_validate(data)

    def kite_configured(self) -> bool:
        """Return True when live Kite API credentials are present."""
        return bool(self.kite.api_key and self.kite.access_token)


def load_settings(path: Path | str | None = None) -> AppSettings:
    """Load application settings (YAML + env)."""
    return AppSettings.from_yaml(path)
