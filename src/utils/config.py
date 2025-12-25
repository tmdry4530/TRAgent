"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Symbol-specific leverage configuration
SYMBOL_LEVERAGE = {
    "BTCUSDT": 50,
    "SOLUSDT": 40,
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    env: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # Binance API
    binance_api_key: SecretStr = Field(default=SecretStr(""))
    binance_secret_key: SecretStr = Field(default=SecretStr(""))
    binance_testnet: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading Parameters
    trading_symbols: list[str] = ["BTCUSDT", "SOLUSDT"]  # 거래 대상 심볼들
    trading_symbol: str = "BTCUSDT"  # 기본 심볼 (하위 호환성)

    # Scalp settings (Wick Reversal)
    scalp_max_position_pct: float = 0.30  # 30% of account
    scalp_leverage: int = 50  # 기본 레버리지 (심볼별 오버라이드 가능)
    max_concurrent_positions: int = 1  # 동시 포지션 수
    max_holding_minutes: int = 5  # 최대 홀딩 시간 (분)
    max_position_usd: float = 50.0  # 고정 포지션 크기 (USD) - 0이면 비율 사용

    # Wick Reversal specific settings
    wick_ema_period: int = 20  # 15분봉 EMA 기간
    wick_volume_multiplier: float = 3.0  # 진입용 거래량 배수
    wick_ratio_threshold: float = 0.6  # 꼬리 비율 임계값 (60%)
    wick_stop_loss_buffer: float = 0.001  # 0.1% 손절 버퍼
    wick_exit_volume_multiplier: float = 2.0  # 익절용 거래량 배수

    # Risk limits
    max_total_exposure_pct: float = 0.30  # 30% of account (단일 포지션)
    daily_loss_limit_pct: float = 0.10  # 10%
    consecutive_loss_cooldown_minutes: int = 60
    max_consecutive_losses: int = 3

    # Legacy settings (for backward compatibility)
    swing_max_position_pct: float = 0.50
    swing_leverage: int = 10
    swing_stop_loss_pct: float = 0.05
    swing_take_profit_pct: float = 0.15
    swing_trailing_stop_pct: float = 0.07
    scalp_stop_loss_pct: float = 0.015
    scalp_take_profit_pct: float = 0.03

    def get_leverage(self, symbol: str) -> int:
        """Get leverage for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Leverage for the symbol
        """
        return SYMBOL_LEVERAGE.get(symbol, self.scalp_leverage)

    @property
    def binance_ws_url(self) -> str:
        """Get Binance WebSocket URL based on testnet setting."""
        if self.binance_testnet:
            return "wss://stream.binancefuture.com"
        return "wss://fstream.binance.com"

    @property
    def binance_rest_url(self) -> str:
        """Get Binance REST URL based on testnet setting."""
        if self.binance_testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
