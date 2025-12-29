"""Signal generators for scalp trading strategies."""

from src.signals.base import BaseSignalGenerator, Signal
from src.signals.scalp import (
    CandleData,
    WickReversalSignal,
    WickSignalContext,
)
from src.signals.high_wr import HighWinRateSignalGenerator

__all__ = [
    "Signal",
    "BaseSignalGenerator",
    "CandleData",
    "WickReversalSignal",
    "WickSignalContext",
    "HighWinRateSignalGenerator",
]
