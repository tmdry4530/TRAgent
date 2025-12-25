"""Signal generators for scalp trading strategies."""

from src.signals.base import BaseSignalGenerator, Signal
from src.signals.scalp import (
    CandleData,
    WickReversalSignal,
    WickSignalContext,
)

__all__ = [
    "Signal",
    "BaseSignalGenerator",
    "CandleData",
    "WickReversalSignal",
    "WickSignalContext",
]
