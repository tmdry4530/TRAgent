"""LLM context filter for signal evaluation."""

from dataclasses import dataclass


@dataclass
class LLMDecision:
    """Decision from LLM context filter."""

    execute: bool
    confidence: float  # 0.0 ~ 1.0
    adjusted_size: float  # 0.0 ~ 1.0 (position size multiplier)
    reason: str


from src.brain.context_filter import (
    LLMContextFilter,
    MarketState,
    evaluate_signal,
)

__all__ = [
    "LLMDecision",
    "LLMContextFilter",
    "MarketState",
    "evaluate_signal",
]
