"""
Data Models for Quantitative Reflection
========================================

Defines the core data structures for decision recording and reflection analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict


class Signal(Enum):
    """Trading signal direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Decision(Enum):
    """Trading decision type"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class DecisionRecord:
    """
    Record of a single decision made by the system.

    Attributes:
        ticker: Stock/asset symbol
        trade_date: Date of the decision
        decision: Final decision (buy/sell/hold)
        confidence: Confidence level (0-1)
        analyst_signals: Dict mapping analyst name to their signal
    """
    ticker: str
    trade_date: str
    decision: str
    confidence: float

    # Individual analyst signals
    analyst_signals: Dict[str, str] = field(default_factory=dict)

    # Metadata
    record_id: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"{self.ticker}_{self.trade_date}_{datetime.now().strftime('%H%M%S')}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ReflectionRecord:
    """
    Record of reflection analysis comparing prediction vs actual outcome.

    Attributes:
        record_id: Reference to the original DecisionRecord
        ticker: Stock/asset symbol
        trade_date: Date of the original decision
        predicted_signal: What the system predicted
        actual_signal: What actually happened
        is_correct: Whether the prediction was correct
        actual_return: Actual return rate
        analyst_accuracy: Dict mapping analyst name to whether they were correct
    """
    record_id: str
    ticker: str
    trade_date: str

    # Prediction vs Reality
    predicted_signal: str
    actual_signal: str
    is_correct: bool

    # Actual performance
    actual_return: float
    actual_return_extended: float = 0.0  # Longer-term return (e.g., 10-day)

    # Per-analyst accuracy
    analyst_accuracy: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    reflection_date: str = ""

    def __post_init__(self):
        if not self.reflection_date:
            self.reflection_date = datetime.now().strftime("%Y-%m-%d")
