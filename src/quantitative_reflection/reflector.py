"""
Quantitative Reflector
======================

Core module for tracking analyst accuracy and dynamically adjusting weights.

The key innovation is using Softmax normalization to adjust weights based on
historical accuracy, enabling the system to learn from past performance.

Algorithm:
    accuracy[analyst] = correct_predictions / total_predictions
    weight[analyst] = exp(accuracy / temperature) / sum(exp(accuracy / temperature))
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .models import DecisionRecord, ReflectionRecord

logger = logging.getLogger(__name__)


class QuantitativeReflector:
    """
    Quantitative Reflector for Multi-Agent Systems

    This class tracks the historical accuracy of multiple analysts/agents and
    dynamically adjusts their weights using Softmax normalization.

    Features:
        1. Record decisions and actual outcomes
        2. Calculate per-analyst accuracy rates
        3. Dynamically adjust weights based on performance
        4. Generate quantitative evaluation reports

    Example:
        >>> reflector = QuantitativeReflector(
        ...     analyst_names=["market", "fundamentals", "news", "social"]
        ... )
        >>> # Record a decision
        >>> record = reflector.record_decision(
        ...     ticker="AAPL",
        ...     trade_date="2024-01-15",
        ...     decision="buy",
        ...     confidence=0.8,
        ...     analyst_signals={
        ...         "market": "bullish",
        ...         "fundamentals": "bullish",
        ...         "news": "neutral",
        ...         "social": "bullish"
        ...     }
        ... )
        >>> # Later, reflect on the outcome
        >>> reflection = reflector.reflect(record, actual_return=0.05)
        >>> # Update weights based on accumulated data
        >>> new_weights = reflector.update_weights()
    """

    def __init__(
        self,
        analyst_names: List[str] = None,
        storage_path: str = None,
        hold_threshold: float = 0.02,
        min_samples_for_update: int = 5,
        softmax_temperature: float = 2.0
    ):
        """
        Initialize the Quantitative Reflector.

        Args:
            analyst_names: List of analyst/agent names to track
            storage_path: Path for persistent storage (JSON files)
            hold_threshold: Return threshold for hold signal (default: 2%)
            min_samples_for_update: Minimum samples before updating weights
            softmax_temperature: Temperature for Softmax (higher = smoother)
        """
        self.analyst_names = analyst_names or ["analyst_1", "analyst_2", "analyst_3", "analyst_4"]
        self.hold_threshold = hold_threshold
        self.min_samples = min_samples_for_update
        self.temperature = softmax_temperature

        # Storage path
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("./reflection_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize equal weights
        self.weights: Dict[str, float] = {
            name: 1.0 / len(self.analyst_names) for name in self.analyst_names
        }

        # Accuracy statistics
        self.stats: Dict[str, Dict[str, int]] = {
            name: {"correct": 0, "total": 0} for name in self.analyst_names
        }

        # Records
        self.decision_records: List[DecisionRecord] = []
        self.reflection_records: List[ReflectionRecord] = []

        # Load historical data
        self._load_data()

        logger.info(f"QuantitativeReflector initialized with {len(self.analyst_names)} analysts")

    def _load_data(self):
        """Load historical data from storage."""
        weights_file = self.storage_path / "weights.json"
        if weights_file.exists():
            try:
                with open(weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.weights = data.get("weights", self.weights)
                    self.stats = data.get("stats", self.stats)
                logger.info(f"Loaded weights: {self.weights}")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

        records_file = self.storage_path / "reflections.json"
        if records_file.exists():
            try:
                with open(records_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    self.reflection_records = [ReflectionRecord(**r) for r in records]
                logger.info(f"Loaded {len(self.reflection_records)} reflection records")
            except Exception as e:
                logger.warning(f"Failed to load reflections: {e}")

    def _save_data(self):
        """Save data to storage."""
        # Save weights
        weights_file = self.storage_path / "weights.json"
        try:
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "weights": self.weights,
                    "stats": self.stats,
                    "updated_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

        # Save reflection records (keep last 100)
        records_file = self.storage_path / "reflections.json"
        try:
            with open(records_file, 'w', encoding='utf-8') as f:
                records = [asdict(r) for r in self.reflection_records[-100:]]
                # Convert numpy types to Python types
                records = self._convert_numpy_types(records)
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reflections: {e}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def record_decision(
        self,
        ticker: str,
        trade_date: str,
        decision: str,
        confidence: float,
        analyst_signals: Dict[str, str]
    ) -> DecisionRecord:
        """
        Record a decision made by the system.

        Args:
            ticker: Stock/asset symbol
            trade_date: Date of decision (YYYY-MM-DD)
            decision: Final decision ("buy", "sell", "hold")
            confidence: Confidence level (0-1)
            analyst_signals: Dict mapping analyst name to signal
                             e.g., {"market": "bullish", "news": "neutral"}

        Returns:
            DecisionRecord object
        """
        record = DecisionRecord(
            ticker=ticker,
            trade_date=trade_date,
            decision=decision,
            confidence=confidence,
            analyst_signals=analyst_signals
        )

        self.decision_records.append(record)
        logger.info(f"Recorded decision: {ticker} @ {trade_date} -> {decision}")

        return record

    def reflect(
        self,
        record: DecisionRecord,
        actual_return: float,
        actual_return_extended: float = None
    ) -> ReflectionRecord:
        """
        Reflect on a past decision by comparing with actual outcome.

        Args:
            record: The original DecisionRecord
            actual_return: Actual return rate (e.g., 0.05 for +5%)
            actual_return_extended: Extended period return (optional)

        Returns:
            ReflectionRecord with analysis results
        """
        if actual_return_extended is None:
            actual_return_extended = actual_return

        # Determine actual signal from return
        actual_signal = self._return_to_signal(actual_return)

        # Determine predicted signal from decision
        predicted_signal = self._decision_to_signal(record.decision)

        # Check if prediction was correct
        is_correct = self._is_prediction_correct(record.decision, actual_return)

        # Analyze per-analyst accuracy
        analyst_accuracy = {}
        for analyst, signal in record.analyst_signals.items():
            analyst_accuracy[analyst] = self._is_signal_correct(signal, actual_signal)

        # Update statistics
        for analyst, correct in analyst_accuracy.items():
            if analyst in self.stats:
                self.stats[analyst]["total"] += 1
                if correct:
                    self.stats[analyst]["correct"] += 1

        # Create reflection record
        reflection = ReflectionRecord(
            record_id=record.record_id,
            ticker=record.ticker,
            trade_date=record.trade_date,
            predicted_signal=predicted_signal,
            actual_signal=actual_signal,
            is_correct=bool(is_correct),  # Ensure Python bool
            actual_return=float(actual_return),
            actual_return_extended=float(actual_return_extended),
            analyst_accuracy=analyst_accuracy
        )

        self.reflection_records.append(reflection)
        self._save_data()

        logger.info(
            f"Reflection: {record.ticker} - "
            f"predicted={predicted_signal}, actual={actual_signal}, "
            f"{'CORRECT' if is_correct else 'WRONG'} (return: {actual_return:+.2%})"
        )

        return reflection

    def _return_to_signal(self, return_rate: float) -> str:
        """Convert return rate to signal."""
        if return_rate > self.hold_threshold:
            return "bullish"
        elif return_rate < -self.hold_threshold:
            return "bearish"
        return "neutral"

    def _decision_to_signal(self, decision: str) -> str:
        """Convert decision to signal."""
        decision_lower = decision.lower()
        if decision_lower in ["buy", "bullish", "long"]:
            return "bullish"
        elif decision_lower in ["sell", "bearish", "short"]:
            return "bearish"
        return "neutral"

    def _is_signal_correct(self, predicted: str, actual: str) -> bool:
        """Check if signal prediction was correct."""
        if predicted == "neutral":
            return True  # Neutral is never wrong
        return predicted == actual

    def _is_prediction_correct(self, decision: str, actual_return: float) -> bool:
        """Check if decision was correct based on actual return."""
        decision_lower = decision.lower()
        if decision_lower in ["buy", "bullish", "long"]:
            return actual_return > 0
        elif decision_lower in ["sell", "bearish", "short"]:
            return actual_return < 0
        return abs(actual_return) < self.hold_threshold

    def update_weights(self) -> Dict[str, float]:
        """
        Update analyst weights based on historical accuracy.

        Uses Softmax normalization to avoid extreme weights:
            weight[i] = exp(accuracy[i] / T) / sum(exp(accuracy / T))

        Returns:
            Updated weights dictionary
        """
        # Check minimum samples
        total_samples = sum(s["total"] for s in self.stats.values())
        required_samples = self.min_samples * len(self.analyst_names)

        if total_samples < required_samples:
            logger.info(
                f"Insufficient samples ({total_samples}/{required_samples}), "
                f"keeping current weights"
            )
            return self.weights

        # Calculate accuracies
        accuracies = {}
        for analyst, stats in self.stats.items():
            if stats["total"] > 0:
                accuracies[analyst] = stats["correct"] / stats["total"]
            else:
                accuracies[analyst] = 0.5  # Default accuracy

        # Softmax normalization
        exp_values = {k: np.exp(v / self.temperature) for k, v in accuracies.items()}
        total = sum(exp_values.values())

        if total > 0:
            self.weights = {k: float(v / total) for k, v in exp_values.items()}

        self._save_data()

        logger.info(f"Updated weights: {self.weights}")
        logger.info(f"Analyst accuracies: {accuracies}")

        return self.weights

    def get_weighted_recommendation(
        self,
        analyst_signals: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        Get weighted recommendation based on analyst signals.

        Args:
            analyst_signals: Dict mapping analyst name to their signal

        Returns:
            Tuple of (recommended_signal, confidence)
        """
        signal_scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}

        for analyst, signal in analyst_signals.items():
            weight = self.weights.get(analyst, 1.0 / len(self.analyst_names))
            if signal in signal_scores:
                signal_scores[signal] += weight

        best_signal = max(signal_scores, key=signal_scores.get)
        confidence = signal_scores[best_signal]

        return best_signal, confidence

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if not self.reflection_records:
            return {"message": "No reflection records yet"}

        total = len(self.reflection_records)
        correct = sum(1 for r in self.reflection_records if r.is_correct)
        returns = [r.actual_return for r in self.reflection_records]

        return {
            "total_decisions": total,
            "correct_decisions": correct,
            "accuracy": correct / total if total > 0 else 0,
            "avg_return": float(np.mean(returns)) if returns else 0,
            "total_return": float(np.sum(returns)) if returns else 0,
            "analyst_stats": self.stats,
            "analyst_weights": self.weights,
        }

    def get_report(self) -> str:
        """Generate a summary report."""
        stats = self.get_statistics()

        if "message" in stats:
            return stats["message"]

        report = f"""
================================================================================
              Quantitative Reflection Report
================================================================================

Overall Performance
-------------------
Total Decisions:  {stats['total_decisions']}
Correct:          {stats['correct_decisions']}
Accuracy:         {stats['accuracy']:.1%}
Average Return:   {stats['avg_return']:.2%}
Total Return:     {stats['total_return']:.2%}

Analyst Accuracy
----------------"""

        for analyst in self.analyst_names:
            s = stats['analyst_stats'].get(analyst, {"correct": 0, "total": 0})
            if s['total'] > 0:
                acc = s['correct'] / s['total']
                report += f"\n  {analyst:20s}: {acc:.1%} ({s['correct']}/{s['total']})"
            else:
                report += f"\n  {analyst:20s}: No data"

        report += """

Current Weights
---------------"""
        for analyst, weight in stats['analyst_weights'].items():
            bar = "#" * int(weight * 40)
            report += f"\n  {analyst:20s}: {weight:.1%} {bar}"

        report += "\n" + "=" * 80

        return report
