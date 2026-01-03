"""
Reflection Evaluator
====================

Tools for evaluating and analyzing reflection performance.
"""

from typing import Dict, List, Any
import numpy as np

from .models import ReflectionRecord


class ReflectionEvaluator:
    """
    Evaluator for analyzing reflection performance over time.

    Provides metrics like:
    - Rolling accuracy
    - Per-analyst performance comparison
    - Weight evolution tracking
    """

    def __init__(self, reflection_records: List[ReflectionRecord] = None):
        self.records = reflection_records or []

    def add_records(self, records: List[ReflectionRecord]):
        """Add reflection records for analysis."""
        self.records.extend(records)

    def calculate_rolling_accuracy(self, window: int = 10) -> List[float]:
        """
        Calculate rolling accuracy over a window.

        Args:
            window: Number of records to average over

        Returns:
            List of rolling accuracy values
        """
        if len(self.records) < window:
            return []

        accuracies = []
        for i in range(window, len(self.records) + 1):
            window_records = self.records[i-window:i]
            correct = sum(1 for r in window_records if r.is_correct)
            accuracies.append(correct / window)

        return accuracies

    def analyze_by_analyst(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance breakdown by analyst.

        Returns:
            Dict with per-analyst statistics
        """
        analyst_data: Dict[str, Dict[str, Any]] = {}

        for record in self.records:
            for analyst, correct in record.analyst_accuracy.items():
                if analyst not in analyst_data:
                    analyst_data[analyst] = {
                        "correct": 0,
                        "total": 0,
                        "returns_when_correct": [],
                        "returns_when_wrong": []
                    }

                analyst_data[analyst]["total"] += 1
                if correct:
                    analyst_data[analyst]["correct"] += 1
                    analyst_data[analyst]["returns_when_correct"].append(record.actual_return)
                else:
                    analyst_data[analyst]["returns_when_wrong"].append(record.actual_return)

        # Calculate summary statistics
        for analyst, data in analyst_data.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
            data["avg_return_when_correct"] = (
                np.mean(data["returns_when_correct"])
                if data["returns_when_correct"] else 0
            )
            data["avg_return_when_wrong"] = (
                np.mean(data["returns_when_wrong"])
                if data["returns_when_wrong"] else 0
            )

        return analyst_data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.records:
            return {"message": "No records to analyze"}

        returns = [r.actual_return for r in self.records]
        correct_count = sum(1 for r in self.records if r.is_correct)

        return {
            "total_records": len(self.records),
            "accuracy": correct_count / len(self.records),
            "total_return": sum(returns),
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "max_return": max(returns),
            "min_return": min(returns),
            "sharpe_ratio": (
                np.mean(returns) / np.std(returns)
                if np.std(returns) > 0 else 0
            )
        }
