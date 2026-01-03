"""
Unit Tests for Quantitative Reflector
======================================
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from quantitative_reflection import QuantitativeReflector, DecisionRecord


class TestQuantitativeReflector:
    """Test suite for QuantitativeReflector."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def reflector(self, temp_storage):
        """Create a reflector instance with temp storage."""
        return QuantitativeReflector(
            analyst_names=["analyst_a", "analyst_b", "analyst_c"],
            storage_path=temp_storage,
            min_samples_for_update=2
        )

    def test_initialization(self, reflector):
        """Test reflector initialization."""
        assert len(reflector.analyst_names) == 3
        assert all(w == pytest.approx(1/3, rel=0.01) for w in reflector.weights.values())

    def test_record_decision(self, reflector):
        """Test recording a decision."""
        record = reflector.record_decision(
            ticker="TEST",
            trade_date="2024-01-01",
            decision="buy",
            confidence=0.8,
            analyst_signals={
                "analyst_a": "bullish",
                "analyst_b": "neutral",
                "analyst_c": "bullish"
            }
        )

        assert record.ticker == "TEST"
        assert record.decision == "buy"
        assert len(reflector.decision_records) == 1

    def test_reflect(self, reflector):
        """Test reflection on a decision."""
        record = reflector.record_decision(
            ticker="TEST",
            trade_date="2024-01-01",
            decision="buy",
            confidence=0.8,
            analyst_signals={
                "analyst_a": "bullish",
                "analyst_b": "neutral",
                "analyst_c": "bullish"
            }
        )

        reflection = reflector.reflect(record, actual_return=0.05)

        assert reflection.predicted_signal == "bullish"
        assert reflection.actual_signal == "bullish"
        assert reflection.is_correct == True
        assert reflection.analyst_accuracy["analyst_a"] == True

    def test_weight_update(self, reflector):
        """Test weight update based on accuracy."""
        # Create records where analyst_a is always correct, analyst_c is wrong
        for i in range(4):
            record = reflector.record_decision(
                ticker=f"TEST{i}",
                trade_date=f"2024-01-0{i+1}",
                decision="buy",
                confidence=0.8,
                analyst_signals={
                    "analyst_a": "bullish",  # Always correct
                    "analyst_b": "neutral",  # Neutral
                    "analyst_c": "bearish"   # Always wrong
                }
            )
            reflector.reflect(record, actual_return=0.05)  # Always positive

        # Update weights
        new_weights = reflector.update_weights()

        # analyst_a should have higher weight than analyst_c
        assert new_weights["analyst_a"] > new_weights["analyst_c"]

    def test_weighted_recommendation(self, reflector):
        """Test weighted recommendation."""
        signal, confidence = reflector.get_weighted_recommendation({
            "analyst_a": "bullish",
            "analyst_b": "bullish",
            "analyst_c": "bearish"
        })

        assert signal == "bullish"
        assert confidence > 0.5

    def test_persistence(self, temp_storage):
        """Test data persistence across instances."""
        # Create and populate first instance
        reflector1 = QuantitativeReflector(
            analyst_names=["a", "b"],
            storage_path=temp_storage,
            min_samples_for_update=1
        )

        record = reflector1.record_decision(
            ticker="TEST",
            trade_date="2024-01-01",
            decision="buy",
            confidence=0.8,
            analyst_signals={"a": "bullish", "b": "bullish"}
        )
        reflector1.reflect(record, actual_return=0.05)
        reflector1.update_weights()

        # Create second instance and verify data loaded
        reflector2 = QuantitativeReflector(
            analyst_names=["a", "b"],
            storage_path=temp_storage
        )

        assert len(reflector2.reflection_records) == 1
        assert reflector2.stats["a"]["total"] == 1


class TestSignalConversion:
    """Test signal conversion methods."""

    @pytest.fixture
    def reflector(self):
        return QuantitativeReflector(min_samples_for_update=1)

    def test_return_to_signal_bullish(self, reflector):
        assert reflector._return_to_signal(0.05) == "bullish"

    def test_return_to_signal_bearish(self, reflector):
        assert reflector._return_to_signal(-0.05) == "bearish"

    def test_return_to_signal_neutral(self, reflector):
        assert reflector._return_to_signal(0.01) == "neutral"

    def test_decision_to_signal(self, reflector):
        assert reflector._decision_to_signal("buy") == "bullish"
        assert reflector._decision_to_signal("sell") == "bearish"
        assert reflector._decision_to_signal("hold") == "neutral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
