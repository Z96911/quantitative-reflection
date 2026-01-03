"""
Basic Usage Example
===================

Demonstrates how to use the Quantitative Reflection module.
"""

from quantitative_reflection import QuantitativeReflector


def main():
    # Initialize reflector with 4 analysts
    reflector = QuantitativeReflector(
        analyst_names=["market", "fundamentals", "news", "social"],
        storage_path="./demo_data",
        min_samples_for_update=2  # Lower threshold for demo
    )

    # Simulate several trading decisions
    decisions_data = [
        {
            "ticker": "AAPL",
            "trade_date": "2024-01-15",
            "decision": "buy",
            "confidence": 0.8,
            "analyst_signals": {
                "market": "bullish",
                "fundamentals": "bullish",
                "news": "neutral",
                "social": "bullish"
            },
            "actual_return": 0.05  # +5%
        },
        {
            "ticker": "GOOGL",
            "trade_date": "2024-01-16",
            "decision": "hold",
            "confidence": 0.6,
            "analyst_signals": {
                "market": "neutral",
                "fundamentals": "bullish",
                "news": "bearish",  # News was wrong
                "social": "neutral"
            },
            "actual_return": 0.03  # +3%
        },
        {
            "ticker": "MSFT",
            "trade_date": "2024-01-17",
            "decision": "buy",
            "confidence": 0.75,
            "analyst_signals": {
                "market": "bullish",
                "fundamentals": "bullish",
                "news": "bearish",  # News was wrong again
                "social": "bullish"
            },
            "actual_return": 0.08  # +8%
        },
        {
            "ticker": "TSLA",
            "trade_date": "2024-01-18",
            "decision": "sell",
            "confidence": 0.7,
            "analyst_signals": {
                "market": "bearish",
                "fundamentals": "bearish",
                "news": "neutral",
                "social": "bearish"
            },
            "actual_return": -0.04  # -4%
        },
    ]

    print("=" * 60)
    print("  Quantitative Reflection Module - Demo")
    print("=" * 60)

    # Process each decision
    for data in decisions_data:
        print(f"\n[Decision] {data['ticker']} @ {data['trade_date']}")
        print(f"  Decision: {data['decision']} (confidence: {data['confidence']})")
        print(f"  Analyst signals: {data['analyst_signals']}")

        # Record the decision
        record = reflector.record_decision(
            ticker=data["ticker"],
            trade_date=data["trade_date"],
            decision=data["decision"],
            confidence=data["confidence"],
            analyst_signals=data["analyst_signals"]
        )

        # Reflect on the outcome
        reflection = reflector.reflect(record, actual_return=data["actual_return"])

        print(f"  Actual return: {data['actual_return']:+.1%}")
        print(f"  Prediction: {'CORRECT' if reflection.is_correct else 'WRONG'}")

    # Update weights based on performance
    print("\n" + "=" * 60)
    print("  Updating Weights")
    print("=" * 60)

    new_weights = reflector.update_weights()

    print("\nUpdated weights:")
    for analyst, weight in new_weights.items():
        bar = "#" * int(weight * 40)
        print(f"  {analyst:15s}: {weight:.1%} {bar}")

    # Show full report
    print("\n" + reflector.get_report())

    # Demonstrate weighted recommendation
    print("\n" + "=" * 60)
    print("  Weighted Recommendation Example")
    print("=" * 60)

    new_signals = {
        "market": "bullish",
        "fundamentals": "neutral",
        "news": "bearish",
        "social": "bullish"
    }

    signal, confidence = reflector.get_weighted_recommendation(new_signals)
    print(f"\nInput signals: {new_signals}")
    print(f"Weighted recommendation: {signal} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()
