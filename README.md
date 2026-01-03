# Quantitative Reflection Module

A self-adaptive weight adjustment system for multi-agent decision systems.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This module enables multi-agent systems to **learn from their past performance** by:

1. **Tracking accuracy** - Recording each agent's prediction accuracy over time
2. **Dynamic weight adjustment** - Using Softmax normalization to adjust agent weights
3. **Self-improvement** - Giving higher influence to better-performing agents

### Core Algorithm

```
accuracy[agent] = correct_predictions / total_predictions
weight[agent] = exp(accuracy / T) / Σexp(accuracy / T)
```

Where `T` is the temperature parameter controlling smoothness.

## Installation

```bash
pip install quantitative-reflection
```

Or install from source:

```bash
git clone https://github.com/yourusername/quantitative-reflection-module.git
cd quantitative-reflection-module
pip install -e .
```

## Quick Start

```python
from quantitative_reflection import QuantitativeReflector

# Initialize with your agents/analysts
reflector = QuantitativeReflector(
    analyst_names=["market", "fundamentals", "news", "social"],
    min_samples_for_update=5
)

# Record a decision
record = reflector.record_decision(
    ticker="AAPL",
    trade_date="2024-01-15",
    decision="buy",
    confidence=0.8,
    analyst_signals={
        "market": "bullish",
        "fundamentals": "bullish",
        "news": "neutral",
        "social": "bullish"
    }
)

# Later, reflect on the outcome
reflection = reflector.reflect(record, actual_return=0.05)

# Update weights based on accumulated data
new_weights = reflector.update_weights()
print(new_weights)
# {'market': 0.26, 'fundamentals': 0.26, 'news': 0.22, 'social': 0.26}
```

## Features

### 1. Decision Recording

Track every decision with full context:

```python
record = reflector.record_decision(
    ticker="AAPL",
    trade_date="2024-01-15",
    decision="buy",           # buy/sell/hold
    confidence=0.8,           # 0-1
    analyst_signals={...}     # Each agent's signal
)
```

### 2. Reflection & Learning

Compare predictions with actual outcomes:

```python
reflection = reflector.reflect(
    record,
    actual_return=0.05,       # +5% return
    actual_return_extended=0.08  # Optional: longer-term return
)

print(f"Prediction was: {'CORRECT' if reflection.is_correct else 'WRONG'}")
print(f"Analyst accuracy: {reflection.analyst_accuracy}")
```

### 3. Dynamic Weight Adjustment

Automatically adjust weights using Softmax:

```python
# After accumulating enough samples
weights = reflector.update_weights()

# Agents with higher accuracy get higher weights
# {'market': 0.26, 'fundamentals': 0.26, 'news': 0.22, 'social': 0.26}
```

### 4. Weighted Recommendations

Get recommendations weighted by agent performance:

```python
signal, confidence = reflector.get_weighted_recommendation({
    "market": "bullish",
    "fundamentals": "neutral",
    "news": "bearish",
    "social": "bullish"
})
# signal: "bullish", confidence: 0.65
```

### 5. Performance Reports

Generate comprehensive reports:

```python
print(reflector.get_report())
```

Output:
```
================================================================================
              Quantitative Reflection Report
================================================================================

Overall Performance
-------------------
Total Decisions:  20
Correct:          15
Accuracy:         75.0%
Average Return:   2.3%

Analyst Accuracy
----------------
  market              : 80.0% (16/20)
  fundamentals        : 75.0% (15/20)
  news                : 55.0% (11/20)
  social              : 70.0% (14/20)

Current Weights
---------------
  market              : 26.2% ##########
  fundamentals        : 25.1% ##########
  news                : 22.5% #########
  social              : 26.2% ##########
================================================================================
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  QuantitativeReflector                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Record    │ -> │   Reflect   │ -> │   Update    │     │
│  │  Decision   │    │  (Compare)  │    │  Weights    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         v                  v                  v             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ DecisionRec │    │ ReflectionR │    │   Softmax   │     │
│  │    ord      │    │    ecord    │    │  Normalize  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `analyst_names` | `["analyst_1", ...]` | List of agent names |
| `storage_path` | `./reflection_data` | Path for persistence |
| `hold_threshold` | `0.02` | Return threshold for "hold" |
| `min_samples_for_update` | `5` | Min samples before weight update |
| `softmax_temperature` | `2.0` | Softmax temperature (higher = smoother) |

## Use Cases

- **Trading Systems** - Adjust analyst weights based on prediction accuracy
- **Recommendation Engines** - Weight different recommendation sources
- **Ensemble Models** - Dynamic ensemble weight adjustment
- **Multi-Agent AI** - Learn optimal agent collaboration weights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This module was developed as part of contributions to [TradingAgents-CN](https://github.com/...), a multi-agent stock analysis system.
