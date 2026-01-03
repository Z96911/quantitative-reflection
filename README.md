# 量化反思模块 (Quantitative Reflection Module)

多智能体决策系统的自适应权重调整系统。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 概述

本模块使多智能体系统能够**从过去的表现中学习**：

1. **追踪准确率** - 记录每个智能体随时间的预测准确率
2. **动态权重调整** - 使用 Softmax 归一化调整智能体权重
3. **自我改进** - 赋予表现更好的智能体更高的影响力

### 核心算法

```
accuracy[智能体] = 正确预测数 / 总预测数
weight[智能体] = exp(accuracy / T) / Σexp(accuracy / T)
```

其中 `T` 是控制平滑度的温度参数。

## 安装

```bash
pip install quantitative-reflection
```

或从源码安装：

```bash
git clone https://github.com/Z96911/quantitative-reflection-module.git
cd quantitative-reflection-module
pip install -e .
```

## 快速开始

```python
from quantitative_reflection import QuantitativeReflector

# 用你的智能体/分析师初始化
reflector = QuantitativeReflector(
    analyst_names=["market", "fundamentals", "news", "social"],
    min_samples_for_update=5
)

# 记录一个决策
record = reflector.record_decision(
    ticker="600519",  # 贵州茅台
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

# 之后，对结果进行反思
reflection = reflector.reflect(record, actual_return=0.05)

# 基于累积数据更新权重
new_weights = reflector.update_weights()
print(new_weights)
# {'market': 0.26, 'fundamentals': 0.26, 'news': 0.22, 'social': 0.26}
```

## 功能特性

### 1. 决策记录

记录每个决策的完整上下文：

```python
record = reflector.record_decision(
    ticker="600519",           # 股票代码
    trade_date="2024-01-15",   # 交易日期
    decision="buy",            # buy/sell/hold
    confidence=0.8,            # 0-1
    analyst_signals={...}      # 各智能体的信号
)
```

### 2. 反思与学习

将预测与实际结果进行对比：

```python
reflection = reflector.reflect(
    record,
    actual_return=0.05,           # +5% 收益
    actual_return_extended=0.08   # 可选: 更长期收益
)

print(f"预测结果: {'正确' if reflection.is_correct else '错误'}")
print(f"分析师准确性: {reflection.analyst_accuracy}")
```

### 3. 动态权重调整

使用 Softmax 自动调整权重：

```python
# 累积足够样本后
weights = reflector.update_weights()

# 准确率更高的智能体获得更高权重
# {'market': 0.26, 'fundamentals': 0.26, 'news': 0.22, 'social': 0.26}
```

### 4. 加权推荐

获取基于智能体表现加权的推荐：

```python
signal, confidence = reflector.get_weighted_recommendation({
    "market": "bullish",
    "fundamentals": "neutral",
    "news": "bearish",
    "social": "bullish"
})
# signal: "bullish", confidence: 0.65
```

### 5. 性能报告

生成综合报告：

```python
print(reflector.get_report())
```

输出示例：
```
================================================================================
                         量化反思报告
================================================================================

整体表现
--------
总决策数:     20
正确决策:     15
准确率:       75.0%
平均收益:     2.3%

分析师准确率
------------
  market              : 80.0% (16/20)
  fundamentals        : 75.0% (15/20)
  news                : 55.0% (11/20)
  social              : 70.0% (14/20)

当前权重
--------
  market              : 26.2% ##########
  fundamentals        : 25.1% ##########
  news                : 22.5% #########
  social              : 26.2% ##########
================================================================================
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                  QuantitativeReflector                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    记录     │ -> │    反思     │ -> │    更新     │     │
│  │    决策     │    │   (对比)    │    │    权重     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         v                  v                  v             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ DecisionRec │    │ ReflectionR │    │   Softmax   │     │
│  │    ord      │    │    ecord    │    │   归一化    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `analyst_names` | `["analyst_1", ...]` | 智能体名称列表 |
| `storage_path` | `./reflection_data` | 持久化存储路径 |
| `hold_threshold` | `0.02` | "持有"信号的收益阈值 |
| `min_samples_for_update` | `5` | 更新权重所需的最小样本数 |
| `softmax_temperature` | `2.0` | Softmax 温度参数（越高越平滑） |

## 应用场景

- **交易系统** - 根据预测准确率调整分析师权重
- **推荐引擎** - 对不同推荐来源进行加权
- **集成模型** - 动态集成权重调整
- **多智能体 AI** - 学习最优的智能体协作权重

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)

## 致谢

本模块是对 [TradingAgents-CN](https://github.com/Z96911/TradingAgents-CN) 多智能体股票分析系统的贡献的一部分。
