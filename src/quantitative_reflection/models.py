"""
量化反思数据模型
================

定义决策记录和反思分析的核心数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict


class Signal(Enum):
    """交易信号方向"""
    BULLISH = "bullish"   # 看涨
    BEARISH = "bearish"   # 看跌
    NEUTRAL = "neutral"   # 中性


class Decision(Enum):
    """交易决策类型"""
    BUY = "buy"      # 买入
    SELL = "sell"    # 卖出
    HOLD = "hold"    # 持有


@dataclass
class DecisionRecord:
    """
    决策记录

    记录系统做出的每一次决策的完整信息。

    属性:
        ticker: 股票/资产代码
        trade_date: 决策日期
        decision: 最终决策 (买入/卖出/持有)
        confidence: 置信度 (0-1)
        analyst_signals: 字典，映射分析师名称到其信号
    """
    ticker: str
    trade_date: str
    decision: str
    confidence: float

    # 各分析师的信号
    analyst_signals: Dict[str, str] = field(default_factory=dict)

    # 元数据
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
    反思记录

    记录预测与实际结果的对比分析。

    属性:
        record_id: 关联的原始决策记录 ID
        ticker: 股票/资产代码
        trade_date: 原始决策日期
        predicted_signal: 系统预测的信号
        actual_signal: 实际发生的信号
        is_correct: 预测是否正确
        actual_return: 实际收益率
        analyst_accuracy: 字典，映射分析师名称到其是否正确
    """
    record_id: str
    ticker: str
    trade_date: str

    # 预测 vs 实际
    predicted_signal: str
    actual_signal: str
    is_correct: bool

    # 实际表现
    actual_return: float
    actual_return_extended: float = 0.0  # 更长期的收益（如10日收益）

    # 各分析师的准确性
    analyst_accuracy: Dict[str, bool] = field(default_factory=dict)

    # 元数据
    reflection_date: str = ""

    def __post_init__(self):
        if not self.reflection_date:
            self.reflection_date = datetime.now().strftime("%Y-%m-%d")
