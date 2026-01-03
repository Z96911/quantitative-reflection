"""
量化反思模块 (Quantitative Reflection Module)
============================================

多智能体决策系统的自适应权重调整系统。

本模块通过追踪多个分析师/智能体的历史准确率，
使用 Softmax 归一化动态调整权重，使系统能够从过去的表现中学习。

作者: Z96911
许可证: MIT
"""

from .reflector import QuantitativeReflector
from .models import DecisionRecord, ReflectionRecord, Signal, Decision
from .evaluator import ReflectionEvaluator

__version__ = "0.1.0"
__all__ = [
    "QuantitativeReflector",
    "DecisionRecord",
    "ReflectionRecord",
    "Signal",
    "Decision",
    "ReflectionEvaluator",
]
