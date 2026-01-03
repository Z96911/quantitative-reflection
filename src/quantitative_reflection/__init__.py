"""
Quantitative Reflection Module
==============================

A self-adaptive weight adjustment system for multi-agent decision systems.

This module tracks the historical accuracy of multiple analysts/agents and
dynamically adjusts their weights using Softmax normalization, enabling
the system to learn from past performance.

Author: [Your Name]
License: MIT
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
