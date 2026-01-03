"""
反思评估器
==========

用于评估和分析反思性能的工具。
"""

from typing import Dict, List, Any
import numpy as np

from .models import ReflectionRecord


class ReflectionEvaluator:
    """
    反思性能评估器

    提供以下指标:
    - 滚动准确率
    - 各分析师性能对比
    - 权重演变追踪
    """

    def __init__(self, reflection_records: List[ReflectionRecord] = None):
        self.records = reflection_records or []

    def add_records(self, records: List[ReflectionRecord]):
        """添加反思记录用于分析。"""
        self.records.extend(records)

    def calculate_rolling_accuracy(self, window: int = 10) -> List[float]:
        """
        计算滚动窗口内的准确率。

        参数:
            window: 计算平均的记录数

        返回:
            滚动准确率值列表
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
        按分析师分解性能分析。

        返回:
            包含各分析师统计数据的字典
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

        # 计算汇总统计
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
        """获取整体性能摘要。"""
        if not self.records:
            return {"message": "暂无记录可分析"}

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
