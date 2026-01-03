"""
量化反思器
==========

追踪分析师准确率并动态调整权重的核心模块。

核心创新是使用 Softmax 归一化基于历史准确率调整权重，
使系统能够从过去的表现中学习。

算法:
    accuracy[分析师] = 正确预测数 / 总预测数
    weight[分析师] = exp(accuracy / 温度) / sum(exp(accuracy / 温度))
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
    多智能体系统的量化反思器

    本类追踪多个分析师/智能体的历史准确率，
    并使用 Softmax 归一化动态调整它们的权重。

    功能:
        1. 记录决策和实际结果
        2. 计算各分析师的准确率
        3. 基于表现动态调整权重
        4. 生成量化评估报告

    使用示例:
        >>> reflector = QuantitativeReflector(
        ...     analyst_names=["market", "fundamentals", "news", "social"]
        ... )
        >>> # 记录一个决策
        >>> record = reflector.record_decision(
        ...     ticker="600519",
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
        >>> # 之后，对结果进行反思
        >>> reflection = reflector.reflect(record, actual_return=0.05)
        >>> # 基于累积数据更新权重
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
        初始化量化反思器。

        参数:
            analyst_names: 要追踪的分析师/智能体名称列表
            storage_path: 持久化存储路径（JSON 文件）
            hold_threshold: 持有信号的收益阈值（默认: 2%）
            min_samples_for_update: 更新权重所需的最小样本数
            softmax_temperature: Softmax 温度参数（越高越平滑）
        """
        self.analyst_names = analyst_names or ["analyst_1", "analyst_2", "analyst_3", "analyst_4"]
        self.hold_threshold = hold_threshold
        self.min_samples = min_samples_for_update
        self.temperature = softmax_temperature

        # 存储路径
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("./reflection_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 初始化均等权重
        self.weights: Dict[str, float] = {
            name: 1.0 / len(self.analyst_names) for name in self.analyst_names
        }

        # 准确率统计
        self.stats: Dict[str, Dict[str, int]] = {
            name: {"correct": 0, "total": 0} for name in self.analyst_names
        }

        # 记录列表
        self.decision_records: List[DecisionRecord] = []
        self.reflection_records: List[ReflectionRecord] = []

        # 加载历史数据
        self._load_data()

        logger.info(f"量化反思器已初始化，共 {len(self.analyst_names)} 个分析师")

    def _load_data(self):
        """从存储加载历史数据。"""
        weights_file = self.storage_path / "weights.json"
        if weights_file.exists():
            try:
                with open(weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.weights = data.get("weights", self.weights)
                    self.stats = data.get("stats", self.stats)
                logger.info(f"已加载权重: {self.weights}")
            except Exception as e:
                logger.warning(f"加载权重失败: {e}")

        records_file = self.storage_path / "reflections.json"
        if records_file.exists():
            try:
                with open(records_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    self.reflection_records = [ReflectionRecord(**r) for r in records]
                logger.info(f"已加载 {len(self.reflection_records)} 条反思记录")
            except Exception as e:
                logger.warning(f"加载反思记录失败: {e}")

    def _save_data(self):
        """保存数据到存储。"""
        # 保存权重
        weights_file = self.storage_path / "weights.json"
        try:
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "weights": self.weights,
                    "stats": self.stats,
                    "updated_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存权重失败: {e}")

        # 保存反思记录（保留最近100条）
        records_file = self.storage_path / "reflections.json"
        try:
            with open(records_file, 'w', encoding='utf-8') as f:
                records = [asdict(r) for r in self.reflection_records[-100:]]
                # 转换 numpy 类型为 Python 原生类型
                records = self._convert_numpy_types(records)
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存反思记录失败: {e}")

    def _convert_numpy_types(self, obj):
        """转换 numpy 类型为 Python 原生类型以便 JSON 序列化。"""
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
        记录系统做出的决策。

        参数:
            ticker: 股票/资产代码
            trade_date: 决策日期 (YYYY-MM-DD)
            decision: 最终决策 ("buy", "sell", "hold")
            confidence: 置信度 (0-1)
            analyst_signals: 字典，映射分析师名称到其信号
                             例如: {"market": "bullish", "news": "neutral"}

        返回:
            DecisionRecord 对象
        """
        record = DecisionRecord(
            ticker=ticker,
            trade_date=trade_date,
            decision=decision,
            confidence=confidence,
            analyst_signals=analyst_signals
        )

        self.decision_records.append(record)
        logger.info(f"已记录决策: {ticker} @ {trade_date} -> {decision}")

        return record

    def reflect(
        self,
        record: DecisionRecord,
        actual_return: float,
        actual_return_extended: float = None
    ) -> ReflectionRecord:
        """
        对过去的决策进行反思，对比实际结果。

        参数:
            record: 原始的 DecisionRecord
            actual_return: 实际收益率（如 0.05 表示 +5%）
            actual_return_extended: 更长期的收益率（可选）

        返回:
            包含分析结果的 ReflectionRecord
        """
        if actual_return_extended is None:
            actual_return_extended = actual_return

        # 根据收益率判断实际信号
        actual_signal = self._return_to_signal(actual_return)

        # 根据决策判断预测信号
        predicted_signal = self._decision_to_signal(record.decision)

        # 检查预测是否正确
        is_correct = self._is_prediction_correct(record.decision, actual_return)

        # 分析各分析师的准确性
        analyst_accuracy = {}
        for analyst, signal in record.analyst_signals.items():
            analyst_accuracy[analyst] = self._is_signal_correct(signal, actual_signal)

        # 更新统计数据
        for analyst, correct in analyst_accuracy.items():
            if analyst in self.stats:
                self.stats[analyst]["total"] += 1
                if correct:
                    self.stats[analyst]["correct"] += 1

        # 创建反思记录
        reflection = ReflectionRecord(
            record_id=record.record_id,
            ticker=record.ticker,
            trade_date=record.trade_date,
            predicted_signal=predicted_signal,
            actual_signal=actual_signal,
            is_correct=bool(is_correct),  # 确保是 Python bool
            actual_return=float(actual_return),
            actual_return_extended=float(actual_return_extended),
            analyst_accuracy=analyst_accuracy
        )

        self.reflection_records.append(reflection)
        self._save_data()

        logger.info(
            f"反思完成: {record.ticker} - "
            f"预测={predicted_signal}, 实际={actual_signal}, "
            f"{'正确' if is_correct else '错误'} (收益: {actual_return:+.2%})"
        )

        return reflection

    def _return_to_signal(self, return_rate: float) -> str:
        """将收益率转换为信号。"""
        if return_rate > self.hold_threshold:
            return "bullish"
        elif return_rate < -self.hold_threshold:
            return "bearish"
        return "neutral"

    def _decision_to_signal(self, decision: str) -> str:
        """将决策转换为信号。"""
        decision_lower = decision.lower()
        if decision_lower in ["buy", "bullish", "long", "买入"]:
            return "bullish"
        elif decision_lower in ["sell", "bearish", "short", "卖出"]:
            return "bearish"
        return "neutral"

    def _is_signal_correct(self, predicted: str, actual: str) -> bool:
        """检查信号预测是否正确。"""
        if predicted == "neutral":
            return True  # 中性信号不算错
        return predicted == actual

    def _is_prediction_correct(self, decision: str, actual_return: float) -> bool:
        """根据实际收益检查决策是否正确。"""
        decision_lower = decision.lower()
        if decision_lower in ["buy", "bullish", "long", "买入"]:
            return actual_return > 0
        elif decision_lower in ["sell", "bearish", "short", "卖出"]:
            return actual_return < 0
        return abs(actual_return) < self.hold_threshold

    def update_weights(self) -> Dict[str, float]:
        """
        基于历史准确率更新分析师权重。

        使用 Softmax 归一化避免极端权重:
            weight[i] = exp(accuracy[i] / T) / sum(exp(accuracy / T))

        返回:
            更新后的权重字典
        """
        # 检查最小样本数
        total_samples = sum(s["total"] for s in self.stats.values())
        required_samples = self.min_samples * len(self.analyst_names)

        if total_samples < required_samples:
            logger.info(
                f"样本不足 ({total_samples}/{required_samples})，保持当前权重"
            )
            return self.weights

        # 计算准确率
        accuracies = {}
        for analyst, stats in self.stats.items():
            if stats["total"] > 0:
                accuracies[analyst] = stats["correct"] / stats["total"]
            else:
                accuracies[analyst] = 0.5  # 默认准确率

        # Softmax 归一化
        exp_values = {k: np.exp(v / self.temperature) for k, v in accuracies.items()}
        total = sum(exp_values.values())

        if total > 0:
            self.weights = {k: float(v / total) for k, v in exp_values.items()}

        self._save_data()

        logger.info(f"权重已更新: {self.weights}")
        logger.info(f"分析师准确率: {accuracies}")

        return self.weights

    def get_weighted_recommendation(
        self,
        analyst_signals: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        基于分析师信号获取加权推荐。

        参数:
            analyst_signals: 字典，映射分析师名称到其信号

        返回:
            元组 (推荐信号, 置信度)
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
        """获取综合统计信息。"""
        if not self.reflection_records:
            return {"message": "暂无反思记录"}

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
        """生成总结报告。"""
        stats = self.get_statistics()

        if "message" in stats:
            return stats["message"]

        report = f"""
================================================================================
                         量化反思报告
================================================================================

整体表现
--------
总决策数:     {stats['total_decisions']}
正确决策:     {stats['correct_decisions']}
准确率:       {stats['accuracy']:.1%}
平均收益:     {stats['avg_return']:.2%}
累计收益:     {stats['total_return']:.2%}

分析师准确率
------------"""

        for analyst in self.analyst_names:
            s = stats['analyst_stats'].get(analyst, {"correct": 0, "total": 0})
            if s['total'] > 0:
                acc = s['correct'] / s['total']
                report += f"\n  {analyst:20s}: {acc:.1%} ({s['correct']}/{s['total']})"
            else:
                report += f"\n  {analyst:20s}: 暂无数据"

        report += """

当前权重
--------"""
        for analyst, weight in stats['analyst_weights'].items():
            bar = "█" * int(weight * 40)
            report += f"\n  {analyst:20s}: {weight:.1%} {bar}"

        report += "\n" + "=" * 80

        return report
