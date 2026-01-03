"""
基础使用示例
============

演示如何使用量化反思模块。
"""

from quantitative_reflection import QuantitativeReflector


def main():
    # 初始化反思器，设置4个分析师
    reflector = QuantitativeReflector(
        analyst_names=["market", "fundamentals", "news", "social"],
        storage_path="./demo_data",
        min_samples_for_update=2  # 为演示降低阈值
    )

    # 模拟几次交易决策
    decisions_data = [
        {
            "ticker": "600519",  # 贵州茅台
            "trade_date": "2024-01-15",
            "decision": "buy",
            "confidence": 0.8,
            "analyst_signals": {
                "market": "bullish",      # 市场分析师看涨
                "fundamentals": "bullish", # 基本面分析师看涨
                "news": "neutral",         # 新闻分析师中性
                "social": "bullish"        # 社交媒体分析师看涨
            },
            "actual_return": 0.05  # 实际收益 +5%
        },
        {
            "ticker": "000001",  # 平安银行
            "trade_date": "2024-01-16",
            "decision": "hold",
            "confidence": 0.6,
            "analyst_signals": {
                "market": "neutral",
                "fundamentals": "bullish",
                "news": "bearish",  # 新闻分析师错误
                "social": "neutral"
            },
            "actual_return": 0.03  # 实际收益 +3%
        },
        {
            "ticker": "300750",  # 宁德时代
            "trade_date": "2024-01-17",
            "decision": "buy",
            "confidence": 0.75,
            "analyst_signals": {
                "market": "bullish",
                "fundamentals": "bullish",
                "news": "bearish",  # 新闻分析师又错了
                "social": "bullish"
            },
            "actual_return": 0.08  # 实际收益 +8%
        },
        {
            "ticker": "002594",  # 比亚迪
            "trade_date": "2024-01-18",
            "decision": "sell",
            "confidence": 0.7,
            "analyst_signals": {
                "market": "bearish",
                "fundamentals": "bearish",
                "news": "neutral",
                "social": "bearish"
            },
            "actual_return": -0.04  # 实际收益 -4%
        },
    ]

    print("=" * 60)
    print("  量化反思模块 - 演示")
    print("=" * 60)

    # 处理每个决策
    for data in decisions_data:
        print(f"\n[决策] {data['ticker']} @ {data['trade_date']}")
        print(f"  决策: {data['decision']} (置信度: {data['confidence']})")
        print(f"  分析师信号: {data['analyst_signals']}")

        # 记录决策
        record = reflector.record_decision(
            ticker=data["ticker"],
            trade_date=data["trade_date"],
            decision=data["decision"],
            confidence=data["confidence"],
            analyst_signals=data["analyst_signals"]
        )

        # 对结果进行反思
        reflection = reflector.reflect(record, actual_return=data["actual_return"])

        print(f"  实际收益: {data['actual_return']:+.1%}")
        print(f"  预测结果: {'正确' if reflection.is_correct else '错误'}")

    # 基于表现更新权重
    print("\n" + "=" * 60)
    print("  更新权重")
    print("=" * 60)

    new_weights = reflector.update_weights()

    print("\n更新后的权重:")
    for analyst, weight in new_weights.items():
        bar = "█" * int(weight * 40)
        print(f"  {analyst:15s}: {weight:.1%} {bar}")

    # 显示完整报告
    print("\n" + reflector.get_report())

    # 演示加权推荐
    print("\n" + "=" * 60)
    print("  加权推荐示例")
    print("=" * 60)

    new_signals = {
        "market": "bullish",
        "fundamentals": "neutral",
        "news": "bearish",
        "social": "bullish"
    }

    signal, confidence = reflector.get_weighted_recommendation(new_signals)
    print(f"\n输入信号: {new_signals}")
    print(f"加权推荐: {signal} (置信度: {confidence:.1%})")
    print(f"\n说明: 由于 news 分析师历史准确率较低，其看跌信号的权重被降低")


if __name__ == "__main__":
    main()
