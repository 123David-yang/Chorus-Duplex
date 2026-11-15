"""一体直出（oneshot_agent）—— 主链路：单次大提示词 → 整条剧本 JSON。

对照 multiagent/（导演 → 泊松调度器 → 逐轮交互的实验臂）：
- generate.py  生成：组装提示词 → LLM → 调用校验 → 返回 Dialogue
- validate.py  校验修复：自动修 / 判失败重试（守门人）

入口：generate.generate_dialogue（由 scripts/text/pipeline.generate_one 分发）。
"""
