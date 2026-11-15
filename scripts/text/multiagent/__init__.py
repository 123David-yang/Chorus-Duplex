"""多 Agent 路径（实验臂，非主链路）——导演 → 泊松调度器 → 逐角色 Agent 逐轮交互。

对照 oneshot（集中式一体直出，放量主力）：论文"三模式协调对比实验"的混合式臂。
入口：generate.generate_dialogue_multi（由 scripts/text/pipeline.generate_one 分发）。
"""
