# scripts/text —— 台词侧（剧本生成）

> 产出标准 jsonl（Dialogue 契约）供 `scripts/sound/` 合成音频；对应论文（arXiv 2607.06540）的对话数据合成流水线。

## 结构（2026-09-16 整理后）

```
scripts/text/
├── voices.py         音色分配 + 性别表（与音频层共用同一函数；assistant 恒 Junhao）
├── pipeline.py       统一入口：generate_one 分发（两个策略）+ sample_persona_pool + CLI
│
├── oneshot_agent/    ★ 主链路：单次大提示词直出整条剧本（放量主力）
│   ├── prompts.py      oneshot_script_prompt + 主题领域 hash
│   ├── generate.py     组装提示词 → LLM → 校验 → Dialogue
│   ├── validate.py     校验修复：自动修（id/kind/clamp/事件限频）+ 判失败重试（守门人）
│   ├── profile.py      参与画像（力度/触发/节奏校验；本路线专用）
│   └── lexicon.py      附和语库加载/抽样（data/lexicons/backchannels.jsonl）
├── multiagent/       实验臂（非主链路）：导演 → 泊松调度器 → 逐角色 Agent 逐轮交互
│   ├── generate.py     导演（mock/LLM）· 逐轮发言 · 驱动循环
│   ├── prompts.py      本路径模板：director_prompt / p1_user_roleplay_multi / assistant_prompt_multi / parse_script
│   └── scheduler.py    泊松调度器（谁发言 / 指向谁 / 参与状态）
├── judge/            质检工具包（不参与生成链路）
│   ├── analyze_data.py 数据统计与多样性检查（CLI）
│   └── review.py       评审提示词（p5 四维评分）与解析
├── utils/            两路线共用（+ 音频侧）
│   ├── schema.py      数据契约：Dialogue / Turn / Event（sound 侧只认它）
│   └── llm.py         LLM 客户端（OpenAI 兼容，env 配置：LLM_BASE_URL/LLM_API_KEY/LLM_MODEL）
└── tmp/              存档（不参与链路，确认无用可整体删除）
    └── legacy_prompts.py  1:1 时代 P1~P4 标签注入式模板（论文 §11.4，已被 JSON 直出取代）
```

场景/主题等**配置不在脚本里**：`scripts/config.py` 读 `data/scenarios.json`（描述/默认人数/画像开关）
与 `data/topics.json`（主题领域）。

## 设计原则（与现状一致）

1. **数据契约先行**：`schema.py` 是唯一数据模型；text 侧输出即 sound 侧输入。
2. **两条生成路线对称**：`oneshot_agent/`（主链路）与 `multiagent/`（实验臂）各自有 `generate.py`，
   产物同构（Dialogue JSONL），下游无感知；`pipeline.generate_one` 负责分发。
3. **配置进数据文件**：主题领域、场景配置、附和语库、人物池均为 `data/` 下数据文件，脚本只留逻辑。
4. **质量两道闸**：生成内联校验（`oneshot_agent/validate.py`，不合格即重试）+ 事后质检
   （`judge/`：统计报告与评审提示词）。
5. **LLM 可插拔**：`utils/llm.py` 走 OpenAI 兼容协议（DeepSeek / 通义 / 本地 vLLM·ollama 皆可）。

## 数据流

```
data/personas（人物池，300→30/条） + data/topics.json + data/scenarios.json + data/lexicons
      → pipeline.generate_one（按 dialogue_id 确定性派生：主题/音色/画像/人数轮数）
      → oneshot_agent.generate（单次 LLM 出整条剧本）→ validate（校验修复/重试）
      → data/annotations/{scenario}_batch.jsonl
      → scripts/sound/（TTS 合成，消费 Dialogue 契约）
```

## 使用约定

- 每个处理任务记录输入、中间结果、运行耗时（logs/run.jsonl 汇总）
- 改行为前先跑 `Experiments/文本侧重构验证/prompt_snapshot.py` 快照，改后 diff 验证（重构/改动自查）
- 新功能先在 `Experiments/` 下验证，成熟后落入本模块
