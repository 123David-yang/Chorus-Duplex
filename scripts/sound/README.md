# scripts/sound —— 音频侧处理模块（双工语音数据 · 音频流）

> 消费 `scripts/text/` 产出的标准 jsonl（对话 + 事件标签 + 时间轴），
> 负责**文本 → 语音 → 双工音频**的全流程，输出可直接用于训练的数据资产。

现行生产链路的模型选择、GPU 配置和音色注册方式见 [TTS 使用说明](TTS.md)。

## 架构设计原则

1. **数据契约先行**：本模块只消费 `scripts/text/schema.py` 定义的标准数据模型（Dialogue / Turn / Event），
   与 text 侧**通过 schema 解耦**，两条线可独立开发、独立验证。
2. **分层**：I/O → TTS 合成 → 时间轴 → 渲染 → 质量，每层单一职责，层间只通过标准类型传递。
3. **引擎可插拔**：TTS 引擎、音色池均为接口 + 实现（当前实现 CosyVoice 2，可换）。
4. **配置驱动**：采样率、路径、引擎参数集中在 `config.py`，脚本不留硬编码。

## 内部结构

```
scripts/sound/
├── __init__.py
├── README.md            # 本文件
├── schema.py            # 数据契约：VoiceProfile / AudioSegment / DualChannel / SoundTrack
├── config.py            # 配置：采样率、路径、TTS 引擎参数、音色池路径
├── io/                  # 音频 I/O 层
│   ├── __init__.py
│   ├── loader.py        #   统一读取（wav/flac/mp3 → numpy），统一采样率
│   ├── writer.py        #   写出（单通道 wav / 双通道打包 wav）
│   └── audio_ops.py     #   基础操作：重采样、音量调整、拼接、重叠
├── tts/                 # 语音合成层（可插拔）
│   ├── __init__.py
│   ├── base.py          #   TTSEngine 抽象接口（synthesize(text, voice) -> AudioSegment）
│   ├── cosyvoice.py     #   CosyVoice 2 实现（对接本地 cosyVoice 仓库）
│   └── voice_pool.py    #   音色池：角色↔音色绑定、零样本克隆管理
├── timeline/            # 时间轴层（文本事件 → 音频时间点）
│   ├── __init__.py
│   ├── builder.py       #   逐句合成后构建时间轴（谁在何时说）
│   └── events.py        #   打断/附和事件 → 音频时间点映射
├── render/              # 渲染层（双工核心）
│   ├── __init__.py
│   ├── channels.py      #   双通道分配（agent 左 / 用户右）
│   ├── overlap.py       #   打断重叠、附和插入（重叠区域处理）
│   └── mix.py           #   混音、响度归一化、背景噪声叠加（鲁棒性）
├── quality/             # 质量层
│   ├── __init__.py
│   ├── metrics.py       #   时长、SNR、能量、削波检测
│   └── filter.py        #   低质样本剔除（jsonl 记录剔除原因）
├── stats.py             # 全链路统计汇总（细粒度 jsonl，符合 CLAUDE.md 规范）
└── pipeline.py          # 编排主入口：io → tts → timeline → render → quality
```

## 数据流

```
scripts/text 输出 jsonl（对话+事件+时间轴）
    → io/loader 读入
    → tts/ 逐句合成（音色池保证角色一致性）
    → timeline/ 事件对齐到时间点
    → render/ 双通道渲染（打断重叠/附和插入/混音）
    → quality/ 检测过滤
    → data/processed/*.wav + data/annotations/*.jsonl
```

## 使用约定

- 每个处理任务记录输入、中间结果、运行耗时（.jsonl 汇总，收尾后汇总分析）
- TTS 模型实例只在进程初始化时加载一次；大批量处理使用多进程/多线程
- 新功能先在 `Experiments/` 下验证，成熟后落入本模块
- 音频大文件不入 git（见根目录 .gitignore），用 DVC/网盘管理
