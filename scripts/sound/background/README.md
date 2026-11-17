# sound/background —— 背景层（babble 池 + 混入）

非参与型场景的"环境人声/背景层"：让模型学"环境人声不响应"。

## 流程

```
babble.py   多音色中文闲聊错位叠加（鸡尾酒会效应）→ output/bg_pool/babble_XX.wav
mix.py      对话 + 背景按目标 SNR 能量调整 → output/{version}/mixed/{id}.wav
                                            + {id}_background.wav（独立背景轨）
```

## 关键决策

- **背景只混入非 assistant 侧**（2026-09-15 用户定）：双声道产物（左=非assistant，右=assistant）
  背景叠加到左声道；**assistant 侧保持纯净**。单声道旧产物（v10 及以前）整轨叠加。
- SNR 参考能量取"背景实际混入的那一侧"（非 assistant 侧）。
- 独立背景轨与成品一起出：训练时模型可学"这段人声不是对我说的"。
- 背景选择：按 `dialogue_id` 的 crc32 稳定哈希从池中选 → 不同对话背景不同、可复现。
- SNR 层级：15dB 轻背景 / 12dB 电视 / 5dB 咖啡厅 / 0dB 嘈杂（`--snr` 控制）。

## 用法

```bash
PY=python
$PY -m scripts.sound.background.babble --speakers 6 --duration 60 --out output/bg_pool/babble_00.wav
$PY -m scripts.sound.background.mix --dialogue output/version_x/audio/side_talk-0001.wav \
    --background output/bg_pool/babble_00.wav --out-dir /tmp/mixed --snr 12
# 或经编排：$PY -m scripts.run --stage bg / --stage mix
```

## 待优化（详见 `docs/audio_background_notes.md` 8 项）

低通滤波（距离感）、混响/RIR、碎片化 babble、能量包络、频段 SNR、立体声分离、随机起点、无缝循环。
