# scripts/ 目录结构

数据生产主链路（剧本 → 音频 → 混音产物）与配套工具。命令均在仓库根目录执行，Python 用 `my_env`。

| 路径 | 作用 |
|---|---|
| `run.py` | **主入口**：text / audio / bg / mix / all / smoke 各 stage 编排（`$PY -m scripts.run --help`） |
| `config.py` | 全局配置（路径、模型位置、常量） |
| `push_dataset.py` | 推送版本产物到 HuggingFace（audio→FLAC、走 hf-mirror；`--count 0` 全量推送） |
| `text/` | 剧本链路：LLM 生成 → 校验修复 → 参与画像（详见 `text/README.md`） |
| `sound/` | 音频链路：TTS → VAD → 响度 → 放置混音 → 时间线（详见 `sound/README.md`） |
| `paper/` | 论文构建：`build.py`（章节合并）/ `build_pdf.py`（PDF 导出 + 页码） |
| `_legacy/` | 已被取代的旧脚本（保留作参考；见 `_legacy/README.md`） |

约定：

- 新功能脚本按模块归入对应子目录；本目录不再新增散放的顶层脚本（入口与全局配置除外）
- 临时/验证性逻辑放 `Experiments/`（中文子目录，见项目 CLAUDE.md）
- 被取代的脚本**移动进 `_legacy/`**（附 README 说明），不直接删除
