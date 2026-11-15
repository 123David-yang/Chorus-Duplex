"""对话数据统计分析（analyze_data.py）—— 检查生成数据的质量与多样性（judge/ 质检工具包）。

用法：
    python -m scripts.text.judge.analyze_data --input data/annotations/side_talk_v17_batch.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="对话数据统计分析")
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    dialogues = []
    for line in open(args.input, encoding="utf-8"):
        if line.strip():
            dialogues.append(json.loads(line))
    if not dialogues:
        print("❌ 空数据"); sys.exit(1)

    n = len(dialogues)
    turns = [len(d["turns"]) for d in dialogues]
    personas = []
    addressee = Counter()
    scenarios = Counter()
    gone = 0
    for d in dialogues:
        scenarios[d["scenario"]] += 1
        for p in d["participants"]:
            personas.append(p["persona"][:40])
            if p.get("state") == "gone":
                gone += 1
        for t in d["turns"]:
            addr = t["addressee"]
            addressee[" + ".join(addr) if isinstance(addr, list) else addr] += 1

    print(f"=== 数据统计 ===")
    print(f"对话条数: {n}")
    print(f"平均轮数: {sum(turns)/n:.1f}（范围 {min(turns)}-{max(turns)}）")
    print(f"场景分布: {dict(scenarios)}")
    print(f"人物人设唯一数: {len(set(personas))} / 总数 {len(personas)}")
    print(f"addressee 分布: {dict(addressee)}")
    print(f"含离场(参与状态管理)的条数: {gone}")

    # 人设多样性抽查
    print(f"\n=== 人设多样性抽查（前 10 个不同人设） ===")
    seen = set()
    for d in dialogues:
        for p in d["participants"]:
            key = p["persona"][:40]
            if key not in seen:
                seen.add(key)
                print(f"  - {key}...")
            if len(seen) >= 10:
                break
        if len(seen) >= 10:
            break


if __name__ == "__main__":
    main()
