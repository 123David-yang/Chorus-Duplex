"""文本侧统一入口（pipeline.py）：generate_one 分发（oneshot 主链路 / multiagent 实验臂）+ CLI。

用法：
    python -m scripts.text.pipeline --count 5 --scenario side_talk --gen-mode oneshot
    python -m scripts.text.pipeline --count 3 --mock   # 无 key 链路自测

实现分居两处（产物同构，下游无感知）：
    oneshot_agent/     一体直出——放量主力（单次大提示词出整条剧本，generate + validate）
    multiagent/        多 Agent——导演 → 泊松调度器 → 逐轮交互（论文对比臂）
场景（三类，配置见 data/scenarios.json）：multi_qa 参与型 / side_talk 非参与型 / social 过渡带
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
import threading
import zlib
import os
from copy import deepcopy
from scripts.config import LOGS_DIR
from .quality import audit, coverage_plan
from .judge.review import review_dialogue, augment_dialogue, revise_dialogue

_QUALITY_LOCK = threading.Lock()
from pathlib import Path

from scripts.config import DEFAULT_USERS, SCENARIOS

from .multiagent.generate import generate_dialogue_multi
from .oneshot_agent.generate import generate_dialogue


def sample_persona_pool(paths: str | list, k: int, rng: random.Random) -> list[str]:
    """从 PersonaHub jsonl 蓄水池抽样 k 条 persona（不整文件载入内存）。

    paths 支持单文件或**多文件列表**（人物池可拆成多份入库，这里合并成同一池采样）。
    """
    import json as _json
    from pathlib import Path as _Path
    if isinstance(paths, (str, _Path)):
        paths = [paths]
    reservoir: list[str] = []
    i = 0  # 已见条目数（跨文件累计）
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if i < k:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, i)
                    if j < k:
                        reservoir[j] = line
                i += 1
    personas = []
    for l in reservoir:
        try:
            personas.append(_json.loads(l)["persona"])
        except Exception:
            continue
    return personas


def generate_one(rng: random.Random, dialogue_id: str, scenario: str,
                 num_users: int, turns: int, mock: bool,
                 persona_pool: list[str] | None = None,
                 gen_mode: str = "multi_agent",
                 quality_mode: str = "llm", augment_probability: float = 0.0,
                 report_sink=None) -> "object | None":
    """单条对话（分发）。

    gen_mode: "oneshot"（一体直出：单次调用产出完整剧本，含时序/打断/附和）
              "multi_agent"（导演→调度器→交互，N+1 次调用）
    """
    if quality_mode not in ("rules", "llm"):
        raise ValueError("quality_mode must be rules or llm")
    if not 0 <= augment_probability <= 1: raise ValueError("augmentation probability outside [0,1]")
    if augment_probability and (mock or quality_mode != "llm"):
        raise ValueError("augmentation requires real LLM review")
    started=time.time()
    if gen_mode == "oneshot" and not mock:
        d = generate_dialogue(dialogue_id, scenario, num_users, turns, persona_pool, rng)
    else:
        d = generate_dialogue_multi(rng, dialogue_id, scenario, num_users, turns, mock, persona_pool)
    if d is None:
        report={"decision":"reject", "reasons":["generation_or_build_validation_failed"]}
    else:
        from .voices import voices_for_dialogue
        d.profile["voice_assignments"] = voices_for_dialogue(dialogue_id, num_users)
        d.profile["tts_backend"] = os.environ.get("TTS_BACKEND", "moss_onnx")
        d.profile["required_modules"] = coverage_plan(dialogue_id, turns)
        checks=audit(d)
        report={"decision":"rules_only" if checks["passed"] else "reject", "deterministic":checks,
                "reasons":[] if checks["passed"] else ["deterministic_failure"]}
        if checks["passed"] and not mock and quality_mode=="llm":
            report=review_dialogue(d)
        if not mock and quality_mode=="llm" and report["decision"] in ("reject","quarantine") and not any(
                str(r).startswith("review_unavailable") for r in report.get("reasons", [])):
            original_report=report
            revised,revision=revise_dialogue(d,report,turns)
            if revised is not None:
                revision["candidate"]=json.loads(revised.to_jsonl_line())
                revised_report=review_dialogue(revised)
                if revised_report["decision"]=="accept":
                    d,report=revised,revised_report
                revision["review"]=deepcopy(revised_report)
            report["revision"]={"original_report":dict(original_report),**revision}
        if report["decision"]=="accept" and augment_probability and len(d.turns)<turns+5:
            d, augmentation=augment_dialogue(d,augment_probability,zlib.crc32(dialogue_id.encode()))
            report["augmentation"]=augmentation
        d.profile["quality_status"] = report["decision"]
    record={"dialogue_id":dialogue_id,"mode":gen_mode,"mock":mock,
            "elapsed_s":round(time.time()-started,3),"report":report,
            "dialogue":json.loads(d.to_jsonl_line()) if d else None}
    LOGS_DIR.mkdir(parents=True,exist_ok=True)
    with _QUALITY_LOCK, (LOGS_DIR/"text_quality_v2.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n"); f.flush()
    if report_sink is not None:
        report_sink(record)
    if report["decision"] in ("reject","quarantine"):
        print(f"    quality {dialogue_id}: {report['decision']} {report.get('reasons', [])}")
        return None
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="ChorusDuplex 文本侧一对多数据合成（oneshot / multi_agent）")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--scenario", choices=SCENARIOS.keys(), default="multi_qa")
    ap.add_argument("--users", type=int, default=None, help="用户数（默认按场景）")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--out", default="data/annotations/multi_test.jsonl")
    ap.add_argument("--log", default="logs/text_pipeline.jsonl")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--gen-mode", choices=["multi_agent", "oneshot"], default="multi_agent",
                    help="oneshot=一体直出（单次调用出完整剧本，推荐）")
    ap.add_argument("--personas", default=None,
                    help="PersonaHub jsonl 路径（导演从池中抽样选人设）；默认 data/personas/ 下自动找")
    ap.add_argument("--quality", choices=["rules", "llm"], default="llm")
    ap.add_argument("--augment-probability", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from scripts.config import load_env
    if not args.mock: load_env()
    rng = random.Random(args.seed)
    num_users = args.users or DEFAULT_USERS[args.scenario]
    out_path, log_path = Path(args.out), Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    # PersonaHub 池：自动找 data/personas/*.jsonl（多文件合并为一池），或 --personas 指定
    persona_paths: list[str] = []
    if args.personas:
        persona_paths = [args.personas]
    elif not args.mock:
        persona_paths = [str(p) for p in sorted(Path("data/personas").glob("*.jsonl"))]
    persona_paths = [p for p in persona_paths if Path(p).exists()]
    persona_pool = None
    if persona_paths:
        print(f"加载 PersonaHub 池: {len(persona_paths)} 个文件（蓄水池抽样 30 条/条数据）")
        persona_pool = sample_persona_pool(persona_paths, k=30, rng=rng)

    with open(out_path, "w", encoding="utf-8") as fout, open(log_path, "w", encoding="utf-8") as flog:
        for i in range(args.count):
            t0 = time.time()
            try:
                d = generate_one(rng, f"{args.scenario}-{i:04d}", args.scenario, num_users,
                                 args.turns, args.mock, persona_pool, args.gen_mode, args.quality, args.augment_probability)
                if d is None:
                    fail += 1
                    continue
                fout.write(d.to_jsonl_line() + "\n")
                ok += 1
                flog.write(json.dumps({
                    "dialogue_id": d.dialogue_id, "scenario": d.scenario,
                    "participants": len(d.participants), "turns": len(d.turns),
                    "cost_s": round(time.time() - t0, 2), "status": "ok",
                }, ensure_ascii=False) + "\n")
            except Exception:
                fail += 1
                traceback.print_exc()
                flog.write(json.dumps({"dialogue_id": f"{args.scenario}-{i:04d}", "status": "fail"}, ensure_ascii=False) + "\n")

    print(f"完成: 成功 {ok} 条 / 失败 {fail} 条 → {out_path}（场景={args.scenario}，用户数={num_users}）")


if __name__ == "__main__":
    sys.exit(main())
