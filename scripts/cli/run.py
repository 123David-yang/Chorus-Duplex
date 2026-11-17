"""统一编排入口（run.py）—— 全链路一条命令，断点续传 + 多进程并行。

框架三原则：
  1. 产物即状态：输出文件存在 = 该条完成（先写 .part 再原子改名，不留半成品）
  2. 单一入口：各 stage 幂等，重跑自动跳过已完成，失败重试
  3. 细粒度记录：每条任务一行 → logs/run.jsonl（实时 flush，含耗时/重试/错误）

数据流（stage 一一对应数据链路节点）：
  text   对话生成        → data/annotations/{scenario}_batch.jsonl
  audio  TTS + VAD 裁剪  → output/{version}/audio/{dialogue_id}.wav
  bg     背景 babble 池  → output/bg_pool/babble_{seed:02d}.wav（跨版本共享）
  mix    背景混入(SNR)   → output/{version}/mixed/{dialogue_id}.wav + _background.wav

版本化输出（--version）：每轮优化迭代一个版本目录 output/version_i/，
内含 audio/、mixed/、annotations.jsonl（快照）与 version.json（各 stage 参数），便于前后对比。
不带 --version 时落到 output/audio、output/mixed（旧布局）。

用法（用 my_env 的 python）：
    PY=python
    $PY -m scripts.cli.run --stage text  --scenario multi_qa --count 100        # ① 文本
    $PY -m scripts.cli.run --stage audio --scenario multi_qa --workers 2 --version version_2
    $PY -m scripts.cli.run --stage bg    --bg-count 4                           # ③ 背景池
    $PY -m scripts.cli.run --stage mix   --scenario multi_qa --snr 12 --version version_2
    $PY -m scripts.cli.run --stage all   --scenario multi_qa --count 100 --version version_3
    $PY -m scripts.cli.run --smoke                                              # 冒烟测试（mock）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import scripts.config as cfg

HEAVY_MODULES = ("numpy", "soundfile", "onnxruntime")


def log_item(stage: str, item_id: str, status: str, cost_s: float | None = None,
             attempts: int | None = None, error: str | None = None) -> None:
    """任务级记录：每条一行，实时 flush（并行下 stdout 会乱，日志是唯一真相源）。"""
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage,
           "item": item_id, "status": status}
    if cost_s is not None:
        rec["cost_s"] = round(cost_s, 2)
    if attempts is not None:
        rec["attempts"] = attempts
    if error:
        rec["error"] = str(error)[:300]
    with open(cfg.RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def check_deps() -> None:
    missing = []
    for mod in HEAVY_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}（当前 python: {sys.executable}）")
        print("   请在当前环境安装缺少的依赖后重新运行")
        sys.exit(1)


def read_dialogues(input_path: Path) -> list[str]:
    """读取对话文件 → 规范 JSONL 行（兼容美化排版单 JSON）。"""
    from scripts.text.utils.schema import load_dialogue_lines
    return load_dialogue_lines(input_path)


def batch_path(scenario: str) -> Path:
    return cfg.ANNOTATIONS_DIR / f"{scenario}_batch.jsonl"


def summarize(stage: str, ok: int, fail: int, skipped: int, t0: float) -> None:
    print(f"\n【{stage}】新增 {ok} / 失败 {fail} / 跳过(已完成) {skipped}"
          f" | 本段耗时 {(time.time() - t0) / 60:.1f}min")
    log_item(stage, "-", "summary", time.time() - t0)


# ------------------------------------------------------------ 版本化输出（--version）
def version_dir(args: argparse.Namespace) -> Path | None:
    return cfg.OUTPUT_DIR / args.version if getattr(args, "version", None) else None


def record_version(args: argparse.Namespace, stage: str, **info) -> None:
    """把该 stage 的生成参数记进 output/{version}/version.json（供版本间对比）。"""
    vd = version_dir(args)
    if not vd:
        return
    vd.mkdir(parents=True, exist_ok=True)
    p = vd / "version.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "version": args.version, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    data[stage] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- text stage
def stage_text(args: argparse.Namespace) -> None:
    from scripts.text import pipeline

    out_path = Path(args.out) if args.out else batch_path(args.scenario)
    if not out_path.is_absolute():
        out_path = cfg.ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_version(args, "text", scenario=args.scenario, count=args.count,
                   start_index=getattr(args, "start_index", 0),
                   turns=args.turns, gen_mode=args.gen_mode, mock=args.mock,
                   out=str(out_path.relative_to(cfg.ROOT)))

    done: set[str] = set()
    if out_path.exists():
        for line in read_dialogues(out_path):
            done.add(json.loads(line)["dialogue_id"])

    lo = getattr(args, "start_index", 0)
    todo = [i for i in range(lo, lo + args.count) if f"{args.scenario}-{i:04d}" not in done]

    print(f"【text】场景={args.scenario} 目标 {args.count} 条 | 已有 {len(done)} 条 | "
          f"待生成 {len(todo)} 条 | {args.workers} 线程并行"
          + ("（mock 模式）" if args.mock else ""))
    if not todo:
        summarize("text", 0, 0, len(done), time.time())
        return

    # 话题/人物多样性（2026-09-15 用户要求）：批内抽 300 人大池，**每条对话再抽 30 人子集**
    # （此前整批共用同一 30 人池 → 人设/话题扎堆"喝茶抗癌研究"）
    persona_big = None
    if not args.mock:
        cand = sorted((cfg.ROOT / "data" / "personas").glob("*.jsonl"))
        if cand:
            print(f"  PersonaHub 池: {len(cand)} 个文件（批内大池 300 条 → 每条对话独立抽 30 条）")
            persona_big = pipeline.sample_persona_pool(
                [str(p) for p in cand], k=300, rng=random.Random(args.seed))

    def gen_one(i: int):
        rng = random.Random(args.seed * 1000 + i)  # 每条独立种子 → 并行下可复现
        persona_pool = random.Random(args.seed * 1000 + i).sample(persona_big, 30) \
            if persona_big else None
        # 内容分散：用户数 1~6 钟形分布（2026-09-16 用户要求）、轮数 ±2
        n_users = args.users or rng.choices([1, 2, 3, 4, 5, 6],
                                            weights=[1, 3, 3, 3, 2, 1])[0]
        n_turns = max(8, args.turns + rng.randint(-2, 2))
        t0 = time.time()
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                d = pipeline.generate_one(rng, f"{args.scenario}-{i:04d}", args.scenario,
                                          n_users, n_turns, args.mock, persona_pool,
                                          args.gen_mode, args.quality, args.augment_probability)
                if d is not None:
                    return d, attempt, time.time() - t0, None
                last_err = "generate_one 返回 None（导演输出不合法）"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            time.sleep(2 * attempt)
        return None, args.retries, time.time() - t0, last_err

    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(out_path, "a", encoding="utf-8") as fout:
        futures = {ex.submit(gen_one, i): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            did = f"{args.scenario}-{i:04d}"
            d, attempts, cost, err = fut.result()
            if d is None:
                fail += 1
                log_item("text", did, "fail", cost, attempts, err)
                print(f"  ❌ {did}（{attempts} 次重试失败: {err}）")
            else:
                fout.write(d.to_jsonl_line() + "\n")
                fout.flush()
                ok += 1
                log_item("text", did, "ok", cost, attempts)
                print(f"  ✅ {did} | {len(d.turns)} 轮 | {cost:.0f}s | {attempts} 次尝试")
    summarize("text", ok, fail, len(done), t0)
    print(f"→ {out_path}")


# --------------------------------------------------------------- audio stage
_W: dict = {}


def _init_model_worker(tmp_dir: str | None = None) -> None:
    """进程初始化：模型只加载一次（CLAUDE.md 规范）。tmp_dir = 该 stage 产物目录。"""
    from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth
    _W["synth"] = DialogueSynth(tmp_dir=tmp_dir)


class _RssWatchdog:
    """内存看门狗：轮询**直接子进程**的 RSS，超限即 SIGKILL。

    TTS 进程内存随合成时长膨胀（devlog 04），任其增长会触发全局 OOM →
    WSL 拆掉整个会话（用户终端/Claude 全死）。这里只杀超限的 worker，
    配合分块建池的"池损坏 → 记账续跑"路径，代价仅是该条对话失败。
    """

    def __init__(self, limit_mb: int) -> None:
        self.limit_mb = limit_mb
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._th.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(5):
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as f:
                        ppid = int(f.read().split()[3])
                    if ppid != os.getpid():
                        continue
                    with open(f"/proc/{pid}/status") as f:
                        rss_kb = int(f.read().split("VmRSS:")[1].split()[0])
                    if rss_kb > self.limit_mb * 1024:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"  ⚠ worker {pid} RSS {rss_kb / 1048576:.1f}GB 超限 → 已杀（改条将记失败并继续）")
                except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                    continue


TTS_WORKER_MB = 2600      # 单个 TTS worker 实测峰值 RSS 量级（MOSS-ONNX + VAD）
SYSTEM_RESERVE_MB = 3200  # 系统/终端/缓存回落预留：2026-09-15 两次全局 OOM 崩机的教训
TTS_WORKER_KILL_MB = 4200  # 看门狗阈值：单 worker 超此值即杀（实测长对话峰值可达 4.6GB；
                           #   原 5.5GB 阈值太迟——全局 OOM 先发生。workers=1 时
                           #   4.2GB + 系统预留仍在安全线内，超限条走逐轮救援）


def _mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            return next(int(l.split()[1]) for l in f
                        if l.startswith("MemAvailable")) // 1024
    except Exception:
        return None


def _clamp_tts_workers(requested: int) -> int:
    """按可用内存收紧并行度（本机 7.8GB：实算 1 个 worker；2 个会被 OOM 杀，曾两次崩机）。"""
    avail_mb = _mem_available_mb()
    if avail_mb is None:
        return requested
    safe = max(1, (avail_mb - SYSTEM_RESERVE_MB) // TTS_WORKER_MB)
    if requested > safe:
        print(f"⚠ 可用内存 {avail_mb}MB → workers {requested} 收紧为 {safe}"
              f"（每 TTS worker 约需 {TTS_WORKER_MB}MB + 系统预留 {SYSTEM_RESERVE_MB}MB）")
        return safe
    return requested


def _wait_for_memory(need_mb: int, timeout_s: int = 600) -> None:
    """块间内存闸：等 MemAvailable ≥ need_mb 再建下一块进程池（池销毁后 RSS 释放有延迟）。"""
    t0 = time.time()
    while True:
        avail = _mem_available_mb()
        if avail is None or avail >= need_mb:
            return
        if time.time() - t0 > timeout_s:
            print(f"  ⚠ 等内存超时（可用 {avail}MB < 需 {need_mb}MB），仍继续")
            return
        print(f"  ⏳ 内存回落中（可用 {avail}MB < 需 {need_mb}MB），等待…")
        time.sleep(10)


def _audio_one(line: str, out_path: str) -> tuple[int, float]:
    from scripts.sound.synthesis import dialogue_to_wav as d2w
    from scripts.text.utils.schema import Dialogue

    t0 = time.time()
    d = Dialogue.from_jsonl_line(line)
    out = Path(out_path)
    tmp = out.with_name(out.name + ".part.wav")
    # 双声道分轨（左=非assistant 用户/第三方，右=assistant）；顺带回写实现时间线，与音频同目录
    d2w.dialogue_to_stereo(d, _W["synth"], tmp,
                           timeline_path=out.with_name(out.stem + ".timeline.json"))
    os.replace(tmp, out)  # 原子改名：存在即为完整产物
    return len(d.turns), time.time() - t0


def stage_audio(args: argparse.Namespace) -> None:
    check_deps()
    input_path = Path(args.input) if args.input else batch_path(args.scenario)
    if not input_path.is_absolute():
        input_path = cfg.ROOT / input_path
    if not input_path.exists():
        print(f"❌ 输入不存在: {input_path}（先跑 --stage text 或用 --input 指定）")
        sys.exit(1)

    cfg.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    tasks, skipped = [], 0
    for line in read_dialogues(input_path):
        did = json.loads(line)["dialogue_id"]
        out = cfg.AUDIO_DIR / f"{did}.wav"
        if out.exists():
            skipped += 1
            continue
        tasks.append((line, str(out)))

    workers = _clamp_tts_workers(args.workers)
    vd = version_dir(args)
    if vd:
        vd.mkdir(parents=True, exist_ok=True)
        snap = (vd / "annotations.jsonl").resolve()
        if input_path.resolve() != snap:  # 输入就是版本快照时不重复拷贝
            shutil.copyfile(input_path, vd / "annotations.jsonl")
    record_version(args, "audio", input=str(input_path.relative_to(cfg.ROOT)),
                   total=len(tasks) + skipped, workers=workers)

    print(f"【audio】输入 {input_path.name} | 待处理 {len(tasks)} 条 | 跳过已完成 {skipped} | "
          f"{workers} 进程并行（每进程模型加载一次）")
    if vd:
        print(f"  版本输出 → {vd.relative_to(cfg.ROOT)}/audio")
    if not tasks:
        summarize("audio", 0, 0, skipped, time.time())
        return

    t0 = time.time()
    ok = fail = 0
    watchdog = _RssWatchdog(limit_mb=int(os.environ.get("TTS_WORKER_MEM_MB",
                                                        str(TTS_WORKER_KILL_MB))))
    watchdog.start()
    # 分块建池（每块 = workers 条）：worker 处理完 1~N 条后随池销毁，
    # 回收内存——TTS 进程内存随合成时长持续增长（见 devlog 04），长任务不回收必 OOM
    for start in range(0, len(tasks), workers):
        _wait_for_memory(SYSTEM_RESERVE_MB + TTS_WORKER_MB)  # 块间内存闸，防叠罗汉
        chunk = tasks[start:start + workers]
        done_outs: set[str] = set()
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_model_worker,
                                     initargs=(str(cfg.AUDIO_DIR),)) as ex:
                futures = {ex.submit(_audio_one, line, out): out for line, out in chunk}
                for fut in as_completed(futures):
                    out = futures[fut]
                    did = Path(out).stem
                    done_outs.add(out)
                    try:
                        turns, cost = fut.result()
                        ok += 1
                        log_item("audio", did, "ok", cost)
                        print(f"  ✅ {did} | {turns} 轮 | {cost:.0f}s")
                    except Exception as e:
                        fail += 1
                        Path(out + ".part.wav").unlink(missing_ok=True)  # 清理半成品
                        log_item("audio", did, "fail", error=e)
                        print(f"  ❌ {did}: {type(e).__name__}: {e}")
        except Exception as e:  # 池损坏（如 worker 触内存上限被杀）→ 记账后继续下一块，不再整批全灭
            for line, out in chunk:
                if out in done_outs:
                    continue
                fail += 1
                Path(out + ".part.wav").unlink(missing_ok=True)
                log_item("audio", Path(out).stem, "fail", error=f"pool broken: {e}")
                print(f"  ❌ {Path(out).stem}: 池损坏（{type(e).__name__}）")
    watchdog.stop()
    summarize("audio", ok, fail, skipped, t0)


# ------------------------------------------------------------------ bg stage
def _bg_one(seed: int, speakers: int, duration: float, out_path: str) -> None:
    from scripts.sound.background.babble import make_babble
    tmp = out_path + ".part.wav"
    make_babble(_W["synth"], speakers, duration, tmp, seed=seed)
    os.replace(tmp, out_path)


def stage_bg(args: argparse.Namespace) -> None:
    check_deps()
    cfg.BG_POOL_DIR.mkdir(parents=True, exist_ok=True)
    all_paths = [cfg.BG_POOL_DIR / f"babble_{s:02d}.wav" for s in range(args.bg_count)]
    tasks = [p for p in all_paths if not p.exists()]

    print(f"【bg】背景池 {args.bg_count} 条 | 待生成 {len(tasks)} | 跳过 {len(all_paths) - len(tasks)} | "
          f"每条约 {args.bg_duration:.0f}s / {args.bg_speakers} 人闲聊")
    record_version(args, "bg", bg_count=args.bg_count,
                   bg_speakers=args.bg_speakers, bg_duration=args.bg_duration)
    if not tasks:
        summarize("bg", 0, 0, len(all_paths), time.time())
        return

    t0 = time.time()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=min(len(tasks), _clamp_tts_workers(args.workers)),
                             initializer=_init_model_worker,
                             initargs=(str(cfg.BG_POOL_DIR),)) as ex:
        futures = {ex.submit(_bg_one, int(p.stem.split("_")[1]), args.bg_speakers,
                             args.bg_duration, str(p)): p for p in tasks}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
                ok += 1
                log_item("bg", p.stem, "ok")
                print(f"  ✅ {p.name}")
            except Exception as e:
                fail += 1
                Path(str(p) + ".part.wav").unlink(missing_ok=True)
                log_item("bg", p.stem, "fail", error=e)
                print(f"  ❌ {p.name}: {type(e).__name__}: {e}")
    summarize("bg", ok, fail, len(all_paths) - len(tasks), t0)


# ----------------------------------------------------------------- mix stage
def _mix_one(dialogue_path: str, out_mixed: str, snr: float,
             bg_path: str | None = None, keep_bg: bool = False) -> None:
    """混入背景：bg_path 为空 → 从 BG_SOURCE_DIR **即时随机拼接**（用完即弃，不落持久副本）。"""
    import random as _random

    import soundfile as sf

    from scripts.sound.background.mix import mix

    did = Path(dialogue_path).stem
    if bg_path is None:
        from scripts.sound.background.clips import stitch_clips
        rooms = sorted(cfg.BG_SOURCE_DIR.glob("rm*/babb"))
        if not rooms:
            raise FileNotFoundError(f"背景素材不存在: {cfg.BG_SOURCE_DIR}/rm*/babb")
        # 随机匹配：按对话 id 随机房间 + 随机片段（稳定可复现），临时文件按进程复用
        room = _random.Random(zlib.crc32(f"room-{did}".encode())).choice(rooms)
        bg_path = str(Path(out_mixed).with_name(f"_tmp_bg_{os.getpid()}.wav"))
        stitch_clips(room, bg_path, sf.info(dialogue_path).duration + 2.0, seed_key=did)

    tmp = out_mixed + ".part.wav"
    out_bg = str(Path(out_mixed).with_name(f"{did}_background.wav")) if keep_bg else False
    mix(dialogue_path, bg_path, snr_db=snr, out_mixed=tmp, out_bg=out_bg, seed_key=did)
    os.replace(tmp, out_mixed)


def stage_mix(args: argparse.Namespace) -> None:
    check_deps()
    live = bool(sorted(cfg.BG_SOURCE_DIR.glob("rm*/babb")))
    pool = [] if live else sorted(cfg.BG_POOL_DIR.glob("*.wav"))
    if not live and not pool:
        print(f"❌ 无背景素材：{cfg.BG_SOURCE_DIR}/rm*/babb 不存在且背景池为空（可跑 --stage bg）")
        sys.exit(1)

    dialogues = sorted(cfg.AUDIO_DIR.glob(f"{args.scenario}-*.wav")) if args.scenario != "all" \
        else sorted(cfg.AUDIO_DIR.glob("*.wav"))
    cfg.MIXED_DIR.mkdir(parents=True, exist_ok=True)
    keep_bg = bool(getattr(args, "keep_bg", False))

    tasks, skipped = [], 0
    for dp in dialogues:
        did = dp.stem
        out_mixed = cfg.MIXED_DIR / f"{did}.wav"
        if out_mixed.exists():
            skipped += 1
            continue
        bg = None if live else str(pool[zlib.crc32(did.encode()) % len(pool)])
        tasks.append((str(dp), str(out_mixed), bg, keep_bg))

    src = "即时随机拼接" if live else f"预拼接池 {len(pool)} 条"
    print(f"【mix】待处理 {len(tasks)} 条 | 跳过已完成 {skipped} | 背景：{src} | SNR {args.snr}dB"
          + ("（保留背景轨）" if keep_bg else ""))
    record_version(args, "mix", snr=args.snr, bg=src, keep_bg=keep_bg, total=len(tasks) + skipped)
    if not tasks:
        summarize("mix", 0, 0, skipped, time.time())
        return

    t0 = time.time()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_mix_one, t[0], t[1], args.snr, t[2], t[3]): t[1] for t in tasks}
        for fut in as_completed(futures):
            out_mixed = futures[fut]
            did = Path(out_mixed).stem
            try:
                fut.result()
                ok += 1
                log_item("mix", did, "ok")
                print(f"  ✅ {did}")
            except Exception as e:
                fail += 1
                Path(out_mixed + ".part.wav").unlink(missing_ok=True)
                log_item("mix", did, "fail", error=e)
                print(f"  ❌ {did}: {type(e).__name__}: {e}")
    summarize("mix", ok, fail, skipped, t0)


# ----------------------------------------------------------------- smoke test
def run_smoke() -> None:
    """1 条 mock 对话全链路 text→audio→bg→mix（验证环境与链路，约 2-3 分钟）。"""
    print("=== 冒烟测试：mock 数据全链路 ===\n")
    ap = build_parser()
    args = ap.parse_args([
        "--stage", "text", "--scenario", "smoke", "--count", "1", "--turns", "3",
        "--users", "2", "--mock", "--workers", "1", "--out", "data/annotations/smoke_batch.jsonl",
    ])
    stage_text(args)
    args = ap.parse_args(["--stage", "audio", "--scenario", "smoke", "--workers", "1"])
    stage_audio(args)
    args = ap.parse_args(["--stage", "bg", "--bg-count", "1", "--bg-speakers", "2",
                          "--bg-duration", "8", "--workers", "1"])
    stage_bg(args)
    args = ap.parse_args(["--stage", "mix", "--scenario", "smoke", "--snr", "12", "--workers", "2"])
    stage_mix(args)

    checks = [cfg.ANNOTATIONS_DIR / "smoke_batch.jsonl",
              cfg.AUDIO_DIR / "smoke-0000.wav",
              cfg.MIXED_DIR / "smoke-0000.wav",
              cfg.MIXED_DIR / "smoke-0000_background.wav"]
    print("\n=== 冒烟结果 ===")
    all_ok = True
    for p in checks:
        exists = p.exists()
        all_ok &= exists
        print(f"  {'✅' if exists else '❌'} {p.relative_to(cfg.ROOT)}")
    print("✅ 全链路通过" if all_ok else "❌ 链路有断点，见上方日志")
    sys.exit(0 if all_ok else 1)


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ChorusDuplex 统一编排入口（断点续传 + 并行）")
    ap.add_argument("--stage", choices=["text", "audio", "bg", "mix", "all"], default="all")
    ap.add_argument("--scenario", default="multi_qa", help="场景名（multi_qa/side_talk/social）")
    ap.add_argument("--count", type=int, default=10, help="text: 目标总条数")
    ap.add_argument("--turns", type=int, default=10, help="text: 每条对话轮数")
    ap.add_argument("--users", type=int, default=None, help="text: 用户数（默认按场景）")
    ap.add_argument("--mock", action="store_true", help="text: 无 LLM 链路自测")
    ap.add_argument("--gen-mode", choices=["multi_agent", "oneshot"], default="oneshot",
                    help="text: oneshot=一体直出（默认，单次调用出完整剧本）")
    ap.add_argument("--quality", choices=["rules", "llm"], default="llm", help="text: 默认七维审核；rules 仅供开发检查")
    ap.add_argument("--augment-probability", type=float, default=0.0, help="text: 可选附和候选采样概率，默认不增广")
    ap.add_argument("--input", default=None, help="audio: 指定 jsonl（默认 {scenario}_batch.jsonl）")
    ap.add_argument("--version", default=None,
                    help="版本目录名（如 version_2）：audio/mix 产物落到 output/{version}/，便于对比优化前后")
    ap.add_argument("--out", default=None, help="text: 输出 jsonl（默认 {scenario}_batch.jsonl）")
    ap.add_argument("--start-index", type=int, default=0,
                    help="text: 起始编号（追加批次用：如 100 → 从 {scenario}-0100 起）")
    ap.add_argument("--bg-count", type=int, default=4, help="bg: 背景池条数")
    ap.add_argument("--bg-speakers", type=int, default=6, help="bg: 每条的说话人数")
    ap.add_argument("--bg-duration", type=float, default=30.0, help="bg: 每条时长（秒）")
    ap.add_argument("--snr", type=float, default=12.0, help="mix: 目标信噪比 dB")
    ap.add_argument("--keep-bg", action="store_true",
                    help="mix: 保留独立背景轨副本（默认不留，省空间）")
    ap.add_argument("--workers", type=int, default=4, help="并行度（text 线程 / audio·bg·mix 进程）")
    ap.add_argument("--retries", type=int, default=3, help="text: 每条最大尝试次数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试（mock 单条全链路）")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    os.chdir(cfg.ROOT)
    cfg.load_env()

    if args.smoke:
        run_smoke()

    if args.version:
        vd = cfg.OUTPUT_DIR / args.version
        cfg.AUDIO_DIR = vd / "audio"
        cfg.MIXED_DIR = vd / "mixed"
        print(f"📦 版本输出 → {vd.relative_to(cfg.ROOT)}/（bg_pool 跨版本共享）")
    elif args.stage in ("audio", "mix", "all"):
        print("ℹ 未指定 --version：audio/mix 产物将落到旧布局 output/audio|mixed/")

    stages = ["text", "audio", "bg", "mix"] if args.stage == "all" else [args.stage]
    t0 = time.time()
    for s in stages:
        {"text": stage_text, "audio": stage_audio, "bg": stage_bg, "mix": stage_mix}[s](args)
        print()
    print(f"总耗时 {(time.time() - t0) / 60:.1f}min | 记录 → {cfg.RUN_LOG.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
