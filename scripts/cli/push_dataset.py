"""推送版本产物到 Hugging Face 数据集仓库（可复用：预览 N 条 / 全量）。

- 音频转 FLAC（无损、体积小；大 wav 在 hf-mirror 上走 multipart 易失败）
- 走 hf-mirror 端点 + 禁用 Xet + 修正 hf-mirror multipart 域名拼写问题（同 push_hf.py）
- 产物（单文件夹布局，2026-09-15 用户定）：
  audio/{id}.flac（干净版）+ audio/{id}_babble.flac（带背景版，来自 {version}/mixed/）
  + audio/{id}.timeline.json + annotations.jsonl + README.md / README_zh.md + scripts.md + stats.md

用法：
    HF_TOKEN_FILE=/path/to/hf_token.txt \\
    python -m scripts.cli.push_dataset \\
        --version version_10 --count 20 --repo BiggestWave/Chorus-Duplex
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_default_hf_home = (Path("D:/DuplexVoice_resources/hf_home") if os.name == "nt"
                    else Path("/mnt/d/DuplexVoice_resources/hf_home")
                    if Path("/mnt/d").is_dir() else Path.home() / ".cache" / "huggingface")
HF_HOME = Path(os.environ.get("HF_HOME", _default_hf_home))


def _setup_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))

    import httpx  # noqa: PLC0415
    _orig_send = httpx.Client.send

    def _patched(self, request, **kwargs):
        if "hf-mirror.org" in str(request.url):
            request.url = httpx.URL(str(request.url).replace("hf-mirror.org", "hf-mirror.com"))
        return _orig_send(self, request, **kwargs)

    httpx.Client.send = _patched


MY_ENV_PY = os.environ.get("CHORUS_AUDIO_PYTHON", sys.executable)


def stats_markdown(picked: list[tuple[str, str]], audio_dirs) -> str:
    """自动统计报告（随推送重新生成，数据分布随音频条数增长）。"""
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    dials = [json.loads(l) for _, l in picked]
    kinds, trig, inten, spk = Counter(), Counter(), Counter(), Counter()
    victims, interrupters, bc_host, bc_giver = Counter(), Counter(), Counter(), Counter()
    asst_single, asst_multi, events = Counter(), Counter(), Counter()
    topic_kw = Counter()
    KEY = ["茶", "癌", "研究", "数据", "实验", "论文", "肠道", "菌群", "睡眠", "运动", "旅行",
           "美食", "育儿", "教育", "工作", "宠物", "健身", "电影", "游戏", "装修",
           "购物", "理财", "数码", "节庆", "邻居", "学习"]
    turns_total = 0
    for d in dials:
        inten[(d.get("profile") or {}).get("intensity", "?")] += 1
        for i, t in enumerate(d["turns"]):
            turns_total += 1
            kinds[t["kind"]] += 1
            spk[t["speaker"]] += 1
            if t["speaker"] == "assistant":
                if t.get("trigger"):
                    trig[t["trigger"]] += 1
                (asst_multi if isinstance(t["addressee"], list) else asst_single)[str(t["addressee"])] += 1
            if t["kind"] == "interruption" and i > 0:
                victims[d["turns"][i - 1]["speaker"]] += 1
                interrupters[t["speaker"]] += 1
            if t["kind"] == "backchannel":
                bc_giver[t["speaker"]] += 1
                host = next((x["speaker"] for x in reversed(d["turns"][:i]) if x["kind"] != "backchannel"), None)
                bc_host[host or "(无)"] += 1
            for e in t.get("events", []):
                events[e["type"]] += 1
        text = d.get("topic", "") + d.get("setting", "") + "".join(t["text"] for t in d["turns"])
        for k in KEY:
            if k in text:
                topic_kw[k] += 1

    durs = []
    if isinstance(audio_dirs, Path):
        audio_dirs = [audio_dirs]
    for did, _ in picked:
        for ad in audio_dirs:                      # 追加模式下旧条目的 timeline 在旧版本目录
            tl = Path(ad) / f"{did}.timeline.json"
            if tl.exists():
                rows = json.loads(tl.read_text(encoding="utf-8"))
                if rows:
                    durs.append(max(r["end_sec"] for r in rows))
                break
    n = len(dials)
    pct = lambda c: f"{c / max(turns_total, 1) * 100:.1f}%"  # noqa: E731

    L = ["# ChorusDuplex-SideTalk · 统计报告（Stats）", "",
         f"> 自动生成：{time.strftime('%Y-%m-%d %H:%M')}；随推送更新（当前 {n} 条对话 / {turns_total} 轮）。", "",
         "## 规模", "",
         f"- 对话：**{n}** 条（每条 1~6 位用户 + 1 个 AI 助手，可选第三方）",
         f"- 轮次：**{turns_total}**",
         f"- 音频：**{sum(durs) / 60:.1f} 分钟**（均值 {sum(durs) / max(len(durs), 1):.0f}s/条，"
         f"48kHz 立体声：左=非assistant（用户/第三方）· 右=assistant）", "",
         "## 轮次类型（kind）", "", "| kind | 次数 | 占比 |", "|---|---|---|"]
    for k, c in kinds.most_common():
        L.append(f"| {k} | {c} | {pct(c)} |")
    L += ["", "## AI 参与画像", "",
          f"- 力度：{' / '.join(f'{k} {v}' for k, v in inten.most_common())}",
          f"- 触发：{' / '.join(f'{k} {v}' for k, v in trig.most_common())}",
          f"- 指向：单目标 {sum(asst_single.values())} / 多目标 {sum(asst_multi.values())}"
          f"（多目标占 {sum(asst_multi.values()) / max(sum(asst_multi.values()) + sum(asst_single.values()), 1) * 100:.0f}%）", "",
          "## 说话人分布（轮数）", "", "| speaker | 轮数 |", "|---|---|"]
    for s, c in spk.most_common():
        L.append(f"| {s} | {c} |")
    L += ["", "## 双工事件分布", "",
          f"- **打断** {sum(victims.values())} 次：受害者 {' / '.join(f'{k} {v}' for k, v in victims.most_common())}；"
          f"打断者 {' / '.join(f'{k} {v}' for k, v in interrupters.most_common())}",
          f"- **附和** {sum(bc_giver.values())} 次：被附和者 {' / '.join(f'{k} {v}' for k, v in bc_host.most_common())}；"
          f"附和者 {' / '.join(f'{k} {v}' for k, v in bc_giver.most_common())}", "",
          "## 非语音事件", ""]
    L.append("- " + (" / ".join(f"{k} {v}" for k, v in events.most_common()) if events else "（暂无）"))
    L += ["", "## 话题关键词命中（多样性监控）", "", "| 关键词 | 命中条数 | 占比 |", "|---|---|---|"]
    for k, c in topic_kw.most_common(10):
        L.append(f"| {k} | {c} | {c / max(n, 1) * 100:.0f}% |")
    asst_int = victims.get("assistant", 0)
    asst_bc = bc_giver.get("assistant", 0)
    L += ["", f"> 说明：统计脚本随 `scripts/push_dataset.py` 自动生成。当前已知缺口（见项目 devlog 06）："
          f"assistant 被打断 {asst_int} 次（barge-in 目标场景）、"
          f"assistant 主动附和 {asst_bc} 次（主动插话能力）。"]
    return "\n".join(L) + "\n"


def scripts_markdown(picked: list[tuple[str, str]]) -> str:
    """按人读格式导出剧本（场景/人物 + 逐轮台词 + 关键标注）。"""
    import json  # noqa: PLC0415
    out = ["# ChorusDuplex-SideTalk · 剧本（Scripts）",
           "",
           "> 人读版；机器可读的完整标注见 `annotations.jsonl`。",
           ""]
    for _, line in picked:
        d = json.loads(line)
        out.append(f"## {d['dialogue_id']}")
        out.append(f"- 场景：{d.get('setting', '')}")
        out.append(f"- 主题：{d.get('topic', '')}")
        out.append(f"- 人物：" + "；".join(
            f"{p['id']}={p.get('persona', '')}（{p.get('style', '')}）" for p in d.get("participants", [])))
        prof = d.get("profile") or {}
        if prof:
            out.append(f"- AI 参与画像：{prof.get('intensity', '')}（触发：{'/'.join(prof.get('triggers', []))}）")
        out.append("")
        out.append("```")
        for t in d["turns"]:
            addr = t["addressee"] if isinstance(t["addressee"], str) else "+".join(t["addressee"])
            tag = t.get("trigger") or t["kind"]
            gap = t.get("gap_before_ms")
            out.append(f"[{t['turn_id']:>2}] {t['speaker']:<15} → {addr:<14} [{tag}]  {t['text']}")
        out.append("```")
        out.append("")
    return "\n".join(out)


def to_flac(wav: Path, flac: Path) -> None:
    """wav → FLAC；当前解释器无 soundfile 时回退用 my_env 子进程转换。"""
    flac.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # noqa: PLC0415
        d, sr = sf.read(str(wav), dtype="float32")
        sf.write(str(flac), d, sr, format="FLAC")
    except ImportError:
        import subprocess  # noqa: PLC0415
        code = ("import soundfile as sf, sys;"
                "d, sr = sf.read(sys.argv[1], dtype='float32');"
                "sf.write(sys.argv[2], d, sr, format='FLAC')")
        subprocess.run([MY_ENV_PY, "-c", code, str(wav), str(flac)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="推送版本产物到 HF 数据集仓库")
    ap.add_argument("--version", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--count", type=int, default=0, help="只推前 N 条（0 = 全部）")
    ap.add_argument("--public", action="store_true", help="仓库设为公开（默认私有）")
    ap.add_argument("--prefix", default="", help="仓库内子目录前缀（如 sidetalk-preview）")
    ap.add_argument("--no-create", action="store_true",
                    help="跳过建仓（hf-mirror 不支持 create API，仓库需先在 HF 网站建好）")
    ap.add_argument("--no-babble", action="store_true",
                    help="不传 _babble 变体（默认 mixed/ 存在即上传，单文件夹 + 后缀区分）")
    ap.add_argument("--append-from", default=None,
                    help="追加模式：旧版 annotations.jsonl 路径（如 output/version_17/annotations.jsonl）——"
                         "音频只传本版本新条目，但 annotations/scripts/stats 按 旧+新 合并生成（不覆盖旧数据）")
    ap.add_argument("--token-file", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    vdir = root / "output" / args.version
    audio_dir = vdir / "audio"
    ann_path = vdir / "annotations.jsonl"
    readme = vdir / "README.md"
    for p in (audio_dir, ann_path):
        if not p.exists():
            print(f"❌ 缺失: {p}"); sys.exit(1)

    token_file = args.token_file or os.environ.get("HF_TOKEN_FILE")
    desktop_token = Path.home() / "Desktop" / "pass-word" / "hf_token.txt"
    if not token_file and desktop_token.is_file():
        token_file = desktop_token
    # Without an explicit file, the Hugging Face SDK can use its own login store.
    token = Path(token_file).read_text(encoding="utf-8-sig").strip() if token_file else None

    _setup_hf_env()
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    if args.no_create:
        print(f"仓库: {args.repo}（跳过建仓；子目录 {args.prefix or '（根）'}）")
    else:
        api.create_repo(repo_id=args.repo, repo_type="dataset",
                        private=not args.public, exist_ok=True)
        print(f"仓库: {args.repo}（{'公开' if args.public else '私有'}）")
    pre = (args.prefix.rstrip("/") + "/") if args.prefix else ""

    # 选取已完成的对话（按 annotations 顺序）
    import json  # noqa: PLC0415
    lines = [l for l in ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = []
    for line in lines:
        did = json.loads(line)["dialogue_id"]
        if (audio_dir / f"{did}.wav").exists():
            picked.append((did, line))
        if args.count and len(picked) >= args.count:
            break

    # 追加模式：文本类产物（annotations/scripts/stats）按"旧 + 新"合并生成；音频只传本版本新条目
    picked_all = picked
    stat_audio_dirs = [audio_dir]
    if args.append_from:
        prev_path = Path(args.append_from)
        if not prev_path.is_absolute():
            prev_path = root / prev_path
        prev_lines = [l for l in prev_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        prev = [(json.loads(l)["dialogue_id"], l) for l in prev_lines]
        prev_dirs = [prev_path.parent / "audio"]          # 旧版的 timeline 用于时长统计
        picked_all = prev + picked
        stat_audio_dirs = [audio_dir] + prev_dirs
        print(f"追加模式：旧 {len(prev)} + 新 {len(picked)} = {len(picked_all)} 条（音频仅上传新 {len(picked)} 条）")

    stage = vdir / "_hf_stage"
    mixed_dir = vdir / "mixed"
    uploads: list[tuple[Path, str]] = []
    n_babble = 0
    for did, _ in picked:
        flac = stage / f"audio/{did}.flac"
        if not flac.exists():
            to_flac(audio_dir / f"{did}.wav", flac)
        uploads.append((flac, f"{pre}audio/{did}.flac"))
        tl = audio_dir / f"{did}.timeline.json"
        if tl.exists():
            uploads.append((tl, f"{pre}audio/{did}.timeline.json"))
        # 带背景说话人版（_babble 后缀）：来自 {version}/mixed/，单文件夹布局
        mix_wav = mixed_dir / f"{did}.wav"
        if mix_wav.exists() and not args.no_babble:
            flac_babble = stage / f"audio/{did}_babble.flac"
            if not flac_babble.exists():
                to_flac(mix_wav, flac_babble)
            uploads.append((flac_babble, f"{pre}audio/{did}_babble.flac"))
            n_babble += 1

    # 预览时同步裁剪 annotations（保持与音频一致）
    ann_out = stage / "annotations.jsonl"
    ann_out.parent.mkdir(parents=True, exist_ok=True)
    ann_out.write_text("\n".join(l for _, l in picked_all) + "\n", encoding="utf-8")
    uploads.append((ann_out, f"{pre}annotations.jsonl"))
    if readme.exists():
        uploads.append((readme, f"{pre}README.md"))
    readme_zh = vdir / "README_zh.md"
    if readme_zh.exists():
        uploads.append((readme_zh, f"{pre}README_zh.md"))
    # 人读版剧本 + 统计报告（每次推送按当前条数重新生成）
    scripts_out = stage / "scripts.md"
    scripts_out.write_text(scripts_markdown(picked_all), encoding="utf-8")
    uploads.append((scripts_out, f"{pre}scripts.md"))
    stats_out = stage / "stats.md"
    stats_out.write_text(stats_markdown(picked_all, stat_audio_dirs), encoding="utf-8")
    uploads.append((stats_out, f"{pre}stats.md"))

    # 卡片与标注先传（页面上先出 dataset card），音频随后
    uploads.sort(key=lambda u: (0 if u[1].endswith((".md", ".jsonl")) else 1))
    print(f"共 {len(picked)} 条对话 / {len(uploads)} 个文件待上传"
          + (f"（其中带背景版 {n_babble} 条）" if n_babble else ""))
    # 批量提交（每批 30 个文件一个 commit）：逐文件提交 = 400+ 次 commit API 调用，
    # hf-mirror 会 429 风暴（2026-09-16 实测），批量后调用数降一个量级
    from huggingface_hub import CommitOperationAdd  # noqa: PLC0415

    BATCH = 30
    ok = fail = 0
    for start in range(0, len(uploads), BATCH):
        batch = uploads[start:start + BATCH]
        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp))
               for lp, rp in batch]
        for attempt in range(6):  # 429 → 指数退避重试
            try:
                api.create_commit(repo_id=args.repo, repo_type="dataset", operations=ops,
                                  commit_message=f"upload batch {start // BATCH + 1}"
                                                 f"（{len(batch)} files）")
                ok += len(batch)
                print(f"  ✅ [{ok}/{len(uploads)}] 批 {start // BATCH + 1}（{len(batch)} 文件）")
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "No files have been modified" in msg:
                    ok += len(batch)  # 内容与远端一致（重推场景）→ 跳过视为完成
                    print(f"  ✅ [{ok}/{len(uploads)}] 批 {start // BATCH + 1}：内容未变，跳过")
                    break
                if "429" in msg and attempt < 5:
                    wait = 10 * (2 ** attempt)
                    print(f"  ⏳ 429 限流 → {wait}s 后重试批 {start // BATCH + 1}")
                    time.sleep(wait)
                    continue
                fail += len(batch)
                print(f"  ❌ 批 {start // BATCH + 1}: {type(e).__name__}: {msg[:100]}")
                break
        time.sleep(1.0)  # 批间节流
    print(f"\n完成: 成功 {ok} / 失败 {fail} → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
