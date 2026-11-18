"""Run Chinese/English producers with one verified incremental uploader.

Use a new durable root for each run. A random namespace is saved in that root
and reused when resuming; --execute is required to call the model or publish.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path

from scripts.config import ROOT, load_env
from scripts.production.state.lock import ProductionLock
from .process_guard import OwnedProcesses
from .storage import atomic_json
from .stream_upload import pending_receipts
from .upload_delivery import MAX_BATCH_BYTES, _hf_token, _matches_remote


def verify_namespace(repo: str, tag: str, roots: dict[str, Path]) -> None:
    """Reject remote IDs in this run namespace unless they match local receipts."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from huggingface_hub import HfApi, hf_hub_download
            token = _hf_token()
            api = HfApi(token=token)
            head = api.repo_info(repo, repo_type="dataset").sha
            if not api.get_paths_info(repo, ["annotations.jsonl"], revision=head, repo_type="dataset"):
                return  # An initialized empty dataset has no conflicting IDs.
            index = hf_hub_download(repo, "annotations.jsonl", repo_type="dataset", revision=head, token=token)
            prefixes = {language: ("moss-scale-en-20261115-" if language == "en" else "moss-scale-20261115-") + tag + "-"
                        for language in roots}
            with open(index, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    annotation = json.loads(line)
                    ident = annotation["dialogue_id"]
                    language = next((lang for lang, prefix in prefixes.items() if ident.startswith(prefix)), None)
                    if language:
                        if not re.fullmatch(re.escape(prefixes[language]) + r"\d{6,}", ident):
                            raise ValueError("Invalid dialogue ID in remote run namespace")
                        local = roots[language] / "completed" / f"{ident}.json"
                        if not local.is_file() or json.loads(local.read_text(encoding="utf-8"))["annotation"] != annotation:
                            raise ValueError("Remote run namespace conflicts with local receipts")
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f", HTTP {status}" if isinstance(status, int) else ""
        raise RuntimeError(f"Remote namespace check failed ({type(exc).__name__}{detail})") from None
    finally:
        logging.disable(previous)


def _stop_owned_producers(active: dict[str, subprocess.Popen], roots: dict[str, Path],
                          owned: OwnedProcesses) -> None:
    for language, process in active.items():
        if process.poll() is None:
            (roots[language] / ".chorus-coordinator-stop").touch(exist_ok=True)
    deadline = time.monotonic() + 150
    while any(process.poll() is None for process in active.values()) and time.monotonic() < deadline:
        time.sleep(2)
    for process in active.values():
        if process.poll() is None:
            owned.kill(process)
            process.wait(timeout=15)


def _clear_owned_stops(roots: dict[str, Path]) -> None:
    for root in roots.values():
        stop = root / ".chorus-coordinator-stop"
        if not stop.exists():
            continue
        with ProductionLock(root):  # This reserved file is separate from the user-controlled STOP.
            stop.unlink(missing_ok=True)


def _uploader_stalled(state: Path, started_wall: float, limit: float) -> bool:
    progress = state / "progress.json"
    last = max(started_wall, progress.stat().st_mtime) if progress.exists() else started_wall
    return time.time() - last > limit


def _produced_dialogues(root: Path) -> int:
    """Read the producer's atomic status; missing/invalid status is not progress."""
    try:
        value = json.loads((root / "status.json").read_text(encoding="utf-8"))["dialogues"]
    except (OSError, ValueError, KeyError, TypeError):
        return 0
    return value if type(value) is int and value >= 0 else 0


class ProducerProgress:
    """Track delivered items, not process liveness or an unrelated uploader heartbeat."""
    def __init__(self, root: Path, now: float):
        self.count = _produced_dialogues(root)
        self.last_progress = now

    def stalled(self, root: Path, now: float, limit: float) -> bool:
        current = _produced_dialogues(root)
        if current > self.count:
            self.count = current
            self.last_progress = now
        return now - self.last_progress > limit


def _remaining(deadline: float, limit: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Run exceeded the configured total time limit; resume from the same root")
    return min(remaining, limit)


def _bounded_remote_check(owned: OwnedProcesses, mode: str, repo: str, tag: str, roots: dict[str, Path],
                          state: Path, env: dict, timeout: int) -> dict:
    command = [sys.executable, "-m", "scripts.production.moss_scale.remote_verify", "--mode", mode,
               "--repo", repo, "--tag", tag, "--zh-root", str(roots["zh"]), "--en-root", str(roots["en"]),
               "--state", str(state)]
    process = owned.popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        owned.kill(process)
        process.communicate()
        raise RuntimeError("Remote verification exceeded the configured time limit") from None
    try:
        result = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError("Remote verification returned an invalid result") from None
    if process.returncode or result.get("status") != "ok":
        raise RuntimeError(f"Remote {mode} verification failed")
    return result


def _verified_hours(roots: dict[str, Path], state: Path, repo: str) -> tuple[dict[str, float], str]:
    if pending_receipts(list(roots.values()), state, repo):
        raise RuntimeError("Some completed receipts have not been remotely verified")
    seen = set()
    clean_hashes = set()
    hours = {}
    expected = {}
    paths = {}
    for language, root in roots.items():
        seconds = 0.0
        for receipt in sorted((root / "completed").glob("*.json")):
            row = json.loads(receipt.read_text(encoding="utf-8"))
            ident = row["annotation"]["dialogue_id"]
            if ident in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
                raise ValueError("Duplicate dialogue ID across production roots")
            seen.add(ident)
            seconds += row["credited_seconds"]
            expected[ident] = row["annotation"]
            names = (f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json")
            if set(row["sha256"]) != set(names) or any(not re.fullmatch(r"[a-f0-9]{64}", row["sha256"][name]) for name in names):
                raise ValueError("Invalid delivery hashes")
            clean_digest = row["sha256"][names[0]]
            if clean_digest in clean_hashes:
                raise ValueError("Duplicate clean audio across production receipts")
            clean_hashes.add(clean_digest)
            for name in names:
                paths[f"audio/{name}"] = (row["sha256"][name], root / "audio" / name)
        hours[language] = seconds / 3600
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from huggingface_hub import HfApi, hf_hub_download
            token = _hf_token()
            api = HfApi(token=token)
            head = api.repo_info(repo, repo_type="dataset").sha
            index = hf_hub_download(repo, "annotations.jsonl", repo_type="dataset", revision=head, token=token)
            found = set()
            with open(index, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    annotation = json.loads(line)
                    ident = annotation["dialogue_id"]
                    if ident in expected:
                        if ident in found or expected[ident] != annotation:
                            raise ValueError("Remote annotations differ from verified receipts")
                        found.add(ident)
            if found != set(expected):
                raise ValueError("Some annotations are missing from the current remote revision")
            path_names = list(paths)
            for offset in range(0, len(path_names), 100):
                chunk = path_names[offset:offset + 100]
                info = api.get_paths_info(repo, chunk, revision=head, repo_type="dataset")
                remote = {item.path: item for item in info}
                for name in chunk:
                    if name not in remote:
                        raise ValueError("An audio or timeline artifact is missing remotely")
                    digest, local = paths[name]
                    lfs = getattr(remote[name], "lfs", None)
                    remote_hash = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                    if remote_hash:
                        matches = remote_hash == digest
                    else:
                        if local.is_file():
                            with local.open("rb") as stream:
                                payload = stream.read(MAX_BATCH_BYTES + 1)
                            matches = (len(payload) <= MAX_BATCH_BYTES
                                       and hashlib.sha256(payload).hexdigest() == digest
                                       and _matches_remote(remote[name], payload))
                        else:
                            matches = False
                    if not matches:
                        raise ValueError("A remote artifact differs from its verified receipt")
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", HTTP {status}" if isinstance(status, int) else ""
        raise RuntimeError(f"Current remote verification failed ({type(exc).__name__}{suffix})") from None
    finally:
        logging.disable(previous)
    return hours, head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="new or resumable durable production directory")
    parser.add_argument("--repo", required=True, help="Hugging Face dataset owner/name")
    parser.add_argument("--hours-zh", type=float, default=10)
    parser.add_argument("--hours-en", type=float, default=10)
    parser.add_argument("--tts-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--draft-mode", choices=("direct", "interactive", "roleplay"), default="direct")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--upload-timeout-seconds", type=int, default=7200,
                        help="maximum time for one upload batch or unresponsive scan; default 2 hours")
    parser.add_argument("--verification-timeout-seconds", type=int, default=900,
                        help="maximum time for each remote read-only verification; default 15 minutes")
    parser.add_argument("--producer-timeout-seconds", type=int, default=1800,
                        help="stop if a producer delivers no new item for this long; default 30 minutes")
    parser.add_argument("--max-run-seconds", type=int, default=86400,
                        help="maximum coordinator runtime before safe stop; default 24 hours")
    parser.add_argument("--run-tag", help="optional unique namespace; generated automatically for a new root")
    parser.add_argument("--execute", action="store_true", help="actually generate and upload")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("Invalid HF dataset repo")
    if any(not math.isfinite(x) or x <= 0 for x in (args.hours_zh, args.hours_en)):
        parser.error("Both language targets must be positive finite hours")
    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", args.run_tag):
        parser.error("Invalid --run-tag")
    if args.upload_timeout_seconds < 120 or args.verification_timeout_seconds < 30:
        parser.error("Upload and verification time limits are too short")
    if args.producer_timeout_seconds < 60 or args.max_run_seconds < 300:
        parser.error("Producer and total time limits are too short")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "hours_zh": args.hours_zh, "hours_en": args.hours_en,
                          "workers": args.workers, "repo": args.repo}))
        return
    run_deadline = time.monotonic() + args.max_run_seconds
    load_env()
    base = args.root.expanduser().resolve()
    if base == ROOT.resolve() or base.is_relative_to(ROOT.resolve()):
        parser.error("Use a production root outside the source repository")
    control = base / "control"
    roots = {language: base / language for language in ("zh", "en")}
    targets = {"zh": args.hours_zh, "en": args.hours_en}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    with ProductionLock(control), OwnedProcesses() as owned:
        plan_file = control / "plan.json"
        settings = {"repo": args.repo, "hours": targets, "tts_device": args.tts_device,
                    "draft_mode": args.draft_mode, "workers": args.workers,
                    "upload_timeout_seconds": args.upload_timeout_seconds,
                    "verification_timeout_seconds": args.verification_timeout_seconds,
                    "producer_timeout_seconds": args.producer_timeout_seconds,
                    "max_run_seconds": args.max_run_seconds}
        if plan_file.exists():
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
            identity = ("repo", "hours", "tts_device", "draft_mode")
            if any(plan.get(key) != settings[key] for key in identity) or (args.run_tag and args.run_tag != plan["tag"]):
                parser.error("Existing run settings differ; use the same settings or a new root")
            tag = plan["tag"]
            if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", tag):
                parser.error("Saved run namespace is invalid")
            plan.update(settings)  # Runtime worker count and time limits may be adjusted on resume.
            atomic_json(plan_file, plan)
        else:
            if any(any(root.iterdir()) for root in roots.values()):
                parser.error("Use an empty root when creating a new run")
            tag = args.run_tag or uuid.uuid4().hex[:16]
            atomic_json(plan_file, {**settings, "tag": tag})
        _clear_owned_stops(roots)
        if any((root / name).exists() for root in roots.values() for name in ("STOP", ".chorus-coordinator-stop")):
            parser.error("A producer STOP file exists; resolve it before resuming")
        from .deduplicate import rebuild as rebuild_deduplication
        from .moss_binding import prepare_moss_models
        dedupe_db = control / "clean-hashes.sqlite3"
        with ProductionLock(roots["zh"]), ProductionLock(roots["en"]):
            _, zh_identity = prepare_moss_models(roots["zh"])
            _, en_identity = prepare_moss_models(roots["en"])
            if zh_identity != en_identity:
                raise ValueError("MOSS model changed between language root checks")
            rebuild_deduplication(dedupe_db, list(roots.values()))
        env = os.environ.copy()
        env["TTS_BACKEND"] = "moss_onnx"
        env["CHORUS_DEDUPE_DB"] = str(dedupe_db)
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        state = control / "upload-state"
        _bounded_remote_check(owned, "namespace", args.repo, tag, roots, state, env,
                              _remaining(run_deadline, args.verification_timeout_seconds))
        for language in roots:
            command = [sys.executable, "-m", "scripts.production.scale_moss", "--check", "--language", language,
                       "--tts-backend", "moss_onnx", "--tts-device", args.tts_device]
            checked = owned.popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                result = checked.wait(timeout=_remaining(run_deadline, 120))
            except subprocess.TimeoutExpired:
                owned.kill(checked)
                checked.wait(timeout=15)
                raise RuntimeError(f"{language} producer preflight timed out") from None
            if result:
                raise RuntimeError(f"{language} producer preflight failed; run scale_moss --check for details")
        done = control / f"producers-done-{uuid.uuid4().hex}.flag"
        upload_command = [sys.executable, "-m", "scripts.production.moss_scale.stream_upload",
                          "--root", str(roots["zh"]), "--root", str(roots["en"]), "--state-dir", str(state),
                          "--repo", args.repo, "--follow", "--done-file", str(done), "--execute"]
        active: dict[str, subprocess.Popen] = {}
        progress: dict[str, ProducerProgress] = {}
        queue = ["zh", "en"]
        failure = None
        with (control / "upload.log").open("a", encoding="utf-8") as upload_log, \
             (control / "zh.log").open("a", encoding="utf-8") as zh_log, \
             (control / "en.log").open("a", encoding="utf-8") as en_log:
            logs = {"zh": zh_log, "en": en_log}
            uploader_started = time.time()
            uploader = owned.popen(upload_command, cwd=ROOT, env=env, stdout=upload_log, stderr=subprocess.STDOUT)
            try:
                while queue or active:
                    _remaining(run_deadline, args.max_run_seconds)
                    if uploader.poll() is not None:
                        raise RuntimeError("Uploader exited before producers finished")
                    if _uploader_stalled(state, uploader_started, args.upload_timeout_seconds):
                        raise RuntimeError("Uploader batch or scan exceeded the configured time limit")
                    while queue and len(active) < args.workers:
                        language = queue.pop(0)
                        command = [sys.executable, "-m", "scripts.production.scale_moss", "--root", str(roots[language]),
                                   "--hours", str(targets[language]), "--language", language, "--run-tag", tag,
                                   "--tts-backend", "moss_onnx", "--tts-device", args.tts_device, "--bind-moss-model",
                                   "--draft-mode", args.draft_mode]
                        active[language] = owned.popen(command, cwd=ROOT, env=env, stdout=logs[language],
                                                       stderr=subprocess.STDOUT)
                        progress[language] = ProducerProgress(roots[language], time.monotonic())
                    for language, process in list(active.items()):
                        code = process.poll()
                        if code is not None:
                            del active[language]
                            del progress[language]
                            if code:
                                raise RuntimeError(f"{language} producer failed; inspect its local log")
                            status = json.loads((roots[language] / "status.json").read_text(encoding="utf-8"))
                            if status["credited_hours"] + 1e-9 < targets[language]:
                                raise RuntimeError(f"{language} producer stopped before its target")
                        elif progress[language].stalled(roots[language], time.monotonic(), args.producer_timeout_seconds):
                            raise RuntimeError(f"{language} producer delivered no new item within the configured time limit")
                    if queue or active:
                        time.sleep(2)
            except BaseException as exc:
                failure = exc
                _stop_owned_producers(active, roots, owned)
            finally:
                done.touch()
            drain_deadline = min(run_deadline, time.monotonic() + 300) if failure else run_deadline
            while uploader.poll() is None:
                if (time.monotonic() >= drain_deadline
                        or _uploader_stalled(state, uploader_started, args.upload_timeout_seconds)):
                    owned.kill(uploader)
                    uploader.wait(timeout=15)
                    if failure is None:
                        failure = RuntimeError("Uploader exceeded the configured time limit; resume from the same root")
                    break
                time.sleep(2)
            upload_code = uploader.returncode
        if failure is not None:
            raise failure
        if upload_code:
            raise RuntimeError("Uploader failed; inspect its local log and resume after fixing the cause")
        checked = _bounded_remote_check(owned, "final", args.repo, tag, roots, state, env,
                                        _remaining(run_deadline, args.verification_timeout_seconds))
        hours, revision = checked["hours"], checked["revision"]
        if any(hours[language] + 1e-9 < targets[language] for language in roots):
            raise RuntimeError("Verified upload did not reach both language targets")
        print(json.dumps({"status": "uploaded_and_verified", "run_tag": tag,
                          "revision": revision,
                          "unique_hours": {key: round(value, 4) for key, value in hours.items()}}), flush=True)


if __name__ == "__main__":
    main()
