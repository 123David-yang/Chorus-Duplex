"""Check a candidate Git history for local-only reports and common credentials."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from scripts.config import ROOT


SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "Hugging Face token": re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}"),
    "API key": re.compile(rb"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def _private_path(path: str) -> bool:
    path = path.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    parts = path.split("/")
    if path.startswith(("docs/paper/", "docs/archive/")) or any(
        part in (".ssh", ".claude") or "tech_report" in part or "techreport" in part for part in parts
    ):
        return True
    if path in ("docs/paper_pipeline.md", "production/continue.md"):
        return True
    if name in ("agents.md", "claude.md", ".env") or name.startswith(".env.") and name != ".env.example":
        return True
    if name.endswith((".pem", ".key", ".p12")):
        return True
    if name.endswith(".txt") and any(word in name for word in ("token", "secret", "password")):
        return True
    return name.startswith("github.txt")


def check_revision(revision: str) -> tuple[list[str], list[str]]:
    """Inspect every reachable tree and text-like blob without printing their contents."""
    commit = _git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    commits = _git("rev-list", commit).decode().splitlines()
    objects: dict[str, set[str]] = {}
    blocked = set()
    for item in commits:
        for entry in _git("ls-tree", "-r", "-z", "--full-tree", item).split(b"\0"):
            if not entry:
                continue
            info, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = info.decode().split()
            path = raw_path.decode("utf-8", "surrogateescape")
            if _private_path(path):
                blocked.add(path)
            if kind == "blob":
                objects.setdefault(oid, set()).add(path)
    if blocked:
        return [f"Local-only path appears in reachable history: {p}" for p in sorted(blocked)], []

    info = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                          cwd=ROOT, input=("\n".join(objects) + "\n").encode(), check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    sizes = {parts[0]: int(parts[2]) for line in info.decode().splitlines()
             if (parts := line.split())[1] == "blob"}
    skipped = [next(iter(objects[oid])) for oid, size in sizes.items() if size > 16 * 1024 * 1024]
    findings = set()
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for oid, size in sizes.items():
            if size > 16 * 1024 * 1024:
                continue
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or int(header[2]) != size:
                raise RuntimeError("Unexpected Git object response")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Incomplete Git object response")
            if b"\0" in data:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    for path in objects[oid]:
                        findings.add(f"{label} pattern in reachable history: {path}")
    finally:
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        process.wait()
    return sorted(findings), sorted(skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="Candidate public commit or branch")
    args = parser.parse_args()
    findings, skipped = check_revision(args.revision)
    for finding in findings:
        print(finding)
    if skipped:
        print(f"Not scanned: {len(skipped)} blobs larger than 16 MiB.")
    if findings or skipped:
        raise SystemExit(1)
    print("No local-only paths or known credential patterns found in reachable Git history.")


if __name__ == "__main__":
    main()
