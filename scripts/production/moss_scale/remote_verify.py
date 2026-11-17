"""Run bounded, read-only Hub checks for the production coordinator."""
import argparse
import json
from pathlib import Path

from .produce_and_upload import _verified_hours, verify_namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("namespace", "final"), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--zh-root", type=Path, required=True)
    parser.add_argument("--en-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    roots = {"zh": args.zh_root, "en": args.en_root}
    try:
        if args.mode == "namespace":
            verify_namespace(args.repo, args.tag, roots)
            result = {"status": "ok"}
        else:
            hours, revision = _verified_hours(roots, args.state, args.repo)
            result = {"status": "ok", "hours": hours, "revision": revision}
    except Exception as exc:
        result = {"status": "error", "type": type(exc).__name__}
    print(json.dumps(result), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
