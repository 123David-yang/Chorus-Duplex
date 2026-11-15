"""Offline or semantic corpus audit. Writes a new directory; never edits source JSONL."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from collections import Counter
from ..utils.schema import Dialogue,load_dialogue_lines
from ..quality import audit
from .review import review_dialogue

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--llm",action="store_true",help="Run semantic reviewer; otherwise rules only, not training approval")
    args=ap.parse_args()
    out=Path(args.out_dir)
    if out.exists(): raise FileExistsError("Choose a new audit output directory")
    out.mkdir(parents=True)
    if args.llm:
        from scripts.config import load_env
        load_env()
    counts=Counter(); issues=Counter()
    with (out/"reports.jsonl").open("w",encoding="utf-8") as reports:
        for line in load_dialogue_lines(args.input):
            start=time.time()
            try:
                d=Dialogue.from_jsonl_line(line)
                check=audit(d)
                report=review_dialogue(d) if args.llm else {"decision":"rules_only" if check["passed"] else "reject","deterministic":check}
                bucket=report["decision"]
                row={"dialogue_id":d.dialogue_id,"elapsed_s":round(time.time()-start,3),"report":report}
                with (out/(bucket+".jsonl")).open("a",encoding="utf-8") as f:
                    f.write(d.to_jsonl_line()+"\n");f.flush()
                issues.update(x["code"] for x in check["issues"])
            except Exception as exc:
                bucket="quarantine";row={"error_type":type(exc).__name__,"source_line":line}
            counts[bucket]+=1
            reports.write(json.dumps(row,ensure_ascii=False)+"\n");reports.flush()
    summary={"counts":dict(counts),"rule_issue_counts":dict(issues),"semantic_review":args.llm}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
