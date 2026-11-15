"""Paired small-batch benchmark: fixed modules/scenarios, full failure retention, bounded calls."""
from __future__ import annotations
import argparse,json,random,time,statistics,traceback
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from scripts.text.pipeline import generate_one,sample_persona_pool
from scripts.text.quality import coverage_plan
from ..utils.llm import capture_calls
from scripts.config import load_env,ROOT


def plan_jobs(seed:int, modes=("oneshot","multi_agent")):
    jobs=[]
    for scenario in ("social","side_talk"):
        for module in ("identity_ask","barge_in","ai_backchannel"):
            ident=next(f"paired-{seed}-{scenario}-{i}" for i in range(1000)
                       if coverage_plan(f"paired-{seed}-{scenario}-{i}",16)==[module])
            for mode in modes:
                jobs.append({"dialogue_id":ident,"scenario":scenario,"module":module,"mode":mode,
                             "users":3 if scenario=="social" else 2,"turns":16})
    return jobs


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups["mode:"+row["mode"]].append(row)
        groups["module:"+row["module"]].append(row)
        groups["scenario:"+row["scenario"]].append(row)
    result={}
    for key,items in groups.items():
        decisions=Counter(r.get("report",{}).get("decision","error") for r in items)
        recovered=sum(r.get("report",{}).get("decision")=="accept" and "revision" in r.get("report",{}) for r in items)
        reasons=Counter()
        for row in items:
            report=row.get("report",{})
            reasons.update(report.get("reasons",[]))
            reasons.update(x["code"] for x in report.get("deterministic",{}).get("issues",[]))
        result[key]={"attempted":len(items),"decisions":dict(decisions),"semantic_revision_recoveries":recovered,
                     "median_s":round(statistics.median(r["elapsed_s"] for r in items),3),
                     "calls":sum(len(r["api_calls"]) for r in items),
                     "total_tokens":sum(c.get("usage",{}).get("total_tokens",0) for r in items for c in r["api_calls"]),
                     "failure_reasons":dict(reasons)}
    return result


def run_job(job,seed,pool,out):
    records=[];started=time.perf_counter()
    with capture_calls() as calls:
        try:
            d=generate_one(random.Random(seed),job["dialogue_id"],job["scenario"],job["users"],
                job["turns"],False,pool,job["mode"],"llm",0,report_sink=records.append)
            row={**job,**(records[-1] if records else {}),"accepted":d is not None}
        except Exception as exc:
            # Type only in the shared report; full stack without remote response bodies is task-local.
            (out/(job["mode"]+"-"+job["dialogue_id"]+".error.txt")).write_text(traceback.format_exc(),encoding="utf-8")
            row={**job,"accepted":False,"error_type":type(exc).__name__}
    row.update(elapsed_s=round(time.perf_counter()-started,3),api_calls=calls)
    (out/(job["mode"]+"-"+job["dialogue_id"]+".json")).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--seed",type=int,default=101)
    ap.add_argument("--workers",type=int,choices=[1,2],default=2)
    ap.add_argument("--personas",default=None)
    ap.add_argument("--modules",nargs="+",choices=["identity_ask","barge_in","ai_backchannel"],default=["identity_ask","barge_in","ai_backchannel"])
    ap.add_argument("--scenarios",nargs="+",choices=["social","side_talk"],default=["social","side_talk"])
    ap.add_argument("--modes",nargs="+",choices=["oneshot","multi_agent"],default=["oneshot","multi_agent"])
    args=ap.parse_args()
    out=Path(args.out_dir)
    if out.exists(): raise FileExistsError("Choose a new benchmark directory")
    out.mkdir(parents=True)
    load_env()
    pool=sample_persona_pool(args.personas,k=12,rng=random.Random(args.seed)) if args.personas else None
    jobs=[job for job in plan_jobs(args.seed,args.modes) if job["module"] in args.modules and job["scenario"] in args.scenarios]
    (out/"plan.json").write_text(json.dumps({"seed":args.seed,"jobs":jobs,"persona_pool":pool},ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor, (out/"results.jsonl").open("w",encoding="utf-8") as log:
        futures=[executor.submit(run_job,job,args.seed,pool,out) for job in jobs]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush()
            print(json.dumps({k:row.get(k) for k in ("mode","scenario","module","accepted","elapsed_s")},ensure_ascii=False),flush=True)
    summary=summarize(rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
