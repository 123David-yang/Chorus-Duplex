"""Bounded, resumable stratified production with periodic verified publishing.

A complete108-item cycle covers2 routes x6 group sizes x3 scenes x3 core modules.
Milestones and the full1000h ledger target remain separate from attempt limits.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,time,traceback,uuid
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT
from scripts.production.state.lock import ProductionLock
from scripts.production.synthesis.producer import create_production_synth,produce,job_spec
from scripts.production.synthesis.asr import SenseVoice
from scripts.production.publishing.publish import publish
from scripts.config import load_env


def stratified_spec(index):
    if index<0:raise ValueError("Negative production index")
    users=1+(index//9)%6
    route="oneshot" if (index//54)%2==0 else "multi_agent"
    # Keep the index stable: each cell rotates seeds/voice assignments across cycles.
    turns=(24,28,32)[(index//108)%3]
    return job_spec(100008+index,users,turns,route)


def recover_batches(ledger):
    results=[]
    for plan in sorted((ledger.root/"publish_staging").glob("*/plan.json")):
        batch=plan.parent.name
        if not (ledger.root/("upload-"+batch+".json")).is_file():
            results.append(publish(ledger,batch,execute=True))
    from scripts.production.publishing.publish_evidence import publish_evidence
    for receipt in sorted(ledger.root.glob("upload-*.json")):
        batch=receipt.stem.removeprefix("upload-")
        if not (ledger.root/("evidence-"+batch+".json")).is_file():results.append(publish_evidence(ledger,batch,execute=True))
    return results


def is_new_attempt(result):
    status=result.get("status","")
    return not (status.startswith("already_") or status=="previous_text_failure")

def run(root,run_id,voices,max_attempts=108,publish_every=4,start_index=0):
    if start_index<0:raise ValueError("start_index must be nonnegative")
    if not 1<=publish_every<=20:raise ValueError("publish_every must be1..20")
    pool=[v.strip() for v in voices.split(",") if v.strip()]
    if len(set(pool))!=len(pool) or len(pool)<7 or "Junhao" in pool:
        raise ValueError("MOSS1..6-user production needs >=7 distinct qualified human voices excluding Junhao")
    if not run_id.replace("-","").replace("_","").isalnum() or not run_id.isascii():raise ValueError("Unsafe run_id")
    ledger=Ledger(root);folder=ledger.root/"runs"/run_id;folder.mkdir(parents=True,exist_ok=True)
    registry_path=Path(os.environ.get("MOSS_QUALIFICATION_REGISTRY",str(ledger.root/"qualification"/"registry.json")))
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("backend")!="moss_onnx":raise ValueError("MOSS production requires MOSS voice qualification; Kokoro evidence is ineligible")
    if not set(pool+["Junhao"])<=set(registry["voices"]):raise ValueError("Missing voice qualification evidence")
    for sid in pool+["Junhao"]:
        for assessment in registry["voices"][str(sid)]["assessments"]:
            path=Path(assessment["report_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=assessment["report_sha256"]:raise ValueError("Voice qualification evidence changed")
    config={"schema_version":1,"qualification_registry_sha256":hashlib.sha256(registry_path.read_bytes()).hexdigest(),"run_id":run_id,"voices":pool,"cycle_size":108,"routes":["oneshot","multi_agent"],"users":[1,2,3,4,5,6],"target_hours":1000,"publish_every":publish_every,"backend":"moss_onnx"}
    if start_index:config["start_index"]=start_index
    config_path=folder/"config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8"))!=config:raise ValueError("Run configuration changed; use a new run ID")
    else:config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    events=folder/"events.jsonl"
    completed=set()
    if events.is_file():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                item=json.loads(line)
                if "index" in item:completed.add(item["index"])
            except json.JSONDecodeError:raise ValueError("Run log has a partial row; preserve and repair it before continuing") from None
    with ProductionLock(ledger.root):
        load_env();os.environ.update(DUPLEX_PRODUCTION="1",TTS_BACKEND="moss_onnx",MOSS_VOICE_POOL=voices,TTS_STRICT="1")
        os.environ.setdefault("VOICE_MANIFEST",r"D:\DuplexVoice_resources\voices\AISHELL-3\manifest.json")
        engine_qualification={"backend":"moss_onnx","registry_sha256":hashlib.sha256(registry_path.read_bytes()).hexdigest()}
        recover_batches(ledger)
        session_id=uuid.uuid4().hex
        project=Path(__file__).parents[3]
        sources={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest() for directory in ("scripts/text","scripts/sound","scripts/production") for path in sorted((project/directory).rglob("*.py"))}
        snapshot={"engine_qualification":engine_qualification,"session_id":session_id,"source_sha256":sources,"runtime_options":{k:os.environ.get(k) for k in ("TTS_BACKEND","MOSS_DIR","MOSS_MODEL_DIR","MOSS_VOICE_POOL","VOICE_MANIFEST","KOKORO_FRONTEND","KOKORO_ENGINE","KOKORO_MODEL_DIR","KOKORO_DEVICE","KOKORO_OV_CACHE_DIR","KOKORO_QUALIFICATION_DIR","TTS_SPEED","TTS_THREADS","TTS_FADE","KOKORO_VOICE_POOL","PRODUCTION_STORAGE","PRODUCTION_NATIVE_CACHE_ROOT","MOSS_SEED","MOSS_QUALIFICATION_REGISTRY")}}
        (folder/("runtime-"+session_id+".json")).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
        asr=SenseVoice();synth=create_production_synth(ledger.root,asr=asr)
        if registry.get("engine_fingerprint")!=synth.fingerprint:raise ValueError("Production runtime differs from the qualified MOSS runtime")
        with events.open("a",encoding="utf-8") as log:
            def record(item):
                log.write(json.dumps(item,ensure_ascii=False)+"\n");log.flush();os.fsync(log.fileno());print(json.dumps(item,ensure_ascii=False),flush=True)
            attempts=0;index=start_index;failure_streak=0;stop_reason="attempt_limit_reached"
            while attempts<max_attempts and ledger.status()["uploaded_credited_seconds"]<3600000:
                if index in completed:index+=1;continue
                if shutil.disk_usage(ledger.root).free<50*1024**3:raise RuntimeError("Disk backpressure: less than50GiB free")
                spec=stratified_spec(index);started=time.perf_counter()
                try:result=produce(spec,ledger,synth,asr)
                except Exception as exc:
                    result={"status":"error","error_type":type(exc).__name__,"dialogue_id":spec["dialogue_id"]};traceback.print_exc()
                record({"session_id":session_id,"index":index,"spec":spec,"result":result,"wall_elapsed_s":round(time.perf_counter()-started,3)})
                completed.add(index);attempts+=1;index+=1
                if is_new_attempt(result):failure_streak=0 if result["status"]=="accepted" else failure_streak+1
                if failure_streak>=8:
                    stop_reason="quality_circuit_breaker"
                    record({"attention_required":stop_reason,"consecutive_failures":failure_streak})
                    break
                pending=ledger.db.execute("SELECT count(*) FROM samples WHERE state='accepted'").fetchone()[0]
                if pending>=publish_every:
                    batch=f"{run_id}-{max(completed):06d}"
                    record({"upload":publish(ledger,batch,limit=publish_every,execute=True)})
                    from scripts.production.publishing.publish_evidence import publish_evidence
                    record({"text_evidence":publish_evidence(ledger,batch,execute=True)})
            pending=ledger.db.execute("SELECT count(*) FROM samples WHERE state='accepted'").fetchone()[0]
            if pending:
                batch=f"{run_id}-tail-{max(completed,default=0):06d}"
                record({"upload":publish(ledger,batch,limit=20,execute=True)})
                from scripts.production.publishing.publish_evidence import publish_evidence
                record({"text_evidence":publish_evidence(ledger,batch,execute=True)})
            record({"run_status":"target_reached_pending_final_audit" if ledger.status()["completed"] else stop_reason,"progress":ledger.status()})
    return ledger.status()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--run-id",required=True);ap.add_argument("--voices",default="Zhiming,Weiguo,Xiaoyu,Yuewen,Lingyu,aishell3_SSB0018,aishell3_SSB0073,aishell3_SSB0139,aishell3_SSB0122");ap.add_argument("--max-attempts",type=int,default=108);ap.add_argument("--publish-every",type=int,default=4);ap.add_argument("--start-index",type=int,default=0)
    args=ap.parse_args()
    if args.max_attempts<1:raise ValueError("max_attempts must be positive")
    print(json.dumps(run(args.root,args.run_id,args.voices,args.max_attempts,args.publish_every,args.start_index),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
