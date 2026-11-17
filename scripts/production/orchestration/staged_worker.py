"""Render independently named accepted text into isolated worker staging; no goal hours counted here.
Useful for testing an additional route/backend while the corpus coordinator continues its own jobs.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,time,traceback
from pathlib import Path
from scripts.production.synthesis.synth import LosslessProductionSynth
from scripts.production.synthesis.producer import qa_audio,digest
from scripts.production.synthesis.asr import SenseVoice
from scripts.production.state.lock import ProductionLock
from scripts.config import load_env
from scripts.text.utils.schema import Dialogue
from scripts.text.judge.review import decide
from scripts.sound.synthesis.dialogue_to_wav import dialogue_to_stereo


def candidate(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"));records=payload.get("records",[payload])
    if len(records)!=1:raise ValueError("One final record per candidate required")
    record=records[0];d=Dialogue.from_jsonl_line(json.dumps(record["dialogue"],ensure_ascii=False))
    if record.get("mock") or decide(d,record["report"].get("review",{}))["decision"]!="accept":raise ValueError("Candidate not accepted by current evidence contract")
    if not re.fullmatch(r"(?:multi-strict|moss-queue)-[A-Za-z0-9_-]+",d.dialogue_id):raise ValueError("This independent worker uses the reserved multi-strict namespace")
    return d,record,payload.get("api_calls",record.get("api_calls",[]))


def render(root,reports):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with ProductionLock(root):
        asr=SenseVoice();synth=LosslessProductionSynth(root,asr=asr)
        with (root/"worker-events.jsonl").open("a",encoding="utf-8") as log:
            for source in reports:
                d,record,calls=candidate(source);folder=root/d.dialogue_id;folder.mkdir(exist_ok=True)
                plan={"source_sha256":digest(source),"dialogue_id":d.dialogue_id,"engine_fingerprint":synth.fingerprint,"counts_towards_goal":False,"qa_source_sha256":{name:digest((Path(__file__).parent if name=="staged_worker.py" else Path(__file__).parents[1]/"synthesis")/name) for name in ("staged_worker.py","producer.py","asr.py")}}
                path=folder/"worker-plan.json"
                if path.exists():
                    if json.loads(path.read_text())!=plan:raise ValueError("Staged candidate/runtime changed")
                    if (folder/"qa.json").exists():continue
                else:path.write_text(json.dumps(plan,indent=2),encoding="utf-8")
                (folder/"annotation.text.json").write_text(d.to_jsonl_line(),encoding="utf-8")
                record["api_calls"]=calls
                (folder/"text-report.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
                started=time.perf_counter();synth.begin(folder)
                try:
                    dialogue_to_stereo(d,synth,folder/"audio.flac",timeline_path=folder/"timeline.json")
                    qa=qa_audio(d,folder,synth,asr,record["report"])
                except Exception as exc:
                    (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8");qa={"passed":False,"error_type":type(exc).__name__,"checks":{}}
                (folder/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
                result={"dialogue_id":d.dialogue_id,"stage_passed":qa["passed"],"counts_towards_goal":False,"duration_s":qa.get("duration_s"),"failed_checks":[k for k,v in qa["checks"].items() if not v],"wall_elapsed_s":time.perf_counter()-started}
                log.write(json.dumps(result,ensure_ascii=False)+"\n");log.flush();print(json.dumps(result,ensure_ascii=False),flush=True)
        if getattr(synth.backend,"frontend",None):synth.backend.frontend.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--reports",nargs="+",required=True)
    args=ap.parse_args();load_env();os.environ.update(TTS_BACKEND="moss_onnx",TTS_STRICT="1",DUPLEX_PRODUCTION="1")
    render(args.root,args.reports)

if __name__=="__main__":main()
