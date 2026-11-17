"""Resumable calibration producer. New audio remains quarantined until every declared QA passes."""
from __future__ import annotations
import argparse,hashlib,json,os,random,time,traceback,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT,text_digest,annotation_digest
from scripts.production.synthesis.asr import SenseVoice
from scripts.production.publishing.publish import publish
from scripts.text.quality import coverage_plan,audit
from scripts.text.judge.review import decide
from scripts.text.pipeline import generate_one
from scripts.text.utils.schema import Dialogue
from scripts.text.utils.llm import capture_calls
from scripts.config import load_env
from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth,dialogue_to_stereo,SAMPLE_RATE


def digest(path):
    with Path(path).open("rb") as file:return hashlib.file_digest(file,"sha256").hexdigest()

class CachedSynth(DialogueSynth):
    def __init__(self,root):
        self.job=None;self.turn_results=[]
        super().__init__(root/"staging")
        selected=os.environ.get("TTS_BACKEND","moss_onnx")
        if selected=="moss_onnx":
            model=self.backend.model_dir.resolve()
            from scripts.config import MOSS_DIR
            runtime_root=Path(os.environ.get("MOSS_DIR",str(MOSS_DIR)))
            names=[p for p in model.rglob("*") if p.is_file() and p.suffix in (".onnx",".data",".json",".model",".npz") and ".cache" not in p.parts]
            files={str(p.relative_to(model)):digest(p) for p in names}
            files.update({"runtime/"+name:digest(runtime_root/name) for name in ("onnx_tts_runtime.py","ort_cpu_runtime.py","text_normalization_pipeline.py")})
            self.provenance={"backend":"moss_onnx","folder":str(runtime_root),"model_folder":str(model),"upstream":"https://github.com/OpenMOSS/MOSS-TTS-Nano","license":"Apache-2.0","source_repositories":["https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX","https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"]}
        elif selected=="kokoro":
            model=self.backend.model_dir
            names=["model.onnx" if (model/"model.onnx").is_file() else "model.int8.onnx","voices.bin","tokens.txt","lexicon-zh.txt","lexicon-us-en.txt"]
            self.provenance=json.loads((model/"asset-provenance.json").read_text(encoding="utf-8"))
            files={name:digest(model/name) for name in names}
            if os.environ.get("KOKORO_EXTRA_LEXICON"):files["extra_lexicon"]=digest(Path(os.environ["KOKORO_EXTRA_LEXICON"]))
        else:raise ValueError("Production provenance currently supports moss_onnx and optional kokoro")
        sound=Path(__file__).parents[2]/"sound"/"synthesis"
        self.engine_configuration={"model":files,"code":{name:digest(sound/name) for name in ("dialogue_to_wav.py","backends.py","vad.py","kokoro_frontend.py","kokoro_frontend_worker.py","kokoro_openvino.py")},
            "speed":os.environ.get("TTS_SPEED","1.0"),"fade":os.environ.get("TTS_FADE","1"),"frontend":"moss_native" if selected=="moss_onnx" else os.environ.get("KOKORO_FRONTEND","lexicon"),
            "threads":os.environ.get("TTS_THREADS","4"),"vad_model":digest(sound.parent/"models"/"silero_vad.onnx"),"backend_runtime":getattr(self.backend,"runtime_provenance",{})}
        self.fingerprint=hashlib.sha256(json.dumps(self.engine_configuration,sort_keys=True).encode()).hexdigest()
    def begin(self,folder):
        self.job=folder;self.turn_results=[];(folder/"segments").mkdir(exist_ok=True);(folder/"turns").mkdir(exist_ok=True)
    def turn_to_mono(self,text,voice="Junhao",jitter_db=0,target_offset_db=0):
        spec={"text":text,"voice":voice,"jitter_db":jitter_db,"offset_db":target_offset_db,"engine":self.fingerprint}
        key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        path=self.job/"segments"/(key+".wav");meta=path.with_suffix(".json")
        candidates=[path]+[Path(folder)/(key+".wav") for folder in getattr(self,"readonly_cache_dirs",[])]
        for candidate in candidates:
            metadata=candidate.with_suffix(".json")
            if candidate.is_file() and metadata.is_file():
                saved=json.loads(metadata.read_text(encoding="utf-8"))
                if saved.get("sha256")==digest(candidate):
                    audio,sr=sf.read(candidate,dtype="float32")
                    if sr==SAMPLE_RATE:
                        self.last_synthesis_metrics={**saved["metrics"],"original_synthesis_elapsed_s":saved["metrics"].get("elapsed_s"),"elapsed_s":0.,"cache_hit":True}
                        return audio
        audio=super().turn_to_mono(text,voice,jitter_db,target_offset_db)
        sf.write(path,audio,SAMPLE_RATE,subtype="FLOAT")
        meta.write_text(json.dumps({"spec":spec,"sha256":digest(path),"metrics":self.last_synthesis_metrics},ensure_ascii=False,indent=2),encoding="utf-8")
        return audio
    def on_turn_complete(self,index,turn,audio,metrics,timing):
        path=self.job/"turns"/f"{index:03d}.wav"
        sf.write(path,audio,SAMPLE_RATE,subtype="PCM_16")
        self.turn_results.append({"turn_id":index,"speaker":turn.speaker,"text":turn.text,"path":str(path),"sha256":digest(path),"tts":metrics,"timing":timing})


def create_production_synth(root,asr=None):
    storage=os.environ.get("PRODUCTION_STORAGE","native_flac")
    if storage=="native_flac":
        from .synth import LosslessProductionSynth
        return LosslessProductionSynth(Path(root),asr=asr)
    if storage=="float_legacy":return CachedSynth(Path(root))
    raise ValueError("Unknown production storage mode")


def job_spec(index,users,turns,gen_mode="oneshot"):
    if gen_mode not in ("oneshot","multi_agent"):raise ValueError("Unknown generation route")
    prefix="prod-v5-moss" if gen_mode=="oneshot" else "prod-ma-v5-moss"
    module=("identity_ask","barge_in","ai_backchannel")[index%3]
    ident=next(f"{prefix}-20261031-{index:06d}-{salt}" for salt in range(1000)
               if coverage_plan(f"{prefix}-20261031-{index:06d}-{salt}",turns)==[module])
    return {"dialogue_id":ident,"module":module,"scenario":("social","side_talk","multi_qa")[(index//3)%3],"users":users,"turns":turns,"seed":10000+index,"gen_mode":gen_mode}


def qa_audio(d,folder,synth,asr,text_report):
    path=folder/"audio.flac";timeline=json.loads((folder/"timeline.json").read_text(encoding="utf-8"))
    audio,sr=sf.read(path,dtype="float32",always_2d=True)
    turns=[]
    for component in synth.turn_results:
        wave,rate=sf.read(component["path"],dtype="float32")
        result=asr.assess(wave,rate,component["text"])
        turns.append({k:v for k,v in component.items() if k!="path"}|{"asr":result})
    no_self=all(not (a["speaker"]==b["speaker"] and a["start_sec"]<b["end_sec"] and b["start_sec"]<a["end_sec"]) for i,a in enumerate(timeline) for b in timeline[:i])
    anchored=True
    for i,t in enumerate(d.turns):
        if t.kind!="backchannel":continue
        row=timeline[i];host=next((j for j in range(i-1,-1,-1) if d.turns[j].kind!="backchannel"),None)
        anchored &= bool(t.anchor_text and row.get("placement_method")=="semantic_anchor" and host is not None and
                          row["start_sec"]>=row.get("anchor_boundary_sec",float("inf")) and row["start_sec"]<timeline[host]["end_sec"])
    frame=960;mono=np.max(np.abs(audio),axis=1);energies=mono[:len(mono)//frame*frame].reshape(-1,frame)
    ratio=float(np.mean(np.sqrt(np.mean(energies.astype(np.float64)**2,axis=1))>10**(-45/20))) if len(energies) else 0.
    licenses=(Path(synth.provenance["folder"])/"LICENSE").read_text(encoding="utf-8",errors="replace")
    max_peak=float(np.max(np.abs(audio))) if audio.size else 0.
    char_errors=sum(t["asr"]["char_errors"] for t in turns);characters=sum(t["asr"]["reference_chars"] for t in turns)
    phone_errors=sum(t["asr"]["phonetic_errors"] for t in turns);phones=sum(t["asr"]["reference_phonemes"] for t in turns)
    weighted_cer=char_errors/max(1,characters);weighted_per=phone_errors/max(1,phones)
    checks={"text_semantics":text_report.get("decision")=="accept" and decide(d,text_report.get("review",{}))["decision"]=="accept",
            "audio_finite":bool(audio.size and np.all(np.isfinite(audio)) and max_peak>1e-4),"no_clipping":bool(max_peak<.999),
            "no_self_overlap":no_self,"asr_agreement":all(row["asr"]["passed"] for row in turns) and (weighted_cer<=.03 or weighted_per<=.025),"timing_anchor":anchored,
            "licenses":"Apache" in licenses,"speech_coverage":ratio>=.5}
    if any(row["tts"].get("truncated") or row["tts"].get("quality_flags") for row in turns):checks["audio_finite"]=False
    return {"schema_version":1,"policy":"production-calibration-v6","dialogue_id":d.dialogue_id,"fixture":False,
            "text_sha256":text_digest(json.loads(d.to_jsonl_line())),"annotation_sha256":annotation_digest(json.loads(d.to_jsonl_line())),"audio_sha256":digest(path),"passed":all(checks.values()),"checks":checks,
            "duration_s":len(audio)/sr,"sample_rate":sr,"channels":audio.shape[1],"peak":max_peak,"speech_coverage":ratio,
            "weighted_cer":weighted_cer,"weighted_phonetic_er":weighted_per,"mix":getattr(synth,"last_mix_metrics",{}),"engine":synth.provenance,"engine_fingerprint":synth.fingerprint,"engine_configuration":getattr(synth,"engine_configuration",None),"timeline_sha256":digest(folder/"timeline.json"),
            "license_sources":synth.provenance.get("source_repositories",["https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh"]),"per_turn":turns}


def produce(spec,ledger,synth,asr):
    ident=spec["dialogue_id"];folder=ledger.root/"samples"/ident;folder.mkdir(parents=True,exist_ok=True)
    old=ledger.db.execute("SELECT state FROM samples WHERE id=?",(ident,)).fetchone()
    if old and old[0] in ("accepted","uploaded","quarantine"):return {"dialogue_id":ident,"status":"already_"+old[0]}
    from scripts.production.state.attempts import preserve_partial
    archived=preserve_partial(folder,ledger.root)
    if archived:
        with ledger.db:ledger.event(ident,"prior_audio_attempt_preserved",{"path":archived})
    report_path=folder/"text-report.json";annotation=folder/"annotation.text.json"
    if annotation.is_file():
        d=Dialogue.from_file(annotation);text_report=json.loads(report_path.read_text(encoding="utf-8"))["report"]
    else:
        if report_path.is_file():return {"dialogue_id":ident,"status":"previous_text_failure"}
        records=[]
        with capture_calls() as calls:
            d=generate_one(random.Random(spec["seed"]),ident,spec["scenario"],spec["users"],spec["turns"],False,gen_mode=spec.get("gen_mode","oneshot"),report_sink=records.append)
        record=records[-1] if records else {"report":{"decision":"reject","reasons":["no_generation_report"]}}
        record["api_calls"]=calls;report_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
        if d is None:return {"dialogue_id":ident,"status":"text_rejected","reasons":record["report"].get("reasons",[])}
        annotation.write_text(d.to_jsonl_line(),encoding="utf-8");text_report=record["report"]
    ledger.register(json.loads(d.to_jsonl_line()))
    current=decide(d,text_report.get("review",{}))
    if current["decision"]!="accept":
        path=folder/"text-current-policy-failure.json"
        if not path.exists():path.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding="utf-8")
        with ledger.db:
            ledger.db.execute("UPDATE samples SET state='quarantine',qa_json=? WHERE id=?",(json.dumps({"passed":False,"stage":"text_revalidation","report":current},ensure_ascii=False),ident))
            ledger.event(ident,"text_policy_quarantine",{"report_path":str(path)})
        ledger.write_status();return {"dialogue_id":ident,"status":"text_policy_quarantine"}
    synth.begin(folder)
    try:
        dialogue_to_stereo(d,synth,folder/"audio.flac",timeline_path=folder/"timeline.json")
        qa=qa_audio(d,folder,synth,asr,text_report)
    except Exception as exc:
        (folder/"audio-error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        qa={"passed":False,"error_type":type(exc).__name__,"tts":getattr(synth,"last_synthesis_metrics",{}),"checks":{}}
    (folder/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    if qa["passed"]:
        ledger.accept_audio(ident,folder/"audio.flac",qa)
        return {"dialogue_id":ident,"status":"accepted","duration_s":qa["duration_s"]}
    with ledger.db:
        ledger.db.execute("UPDATE samples SET state='quarantine',qa_json=? WHERE id=?",(json.dumps(qa,ensure_ascii=False),ident))
        ledger.event(ident,"audio_quarantine",{"failed_checks":[k for k,v in qa.get("checks",{}).items() if not v],"error_type":qa.get("error_type")})
    ledger.write_status()
    return {"dialogue_id":ident,"status":"audio_quarantine","failed_checks":[k for k,v in qa.get("checks",{}).items() if not v],"error_type":qa.get("error_type")}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--start",type=int,default=0);ap.add_argument("--count",type=int,default=3)
    ap.add_argument("--users",type=int,default=2);ap.add_argument("--turns",type=int,default=24);ap.add_argument("--voice-pool",default="3,4,59")
    ap.add_argument("--gen-mode",choices=["oneshot","multi_agent"],default="oneshot");ap.add_argument("--publish-batch",default=None);ap.add_argument("--execute-publish",action="store_true")
    args=ap.parse_args();load_env()
    os.environ.update(DUPLEX_PRODUCTION="1",TTS_BACKEND="moss_onnx",TTS_STRICT="1")
    os.environ.setdefault("VOICE_MANIFEST",r"D:\DuplexVoice_resources\voices\AISHELL-3\manifest.json")
    from scripts.production.state.lock import ProductionLock
    ledger=Ledger(args.root)
    with ProductionLock(ledger.root):run(args,ledger)


def run(args,ledger):
    asr=SenseVoice();synth=create_production_synth(ledger.root,asr=asr)
    with (ledger.root/"production-events.jsonl").open("a",encoding="utf-8") as log:
        for index in range(args.start,args.start+args.count):
            if shutil.disk_usage(ledger.root).free<50*1024**3:raise RuntimeError("Disk backpressure: less than50GiB available")
            spec=job_spec(index,args.users,args.turns,args.gen_mode);started=time.time()
            try:result=produce(spec,ledger,synth,asr)
            except Exception as exc:result={"dialogue_id":spec["dialogue_id"],"status":"error","error_type":type(exc).__name__};traceback.print_exc()
            result.update(elapsed_s=round(time.time()-started,3),spec=spec)
            log.write(json.dumps(result,ensure_ascii=False)+"\n");log.flush();print(json.dumps(result,ensure_ascii=False),flush=True)
    if args.publish_batch:print(json.dumps(publish(ledger,args.publish_batch,execute=args.execute_publish)),flush=True)
    print(json.dumps(ledger.status(),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
