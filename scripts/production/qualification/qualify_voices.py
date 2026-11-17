"""Resumable local voice qualification. Diagnostics are never counted as production hours."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import soundfile as sf
from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth,SAMPLE_RATE
from scripts.sound.diagnostics.benchmark import waveform_metrics
from scripts.production.synthesis.asr import SenseVoice

CASES=[
    ("statement","我们先确认时间，再讨论具体安排。"),
    ("question","你想先去公园，还是先找个地方吃饭？"),
    ("numbers","下午三点半出发，预算一百五十元，两个人一起去。"),
    ("negation","不是取消这次活动，只是把时间改到下周六。"),
    ("bc_agree","对呀"),("bc_understood","明白"),
    ("bc_yes","是啊"),("bc_true","确实"),("bc_correct","没错"),
]
CORE={"statement","question","numbers","negation","bc_agree","bc_understood"}

def qualify(out,voices,backend="moss_onnx"):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan={"schema_version":1,"fixture":True,"voices":voices,"cases":CASES,"core_cases":sorted(CORE),
          "engine":backend,"native_sample_rate":48000 if backend=="moss_onnx" else 24000,"backend_source_sha256":hashlib.sha256((Path(__file__).parents[2]/"sound"/"synthesis"/"backends.py").read_bytes()).hexdigest(),"note":"Small objective qualification; not human listening, MOS, or a guarantee for arbitrary text."}
    path=out/"plan.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8"))!=json.loads(json.dumps(plan)):raise ValueError("Choose a fresh directory for a changed plan")
    if not path.is_file():path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    os.environ.update(TTS_BACKEND=backend,TTS_STRICT="1" if backend=="moss_onnx" else "0")
    asr=SenseVoice()
    if backend=="moss_onnx":
        from scripts.production.synthesis.synth import LosslessProductionSynth
        synth=LosslessProductionSynth(out,asr=asr)
    else:synth=DialogueSynth(out/"staging")
    rows=[]
    with (out/"events.jsonl").open("a",encoding="utf-8") as log:
        for sid in voices:
            for name,text in CASES:
                key=f"{sid:03d}-{name}" if backend=="kokoro" else f"{sid}-{name}";report=out/(key+".json");wave=out/(key+".flac")
                if report.is_file():
                    row=json.loads(report.read_text(encoding="utf-8"))
                    if row.get("audio_sha256"):
                        if not wave.is_file() or hashlib.sha256(wave.read_bytes()).hexdigest()!=row["audio_sha256"]:raise ValueError("Qualification audio changed")
                    rows.append(row);continue
                started=time.perf_counter()
                try:
                    audio=synth.turn_to_mono(text,f"kokoro_{sid:03d}" if backend=="kokoro" else sid)
                    tts=dict(synth.last_synthesis_metrics)
                    sf.write(wave,audio,SAMPLE_RATE,format="FLAC",subtype="PCM_16")
                    actual,rate=sf.read(wave,dtype="float32")
                    diagnostic=asr.assess(actual,rate,text);metrics=waveform_metrics(actual)
                    passed=bool(diagnostic["passed"] and metrics["valid_waveform"] and metrics["peak"]<.999 and not tts.get("quality_flags") and not tts.get("truncated"))
                    row={"voice":sid,"case":name,"text":text,"fixture":True,"passed":passed,"tts":tts,"asr":diagnostic,"waveform":metrics,
                         "engine_fingerprint":getattr(synth,"fingerprint",None),"engine_configuration":getattr(synth,"engine_configuration",None),"tts_rtf":tts["elapsed_s"]/max(tts["duration_s"],1e-6),"audio_sha256":hashlib.sha256(wave.read_bytes()).hexdigest()}
                except Exception as exc:
                    (out/(key+".error.txt")).write_text(traceback.format_exc(),encoding="utf-8")
                    row={"voice":sid,"case":name,"fixture":True,"passed":False,"error_type":type(exc).__name__}
                row["wall_elapsed_s"]=round(time.perf_counter()-started,3)
                report.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
                log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush();rows.append(row)
                print(json.dumps({k:row.get(k) for k in ("voice","case","passed","tts_rtf","wall_elapsed_s")}),flush=True)
    summary={"fixture":True,"qualified_voices":[v for v in voices if all(r["passed"] for r in rows if r["voice"]==v and r["case"] in CORE)],
             "passed_cases_by_voice":{str(v):[r["case"] for r in rows if r["voice"]==v and r["passed"]] for v in voices},"attempted":len(rows),"note":plan["note"]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--voices",default="Junhao,Zhiming,Weiguo,Xiaoyu,Yuewen,Lingyu,aishell3_SSB0018,aishell3_SSB0073");ap.add_argument("--backend",choices=["moss_onnx","kokoro"],default="moss_onnx")
    args=ap.parse_args()
    voices=[int(v) for v in args.voices.split(",")] if args.backend=="kokoro" else args.voices.split(",")
    if len(set(voices))!=len(voices) or any(not str(v).replace("_","").isalnum() for v in voices):raise ValueError("Invalid voice IDs")
    print(json.dumps(qualify(args.out,voices,args.backend),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
