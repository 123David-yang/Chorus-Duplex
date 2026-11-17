"""Small CPU TTS/clone benchmark, one model instance, explicit cold/warm timing."""
from __future__ import annotations
import argparse,json,time,traceback,hashlib,os
from pathlib import Path
import numpy as np
import soundfile as sf
from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth,SAMPLE_RATE
from scripts.text.voices import reference_for_voice


def waveform_metrics(audio):
    audio=np.asarray(audio,dtype=np.float32)
    if not audio.size or not np.all(np.isfinite(audio)): return {"valid_waveform":False}
    frame=960
    frames=audio[:len(audio)//frame*frame].reshape(-1,frame)
    rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1)) if len(frames) else np.array([])
    quiet=rms<10**(-45/20)
    longest=run=0
    for value in quiet:
        run=run+1 if value else 0;longest=max(longest,run)
    peak=float(np.max(np.abs(audio)))
    return {"valid_waveform":True,"samples":len(audio),"peak":round(peak,6),
            "fullscale_fraction":round(float(np.mean(np.abs(audio)>=.999)),7),
            "rms_dbfs":round(float(20*np.log10(max(1e-10,np.sqrt(np.mean(audio.astype(np.float64)**2))))),3),
            "quiet_fraction":round(float(quiet.mean()),4) if len(quiet) else None,
            "longest_quiet_run_s":round(longest*frame/SAMPLE_RATE,4)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--voices",default="Junhao,Xiaoyu,aishell3_SSB0073,aishell3_SSB0005")
    args=ap.parse_args();out=Path(args.out_dir)
    if out.exists(): raise FileExistsError("Choose a new audio benchmark directory")
    out.mkdir(parents=True)
    voices=args.voices.split(",")
    texts=[("backchannel","嗯嗯"),("statement","我们先确认时间，再讨论具体安排。")]
    plan=[{"voice":v,"case":name,"text":text} for v in voices for name,text in texts]
    (out/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    os.environ.setdefault("TTS_STRICT","0")  # analysis retains flagged clips; production defaults to strict rejection
    start=time.perf_counter();synth=DialogueSynth(out);load_s=round(time.perf_counter()-start,4)
    print(json.dumps({"model_load_s":load_s}),flush=True)
    rows=[];seen=set()
    with (out/"results.jsonl").open("w",encoding="utf-8") as log:
        for i,case in enumerate(plan):
            start=time.perf_counter()
            try:
                ref=reference_for_voice(case["voice"])
                audio=synth.turn_to_mono(case["text"],case["voice"])
                path=out/(str(i)+"-"+case["voice"]+"-"+case["case"]+".wav")
                sf.write(path,audio,SAMPLE_RATE)
                metrics=synth.last_synthesis_metrics.copy()
                row={**case,**metrics,**waveform_metrics(audio),"path":str(path.resolve()),
                     "first_use_of_voice":case["voice"] not in seen,
                     "reference_sha256":hashlib.sha256(Path(ref).read_bytes()).hexdigest() if ref else None}
                row["rtf"]=round(metrics["elapsed_s"]/metrics["duration_s"],4) if metrics["duration_s"] else None
                seen.add(case["voice"])
            except Exception as exc:
                (out/(str(i)+".error.txt")).write_text(traceback.format_exc(),encoding="utf-8")
                row={**case,"status":"error","error_type":type(exc).__name__,"elapsed_s":round(time.perf_counter()-start,4)}
            rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush()
            print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={"model_load_s":load_s,"attempted":len(rows),"successful":sum(r.get("status")=="ok" for r in rows),
             "truncated":sum(r.get("truncated",False) for r in rows),"note":"Waveform metrics alone do not verify pronunciation, missing words, speaker similarity or subjective quality."}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
