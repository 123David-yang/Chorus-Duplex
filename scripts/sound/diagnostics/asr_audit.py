"""Local ASR-assisted TTS inspection. CER is diagnostic, never a subjective quality score."""
from __future__ import annotations
import argparse,json,time,unicodedata,traceback
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def normalize(text):
    try:
        from opencc import OpenCC
        text=OpenCC("t2s").convert(text)
    except ImportError:
        pass
    return "".join(c.lower() for c in unicodedata.normalize("NFKC",text) if c.isalnum())


def edit_distance(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="TTS benchmark results.jsonl")
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--model-dir",default=r"D:\DuplexVoice_resources\models\whisper-tiny")
    args=ap.parse_args();out=Path(args.out_dir)
    if out.exists():raise FileExistsError("Choose a new ASR audit directory")
    out.mkdir(parents=True)
    import torch
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    torch.set_num_threads(4)
    processor=WhisperProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True).eval()
    rows=[]
    with (out/"results.jsonl").open("w",encoding="utf-8") as log:
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            source=json.loads(line)
            if source.get("status")!="ok":continue
            start=time.perf_counter()
            try:
                audio,sr=sf.read(source["path"],dtype="float32")
                if audio.ndim>1:audio=audio.mean(axis=1)
                if sr!=16000:
                    common=gcd(sr,16000);audio=resample_poly(audio,16000//common,sr//common)
                inputs=processor(audio,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
                with torch.inference_mode():
                    generated=model.generate(inputs.input_features,attention_mask=inputs.attention_mask,
                        language="zh",task="transcribe",max_new_tokens=100,do_sample=False)
                transcription=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
                expected,actual=normalize(source["text"]),normalize(transcription)
                errors=edit_distance(expected,actual)
                cer=errors/max(1,len(expected))
                row={"voice":source["voice"],"case":source["case"],"path":source["path"],
                     "expected":source["text"],"transcript":transcription,"cer":round(cer,4),
                     "elapsed_s":round(time.perf_counter()-start,3),"recognizer":"whisper-tiny-local",
                     "disposition":"short_utterance_manual_review" if len(expected)<8 else "asr_supports" if cer<=.25 else "manual_review",
                     "tts_truncated":source.get("truncated",False)}
            except Exception as exc:
                (out/(str(len(rows))+".error.txt")).write_text(traceback.format_exc(),encoding="utf-8")
                row={"path":source["path"],"error_type":type(exc).__name__}
            rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush()
            print(json.dumps(row,ensure_ascii=False),flush=True)
    (out/"summary.json").write_text(json.dumps({"attempted":len(rows),"note":"Tiny ASR errors and Chinese script variants can inflate CER. Short backchannels require listening; do not reject TTS solely on this recognizer."},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
