"""Kokoro1.1 CPU inference through OpenVINO, preserving the existing Chinese voice bank.
Requires the SHA-pinned vectorized ONNX derivative and isolated Misaki frontend.
"""
from __future__ import annotations
import os,re,hashlib
from pathlib import Path
import numpy as np
from .backends import AudioResult
from .kokoro_frontend import ChineseFrontend

class OpenVINOKokoroBackend:
    capabilities={"cpu":True,"voice_cloning":False,"native_sample_rate":24000}
    def __init__(self,tmp):
        import openvino as ov
        folder=Path(os.environ.get("KOKORO_MODEL_DIR","resources/models/kokoro-vectorized-v1_1"));self.model_dir=folder
        path=folder/"model.onnx"
        with path.open("rb") as stream:sha=hashlib.file_digest(stream,"sha256").hexdigest()
        if sha!="523b595681c7e4c7b9797a166f0971df63ec06b005e936339693bb461f871d64":raise ValueError("Unqualified OpenVINO Kokoro graph; use the pinned vectorized model")
        if os.environ.get("KOKORO_FRONTEND","misaki")!="misaki":raise ValueError("OpenVINO Kokoro requires the validated Misaki phoneme frontend")
        device=os.environ.get("KOKORO_DEVICE","CPU")
        if device!="CPU":raise ValueError("Only the CPU path is currently qualified; GPU remains an isolated experiment")
        core=ov.Core();model=core.read_model(path);model.reshape({"tokens":ov.PartialShape([1,ov.Dimension(3,511)])})
        cache=Path(os.environ.get("KOKORO_OV_CACHE_DIR","output/openvino-cache"));cache.mkdir(parents=True,exist_ok=True)
        self.engine=core.compile_model(model,"CPU",{"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f32","INFERENCE_NUM_THREADS":int(os.environ.get("TTS_THREADS","4")),"CACHE_DIR":str(cache)})
        self.runtime_provenance={"engine":"openvino","version":ov.__version__,"execution_devices":self.engine.get_property("EXECUTION_DEVICES"),"precision_hint":"f32","token_length_bounds":[3,511],"model_sha256":sha}
        self.voices=np.memmap(folder/"voices.bin",dtype="<f4",mode="r",shape=(103,510,256))
        self.tokens={}
        for line in (folder/"tokens.txt").read_text(encoding="utf-8").splitlines():
            symbol,index=line.rsplit(" ",1);self.tokens[symbol]=int(index)
        self.frontend=ChineseFrontend(Path(tmp).parent);self.english=None
    def english_phones(self,text):
        if self.english is None:
            self.english={}
            for line in (self.model_dir/"lexicon-us-en.txt").read_text(encoding="utf-8").splitlines():
                parts=line.split()
                if len(parts)>1:self.english[parts[0].lower()]="".join(parts[1:])
        words=[]
        for word in re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*",text):
            if word.isupper() and 1<len(word)<=5:
                if any(c.lower() not in self.english for c in word):raise ValueError("Initialism letter absent from English lexicon")
                words.append(" ".join(self.english[c.lower()] for c in word))
            elif word.lower() in self.english:words.append(self.english[word.lower()])
            else:
                pieces=re.split(r"(?<=[a-z])(?=[A-Z])|-",word)
                if len(pieces)>1 and all(p.lower() in self.english for p in pieces):words.append(" ".join(self.english[p.lower()] for p in pieces))
                else:raise ValueError("Unknown English token; refusing to silently drop text")
        return " ".join(words)
    def phones(self,text):
        if not re.search(r"[A-Za-z]",text):return self.frontend(text)
        parts=[]
        for segment in re.split(r"([A-Za-z]+(?:['-][A-Za-z]+)*)",text):
            if not segment:continue
            parts.append(self.english_phones(segment) if re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*",segment) else self.frontend(segment)["phonemes"])
        phones=" ".join(parts)
        return {"frontend":"misaki-zh1.1+native-English-lexicon-v1","phonemes":phones,"phonemes_sha256":hashlib.sha256(phones.encode()).hexdigest()}
    def synthesize(self,text,voice,reference_audio,max_seconds):
        if reference_audio:raise ValueError("Kokoro does not support reference cloning")
        sid=int(voice.removeprefix("kokoro_"))
        if not 0<=sid<103:raise ValueError("Kokoro speaker out of range")
        provenance=self.phones(text);phones=provenance["phonemes"]
        if not 1<=len(phones)<510:raise ValueError("Phoneme count outside the qualified model contract")
        if set(phones)-set(self.tokens):raise ValueError("Unsupported phoneme; refusing to drop it")
        tokens=np.asarray([[0]+[self.tokens[p] for p in phones]+[0]],dtype=np.int64)
        inputs={"tokens":tokens,"style":np.asarray(self.voices[sid,len(phones)][None,:]),"speed":np.asarray([float(os.environ.get("TTS_SPEED","1.0"))],dtype=np.float32)}
        result=self.engine(inputs);audio=np.asarray(result[self.engine.output(0)]).reshape(-1).astype(np.float32)
        durations=np.asarray(result[self.engine.output(1)]).reshape(-1).astype(np.int64)
        provenance.update(self.runtime_provenance)
        provenance.update(token_duration_units=durations.tolist(),samples_per_duration_unit=float(len(audio)/durations.sum()) if durations.sum() else None)
        return AudioResult(audio,24000,provenance=provenance)
