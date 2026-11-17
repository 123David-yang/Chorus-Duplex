"""Immutable native-rate PCM24 FLAC inference cache. Fresh and resumed callers use identical decoded samples.
The codec preserves the PCM24 stream losslessly; float-to-PCM24 quantization is explicit and bounded.
No previously generated data is removed. Jitter/VAD/mixing stay downstream of this cache.
"""
from __future__ import annotations
import hashlib,json,os,time,uuid
from pathlib import Path
import numpy as np
import soundfile as sf
from scripts.sound.synthesis.backends import AudioResult


def sha256(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

class NativeCacheBackend:
    def __init__(self,backend,root,signature,max_text_chars=None):
        self.backend=backend;self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.signature=signature;self.max_text_chars=max_text_chars
    def __getattr__(self,name):return getattr(self.backend,name)
    def _load(self,path,metadata,key,hit):
        row=json.loads(metadata.read_text(encoding="utf-8"))
        if row["key"]!=key or sha256(path)!=row["audio_sha256"]:raise ValueError("Native inference cache integrity check failed")
        audio,sr=sf.read(path,dtype="float32")
        if audio.ndim not in (1,2) or (1 if audio.ndim==1 else audio.shape[1])!=row.get("channels",1) or sr!=row["sample_rate"] or len(audio)!=row["frames"]:raise ValueError("Native cache audio contract changed")
        audio=(audio*np.float32(row["restore_scale"])).astype(np.float32)
        provenance={**(row.get("backend_provenance") or {}),"native_cache":{"stored":True,"key":key,"hit":hit,"audio_sha256":row["audio_sha256"],
                    "encoding":"FLAC_PCM24","sample_rate":sr,"restore_scale":row["restore_scale"],"source_inference_s":row["source_inference_s"]}}
        return AudioResult(audio,sr,truncated=bool(row.get("truncated",False)),provenance=provenance)
    def synthesize(self,text,voice,reference_audio,max_seconds):
        spec={"version":"native-pcm24-v1","signature":self.signature,"text":text,"voice":voice,"max_seconds":max_seconds,
              "moss_seed":os.environ.get("MOSS_SEED","1234"),"speed":os.environ.get("TTS_SPEED","1.0"),"reference_sha256":sha256(reference_audio) if reference_audio else None}
        key=hashlib.sha256(json.dumps(spec,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        folder=self.root/key[:2];folder.mkdir(exist_ok=True);path=folder/(key+".flac");metadata=folder/(key+".json")
        if metadata.is_file():
            if not path.is_file():raise ValueError("Native cache metadata has no audio")
            return self._load(path,metadata,key,True)
        if path.is_file():
            # Recover a crash between the final audio link and its metadata link.
            for candidate in sorted(folder.glob(key+"-*.pending.json")):
                row=json.loads(candidate.read_text(encoding="utf-8"))
                if row.get("key")==key and row.get("audio_sha256")==sha256(path):
                    try:os.link(candidate,metadata)
                    except FileExistsError:pass
                    return self._load(path,metadata,key,True)
            raise ValueError("Unfinished native cache audio lacks matching recovery metadata")
        started=time.perf_counter();result=self.backend.synthesize(text,voice,reference_audio,max_seconds)
        elapsed=time.perf_counter()-started;audio=np.asarray(result.waveform,dtype=np.float32)
        if audio.ndim not in (1,2) or not audio.size or not np.all(np.isfinite(audio)):return result
        if self.max_text_chars is not None and len(text)>self.max_text_chars:
            result.provenance={**(result.provenance or {}),"native_cache":{"stored":False,"request_sha256":key,"reason":"long_unique_input","max_cached_text_chars":self.max_text_chars,"source_inference_s":elapsed}}
            return result
        peak=float(np.max(np.abs(audio)));scale=max(1.,peak/.98)
        pending=folder/(key+"-"+uuid.uuid4().hex+".pending.flac")
        sf.write(pending,audio/scale,int(result.sample_rate),format="FLAC",subtype="PCM_24")
        with pending.open("r+b") as stream:os.fsync(stream.fileno())
        row={"key":key,"spec":spec,"truncated":bool(result.truncated),"channels":1 if audio.ndim==1 else audio.shape[1],"frames":len(audio),"sample_rate":int(result.sample_rate),"restore_scale":scale,
             "max_quantization_error_bound":scale/(2**23),"backend_provenance":result.provenance,"source_inference_s":elapsed,"audio_sha256":sha256(pending)}
        pending_meta=folder/(key+"-"+uuid.uuid4().hex+".pending.json")
        with pending_meta.open("x",encoding="utf-8") as stream:
            json.dump(row,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.flush();os.fsync(stream.fileno())
        # Hardlink publication is atomic and never replaces an existing immutable object.
        try:os.link(pending,path)
        except FileExistsError:
            if sha256(path)!=row["audio_sha256"]:raise ValueError("Different unfinished native cache object already exists")
        try:os.link(pending_meta,metadata)
        except FileExistsError:pass
        return self._load(path,metadata,key,False)
