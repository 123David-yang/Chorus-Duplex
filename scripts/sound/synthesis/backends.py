"""Pluggable TTS contract: text + voice/reference -> mono waveform and native sample rate.
Backends are loaded once per worker; all alignment/VAD/mixing remains backend independent.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import importlib
import os
import re
import sys
import numpy as np

@dataclass
class AudioResult:
    waveform: np.ndarray
    sample_rate: int
    truncated: bool = False
    provenance: dict | None = None

class TTSBackend(Protocol):
    def synthesize(self, text: str, voice: str, reference_audio: str | None,
                   max_seconds: float) -> AudioResult: ...

def prepare_reference(path: str | Path, target_sr: int, target_channels: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    data,sr=sf.read(str(path),dtype="float32",always_2d=True)
    if not len(data) or not np.all(np.isfinite(data)): raise ValueError("Invalid reference audio")
    if sr!=target_sr:
        common=gcd(sr,target_sr)
        data=resample_poly(data,target_sr//common,sr//common,axis=0)
    if data.shape[1]!=target_channels:
        data=np.repeat(data.mean(axis=1,keepdims=True),target_channels,axis=1)
    return np.ascontiguousarray(data.T[None,:,:],dtype=np.float32)

def moss_synthesis_text(text):
    # A terminal boundary helps the autoregressive model end short replies instead of continuing.
    # No lexical content is inserted, and interrupted half-sentences retain their dash.
    stripped=text.rstrip()
    return stripped+"。" if stripped and stripped[-1].isalnum() else text

class MossOnnxBackend:
    capabilities={"cpu":True,"voice_cloning":True,"reference_text_required":False}
    def __init__(self, tmp: Path):
        from scripts.config import MOSS_DIR
        sys.path.insert(0,os.environ.get("MOSS_DIR",str(MOSS_DIR)))
        from onnx_tts_runtime import OnnxTtsRuntime
        import soundfile as sf
        self.sf=sf; self.tmp=tmp
        opts={"thread_count":int(os.environ.get("TTS_THREADS","4"))}
        if os.environ.get("MOSS_MODEL_DIR"):opts["model_dir"]=os.environ["MOSS_MODEL_DIR"]
        class PortableOnnxRuntime(OnnxTtsRuntime):
            def _load_reference_audio(self, reference_audio_path):
                # Local upstream fork still had tensor-only calls on a numpy waveform.
                cc=self.codec_meta["codec_config"]
                return prepare_reference(reference_audio_path,int(cc["sample_rate"]),int(cc["channels"]))
            def encode_reference_audio(self, reference_audio_path):
                path=Path(reference_audio_path).resolve()
                key=(str(path),path.stat().st_mtime_ns)
                if not hasattr(self,"_reference_code_cache"): self._reference_code_cache={}
                if key not in self._reference_code_cache:
                    self._reference_code_cache[key]=super().encode_reference_audio(path)
                return self._reference_code_cache[key]
        self.model_dir=Path(opts.get("model_dir",Path(os.environ.get("MOSS_DIR",str(MOSS_DIR)))/"models"))
        self.runtime=PortableOnnxRuntime(**opts)
        cc=self.runtime.codec_meta["codec_config"]
        self.fps=int(cc["sample_rate"])/int(cc["downsample_rate"])
        self.max_frames=int(self.runtime.manifest["generation_defaults"]["max_new_frames"])
        import onnxruntime
        self.runtime_provenance={"engine":"moss_onnx","onnxruntime":onnxruntime.__version__,"native_sample_rate":int(cc["sample_rate"]),"native_channels":int(cc["channels"]),"sample_mode":"fixed","request_seed":int(os.environ.get("MOSS_SEED","1234"))}
    def synthesize(self,text,voice,reference_audio,max_seconds):
        cap=max(1,min(self.max_frames,round(max_seconds*self.fps)))
        spoken=moss_synthesis_text(text)
        res=self.runtime.synthesize(text=spoken,voice=voice if not reference_audio else None,
            prompt_audio_path=reference_audio,output_audio_path=str(self.tmp),
            sample_mode="fixed",do_sample=False,seed=int(os.environ.get("MOSS_SEED","1234")),enable_wetext=False,
            enable_normalize_tts_text=False,max_new_frames=cap)
        audio=np.asarray(res["waveform"],dtype=np.float32);sr=int(res["sample_rate"])
        chunks=res.get("chunk_results",[])
        exhausted=any(len(chunk.get("generated_frames",[]))>=cap for chunk in chunks) or (not chunks and len(res.get("audio_token_ids",[]))>=cap)
        return AudioResult(audio,sr,exhausted,provenance={**self.runtime_provenance,"voice":voice,"voice_cloning":bool(reference_audio),"request_seed":int(os.environ.get("MOSS_SEED","1234")),"source_text":text,"synthesized_text":spoken,"terminal_punctuation_added":spoken!=text,"generation_frame_cap_per_chunk":cap,"generated_frames":len(res.get("audio_token_ids",[]))})

class QwenBackend:
    """Optional larger model; no import/download until explicitly selected.
    Use voice clone x-vector-only mode so reference transcripts are not required.
    Install qwen-tts and select TTS_DEVICE=cpu/cuda:0 on the deployment host.
    """
    capabilities={"cpu":True,"gpu":True,"voice_cloning":True,"reference_text_required":False}
    def __init__(self,tmp:Path):
        import torch
        from qwen_tts import Qwen3TTSModel
        device=os.environ.get("TTS_DEVICE","cpu")
        self.model=Qwen3TTSModel.from_pretrained(
            os.environ.get("TTS_MODEL","Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            device_map=device,dtype=torch.float32 if device=="cpu" else torch.bfloat16,
            attn_implementation="eager")
        self.prompts={}
    def synthesize(self,text,voice,reference_audio,max_seconds):
        if not reference_audio: raise ValueError("Qwen clone backend requires a reference voice manifest")
        # Cache conditioning once per reference; do not reload models per turn.
        if reference_audio not in self.prompts:
            self.prompts[reference_audio]=self.model.create_voice_clone_prompt(
                ref_audio=reference_audio,x_vector_only_mode=True)
        waves,sr=self.model.generate_voice_clone(text=text,language="Chinese",
            voice_clone_prompt=self.prompts[reference_audio])
        return AudioResult(np.asarray(waves[0],dtype=np.float32),int(sr))

class KokoroBackend:
    """Chinese/English sherpa ONNX backend; presets only, native 24kHz."""
    capabilities={"cpu":True,"voice_cloning":False,"native_sample_rate":24000}
    def __init__(self,tmp:Path):
        import sherpa_onnx
        folder=Path(os.environ.get("KOKORO_MODEL_DIR","resources/models/kokoro-multi-lang-v1_1"))
        self.model_dir=folder
        model_name="model.onnx" if (folder/"model.onnx").is_file() else "model.int8.onnx"
        required=[model_name,"voices.bin","tokens.txt","lexicon-us-en.txt","lexicon-zh.txt"]
        for name in required:
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
        lexicons=[folder/p for p in ("lexicon-us-en.txt","lexicon-zh.txt")]
        if os.environ.get("KOKORO_EXTRA_LEXICON"):
            extra=Path(os.environ["KOKORO_EXTRA_LEXICON"])
            if not extra.is_file():raise FileNotFoundError(extra)
            lexicons.append(extra)
        self.lexicon_paths=lexicons
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(model=str(folder/model_name),voices=str(folder/"voices.bin"),
            tokens=str(folder/"tokens.txt"),lexicon=",".join(str(p) for p in lexicons))
        config=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro,
            num_threads=int(os.environ.get("TTS_THREADS","4")),provider="cpu"),max_num_sentences=1,
            rule_fsts=",".join(str(folder/p) for p in ("date-zh.fst","number-zh.fst","phone-zh.fst") if (folder/p).is_file()))
        if not config.validate():raise ValueError("Kokoro model configuration is invalid")
        self.engine=sherpa_onnx.OfflineTts(config)
        self.frontend=None
        if os.environ.get("KOKORO_FRONTEND","lexicon")=="misaki":
            from .kokoro_frontend import ChineseFrontend
            self.frontend=ChineseFrontend(tmp.parent)
        elif os.environ.get("KOKORO_FRONTEND","lexicon")!="lexicon":raise ValueError("Unsupported Kokoro frontend")
        self.token_symbols={line.rsplit(" ",1)[0] for line in (folder/"tokens.txt").read_text(encoding="utf-8").splitlines()}
    def synthesize(self,text,voice,reference_audio,max_seconds):
        if reference_audio:raise ValueError("Kokoro does not support reference voice cloning")
        try:sid=int(voice.removeprefix("kokoro_"))
        except (ValueError,AttributeError):raise ValueError("Kokoro voice must be kokoro_003..kokoro_102") from None
        if not 0<=sid<103:raise ValueError("Kokoro speaker index out of range")
        speed=float(os.environ.get("TTS_SPEED","1.0"));provenance={"frontend":"native_lexicon"}
        if self.frontend is not None and not re.search(r"[A-Za-z]",text):
            import sherpa_onnx
            provenance=self.frontend(text);phones=provenance["phonemes"]
            unknown=set(phones)-self.token_symbols
            if unknown:raise ValueError("G2P emitted unsupported tokens; refusing to drop phonemes")
            if not phones.strip():raise ValueError("G2P produced no phonemes")
            config=sherpa_onnx.GenerationConfig();config.sid=sid;config.speed=speed;config.tokens=[list(phones)]
            generated=self.engine.generate(text="",config=config)
        else:
            if self.frontend is not None:provenance["reason"]="English-containing text retains native bilingual lexicon, never silently dropped"
            generated=self.engine.generate(text=text,sid=sid,speed=speed)
        return AudioResult(np.asarray(generated.samples,dtype=np.float32),int(generated.sample_rate),provenance=provenance)


def create_backend(tmp:Path) -> TTSBackend:
    name=os.environ.get("TTS_BACKEND","moss_onnx")
    if name=="moss_onnx": return MossOnnxBackend(tmp)
    if name=="qwen3": return QwenBackend(tmp)
    if name=="kokoro":
        engine=os.environ.get("KOKORO_ENGINE","onnxruntime")
        if engine=="openvino":
            from .kokoro_openvino import OpenVINOKokoroBackend
            return OpenVINOKokoroBackend(tmp)
        if engine!="onnxruntime":raise ValueError("KOKORO_ENGINE must be onnxruntime or openvino")
        return KokoroBackend(tmp)
    if ":" not in name: raise ValueError("TTS_BACKEND: moss_onnx, qwen3, or module:factory")
    module,factory=name.split(":",1)
    return getattr(importlib.import_module(module),factory)(tmp)
