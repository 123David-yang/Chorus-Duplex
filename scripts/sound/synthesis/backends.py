"""Pluggable TTS contract: text + voice/reference -> mono waveform and native sample rate.
Backends are loaded once per worker; all alignment/VAD/mixing remains backend independent.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import importlib
import math
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

def moss_synthesis_text(text, language="zh"):
    # A terminal boundary helps the autoregressive model end short replies instead of continuing.
    # No lexical content is inserted, and interrupted half-sentences retain their dash.
    stripped=text.rstrip()
    if not stripped or not stripped[-1].isalnum():
        return text
    return stripped + ("." if language == "en" else "。")

class MossOnnxBackend:
    capabilities={"cpu":True,"gpu":True,"voice_cloning":True,"reference_text_required":False}
    def __init__(self, tmp: Path):
        from scripts.config import MOSS_DIR
        sys.path.insert(0,os.environ.get("MOSS_DIR",str(MOSS_DIR)))
        import soundfile as sf
        self.sf=sf; self.tmp=tmp
        opts={"thread_count":int(os.environ.get("TTS_THREADS","4"))}
        device = os.environ.get("TTS_DEVICE", "cpu").lower()
        if device in ("cuda", "cuda:0", "gpu"):
            # ORT can load CUDA/cuDNN libraries from an installed compatible runtime.
            # This must happen before the MOSS sessions are created.
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            opts["execution_provider"] = "cuda"
        elif device != "cpu":
            raise ValueError("MOSS ONNX device must be cpu or cuda:0")
        from onnx_tts_runtime import OnnxTtsRuntime
        from scripts.production.moss_scale.config import moss_model_root
        opts["model_dir"] = str(moss_model_root(Path(os.environ.get("MOSS_DIR", str(MOSS_DIR)))))
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
        if device != "cpu" and getattr(self.runtime, "execution_provider", None) != "cuda":
            raise RuntimeError("MOSS CUDA was requested but the runtime did not select CUDA")
        cc=self.runtime.codec_meta["codec_config"]
        self.fps=int(cc["sample_rate"])/int(cc["downsample_rate"])
        self.max_frames=int(self.runtime.manifest["generation_defaults"]["max_new_frames"])
        import onnxruntime
        self.runtime_provenance={"engine":"moss_onnx","onnxruntime":onnxruntime.__version__,"execution_provider":"cuda" if device != "cpu" else "cpu","native_sample_rate":int(cc["sample_rate"]),"native_channels":int(cc["channels"]),"sample_mode":"fixed","request_seed":int(os.environ.get("MOSS_SEED","1234"))}
    def synthesize(self,text,voice,reference_audio,max_seconds):
        cap=max(1,min(self.max_frames,round(max_seconds*self.fps)))
        spoken=moss_synthesis_text(text, os.environ.get("MOSS_TEXT_LANGUAGE", "zh"))
        res=self.runtime.synthesize(text=spoken,voice=voice if not reference_audio else None,
            prompt_audio_path=reference_audio,output_audio_path=str(self.tmp),
            sample_mode="fixed",do_sample=False,seed=int(os.environ.get("MOSS_SEED","1234")),enable_wetext=False,
            enable_normalize_tts_text=False,max_new_frames=cap)
        audio=np.asarray(res["waveform"],dtype=np.float32);sr=int(res["sample_rate"])
        chunks=res.get("chunk_results",[])
        exhausted=any(len(chunk.get("generated_frames",[]))>=cap for chunk in chunks) or (not chunks and len(res.get("audio_token_ids",[]))>=cap)
        return AudioResult(audio,sr,exhausted,provenance={**self.runtime_provenance,"voice":voice,"voice_cloning":bool(reference_audio),"request_seed":int(os.environ.get("MOSS_SEED","1234")),"source_text":text,"synthesized_text":spoken,"terminal_punctuation_added":spoken!=text,"generation_frame_cap_per_chunk":cap,"generated_frames":len(res.get("audio_token_ids",[]))})

class QwenBackend:
    """Qwen preset and clone models, loaded lazily when a voice needs each model."""
    capabilities={"cpu":True,"gpu":True,"voice_cloning":True,"reference_text_required":False}
    def __init__(self,tmp:Path):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.device=os.environ.get("TTS_DEVICE","cpu").lower()
        if self.device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", self.device):
            raise ValueError("Qwen device must be cpu or cuda[:index]")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Qwen CUDA was requested but CUDA is unavailable")
        self.torch=torch
        self.factory=Qwen3TTSModel
        self.models={}
        self.prompts={}

    def _model(self, kind):
        if kind not in self.models:
            if kind == "clone":
                name=os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
            else:
                name=os.environ.get("QWEN_PRESET_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
            self.models[kind]=self.factory.from_pretrained(
                name, device_map=self.device,
                dtype=self.torch.float32 if self.device=="cpu" else self.torch.float16,
                attn_implementation="eager")
        return self.models[kind]

    def synthesize(self,text,voice,reference_audio,max_seconds):
        language = os.environ.get("MOSS_TEXT_LANGUAGE", "zh")
        if language not in ("zh", "en"):
            raise ValueError("Qwen language must be zh or en")
        language = {"zh": "Chinese", "en": "English"}[language]
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Qwen max_seconds must be positive and finite")
        model=self._model("clone" if reference_audio else "preset")
        try:
            config=model.model.speech_tokenizer.config
            fps=float(config.output_sample_rate) / float(config.decode_upsample_rate)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            raise ValueError("Qwen speech tokenizer does not expose a valid codec frame rate") from None
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Qwen codec frame rate must be positive and finite")
        max_tokens=max(1, math.ceil(max_seconds * fps))
        if reference_audio:
            path=Path(reference_audio).resolve()
            key=(str(path), path.stat().st_mtime_ns, path.stat().st_size)
            if key not in self.prompts:
                self.prompts[key]=model.create_voice_clone_prompt(
                    ref_audio=str(path),x_vector_only_mode=True)
            waves,sr=model.generate_voice_clone(text=text,language=language,
                voice_clone_prompt=self.prompts[key],max_new_tokens=max_tokens)
        else:
            waves,sr=model.generate_custom_voice(text=text,language=language,speaker=voice,
                                                max_new_tokens=max_tokens)
        samples=np.asarray(waves[0],dtype=np.float32)
        rate=int(sr)
        if rate <= 0:
            raise ValueError("Qwen returned an invalid sample rate")
        # The high-level API returns audio without an EOS flag. A waveform within
        # two codec frames of the limit is conservatively treated as truncated.
        truncated=len(samples)/rate >= max(0, (max_tokens - 2)/fps)
        return AudioResult(samples,rate,truncated,
                           provenance={"engine":"qwen3","voice":voice,
                                       "voice_cloning":bool(reference_audio),
                                       "language":language,"device":self.device,
                                       "codec_frames_per_second":fps,"max_new_tokens":max_tokens,
                                       "truncation_detection":"duration_near_token_limit"})

class KokoroBackend:
    """Chinese/English sherpa ONNX backend; presets only, native 24kHz."""
    capabilities={"cpu":True,"voice_cloning":False,"native_sample_rate":24000}
    def __init__(self,tmp:Path):
        import sherpa_onnx
        from scripts.production.moss_scale.config import kokoro_model_root
        folder=kokoro_model_root()
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

class RegistryBackend:
    """Resolve registered preset aliases while keeping model-specific logic in each backend."""
    def __init__(self, backend):
        from scripts.text.voices import voice_bindings_snapshot, voice_registry
        self.backend = backend
        self.bindings = voice_bindings_snapshot(voice_registry()["voices"])

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def synthesize(self, text, voice, reference_audio, max_seconds):
        from scripts.text.voices import preset_for_voice, voice_bindings_snapshot
        if voice_bindings_snapshot([voice]).get(voice) != self.bindings.get(voice):
            raise ValueError(f"Registered voice changed during synthesis: {voice}")
        return self.backend.synthesize(text, preset_for_voice(voice), reference_audio, max_seconds)


def create_backend(tmp:Path) -> TTSBackend:
    name=os.environ.get("TTS_BACKEND","moss_onnx")
    if name=="moss_onnx":
        backend = MossOnnxBackend(tmp)
    elif name=="qwen3":
        backend = QwenBackend(tmp)
    elif name=="kokoro":
        if os.environ.get("TTS_DEVICE", "cpu").lower() != "cpu":
            raise ValueError("Kokoro uses CPU here; configure OpenVINO device separately")
        engine=os.environ.get("KOKORO_ENGINE","onnxruntime")
        if engine=="openvino":
            from .kokoro_openvino import OpenVINOKokoroBackend
            backend = OpenVINOKokoroBackend(tmp)
        elif engine=="onnxruntime":
            backend = KokoroBackend(tmp)
        else:
            raise ValueError("KOKORO_ENGINE must be onnxruntime or openvino")
    else:
        if ":" not in name: raise ValueError("TTS_BACKEND: moss_onnx, qwen3, kokoro, or module:factory")
        module,factory=name.split(":",1)
        backend = getattr(importlib.import_module(module),factory)(tmp)
    if os.environ.get("VOICE_REGISTRY"):
        return RegistryBackend(backend)
    return backend
