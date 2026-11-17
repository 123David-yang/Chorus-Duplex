"""Production synthesis with native inference FLAC cache and per-turn PCM16 FLAC evidence.
Previous FLOAT caches/WAV files are untouched; this storage layout applies only to new jobs.
"""
from __future__ import annotations
import hashlib,json,os,io
from pathlib import Path
import soundfile as sf
from .producer import CachedSynth,digest
from .native_cache import NativeCacheBackend
from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth,SAMPLE_RATE,GEN_SEC_PER_CHAR,GEN_PAD_SEC
from scripts.sound.synthesis.backends import AudioResult

class ProcessedMossBackend:
    """Expose the actual final mono waveform to the bounded pronunciation auditor."""
    def __init__(self,synth):self.synth=synth
    def synthesize(self,text,voice,reference_audio,max_seconds):
        previous=os.environ.get("TTS_STRICT")
        try:
            # Defer raising until the bounded auditor has inspected flags and exhausted retries.
            os.environ["TTS_STRICT"]="0"
            audio=DialogueSynth.turn_to_mono(self.synth,text,voice,self.synth._request_jitter,self.synth._request_offset,synthesis_budget=max_seconds)
        finally:
            if previous is None:os.environ.pop("TTS_STRICT",None)
            else:os.environ["TTS_STRICT"]=previous
        metrics=dict(self.synth.last_synthesis_metrics)
        encoded=io.BytesIO();sf.write(encoded,audio,SAMPLE_RATE,format="FLAC",subtype="PCM_16");encoded.seek(0)
        audio,_=sf.read(encoded,dtype="float32")
        metrics["audited_waveform_format"]="FLAC_PCM16"
        return AudioResult(audio,SAMPLE_RATE,truncated=bool(metrics.get("truncated")),provenance={"postprocess_metrics":metrics,"quality_flags":metrics.get("quality_flags",[])})

class LosslessProductionSynth(CachedSynth):
    def __init__(self,root,asr=None):
        super().__init__(root)
        self.engine_configuration.update(storage={"native_cache":"native-pcm24-v1","components":"FLAC_PCM16"},
            native_cache_code=digest(Path(__file__).with_name("native_cache.py")),production_synth_code=digest(Path(__file__)),moss_audit_code=digest(Path(__file__).with_name("moss_audit.py")),acoustic_anchor_code=digest(Path(__file__).with_name("acoustic_anchor.py")),asr_code=digest(Path(__file__).with_name("asr.py")),whole_turn_anchors=os.environ.get("MOSS_WHOLE_TURN_ANCHORS","1"),max_cached_text_chars=int(os.environ.get("PRODUCTION_NATIVE_CACHE_MAX_CHARS","16")))
        self.fingerprint=hashlib.sha256(json.dumps(self.engine_configuration,sort_keys=True).encode()).hexdigest()
        native={k:self.engine_configuration[k] for k in ("model","frontend","speed","threads","backend_runtime")}
        native["backend_sources"]={k:v for k,v in self.engine_configuration["code"].items() if k in ("backends.py","kokoro_frontend.py","kokoro_frontend_worker.py","kokoro_openvino.py")}
        import importlib.metadata
        native["runtime_versions"]={name:importlib.metadata.version(name) for name in ("onnxruntime","numpy","soundfile")}
        cache_root=Path(os.environ.get("PRODUCTION_NATIVE_CACHE_ROOT",str(root/"shared_native_cache")))
        self.backend=NativeCacheBackend(self.backend,cache_root,native,max_text_chars=self.engine_configuration["max_cached_text_chars"])
        self.moss_auditor=None
        if os.environ.get("TTS_BACKEND","moss_onnx")=="moss_onnx":
            from .moss_audit import AuditedMossBackend
            if asr is None:
                from .asr import SenseVoice
                asr=SenseVoice()
            self.asr=asr
            self._anchor_check=None
            self.moss_auditor=AuditedMossBackend(ProcessedMossBackend(self),asr,root/"local_tts_audits")
    def begin(self,folder):
        self.job=folder;self.turn_results=[];(folder/"turns").mkdir(exist_ok=True)
    def turn_to_mono(self,text,voice="Junhao",jitter_db=0,target_offset_db=0):
        if self.moss_auditor is None:return DialogueSynth.turn_to_mono(self,text,voice,jitter_db,target_offset_db)
        self._request_jitter=jitter_db;self._request_offset=target_offset_db
        from scripts.text.voices import reference_for_voice
        result=self.moss_auditor.synthesize(text,voice,reference_for_voice(voice),len(text)*GEN_SEC_PER_CHAR+GEN_PAD_SEC,extra_check=getattr(self,"_anchor_check",None))
        self.last_synthesis_metrics=dict(result.provenance["postprocess_metrics"])
        self.last_synthesis_metrics["accepted_attempt_elapsed_s"]=self.last_synthesis_metrics.get("elapsed_s")
        self.last_synthesis_metrics["elapsed_s"]=result.provenance["local_asr_audit"]["elapsed_s"]
        self.last_synthesis_metrics["backend_provenance"]={**(self.last_synthesis_metrics.get("backend_provenance") or {}),"local_asr_audit":result.provenance["local_asr_audit"]}
        return result.waveform
    def synthesize_with_anchor_offsets(self,text,voice,anchors,jitter,offset):
        self.last_anchor_attempt=None
        if self.moss_auditor is None or self.engine_configuration["whole_turn_anchors"]!="1":return None
        from .acoustic_anchor import verify_anchors
        from .moss_audit import MossAuditExhausted
        self._anchor_check=lambda audio,sr:verify_anchors(audio,sr,anchors,self.asr)
        try:
            audio=self.turn_to_mono(text,voice,jitter,offset)
            validation=self.last_synthesis_metrics["backend_provenance"]["local_asr_audit"]["additional_validation"]
            self.last_synthesis_metrics.update(semantic_split=False,acoustic_anchors=validation)
            return audio,validation["offsets_samples"]
        except MossAuditExhausted as exc:
            self.last_anchor_attempt={"method":"whole_turn_ctc","status":"fallback_to_exact_split","reason":str(exc)}
            return None
        finally:self._anchor_check=None

    def on_turn_complete(self,index,turn,audio,metrics,timing):
        path=self.job/"turns"/f"{index:03d}.flac"
        sf.write(path,audio,SAMPLE_RATE,format="FLAC",subtype="PCM_16")
        self.turn_results.append({"turn_id":index,"speaker":turn.speaker,"text":turn.text,"path":str(path),"audio_file":"turns/"+path.name,
                                  "sha256":digest(path),"tts":metrics,"timing":timing})
