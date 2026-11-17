"""Bounded deterministic MOSS pronunciation retries. Failed attempts remain auditable; no gate is relaxed."""
from __future__ import annotations
import hashlib,json,os,uuid,time
from pathlib import Path
import numpy as np
import soundfile as sf
from .native_cache import sha256

class MossAuditExhausted(RuntimeError):pass

class AuditedMossBackend:
    def __init__(self,backend,asr,root):
        self.backend=backend;self.asr=asr;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def __getattr__(self,name):return getattr(self.backend,name)
    def synthesize(self,text,voice,reference_audio,max_seconds,extra_check=None):
        spec={"text":text,"voice":voice,"reference_sha256":sha256(reference_audio) if reference_audio else None,"max_seconds":max_seconds}
        key=hashlib.sha256(json.dumps(spec,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        started=time.perf_counter()
        base=int(os.environ.get("MOSS_SEED","1234"));seeds=list(dict.fromkeys([base,int(key[:8],16),int(key[8:16],16)]))
        folder=self.root/(key+"-"+uuid.uuid4().hex);folder.mkdir();rows=[]
        previous=os.environ.get("MOSS_SEED")
        try:
            with (folder/"attempts.jsonl").open("x",encoding="utf-8") as log:
                for index,seed in enumerate(seeds):
                    os.environ["MOSS_SEED"]=str(seed)
                    attempt_budget=max_seconds if index==0 else max(max_seconds,6.0)
                    result=self.backend.synthesize(text,voice,reference_audio,attempt_budget)
                    audio=np.asarray(result.waveform,dtype=np.float32)
                    valid=audio.ndim in (1,2) and audio.size>0 and bool(np.all(np.isfinite(audio)))
                    assessment=self.asr.assess(audio,result.sample_rate,text) if valid else {"passed":False,"reason":"invalid_waveform"}
                    passed=bool(valid and not result.truncated and not (result.provenance or {}).get("quality_flags") and assessment["passed"])
                    additional=extra_check(audio,result.sample_rate) if passed and extra_check is not None else None
                    if additional is not None:passed=bool(additional["passed"])
                    row={"additional_validation":additional,"attempt":index,"seed":seed,"effective_budget_s":attempt_budget,"source_text":text,"voice":voice,"passed":passed,"truncated":result.truncated,"asr":assessment,"backend":result.provenance}
                    if not passed and valid:
                        scale=max(1.,float(np.max(np.abs(audio)))/.98);path=folder/f"failed-{index}.flac"
                        sf.write(path,audio/scale,result.sample_rate,format="FLAC",subtype="PCM_24")
                        row.update(audio_file=str(path),audio_sha256=sha256(path),restore_scale=scale)
                    rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush();os.fsync(log.fileno())
                    if passed:
                        result.provenance={**(result.provenance or {}),"local_asr_audit":{"policy":"moss-final-waveform-bounded-seed-budget-retry-v3","attempts":len(rows),"additional_validation":additional,"accepted_seed":seed,"accepted_budget_s":attempt_budget,"elapsed_s":time.perf_counter()-started,"evidence_file":str(folder/"attempts.jsonl"),"assessment":assessment}}
                        return result
            raise MossAuditExhausted(f"MOSS failed all {len(seeds)} bounded attempts; evidence: {folder}")
        finally:
            if previous is None:os.environ.pop("MOSS_SEED",None)
            else:os.environ["MOSS_SEED"]=previous
