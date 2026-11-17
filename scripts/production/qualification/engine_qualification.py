"""Bind production OpenVINO startup to actual current core-voice evidence."""
import hashlib,json
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_openvino(folder,voices):
    if not folder:raise ValueError("Set KOKORO_QUALIFICATION_DIR to completed current OpenVINO voice tests")
    folder=Path(folder);summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if not set(voices)<=set(summary["qualified_voices"]):raise ValueError("Unqualified OpenVINO voice")
    worker=Path(__file__).parents[2]/"sound"/"synthesis"/"kokoro_frontend_worker.py"
    attestation=json.loads((folder/"frontend-source.json").read_text(encoding="utf-8"))
    if attestation["sha256"]!=digest(worker):raise ValueError("Frontend changed after voice qualification")
    current_frontend="misaki-0.9.4-zh1.1-context-v2"
    evidence={}
    for voice in set(voices):
        for case in ("statement","question","numbers","negation","bc_agree","bc_understood"):
            path=folder/f"{voice:03d}-{case}.json";row=json.loads(path.read_text(encoding="utf-8"))
            backend=row["tts"]["backend_provenance"]
            if not row["passed"] or backend.get("engine")!="openvino" or backend.get("frontend")!=current_frontend:
                raise ValueError("Core voice evidence uses a different engine/frontend")
            if backend.get("model_sha256")!="523b595681c7e4c7b9797a166f0971df63ec06b005e936339693bb461f871d64":raise ValueError("Different qualified model")
            if digest(path.with_suffix(".flac"))!=row["audio_sha256"]:raise ValueError("Qualified waveform changed")
            evidence[str(path)]=digest(path)
    return {"reports_sha256":evidence,"summary_sha256":digest(folder/"summary.json"),"current_frontend_source_sha256":digest(worker)}
