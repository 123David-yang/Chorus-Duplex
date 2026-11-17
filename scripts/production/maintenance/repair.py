"""One bounded acoustic repair of quarantined speech; original bytes and failed QA stay intact.
Only failed ASR turns are re-synthesized, with unchanged lexical text and voice. No gate relaxation.
"""
from __future__ import annotations
import argparse,hashlib,json,os,traceback
from pathlib import Path
from scripts.production.synthesis.producer import CachedSynth,qa_audio,digest
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT,annotation_digest
from scripts.production.state.lock import ProductionLock
from scripts.production.synthesis.asr import SenseVoice
from scripts.sound.synthesis.dialogue_to_wav import dialogue_to_stereo
from scripts.text.utils.schema import Dialogue
from scripts.config import load_env

class RepairSynth(CachedSynth):
    def __init__(self,root,source,failed,speed):
        super().__init__(root)
        self.readonly_cache_dirs=[source/"segments"];self.failed=set(failed);self.speed=speed
        self.base_fingerprint=self.fingerprint
        self.repair_fingerprint=hashlib.sha256((self.fingerprint+f":bounded-repair-v1:speed={speed}:clause-pause").encode()).hexdigest()
    def turn_to_mono(self,text,voice="Junhao",jitter_db=0,target_offset_db=0):
        selected=len(self.turn_results) in self.failed
        previous=os.environ.get("TTS_SPEED")
        self.fingerprint=self.repair_fingerprint if selected else self.base_fingerprint
        if selected:
            os.environ["TTS_SPEED"]=str(self.speed)
            actual=text.replace("，","。")
        else:actual=text
        try:
            audio=super().turn_to_mono(actual,voice,jitter_db,target_offset_db)
            self.last_synthesis_metrics={**self.last_synthesis_metrics,"repair_selected":selected,"actual_tts_input":actual}
            return audio
        finally:
            self.fingerprint=self.base_fingerprint
            if previous is None:os.environ.pop("TTS_SPEED",None)
            else:os.environ["TTS_SPEED"]=previous


def repair(ledger,ident,speed=.95):
    row=ledger.db.execute("SELECT state,dialogue_json,qa_json FROM samples WHERE id=?",(ident,)).fetchone()
    if not row or row[0]!="quarantine":raise ValueError("Repair requires an existing quarantined dialogue")
    d=Dialogue.from_jsonl_line(row[1]);old=json.loads(row[2]);source=ledger.root/"samples"/ident
    if old.get("annotation_sha256")!=annotation_digest(json.loads(row[1])):raise ValueError("Original annotation changed")
    if old.get("audio_sha256")!=digest(source/"audio.flac"):raise ValueError("Original quarantined waveform changed")
    if any(not v for k,v in old["checks"].items() if k!="asr_agreement"):raise ValueError("This repair only handles isolated ASR agreement failures")
    failed=[t["turn_id"] for t in old["per_turn"] if not t["asr"]["passed"]]
    if not failed or len(failed)>3:raise ValueError("Bounded repair supports1..3 failed turns; investigate systematic failures separately")
    if any(d.turns[i].kind=="backchannel" for i in failed):raise ValueError("Short backchannel failures need explicit voice calibration")
    for t in old["per_turn"]:
        path=source/"turns"/f"{t['turn_id']:03d}.wav"
        if digest(path)!=t["sha256"]:raise ValueError("Original component changed")
    folder=source/"repairs"/"acoustic-v1"
    if folder.exists():raise FileExistsError("This bounded repair was already attempted; inspect preserved results")
    folder.mkdir(parents=True)
    metadata={"policy":"bounded-acoustic-repair-v1","source_audio_sha256":old["audio_sha256"],"source_qa_sha256":digest(source/"qa.json"),
              "failed_turn_ids":failed,"speed":speed,"lexical_text_unchanged":True,"voice_assignments_unchanged":True,
              "punctuation_transform":"comma to period only on selected TTS inputs; actual inputs recorded per segment"}
    (folder/"repair-plan.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    synth=RepairSynth(ledger.root,source,failed,speed)
    if synth.base_fingerprint!=old["engine_fingerprint"]:raise ValueError("Original runtime fingerprint changed; do not silently replay a different engine")
    synth.begin(folder)
    try:
        dialogue_to_stereo(d,synth,folder/"audio.flac",timeline_path=folder/"timeline.json")
        text_report=json.loads((source/"text-report.json").read_text(encoding="utf-8"))["report"]
        qa=qa_audio(d,folder,synth,SenseVoice(),text_report)
        qa["repair"]=metadata
        (folder/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
        with ledger.db:ledger.event(ident,"bounded_acoustic_repair",{"passed":qa["passed"],"folder":str(folder),"failed_turns":failed})
        if qa["passed"]:ledger.accept_audio(ident,folder/"audio.flac",qa)
        return {"id":ident,"passed":qa["passed"],"checks":qa["checks"],"weighted_cer":qa["weighted_cer"],"weighted_phonetic_er":qa["weighted_phonetic_er"],"duration_s":qa["duration_s"]}
    except Exception:
        (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8");raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--id",required=True)
    args=ap.parse_args();load_env();os.environ.update(DUPLEX_PRODUCTION="1",TTS_BACKEND="moss_onnx",TTS_STRICT="1",TTS_SPEED="1.0")
    ledger=Ledger(args.root)
    with ProductionLock(ledger.root):print(json.dumps(repair(ledger,args.id),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
