"""Explicit bounded acoustic revision of unpublished quarantines. Keep old artifacts and original text."""
import argparse,json,hashlib,os,time
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT
from scripts.production.state.lock import ProductionLock
from scripts.production.synthesis.asr import SenseVoice
from scripts.production.synthesis.producer import create_production_synth,produce
from scripts.config import load_env
from scripts.text.utils.schema import Dialogue

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=str(ROOT_DEFAULT));p.add_argument("--ids",nargs="+",required=True);a=p.parse_args()
    load_env();os.environ.update(TTS_BACKEND="moss_onnx",DUPLEX_PRODUCTION="1",TTS_STRICT="1");os.environ.setdefault("VOICE_MANIFEST",r"D:\DuplexVoice_resources\voices\AISHELL-3\manifest.json")
    ledger=Ledger(a.root)
    with ProductionLock(ledger.root):
        asr=SenseVoice();synth=create_production_synth(ledger.root,asr=asr)
        registry=Path(os.environ.get("MOSS_QUALIFICATION_REGISTRY",str(ledger.root/"qualification"/"registry.json")))
        qualified=json.loads(registry.read_text(encoding="utf-8"))
        if qualified["engine_fingerprint"]!=synth.fingerprint:raise ValueError("Acoustic revision runtime is not qualified")
        for ident in a.ids:
            row=ledger.db.execute("SELECT state,dialogue_json,qa_json FROM samples WHERE id=?",(ident,)).fetchone()
            if not row:raise ValueError("Unknown sample")
            if row[0] in ("accepted","uploaded"):
                print(json.dumps({"id":ident,"status":"already_eligible_unchanged"}),flush=True);continue
            if row[0]!="quarantine":raise ValueError("Only unpublished audio quarantines can be explicitly revised")
            previous=json.loads(row[2]);checks=previous.get("checks",{})
            if not checks or checks.get("text_semantics") is not True:raise ValueError("Text rejection is not an acoustic revision")
            d=Dialogue.from_jsonl_line(row[1]);reason={"prior_qa_json_sha256":hashlib.sha256(row[2].encode()).hexdigest(),"new_engine_fingerprint":synth.fingerprint,"reason":"Qualified whole-turn acoustic alignment and actual PCM16 audit; original words unchanged"}
            with ledger.db:
                ledger.db.execute("UPDATE samples SET state='text_accepted' WHERE id=?",(ident,));ledger.event(ident,"explicit_acoustic_revision",reason)
            ledger.write_status()
            result=produce({"dialogue_id":ident,"scenario":d.scenario,"users":len(d.participants),"turns":len(d.turns),"seed":0},ledger,synth,asr)
            print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
