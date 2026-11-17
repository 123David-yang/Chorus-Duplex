"""Re-assess unchanged quarantined audio under a new documented QA policy, preserving prior reports."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT
from scripts.production.synthesis.producer import qa_audio,digest
from scripts.production.synthesis.asr import SenseVoice
from scripts.production.publishing.publish import publish
from scripts.text.utils.schema import Dialogue

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--id",required=True);ap.add_argument("--publish-batch",default=None);ap.add_argument("--execute-publish",action="store_true")
    args=ap.parse_args();ledger=Ledger(args.root)
    row=ledger.db.execute("SELECT state,dialogue_json,qa_json FROM samples WHERE id=?",(args.id,)).fetchone()
    if not row or row[0]!="quarantine":raise ValueError("Select an existing quarantined sample")
    d=Dialogue.from_jsonl_line(row[1]);previous=json.loads(row[2]);folder=ledger.root/"samples"/args.id
    components=[]
    for saved in previous.get("per_turn",[]):
        relative=saved.get("audio_file",f"turns/{saved['turn_id']:03d}.wav")
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()):raise ValueError("Component escapes sample folder")
        if digest(path)!=saved["sha256"]:raise ValueError("Original component bytes changed")
        components.append({**saved,"path":str(path)})
    if len(components)!=len(d.turns):raise ValueError("Missing original audio evidence")
    evidence=SimpleNamespace(turn_results=components,provenance=previous["engine"],fingerprint=previous["engine_fingerprint"],last_mix_metrics=previous["mix"])
    report=json.loads((folder/"text-report.json").read_text(encoding="utf-8"))
    qa=qa_audio(d,folder,evidence,SenseVoice(),report["report"])
    qa["supersedes_qa_sha256"]=digest(folder/"qa.json")
    path=folder/("qa-"+qa["policy"]+".json")
    if path.exists():raise FileExistsError("This policy already has an assessment")
    path.write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    if qa["passed"]:ledger.accept_audio(args.id,folder/"audio.flac",qa)
    else:
        with ledger.db:
            ledger.db.execute("UPDATE samples SET qa_json=? WHERE id=?",(json.dumps(qa,ensure_ascii=False),args.id))
            ledger.event(args.id,"qa_recheck_quarantine",{"policy":qa["policy"]})
    print(json.dumps({"dialogue_id":args.id,"passed":qa["passed"],"checks":qa["checks"],"weighted_cer":qa["weighted_cer"],"weighted_phonetic_er":qa["weighted_phonetic_er"]}),flush=True)
    if args.publish_batch:print(json.dumps(publish(ledger,args.publish_batch,execute=args.execute_publish)),flush=True)
    print(json.dumps(ledger.status()),flush=True)

if __name__=="__main__":main()
