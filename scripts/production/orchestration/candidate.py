"""Import a real accepted pipeline report without repeating generation or counting text as audio."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT
from scripts.text.utils.schema import Dialogue
from scripts.text.quality import audit
from scripts.text.judge.review import decide


def import_candidate(ledger,path):
    path=Path(path);payload=json.loads(path.read_text(encoding="utf-8"))
    records=payload.get("records",[payload])
    if len(records)!=1:raise ValueError("Import exactly one final pipeline record")
    record=records[0];raw=record.get("dialogue")
    if record.get("mock") or record.get("report",{}).get("decision")!="accept" or not raw:raise ValueError("Only real semantically accepted text is eligible")
    d=Dialogue.from_jsonl_line(json.dumps(raw,ensure_ascii=False))
    if decide(d,record["report"].get("review",{}))["decision"]!="accept":raise ValueError("Current semantic evidence contract failed")
    ident=d.dialogue_id
    if not re.fullmatch(r"[A-Za-z0-9_-]+",ident):raise ValueError("Unsafe ID")
    folder=ledger.root/"samples"/ident
    if folder.exists():raise FileExistsError("Candidate destination already exists")
    ledger.register(raw);folder.mkdir(parents=True)
    (folder/"annotation.text.json").write_text(d.to_jsonl_line(),encoding="utf-8")
    if payload.get("api_calls") is not None:record["api_calls"]=payload["api_calls"]
    record["import_provenance"]={"source":str(path.resolve()),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"text_only_no_hours_counted":True}
    (folder/"text-report.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"dialogue_id":ident,"scenario":d.scenario,"users":len(d.participants),"turns":len(d.turns),"gen_mode":record["mode"],"seed":0}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--report",required=True)
    args=ap.parse_args();print(json.dumps(import_candidate(Ledger(args.root),args.report),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
