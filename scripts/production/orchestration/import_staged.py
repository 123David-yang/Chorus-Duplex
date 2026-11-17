"""Verify and import independently named staged results into the authoritative corpus ledger.
Distinct multi-strict IDs cannot collide with coordinator prod-v4/prod-ma-v4 jobs. SQLite unique
text/PCM constraints and per-ID OS locks protect ingestion; staging never contributes hours itself.
"""
from __future__ import annotations
import argparse,hashlib,json,re,shutil
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT,annotation_digest
from scripts.production.state.lock import ProductionLock
from scripts.production.synthesis.producer import digest
from scripts.text.utils.schema import Dialogue
from scripts.text.judge.review import decide


def copy_verified(source,target,root):
    source=Path(source).resolve();target=Path(target).resolve()
    if not target.is_relative_to(root):raise ValueError("Import target escapes the explicit corpus root")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=digest(source):raise ValueError("Refusing to replace different imported evidence")
    else:shutil.copy2(source,target)
    if digest(target)!=digest(source):raise ValueError("Imported bytes do not match source")


def ingest(ledger,source):
    source=Path(source).resolve();qa=json.loads((source/"qa.json").read_text(encoding="utf-8"))
    raw=json.loads((source/"annotation.text.json").read_text(encoding="utf-8"));d=Dialogue.from_jsonl_line(json.dumps(raw,ensure_ascii=False));ident=d.dialogue_id
    if not re.fullmatch(r"(?:multi-strict|moss-queue)-[A-Za-z0-9_-]+",ident):raise ValueError("Independent staging namespace required")
    if not qa.get("passed") or qa.get("fixture"):raise ValueError("Failed/fixture staging cannot count")
    if qa.get("annotation_sha256")!=annotation_digest(raw) or qa.get("audio_sha256")!=digest(source/"audio.flac") or qa.get("timeline_sha256")!=digest(source/"timeline.json"):
        raise ValueError("Staged QA no longer binds its exact evidence")
    record=json.loads((source/"text-report.json").read_text(encoding="utf-8"))
    if record.get("mock") or annotation_digest(record["dialogue"])!=annotation_digest(raw) or decide(d,record["report"].get("review",{}))["decision"]!="accept":raise ValueError("Staged text evidence is not valid")
    if len(qa.get("per_turn",[]))!=len(d.turns):raise ValueError("Missing component evidence")
    files=["annotation.text.json","text-report.json","audio.flac","timeline.json","qa.json","worker-plan.json"]
    for expected_id,t in enumerate(qa["per_turn"]):
        if t["turn_id"]!=expected_id or t["speaker"]!=d.turns[expected_id].speaker or t["text"]!=d.turns[expected_id].text:raise ValueError("Staged component annotations differ from the dialogue")
        if t.get("asr",{}).get("expected")!=t["text"]:raise ValueError("ASR covers different reference text")
        relative=t.get("audio_file",f"turns/{t['turn_id']:03d}.wav")
        if not re.fullmatch(r"turns/"+f"{expected_id:03d}"+r"\.(?:flac|wav)",relative):raise ValueError("Unsafe staged component path")
        path=(source/relative).resolve()
        if not path.is_relative_to(source) or digest(path)!=t["sha256"]:raise ValueError("Staged component changed")
        if not t.get("asr",{}).get("passed") or t["asr"].get("numeric_match") is not True:raise ValueError("Staged ASR/numeric gate failed")
        files.append(relative)
    destination=ledger.root/"samples"/ident
    with ProductionLock(ledger.root/"ingestion_locks"/ident):
        for name in files:copy_verified(source/name,destination/name,ledger.root)
        ledger.register(raw)
        enriched={**qa,"staging_import":{"original_qa_sha256":digest(source/"qa.json"),"worker_plan_sha256":digest(source/"worker-plan.json"),"independent_staging_id":ident}}
        path=destination/"qa.imported.json";content=json.dumps(enriched,ensure_ascii=False,indent=2)
        if path.exists():
            if path.read_text(encoding="utf-8")!=content:raise ValueError("Import QA changed on retry")
        else:path.write_text(content,encoding="utf-8")
        ledger.accept_audio(ident,destination/"audio.flac",enriched)
        result={"status":"imported_accepted","dialogue_id":ident,"audio_sha256":qa["audio_sha256"],"duration_s":qa["duration_s"],"source_stage":str(source)}
        receipt=ledger.root/("import-"+ident+".json")
        if not receipt.exists():receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--source",required=True)
    args=ap.parse_args();print(json.dumps(ingest(Ledger(args.root),args.source),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
