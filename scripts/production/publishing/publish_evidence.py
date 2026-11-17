"""Append full text-review evidence to already verified batches without repacking or overwriting."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT,annotation_digest
from .publish import api_client,remote_matches,REPO,PREFIX
from scripts.text.utils.schema import Dialogue
from scripts.text.judge.review import decide


def evidence_artifacts(ledger,batch):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",batch):raise ValueError("Unsafe batch")
    receipt=json.loads((ledger.root/("upload-"+batch+".json")).read_text(encoding="utf-8"))
    if receipt.get("status")!="uploaded_and_verified" or receipt.get("repo")!=REPO or receipt.get("prefix")!=PREFIX:raise ValueError("Receipt destination/status mismatch")
    artifacts={}
    for ident in receipt["sample_ids"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+",ident):raise ValueError("Unsafe sample ID")
        row=ledger.db.execute("SELECT state,dialogue_json,audio_sha,qa_json,annotation_sha FROM samples WHERE id=?",(ident,)).fetchone()
        if not row or row[0]!="uploaded":raise ValueError("Only currently eligible verified data gets acceptance evidence")
        path=ledger.root/"samples"/ident/"text-report.json";raw=path.read_bytes();record=json.loads(raw)
        if annotation_digest(record["dialogue"])!=row[4]:raise ValueError("Text report covers different annotations")
        d=Dialogue.from_jsonl_line(row[1]);report=record["report"]
        if decide(d,report.get("review",{}))["decision"]!="accept":raise ValueError("Current text-review evidence is not accepted")
        # The captured API call schema contains counts/model/latency only; never publish environment/configuration.
        safe_keys={"model","prompt_tokens","completion_tokens","total_tokens","finish_reason","elapsed_s","status","error_type","attempt"}
        usage=[{k:v for k,v in call.items() if k in safe_keys} for call in record.get("api_calls",[])]
        evidence={"schema_version":1,"dialogue_id":ident,"annotation_sha256":row[4],"audio_sha256":row[2],
                  "qa_bytes_sha256":hashlib.sha256(row[3].encode()).hexdigest(),"local_source_report_sha256":hashlib.sha256(raw).hexdigest(),
                  "report":report,"secret_free_api_usage":usage,"note":"Automatic evidence-grounded review, not human MOS/listening."}
        base=f"{PREFIX}/batches/{batch}/samples/{ident}"
        artifacts[base+"/text-review.json"]=json.dumps(evidence,ensure_ascii=False,sort_keys=True,indent=2).encode()
    return artifacts


def publish_evidence(ledger,batch,execute=False):
    from huggingface_hub import CommitOperationAdd
    artifacts=evidence_artifacts(ledger,batch)
    if not execute:return {"status":"prepared","files":list(artifacts)}
    api=api_client();head=api.repo_info(REPO,repo_type="dataset").sha
    existing={x.path:x for x in api.get_paths_info(REPO,list(artifacts),repo_type="dataset",revision=head)}
    operations=[]
    for name,data in artifacts.items():
        if name in existing:
            if not remote_matches(existing[name],data):raise ValueError("Different immutable review artifact already exists")
        else:operations.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=data))
    if operations:head=api.create_commit(REPO,repo_type="dataset",operations=operations,parent_commit=head,commit_message=f"Add full text-review provenance for {batch}").oid
    infos={x.path:x for x in api.get_paths_info(REPO,list(artifacts),repo_type="dataset",revision=head)}
    if any(name not in infos or not remote_matches(infos[name],data) for name,data in artifacts.items()):raise RuntimeError("Evidence remote verification failed")
    result={"status":"evidence_uploaded_and_verified","batch":batch,"files":list(artifacts),"commit":head}
    target=ledger.root/("evidence-"+batch+".json")
    if not target.exists():target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--batch",required=True);ap.add_argument("--execute",action="store_true")
    args=ap.parse_args();print(json.dumps(publish_evidence(Ledger(args.root),args.batch,args.execute)),flush=True)

if __name__=="__main__":main()
