"""Append-only verified HF publishing of accepted production batches. Never deletes legacy files."""
from __future__ import annotations
import argparse,hashlib,json,os,re,uuid
from pathlib import Path
from scripts.production.state.ledger import Ledger,ROOT_DEFAULT

REPO="BiggestWave/Chorus-Duplex"
PREFIX="releases/2026-10-31-v5-moss"

def api_client():
    from .network import configure_hf_transport
    configure_hf_transport()
    from scripts.config import load_env
    from huggingface_hub import HfApi,get_token
    load_env()
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    paths=(Path.home()/"Desktop"/"pass-word"/"hf_token.txt",Path.home()/"Desktop"/"hf_token.txt")
    for path in paths:
        if not token and path.is_file():token=path.read_text(encoding="utf-8-sig").strip()
    if not token:token=get_token()
    if not token:raise RuntimeError("No existing authorized HF credential is available")
    return HfApi(endpoint="https://huggingface.co",token=token)

def remote_matches(info,data):
    if isinstance(data,Path):
        with data.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        size=data.stat().st_size
    else:sha=hashlib.sha256(data).hexdigest();size=len(data)
    lfs=getattr(info,"lfs",None)
    if lfs:
        actual=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None)
        return actual==sha
    digest=hashlib.sha1(f"blob {size}\0".encode())
    if isinstance(data,Path):
        with data.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
    else:digest.update(data)
    return getattr(info,"blob_id",None)==digest.hexdigest()

def batch_identity(row):
    ident,dialogue,audio,sha,qa,frames,sr=row
    return {"id":ident,"audio_sha256":sha,"annotation_sha256":hashlib.sha256(dialogue.encode("utf-8")).hexdigest(),
            "qa_sha256":hashlib.sha256(qa.encode("utf-8")).hexdigest(),"frames":frames,"sample_rate":sr}


def publish(ledger,batch,limit=20,execute=False):
    if (ledger.root/"PUBLISH_HOLD.json").exists():raise RuntimeError("Corpus publication held pending speech naturalness acceptance")
    goal_path=ledger.root/"goal.json"
    if goal_path.exists() and json.loads(goal_path.read_text(encoding="utf-8")).get("remote_prefix",PREFIX)!=PREFIX:raise ValueError("Release destination differs from the corpus goal; refusing cross-release publish")
    from huggingface_hub import CommitOperationAdd
    if not re.fullmatch(r"[A-Za-z0-9_-]+",batch):raise ValueError("Unsafe batch identifier")
    if not 1<=limit<=100:raise ValueError("Batch limit must be 1..100")
    stage=ledger.root/"publish_staging"/batch;stage.mkdir(parents=True,exist_ok=True)
    plan_path=stage/"plan.json"
    fields="id,dialogue_json,audio_path,audio_sha,qa_json,frames,sample_rate"
    if plan_path.is_file():
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("batch")!=batch or plan.get("repo")!=REPO or plan.get("prefix")!=PREFIX:raise ValueError("Batch destination changed")
        rows=[]
        for item in plan["samples"]:
            row=ledger.db.execute(f"SELECT {fields} FROM samples WHERE id=? AND state IN ('accepted','uploaded')",(item["id"],)).fetchone()
            if not row:raise ValueError("Planned sample is no longer accepted")
            if batch_identity(row)!=item:raise ValueError("Planned immutable sample changed")
            rows.append(row)
    else:
        rows=ledger.db.execute(f"SELECT {fields} FROM samples WHERE state='accepted' ORDER BY id LIMIT ?",(limit,)).fetchall()
        if not rows:return {"status":"nothing_accepted_to_upload","count":0}
        plan={"schema_version":1,"batch":batch,"repo":REPO,"prefix":PREFIX,"samples":[batch_identity(r) for r in rows]}
        # Freeze membership before any remote commit or per-sample acknowledgement.
        pending=stage/("plan-"+uuid.uuid4().hex+".pending.json")
        with pending.open("x",encoding="utf-8") as stream:
            json.dump(plan,stream,ensure_ascii=False,sort_keys=True,indent=2)
            stream.flush();os.fsync(stream.fileno())
        os.link(pending,plan_path)  # atomic, fails rather than replacing an existing plan
    artifacts={};ids=[]
    for ident,dialogue,audio,sha,qa,frames,sr in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+",ident):raise ValueError("Unsafe dialogue identifier")
        path=Path(audio)
        if not path.resolve().is_relative_to(ledger.root):raise ValueError("Audio outside release root")
        with path.open("rb") as f:current=hashlib.file_digest(f,"sha256").hexdigest()
        if current!=sha:raise ValueError("Accepted audio changed before publication")
        base=f"{PREFIX}/batches/{batch}/samples/{ident}"
        manifest={"dialogue_id":ident,"audio_sha256":sha,"frames":frames,"sample_rate":sr,"channels":2,"unique_seconds":frames/sr,"counts_once":True}
        if json.loads(qa).get("timeline_sha256"):
            timeline=path.parent/"timeline.json"
            with timeline.open("rb") as stream:timeline_sha=hashlib.file_digest(stream,"sha256").hexdigest()
            if timeline_sha!=json.loads(qa)["timeline_sha256"]:raise ValueError("Accepted timeline changed before publication")
            artifacts[base+"/timeline.json"]=timeline
            manifest["timeline_sha256"]=timeline_sha
        manifest["annotation_bytes_sha256"]=hashlib.sha256(dialogue.encode()).hexdigest()
        manifest["qa_bytes_sha256"]=hashlib.sha256(qa.encode()).hexdigest()
        artifacts[base+"/audio.flac"]=path
        artifacts[base+"/annotation.json"]=dialogue.encode("utf-8")
        artifacts[base+"/qa.json"]=qa.encode("utf-8")
        artifacts[base+"/manifest.json"]=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode()
        ids.append((ident,sha,base+"/audio.flac"))
    # Self-contained Arrow Audio column: keeps the new release directly loadable as Parquet.
    import pyarrow as pa
    import pyarrow.parquet as pq
    features={"dialogue_id":{"dtype":"string","_type":"Value"},"audio":{"sampling_rate":48000,"_type":"Audio"},
              "duration_s":{"dtype":"float64","_type":"Value"},"annotation_json":{"dtype":"string","_type":"Value"},
              "qa_json":{"dtype":"string","_type":"Value"}}
    schema=pa.schema([("dialogue_id",pa.string()),("audio",pa.struct([("bytes",pa.binary()),("path",pa.string())])),
                     ("duration_s",pa.float64()),("annotation_json",pa.string()),("qa_json",pa.string())],
                    metadata={b"huggingface":json.dumps({"info":{"features":features}}).encode()})
    data=[]
    for ident,dialogue,audio,sha,qa,frames,sr in rows:
        data.append({"dialogue_id":ident,"audio":{"bytes":Path(audio).read_bytes(),"path":ident+".flac"},
                     "duration_s":frames/sr,"annotation_json":dialogue,"qa_json":qa})
    parquet=stage/"data.parquet";table=pa.Table.from_pylist(data,schema=schema)
    if not parquet.is_file():
        pending=stage/("data-"+uuid.uuid4().hex+".pending.parquet")
        pq.write_table(table,pending,compression="zstd")
        with pending.open("r+b") as stream:os.fsync(stream.fileno())
        os.link(pending,parquet)  # a crash cannot expose a partially written final shard
    else:
        prior=pq.read_table(parquet)
        if not prior.equals(table,check_metadata=True):raise ValueError("Batch ID reused for different Parquet contents")
    artifacts[f"{PREFIX}/batches/{batch}/data.parquet"]=parquet
    card=("# ChorusDuplex v5 MOSS: ongoing 1,000-hour production release\n\n"
          "Target: 1,000 newly accepted unique hours. This is an incremental release, not a claim that the target is complete. "
          "The preliminary v3 pilot was excluded after a stricter per-request audit; see its VALIDATION_UPDATE.json. Existing root-level data is preserved. Clean/noisy derivatives and component tracks never add duplicate hours.\n\n"
          "Each published batch passed versioned automatic text and audio checks, with raw ASR diagnostics and provenance. "
          "Automatic acceptance is not per-item human listening or a guarantee of zero pronunciation errors. "
          "MOSS-TTS-Nano ONNX output is native 48 kHz, mixed into the common 48 kHz stereo contract: left=users/third parties, right=assistant.\n\n"
          "Models: https://github.com/OpenMOSS/MOSS-TTS-Nano (Apache-2.0); "
          "runtime: MOSS ONNX CPU runtime with a local portable reference-loading adapter. Optional reference voices: AISHELL-3 (Apache-2.0), individually documented in provenance. Synthetic personas and scripts; no user recordings. "
          "These initial pilot batches contain no VOiCES background material. Any later noisy release must include its CC-BY-4.0 attribution.\n\n"
          "## Load\n\n```python\nfrom datasets import load_dataset\nds = load_dataset(\n"
          "    'parquet',\n    data_files='hf://datasets/"+REPO+"/"+PREFIX+"/batches/*/data.parquet',\n    split='train',\n)\n```\n\n"
          "Read annotation_json and qa_json with json.loads. Per-dialogue source FLAC, annotations and QA reports are also available under each batch.\n")
    artifacts[f"{PREFIX}/README.md"]=card.encode("utf-8")
    if not execute:return {"status":"prepared","count":len(ids),"files":len(artifacts),"repo":REPO,"prefix":PREFIX,"ids":[x[0] for x in ids]}
    api=api_client();head=api.repo_info(REPO,repo_type="dataset").sha
    existing={r.path:r for r in api.get_paths_info(REPO,list(artifacts),repo_type="dataset",revision=head)}
    operations=[]
    for name,data in artifacts.items():
        if name in existing:
            if not remote_matches(existing[name],data):raise ValueError("Refusing to overwrite a different immutable release object")
        else:operations.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(data) if isinstance(data,Path) else data))
    if operations:
        commit=api.create_commit(REPO,repo_type="dataset",operations=operations,
            commit_message=f"Add verified MOSS batch {batch} ({len(ids)} dialogues)",parent_commit=head)
        head=commit.oid
    verified={r.path:r for r in api.get_paths_info(REPO,list(artifacts),repo_type="dataset",revision=head)}
    for name,data in artifacts.items():
        if name not in verified or not remote_matches(verified[name],data):raise RuntimeError("Remote object verification failed")
    for ident,sha,name in ids:ledger.mark_uploaded(ident,sha,head,verified=True)
    result={"status":"uploaded_and_verified","count":len(ids),"commit":head,"repo":REPO,"prefix":PREFIX,"sample_ids":[x[0] for x in ids]}
    (ledger.root/("upload-"+batch+".json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));ap.add_argument("--batch",required=True);ap.add_argument("--limit",type=int,default=20);ap.add_argument("--execute",action="store_true")
    args=ap.parse_args();print(json.dumps(publish(Ledger(args.root),args.batch,args.limit,args.execute),ensure_ascii=False))

if __name__=="__main__":main()
