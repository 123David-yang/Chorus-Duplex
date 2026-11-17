"""Transactional, unique-content accounting for the 1000h release. Fixtures never count."""
from __future__ import annotations
import argparse,hashlib,json,sqlite3,time,uuid
from pathlib import Path

ROOT_DEFAULT=Path(r"D:\DuplexVoice_production\release_20261031_v5_moss")
REQUIRED_CHECKS={"text_semantics","audio_finite","no_clipping","no_self_overlap","asr_agreement","timing_anchor","licenses","speech_coverage"}

def annotation_digest(dialogue):
    return hashlib.sha256(json.dumps(dialogue,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def text_digest(dialogue):
    rows=[{"speaker":t["speaker"],"text":t["text"],"addressee":t["addressee"]} for t in dialogue["turns"]]
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

class Ledger:
    def __init__(self,root=ROOT_DEFAULT):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/"ledger.sqlite",timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS samples(
          id TEXT PRIMARY KEY,text_sha TEXT NOT NULL UNIQUE,dialogue_json TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'text_accepted',audio_path TEXT,audio_sha TEXT,pcm_sha TEXT UNIQUE,
          frames INTEGER,sample_rate INTEGER,channels INTEGER,qa_json TEXT,remote_commit TEXT,created_utc TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS exclusions(text_sha TEXT PRIMARY KEY,source TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, sample_id TEXT,kind TEXT,payload_json TEXT,utc TEXT);
        """)
        columns={row[1] for row in self.db.execute("PRAGMA table_info(samples)")}
        if "annotation_sha" not in columns:self.db.execute("ALTER TABLE samples ADD COLUMN annotation_sha TEXT")
        for ident,payload in self.db.execute("SELECT id,dialogue_json FROM samples WHERE annotation_sha IS NULL").fetchall():
            self.db.execute("UPDATE samples SET annotation_sha=? WHERE id=?",(annotation_digest(json.loads(payload)),ident))
        self.db.commit()
        from .dedup import NearDuplicateIndex
        self.near=NearDuplicateIndex(self.db)
        with self.db:
            for sha,payload in self.db.execute("SELECT text_sha,dialogue_json FROM samples WHERE text_sha NOT IN (SELECT text_sha FROM near_documents)").fetchall():
                self.near.add(sha,json.loads(payload),"production")
    def event(self,ident,kind,payload):
        self.db.execute("INSERT INTO events(sample_id,kind,payload_json,utc) VALUES(?,?,?,?)",
            (ident,kind,json.dumps(payload,ensure_ascii=False),time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    def exclude(self,dialogue,source):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO exclusions VALUES(?,?)",(text_digest(dialogue),source))
            self.near.add(text_digest(dialogue),dialogue,source)
    def register(self,dialogue):
        if dialogue.get("profile",{}).get("quality_status")!="accept":raise ValueError("Text lacks semantic acceptance")
        sha=text_digest(dialogue)
        if self.db.execute("SELECT 1 FROM exclusions WHERE text_sha=?",(sha,)).fetchone():raise ValueError("Legacy/previously published text is excluded")
        ident=dialogue["dialogue_id"]
        old=self.db.execute("SELECT text_sha,annotation_sha FROM samples WHERE id=?",(ident,)).fetchone()
        if old:
            if old[0]!=sha or old[1]!=annotation_digest(dialogue):raise ValueError("Immutable ID reused with different annotations")
            return ident
        with self.db:
            similar=self.near.closest(dialogue)
            if similar:raise ValueError("Near-copy of existing/excluded dialogue: "+json.dumps(similar))
            self.db.execute("INSERT INTO samples(id,text_sha,dialogue_json,created_utc,annotation_sha) VALUES(?,?,?,?,?)",
                (ident,sha,json.dumps(dialogue,ensure_ascii=False),time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),annotation_digest(dialogue)))
            self.near.add(sha,dialogue,"production")
            self.event(ident,"text_registered",{"text_sha":sha})
        self.write_status();return ident
    def accept_audio(self,ident,audio_path,qa):
        goal_path=self.root/"goal.json"
        if goal_path.exists():
            expected=json.loads(goal_path.read_text(encoding="utf-8")).get("selected_backend")
            if expected and qa.get("engine",{}).get("backend")!=expected:raise ValueError("Audio backend differs from the selected production backend")
        import soundfile as sf
        path=Path(audio_path).resolve()
        if not path.is_relative_to(self.root):raise ValueError("Production audio must reside under the dedicated release root")
        if qa.get("fixture") or qa.get("passed") is not True:raise ValueError("Quarantined/fixture audio cannot count")
        checks=qa.get("checks",{})
        if not REQUIRED_CHECKS<=set(checks) or any(checks[k] is not True for k in REQUIRED_CHECKS):raise ValueError("Missing or failed mandatory audio QA")
        row=self.db.execute("SELECT text_sha,annotation_sha FROM samples WHERE id=?",(ident,)).fetchone()
        if not row or qa.get("text_sha256")!=row[0] or qa.get("annotation_sha256")!=row[1] or qa.get("dialogue_id")!=ident:
            raise ValueError("QA does not cover the registered text")
        info=sf.info(path)
        if not 15<=info.frames/info.samplerate<=300:raise ValueError("Production dialogue duration must be 15..300 seconds")
        if info.frames<=0 or info.samplerate!=48000 or info.channels!=2:raise ValueError("Unexpected production audio format")
        if not self.db.execute("SELECT 1 FROM samples WHERE id=?",(ident,)).fetchone():raise ValueError("Text must be registered first")
        with path.open("rb") as file:
            sha=hashlib.file_digest(file,"sha256").hexdigest()
        if qa.get("audio_sha256")!=sha:raise ValueError("QA does not cover the current audio bytes")
        pcm=hashlib.sha256()
        with sf.SoundFile(path) as f:
            for block in f.blocks(blocksize=65536,dtype="int16",always_2d=True):pcm.update(block.astype("<i2",copy=False).tobytes())
        pcm_sha=pcm.hexdigest()
        old=self.db.execute("SELECT state,audio_sha FROM samples WHERE id=?",(ident,)).fetchone()
        if old[0] in ("accepted","uploaded"):
            if old[1]!=sha:raise ValueError("Accepted audio is immutable")
            return
        with self.db:
            self.db.execute("UPDATE samples SET state='accepted',audio_path=?,audio_sha=?,pcm_sha=?,frames=?,sample_rate=?,channels=?,qa_json=? WHERE id=?",
                (str(path),sha,pcm_sha,info.frames,info.samplerate,info.channels,json.dumps(qa,ensure_ascii=False),ident))
            self.event(ident,"audio_accepted",{"sha256":sha,"frames":info.frames,"sample_rate":info.samplerate})
        self.write_status()
    def mark_uploaded(self,ident,sha,commit,verified=False):
        if not verified or not commit:raise ValueError("Remote hash verification is required")
        row=self.db.execute("SELECT state,audio_sha FROM samples WHERE id=?",(ident,)).fetchone()
        if not row or row[0] not in ("accepted","uploaded") or row[1]!=sha:raise ValueError("Only the accepted immutable audio may be uploaded")
        with self.db:
            self.db.execute("UPDATE samples SET state='uploaded',remote_commit=? WHERE id=?",(commit,ident))
            self.event(ident,"upload_verified",{"sha256":sha,"commit":commit})
        self.write_status()
    def withdraw(self,ident,reason,evidence):
        row=self.db.execute("SELECT state,audio_sha,remote_commit FROM samples WHERE id=?",(ident,)).fetchone()
        if not row or row[0] not in ("accepted","uploaded"):raise ValueError("Only an accepted/published sample can be withdrawn")
        record={"dialogue_id":ident,"reason":reason,"evidence":evidence,"prior_state":row[0],"audio_sha256":row[1],"remote_commit":row[2],"old_bytes_preserved":True}
        path=self.root/("withdrawn-"+ident+".json")
        with path.open("x",encoding="utf-8") as f:json.dump(record,f,ensure_ascii=False,indent=2)
        with self.db:
            self.db.execute("UPDATE samples SET state='withdrawn' WHERE id=?",(ident,))
            self.event(ident,"withdrawn_from_target",record)
        self.write_status();return record

    def status(self):
        rows=self.db.execute("SELECT state,count(*),coalesce(sum(cast(frames as real)/sample_rate),0) FROM samples GROUP BY state").fetchall()
        counts={s:n for s,n,_ in rows};accepted=sum(sec for s,_,sec in rows if s in ("accepted","uploaded"));uploaded=sum(sec for s,_,sec in rows if s=="uploaded")
        from .credit import credited_frames
        clean_accepted=0.;clean_uploaded=0.;unverified=[]
        for ident,state,frames,sr,qa_raw in self.db.execute("SELECT id,state,frames,sample_rate,qa_json FROM samples WHERE state IN ('accepted','uploaded')"):
            credit=credited_frames(json.loads(qa_raw or '{}'),frames,sr)
            if credit is None:unverified.append(ident);continue
            clean_accepted+=credit/sr
            if state=='uploaded':clean_uploaded+=credit/sr
        return {"target_hours":1000,"counts":counts,"accepted_unique_seconds":accepted,"uploaded_unique_seconds":uploaded,
                "accepted_credited_seconds":clean_accepted,"uploaded_credited_seconds":clean_uploaded,
                "accepted_credited_hours":round(clean_accepted/3600,6),"uploaded_credited_hours":round(clean_uploaded/3600,6),
                "credited_accounting_policy":"exact_component_frame_span_v1","unverified_credit_ids":unverified,
                "accepted_hours":round(accepted/3600,6),"uploaded_hours":round(uploaded/3600,6),"completed":clean_uploaded>=3600000 and not unverified,
                "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    def write_status(self):
        target=self.root/"status.json";tmp=self.root/("status-"+uuid.uuid4().hex+".tmp.json")
        tmp.write_text(json.dumps(self.status(),ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(target)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(ROOT_DEFAULT));args=ap.parse_args()
    ledger=Ledger(args.root);ledger.write_status();print(json.dumps(ledger.status(),ensure_ascii=False))

if __name__=="__main__":main()
