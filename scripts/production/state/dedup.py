"""Conservative corpus near-copy filter: MinHash candidates, exact character-shingle Jaccard.
This is a heuristic candidate search, not proof of semantic uniqueness. Exact text/PCM checks remain mandatory.
"""
from __future__ import annotations
import hashlib,random,struct,unicodedata

_PRIME=(1<<61)-1
_rng=random.Random(20261031)
_PERM=[(_rng.randrange(1,_PRIME),_rng.randrange(_PRIME)) for _ in range(64)]


def document(dialogue):
    text=''.join(t['text'] for t in dialogue['turns'] if t.get('kind')!='backchannel')
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC',text) if c.isalnum())


def shingles(text):return {text[i:i+5] for i in range(max(0,len(text)-4))}


def signature(parts):
    hashes=[int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')%_PRIME for s in parts]
    return [min((a*x+b)%_PRIME for x in hashes) for a,b in _PERM] if hashes else []


def buckets(values):
    return [str(i//4)+':'+hashlib.sha256(struct.pack('<4Q',*values[i:i+4])).hexdigest() for i in range(0,len(values),4)]

class NearDuplicateIndex:
    def __init__(self,db):
        self.db=db
        db.executescript("""CREATE TABLE IF NOT EXISTS near_documents(text_sha TEXT PRIMARY KEY,document TEXT NOT NULL,source TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS near_bands(bucket TEXT NOT NULL,text_sha TEXT NOT NULL,PRIMARY KEY(bucket,text_sha));""")
    def add(self,text_sha,dialogue,source):
        if self.db.execute('SELECT 1 FROM near_documents WHERE text_sha=?',(text_sha,)).fetchone():return
        text=document(dialogue)
        self.db.execute('INSERT INTO near_documents VALUES(?,?,?)',(text_sha,text,source))
        self.db.executemany('INSERT OR IGNORE INTO near_bands VALUES(?,?)',[(b,text_sha) for b in buckets(signature(shingles(text)))])
    def closest(self,dialogue,exclude_sha=None,threshold=.85):
        text=document(dialogue);parts=shingles(text)
        if len(parts)<30:return None
        keys=buckets(signature(parts));candidates=self.db.execute('SELECT DISTINCT d.text_sha,d.document,d.source FROM near_documents d JOIN near_bands b ON b.text_sha=d.text_sha WHERE b.bucket IN ('+','.join('?' for _ in keys)+')',keys).fetchall()
        best=None
        for sha,other,source in candidates:
            if sha==exclude_sha:continue
            other_parts=shingles(other);score=len(parts&other_parts)/max(1,len(parts|other_parts))
            if score>=threshold and (best is None or score>best['jaccard']):best={'text_sha':sha,'source':source,'jaccard':score,'policy':'char5-minhash64-jaccard85-v1'}
        return best
