"""OS-held single-producer lock; released by the OS even after a crash."""
from __future__ import annotations
import json,os,time
from pathlib import Path

class ProductionLock:
    def __init__(self,root):self.path=Path(root)/"production.lock";self.stream=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open("a+b")
        if self.path.stat().st_size==0:self.stream.write(b" ");self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close();self.stream=None
            raise RuntimeError("Another production worker holds this release root") from None
        self.stream.seek(0);self.stream.truncate()
        self.stream.write(json.dumps({"pid":os.getpid(),"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}).encode())
        self.stream.flush()
        return self
    def __exit__(self,*exc):
        self.stream.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
        self.stream.close();self.stream=None
