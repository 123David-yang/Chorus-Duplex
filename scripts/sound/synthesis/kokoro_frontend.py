"""JSON-lines client for a single reusable isolated G2P subprocess."""
from __future__ import annotations
import atexit,json,os,queue,subprocess,sys,threading,uuid
from pathlib import Path

class ChineseFrontend:
    def __init__(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.log=(folder/("g2p-"+uuid.uuid4().hex+".stderr.log")).open("x",encoding="utf-8")
        self.process=subprocess.Popen([sys.executable,"-m","scripts.sound.synthesis.kokoro_frontend_worker"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
              text=True,encoding="utf-8",bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        self.responses=queue.Queue()
        def reader():
            try:
                for line in self.process.stdout:self.responses.put(json.loads(line))
            except Exception as exc:self.responses.put({"ok":False,"error_type":type(exc).__name__})
            finally:self.responses.put({"ok":False,"error_type":"FrontendClosed"})
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start();atexit.register(self.close)
        response=self.responses.get(timeout=60)
        if not response.get("ready"):self.close();raise RuntimeError("Chinese frontend failed to initialize")
        self.version=response["frontend"]
    def __call__(self,text):
        if self.process.poll() is not None:raise RuntimeError("Chinese frontend is not live")
        self.process.stdin.write(json.dumps({"text":text},ensure_ascii=False)+"\n");self.process.stdin.flush()
        response=self.responses.get(timeout=60)
        if not response.get("ok"):raise RuntimeError("Chinese frontend failed: "+response.get("error_type","Unknown"))
        return response
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)
        if not self.log.closed:self.log.close()
