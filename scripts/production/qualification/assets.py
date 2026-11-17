"""Download public official runtime assets to D; validate archive paths before extraction."""
from __future__ import annotations
import argparse,hashlib,json,tarfile,time,urllib.request
from pathlib import Path

ASSETS={
 "paraformer":("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2","sherpa-onnx-paraformer-zh-2024-03-09"),
 "kokoro_int8":("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2","kokoro-int8-multi-lang-v1_1"),
 "kokoro":("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2","kokoro-multi-lang-v1_1"),
 "sensevoice":("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2","sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")}

def obtain(name,root):
    url,dirname=ASSETS[name]
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    folder=root/dirname
    marker=folder/"asset-provenance.json"
    if marker.is_file():return json.loads(marker.read_text(encoding="utf-8"))
    archive=root/(dirname+".tar.bz2");part=root/(dirname+".tar.bz2.part")
    if not archive.is_file():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=120) as response,part.open("wb") as f:
                    expected=int(response.headers.get("Content-Length",0));count=0
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        f.write(chunk);count+=len(chunk)
                    f.flush()
                    if expected and count!=expected:raise ValueError("incomplete_download")
                part.replace(archive);break
            except Exception as exc:
                print(json.dumps({"asset":name,"attempt":attempt+1,"error_type":type(exc).__name__}),flush=True)
                if attempt==2:raise
    with tarfile.open(archive,"r:bz2") as tar:
        for member in tar.getmembers():
            destination=(root/member.name).resolve()
            if not destination.is_relative_to(root):raise ValueError("archive_path_escape")
        tar.extractall(root,filter="data")
    result={"name":name,"url":url,"archive_sha256":hashlib.file_digest(archive.open("rb"),"sha256").hexdigest(),"folder":str(folder),"downloaded_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    marker.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

HF_REPOS={"paraformer":"csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09","kokoro_int8":"csukuangfj/kokoro-int8-multi-lang-v1_1","kokoro":"csukuangfj/kokoro-multi-lang-v1_1","sensevoice":"csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"}

def obtain_hf(name,root):
    from huggingface_hub import HfApi,snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    repo=HF_REPOS[name];folder=root.resolve()/ASSETS[name][1]
    marker=folder/"asset-provenance.json"
    if marker.is_file():return json.loads(marker.read_text(encoding="utf-8"))
    info=HfApi(token=False).model_info(repo)
    patterns=["model.int8.onnx","tokens.txt","LICENSE","README.md"] if name in ("sensevoice","paraformer") else ["model.onnx","voices.bin","tokens.txt","*.fst","*.txt","LICENSE","README.md","dict/**","espeak-ng-data/**"]
    snapshot_download(repo,revision=info.sha,local_dir=folder,allow_patterns=patterns,token=False,max_workers=4)
    required=["model.int8.onnx","tokens.txt"] if name in ("sensevoice","paraformer") else ["model.onnx","voices.bin","tokens.txt","lexicon-zh.txt"]
    hashes={}
    for filename in required:
        with (folder/filename).open("rb") as f:hashes[filename]=hashlib.file_digest(f,"sha256").hexdigest()
    result={"name":name,"repo":repo,"revision":info.sha,"folder":str(folder),"sha256":hashes,"source":"sherpa-onnx maintainer HF model mirror"}
    marker.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def segmented_weight(repo,revision,filename,folder,size,expected_sha):
    from concurrent.futures import ThreadPoolExecutor
    chunk_size=2*1024*1024
    destination=folder/filename
    if destination.is_file():
        with destination.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()==expected_sha:return
    parts=folder/".range-parts"/filename;parts.mkdir(parents=True,exist_ok=True)
    jobs=[(i,start,min(size-1,start+chunk_size-1)) for i,start in enumerate(range(0,size,chunk_size))]
    def fetch(job):
        i,start,end=job;part=parts/f"{i:06d}.part";length=end-start+1
        if part.is_file() and part.stat().st_size==length:return part
        for attempt in range(8):
            host="https://huggingface.co" if attempt%2==0 else "https://hf-mirror.com"
            url=f"{host}/{repo}/resolve/{revision}/{filename}?download=true&segment={i}&attempt={attempt}&ts={time.time_ns()}"
            try:
                request=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request,timeout=45) as response:
                    actual_range=response.headers.get("Content-Range","").split("/")[0]
                    if response.status!=206 or actual_range!=f"bytes {start}-{end}":
                        raise ValueError("range_mismatch")
                    data=response.read(length+1)
                    if len(data)!=length:raise ValueError("incomplete_range")
                part.write_bytes(data)
                if i%25==0:print(json.dumps({"asset_file":filename,"chunk":i,"chunks":len(jobs)}),flush=True)
                return part
            except Exception as exc:
                if attempt in (3,7):print(json.dumps({"asset_file":filename,"chunk":i,"attempt":attempt+1,"error_type":type(exc).__name__,"validation_error":str(exc) if isinstance(exc,ValueError) and str(exc) in ("range_mismatch","incomplete_range") else "transport"}),flush=True)
                if attempt==7:raise RuntimeError("Asset range download failed: "+type(exc).__name__) from None
    with ThreadPoolExecutor(max_workers=3) as pool:chunks=list(pool.map(fetch,jobs))
    merged=folder/(filename+".verified-part");digest=hashlib.sha256()
    with merged.open("wb") as f:
        for chunk in chunks:
            data=chunk.read_bytes();digest.update(data);f.write(data)
    if digest.hexdigest()!=expected_sha:raise ValueError("asset_checksum_mismatch")
    merged.replace(destination)


def obtain_segmented(name,root):
    from huggingface_hub import HfApi,snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars();repo=HF_REPOS[name];folder=root.resolve()/ASSETS[name][1];folder.mkdir(parents=True,exist_ok=True)
    info=HfApi(token=False).model_info(repo,files_metadata=True)
    filename="model.onnx" if name=="kokoro" else "model.int8.onnx"
    entry=next(x for x in info.siblings if x.rfilename==filename)
    sha=entry.lfs.sha256 if hasattr(entry.lfs,"sha256") else entry.lfs["sha256"]
    segmented_weight(repo,info.sha,filename,folder,entry.size,sha)
    patterns=["tokens.txt","LICENSE","README.md"] if name in ("sensevoice","paraformer") else ["voices.bin","tokens.txt","*.fst","*.txt","LICENSE","README.md","dict/**","espeak-ng-data/**"]
    snapshot_download(repo,revision=info.sha,local_dir=folder,allow_patterns=patterns,ignore_patterns=["*.onnx"],token=False,max_workers=4)
    result={"name":name,"repo":repo,"revision":info.sha,"folder":str(folder),"sha256":{filename:sha},"transfer":"verified_2MiB_ranges","source":"sherpa-onnx maintainer HF mirror; weight hash checked against official HF metadata"}
    (folder/"asset-provenance.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("name",choices=ASSETS);ap.add_argument("--root",default=r"D:\DuplexVoice_resources\models");ap.add_argument("--source",choices=["hf","github","segmented"],default="segmented")
    args=ap.parse_args();obtain_fn={"hf":obtain_hf,"github":obtain,"segmented":obtain_segmented}[args.source]
    print(json.dumps(obtain_fn(args.name,Path(args.root))),flush=True)

if __name__=="__main__":main()
