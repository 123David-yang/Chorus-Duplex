"""Build immutable MOSS qualification evidence; validate complete core coverage, hashes and runtime identity."""
import argparse,json,hashlib
from pathlib import Path
from .qualify_voices import CORE

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(folders,destination):
    voices={};fingerprint=None
    for folder in map(Path,folders):
        summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        for voice in summary["qualified_voices"]:
            assessments=[]
            for case in sorted(CORE):
                path=folder/f"{voice}-{case}.json";row=json.loads(path.read_text(encoding="utf-8"))
                if row.get("voice")!=voice or row.get("case")!=case or not row.get("passed"):raise ValueError("Incomplete or failed core qualification")
                if row["tts"]["backend_provenance"].get("engine")!="moss_onnx":raise ValueError("Non-MOSS evidence")
                if digest(path.with_suffix(".flac"))!=row["audio_sha256"]:raise ValueError("Qualification audio changed")
                current=row.get("engine_fingerprint")
                if not current:raise ValueError("Missing exact production runtime qualification")
                if fingerprint is None:fingerprint=current
                if current!=fingerprint:raise ValueError("Cannot mix different production runtimes")
                assessments.append({"case":case,"report_path":str(path.resolve()),"report_sha256":digest(path),"audio_sha256":row["audio_sha256"]})
            voices[voice]={"assessments":assessments}
    if not voices:raise ValueError("No qualified MOSS voices")
    result={"schema_version":2,"backend":"moss_onnx","engine_fingerprint":fingerprint,"voices":voices,"note":"Objective core tests plus bounded retries; not human MOS or a guarantee for arbitrary text."}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(result,ensure_ascii=False,indent=2)
    if destination.exists():
        if destination.read_text(encoding="utf-8")!=text:raise ValueError("Choose a new immutable registry path")
    else:destination.write_text(text,encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--folders",nargs="+",required=True);p.add_argument("--out",required=True);args=p.parse_args()
    result=build(args.folders,args.out);print(json.dumps({"backend":result["backend"],"qualified_voices":list(result["voices"])}))
if __name__=="__main__":main()
