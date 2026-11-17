"""Preserve interrupted audio attempts before a retry, without replacing text or accepted artifacts."""
from pathlib import Path
import shutil,time,json

def preserve_partial(folder,root):
    root=Path(root).resolve();folder=Path(folder).resolve();samples=root/"samples"
    if folder.parent!=samples:raise ValueError("Partial attempt must be one sample directly inside the corpus samples directory")
    names=[name for name in ("turns","segments","audio.flac","timeline.json","qa.json","audio-error.txt") if (folder/name).exists()]
    if not names:return None
    destination=(root/"superseded_attempts"/(folder.name+"-"+str(time.time_ns()))).resolve()
    if not destination.is_relative_to(root):raise ValueError("Attempt archive escapes corpus root")
    destination.mkdir(parents=True)
    for name in names:
        source=(folder/name).resolve();target=(destination/name).resolve()
        if not source.is_relative_to(folder) or not target.is_relative_to(destination):raise ValueError("Unsafe audio artifact boundary")
        shutil.move(str(source),str(target))
    for name in ("annotation.text.json","text-report.json"):
        if (folder/name).is_file():shutil.copy2(folder/name,destination/name)
    (destination/"preserved-attempt.json").write_text(json.dumps({"sample_id":folder.name,"reason":"Interrupted attempt or explicit acoustic revision; previous bytes preserved","artifacts":names},indent=2),encoding="utf-8")
    return str(destination)
