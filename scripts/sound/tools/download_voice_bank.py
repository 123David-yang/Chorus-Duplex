"""Download a small public reference bank, with immutable provenance."""

def main():
    from pathlib import Path
    from scripts.production.moss_scale.config import resources_root
    import urllib.request,json,wave,io,hashlib,concurrent.futures,argparse
    def fetch(url):
     for attempt in range(4):
      try:
       with urllib.request.urlopen(url,timeout=40) as response: return response.read()
      except Exception:
       if attempt==3: raise

    ap=argparse.ArgumentParser(description='Download a small licensed AISHELL-3 reference bank, not the full corpus')
    ap.add_argument('--out',default=str(resources_root()/'voices'/'AISHELL-3'))
    args=ap.parse_args()
    out=Path(args.out)
    if (out/'manifest.json').exists(): raise FileExistsError('Choose a new bank directory to preserve provenance')
    out.mkdir(parents=True,exist_ok=True)
    api='https://huggingface.co/api/datasets/AISHELL/AISHELL-3'
    head=urllib.request.urlopen(urllib.request.Request('https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/main/spk-info.txt',method='HEAD'),timeout=30)
    revision=head.headers.get('x-repo-commit')
    if not revision: raise RuntimeError('missing immutable revision')
    base=f'https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{revision}/'
    info=urllib.request.urlopen(base+'spk-info.txt',timeout=30).read().decode('utf-8-sig')
    (out/'spk-info.txt').write_text(info,encoding='utf-8')
    print('speaker metadata sample:',info.splitlines()[:6])
    folders=json.loads(fetch(api+'/tree/'+revision+'/test/wav?limit=100'))
    def grab(folder):
     sid=folder['path'].split('/')[-1]
     rows=json.loads(fetch(api+'/tree/'+revision+'/'+folder['path']+'?limit=100'))
     for row in sorted((r for r in rows if r['path'].endswith('.wav')),key=lambda r:r.get('size',0),reverse=True)[:3]:
      try: data=fetch(base+row['path'])
      except Exception: continue
      with wave.open(io.BytesIO(data)) as w: duration=w.getnframes()/w.getframerate(); sr=w.getframerate(); channels=w.getnchannels()
      if 3<=duration<=15:
       p=out/(sid+'.wav'); p.write_bytes(data)
       attrs=next((line.split() for line in info.splitlines() if line.startswith(sid)),[])
       gender='女声' if 'female' in attrs else '男声' if 'male' in attrs else '未知'
       return {'id':'aishell3_'+sid,'speaker_id':sid,'gender':gender,'attributes':attrs,'reference_audio':p.name,'duration_s':round(duration,3),'sample_rate':sr,'channels':channels,'sha256':hashlib.sha256(data).hexdigest(),'source_url':base+row['path'],'license':'Apache-2.0'}
     return None
    # Download a small reference bank rather than the entire 19GB corpus.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
     voices=[v for v in pool.map(grab,folders[:40]) if v]
    manifest={'schema_version':1,'source_dataset':'AISHELL/AISHELL-3','revision':revision,'upstream':'https://www.openslr.org/93/','license':'Apache-2.0','purpose':'Optional voice-cloning references; subject to synthesis QA before production','voices':voices}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    license_text=urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt',timeout=30).read()
    (out/'LICENSE-2.0.txt').write_bytes(license_text)
    (out/'NOTICE.txt').write_text('AISHELL-3: A Multi-Speaker Mandarin TTS Corpus. Beijing Shell Shell Technology Co., Ltd.\nYao Shi, Hui Bu, Xin Xu, Shaoji Zhang, Ming Li. Interspeech 2021.\nUpstream: https://www.openslr.org/93/\nHF mirror: https://huggingface.co/datasets/AISHELL/AISHELL-3\nReference bank only, not an endorsement or identity claim. See manifest for immutable source revision and hashes.\n',encoding='utf-8')
    print(json.dumps({'voices':len(voices),'genders':{g:sum(v['gender']==g for v in voices) for g in ['男声','女声','未知']},'bytes':sum(p.stat().st_size for p in out.glob('*') if p.is_file()),'manifest':str(out/'manifest.json')},ensure_ascii=False))


if __name__ == "__main__":
    main()
