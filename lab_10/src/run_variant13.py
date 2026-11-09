"""Измерения варианта 13 с первичными PNG и отдельной проверкой seed 101."""
import argparse, gc, hashlib, json, os, platform, sys, time, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'original'))

def stamp():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/main'); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    import lab10_lib as L
    import torch, diffusers, transformers, numpy, scipy, PIL
    hyp=Path('docs/hypotheses_before_run.json')
    hypotheses=json.loads(hyp.read_text())
    started=stamp()
    assert hypotheses['registered_at'] < started
    passport=dict(started_at=started,variant=13,context=3,prompt=L.BRIEFS[2],probe=L.TEXT_PROBE,
        seeds=list(L.SEEDS),device='cpu',dtype='float32',threads=torch.get_num_threads(),
        python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
        torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,
        numpy=numpy.__version__,scipy=scipy.__version__,pillow=PIL.__version__,
        cpu_count=os.cpu_count(),clip_id=L.CLIP_ID,clip_revision=L.CLIP_REVISION,
        configs={k:L.MODELS[k] for k in ('turbo','lcm')},hypotheses_sha256=sha(hyp),
        library_sha256=sha(Path(L.__file__)),runner_sha256=sha(Path(__file__)),schedulers={})
    write(out/'passport.json',passport); print('START',started,flush=True)
    t=time.perf_counter(); clip=L.load_clip(); print('CLIP_LOAD_SECONDS',time.perf_counter()-t,flush=True)
    tokens=clip[1].tokenizer(L.BRIEFS[2],truncation=False)['input_ids']
    passport['clip_prompt_tokens']=len(tokens); passport['clip_max_tokens']=77
    records={}; repeats={}
    for key in ('turbo','lcm'):
        t=time.perf_counter(); pipe=L.load_model(key)
        print('MODEL_LOAD',key,time.perf_counter()-t,flush=True)
        passport['schedulers'][key]={'class':type(pipe.scheduler).__name__,'config':dict(pipe.scheduler.config)}
        write(out/'passport.json',passport)
        _,warm=L.generate(pipe,key,L.BRIEFS[2],0)
        print('WARMUP',key,warm,flush=True); records[key]=[]
        for seed in L.SEEDS:
            row,img=L.measure(pipe,clip,key,2,seed)
            png=out/f'{key}_s{seed}.png'; img.save(png)
            img.save(out/f'{key}_s{seed}.jpg',quality=90)
            row.update(seed=seed,image=png.name,sha256=sha(png),measured_at=stamp())
            records[key].append(row)
            with (out/'measurements.jsonl').open('a') as f:f.write(json.dumps(dict(model=key,**row))+'\n')
            print('MEASURE',key,seed,json.dumps(row),flush=True)
        img,dt=L.generate(pipe,key,L.BRIEFS[2],101)
        dest=out/f'repeat_{key}_s101.png'; img.save(dest)
        repeats[key]=dict(image=dest.name,sha256=sha(dest),seconds=dt,completed_at=stamp(),
                          matches_primary=sha(dest)==records[key][0]['sha256'])
        print('REPEAT',key,json.dumps(repeats[key]),flush=True)
        del pipe; gc.collect()
    result=dict(variant=13,context=3,pair=['turbo','lcm'],completed_at=stamp(),per_seed=records,
                rows=L.compare(records['turbo'],records['lcm']),repeats=repeats)
    write(out/'results.json',result)
    L.blind_sheets(13,str(out))
    passport['completed_at']=result['completed_at']; write(out/'passport.json',passport)
    print('COMPLETE',stamp(),flush=True)
    pack=Path('artifacts/engee_results.zip')
    with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(out.parent))
        for f in [Path('src/run_variant13.py'),Path('src/original/lab10_lib.py'),hyp]:z.write(f,str(f))
        if Path('environment.log').exists():z.write('environment.log','environment.log')
    print('EXPORT',pack,sha(pack),flush=True)

if __name__=='__main__': main()
