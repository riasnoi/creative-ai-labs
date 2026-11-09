"""Независимая проверка статистик, пиксельных метрик и первичных файлов ЛР10."""
import argparse, hashlib, json, math, statistics, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def pixels(img):
    x=np.asarray(img.convert('RGB').resize((256,256)),dtype=float)/255
    y=np.sum(x*np.array([.299,.587,.114]),axis=2)
    z=np.pad(y,1,mode='symmetric')
    gx=(z[2:,:-2]+2*z[2:,1:-1]+z[2:,2:] - z[:-2,:-2]-2*z[:-2,1:-1]-z[:-2,2:])/8
    gy=(z[:-2,2:]+2*z[1:-1,2:]+z[2:,2:] - z[:-2,:-2]-2*z[1:-1,:-2]-z[2:,:-2])/8
    return float(np.hypot(gx,gy).mean()),float(y.std())

def near(x,y,tol=1e-12): assert math.isclose(x,y,rel_tol=tol,abs_tol=tol),(x,y)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/main')
    parser.add_argument('--require-blind',action='store_true')
    args=parser.parse_args(); root=args.data
    r=json.loads((root/'results.json').read_text()); p=json.loads((root/'passport.json').read_text())
    hyp=json.loads((ROOT/'docs/hypotheses_before_run.json').read_text())
    assert hyp['registered_at'] < p['started_at'] <= r['completed_at']
    assert r['variant']==13 and r['context']==3 and r['pair']==['turbo','lcm']
    assert p['threads']==4 and p['dtype']=='float32' and p['device']=='cpu'
    assert p['configs']['turbo']['steps']==1 and p['configs']['turbo']['guidance']==0
    assert p['configs']['lcm']['steps']==4 and p['configs']['lcm']['guidance']==8
    for name,version in dict(python='3.11.11',torch='2.4.1+cpu',diffusers='0.32.2',transformers='4.57.6',numpy='2.1.3',scipy='1.15.2',pillow='11.1.0').items():
        assert p[name]==version
    for model,repo,rev,scheduler in [
        ('turbo','stabilityai/sd-turbo','b261bac6fd2cf515557d5d0707481eafa0485ec2','EulerDiscreteScheduler'),
        ('lcm','SimianLuo/LCM_Dreamshaper_v7','a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5','LCMScheduler')]:
        assert p['configs'][model]['repo']==repo and p['configs'][model]['revision']==rev
        assert p['configs'][model]['size']==512 and p['schedulers'][model]['class']==scheduler
    assert p['clip_revision']=='32bd64288804d66eefd0ccbe215aa642df71cc41'
    assert digest(ROOT/'docs/hypotheses_before_run.json')==p['hypotheses_sha256']
    assert digest(ROOT/'src/original/lab10_lib.py')==p['library_sha256']
    assert digest(ROOT/'src/run_variant13.py')==p['runner_sha256']
    for model in r['pair']:
        rows=r['per_seed'][model]; assert [x['seed'] for x in rows]==list(range(101,106))
        for row in rows:
            path=root/row['image']; assert digest(path)==row['sha256']
            with Image.open(path) as img:
                assert img.size==(512,512); e,c=pixels(img)
            near(e,row['edge']); near(c,row['contrast']); assert row['seconds']>0
        repeat=r['repeats'][model]
        assert digest(root/repeat['image'])==repeat['sha256']
        assert repeat['matches_primary']==(repeat['sha256']==rows[0]['sha256'])
    for row in r['rows']:
        m,avg_a,s_a,avg_b,delta,ratio,flag=row
        a=[x[m] for x in r['per_seed']['turbo']]; b=[x[m] for x in r['per_seed']['lcm']]
        av=statistics.fmean(a); bv=statistics.fmean(b); sd=statistics.stdev(a)
        for x,y in [(avg_a,av),(s_a,sd),(avg_b,bv),(delta,bv-av),(ratio,abs(bv-av)/sd)]:near(x,y)
        assert flag==(abs(bv-av)>2*sd)
    journal=[json.loads(x) for x in (root/'measurements.jsonl').read_text().splitlines() if x.strip()]
    assert journal==[dict(model=model,**row) for model in r['pair'] for row in r['per_seed'][model]]
    key=json.loads((root/'blind_key.json').read_text())
    import random
    rnd=random.Random(13)
    for seed in range(101,106):
        left,right=('turbo','lcm') if rnd.random()<.5 else ('lcm','turbo')
        assert key[str(seed)]==left
        with Image.open(root/f'blind_{seed}.png') as im:
            assert im.size==(780,384)
            for model,x in [(left,0),(right,396)]:
                with Image.open(root/f'{model}_s{seed}.jpg') as jpg:
                    assert np.array_equal(np.asarray(im.crop((x,0,x+384,384))),np.asarray(jpg.convert('RGB').resize((384,384))))
    assessment=ROOT/'artifacts/visual_assessment.json'
    if args.require_blind and not assessment.exists():
        raise SystemExit('NUMERICAL PASS; BLOCKED: blinded choices have not been recorded')
    if assessment.exists():
        v=json.loads(assessment.read_text())
        original=json.loads((ROOT/'artifacts/choices_before_key.json').read_text())
        assert original['assessed_at']==v['assessed_at'] and original['observer']==v['observer']
        assert original['choices']==[{k:x[k] for k in ('seed','side','reason')} for x in v['choices']]
        assert v['assessed_at'] < v['key_opened_at']
        assert [x['seed'] for x in v['choices']]==list(range(101,106))
        key=json.loads((root/'blind_key.json').read_text())
        wins=0
        for x in v['choices']:
            assert x['side'] in ('left','right') and x['reason'].strip()
            assert x['left_model']==key[str(x['seed'])]
            chosen=key[str(x['seed'])] if x['side']=='left' else ('lcm' if key[str(x['seed'])]=='turbo' else 'turbo')
            assert chosen==x['chosen_model']; wins+=chosen=='lcm'
        assert wins==v['wins_B']
    print('PASS: 10 primary PNG, 2 repeat PNG, 5 summaries, pixel metrics, 5 blind sheets, versions/configs and source hashes')

if __name__=='__main__':main()
