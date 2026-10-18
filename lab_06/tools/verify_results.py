"""Независимая проверка PNG и агрегатов без GPU и без учебных библиотек."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/'artifacts'/name).read_text())
def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(img):
    a=np.asarray(img,dtype=np.float64)/255.0
    y=.299*a[:,:,0]+.587*a[:,:,1]+.114*a[:,:,2]
    # np.pad symmetric соответствует scipy ndimage mode='reflect' для ядра 3×3.
    p=np.pad(y,1,mode='symmetric')
    gx=(p[:-2,2:]+2*p[1:-1,2:]+p[2:,2:])-(p[:-2,:-2]+2*p[1:-1,:-2]+p[2:,:-2])
    gy=(p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:])-(p[:-2,:-2]+2*p[:-2,1:-1]+p[:-2,2:])
    return float(np.hypot(gx,gy).mean()/8),float(y.std())
def main():
    plan=read('expectations_before_run.json'); raw=read('raw_metrics.json'); summary=read('summary.json')
    assert plan['variant']==13 and plan['seeds']==list(range(101,106))
    assert [k for k in plan['A'] if plan['A'][k]!=plan['B'][k]]==['steps']
    assert plan['A']['steps']==1 and plan['B']['steps']==4
    assert len(raw)==10 and {(x['mode'],x['seed']) for x in raw}=={(m,s) for m in ['A','B'] for s in range(101,106)}
    for x in raw:
        p=ROOT/'artifacts'/x['image']; assert hashfile(p)==x['sha256']
        im=Image.open(p); assert im.mode=='RGB' and im.size==(512,512)
        e,c=metrics(im)
        assert np.isclose(e,x['metrics']['edge'],atol=1e-12,rtol=0)
        assert np.isclose(c,x['metrics']['contrast'],atol=1e-12,rtol=0)
        assert x['settings']==plan[x['mode']] and plan['recorded_at']<x['recorded_at']
    for m,a,s,b,d,r,sig in summary['rows']:
        av=np.array([x['metrics'][m] for x in raw if x['mode']=='A']);bv=np.array([x['metrics'][m] for x in raw if x['mode']=='B'])
        delta=bv.mean()-av.mean(); std=av.std(ddof=1)
        assert np.allclose([a,s,b,d,r],[av.mean(),std,bv.mean(),delta,abs(delta)/std],rtol=1e-12,atol=1e-12)
        assert sig==bool(abs(delta)>2*std)
    sheet=Image.open(ROOT/'artifacts/comparison.png'); assert sheet.size==(768,384)
    for i,m in enumerate(['A','B']):
        expected=Image.open(ROOT/f'artifacts/{m}/seed_101.png').resize((384,384))
        assert np.array_equal(np.array(sheet.crop((384*i,0,384*(i+1),384))),np.array(expected))
    err=read('guidance_error.json')
    assert [r['cfg_active'] for r in err['runs']]==[False,False,True]
    for r in err['runs']: assert hashfile(ROOT/'artifacts'/r['image'])==r['sha256']
    assert err['equal_sha_0_1'] and err['runs'][0]['sha256']==err['runs'][1]['sha256']
    a=np.array(Image.open(ROOT/'artifacts/guidance_0.png'),dtype=float);b=np.array(Image.open(ROOT/'artifacts/guidance_3.png'),dtype=float)
    psnr=10*np.log10(255**2/np.mean((a-b)**2));assert np.isclose(psnr,err['psnr_0_3'],atol=1e-12)
    assert all(x['matches_saved'] for x in read('repeat_check.json'))
    passport=read('passport.json'); assert passport['summary']==summary and passport['error']==err
    assert passport['started_at']==plan['recorded_at'] and passport['finished_at']>max(x['recorded_at'] for x in raw)
    env=read('environment.json')
    for name,h in env['libraries'].items(): assert hashfile(ROOT/'src/original'/name)==h
    print('PASS: 10 PNG, Sobel/contrast, 4 aggregates, comparison, CFG/SHA/PSNR, repeat, chronology, source hashes')
if __name__=='__main__':main()
