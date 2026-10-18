"""Независимая проверка первичных метрик; CUDA и обучение не требуются."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
A=R/'artifacts'
def read(n):return json.loads((A/n).read_text())
def check():
    raw=read('raw_metrics.json');s=read('summary.json');pas=read('passport.json');plan=read('expectations_before_run.json')
    z=np.load(A/'measurements.npz',allow_pickle=False)
    assert len(raw)==10 and {(x['mode'],x['seed']) for x in raw}=={(m,k) for m in ['A','B'] for k in range(101,106)}
    assert plan['recorded_at']<raw[0]['started_at']<pas['finished_at']
    assert pas['configs']['A']['d']==8 and pas['configs']['B']['d']==2
    assert [k for k in pas['configs']['A'] if pas['configs']['A'][k]!=pas['configs']['B'][k]]==['d']
    errors=[]
    for row in raw:
        m,k=row['mode'],row['seed'];test=z[f'{m}_{k}_test_mse'];train=z[f'{m}_{k}_train_mse']
        assert test.shape==train.shape==(1024,) and np.all(test>0)
        psnr=float((-10*np.log10(test.astype('float64'))).mean());ptr=float((-10*np.log10(train.astype('float64'))).mean())
        errors.extend([abs(psnr-row['metrics']['psnr']),abs(ptr-psnr-row['metrics']['gap'])])
        assert abs(psnr-row['metrics']['psnr'])<1e-5
        assert abs(ptr-psnr-row['metrics']['gap'])<1e-5
        assert row['metrics']['params']==(101.035 if m=='A' else 88.741)
    for name,a,sd,b,delta,q,sig in s['rows']:
        av=[r['metrics'][name] for r in raw if r['mode']=='A'];bv=[r['metrics'][name] for r in raw if r['mode']=='B']
        expected=[np.mean(av),np.std(av,ddof=1),np.mean(bv),np.mean(bv)-np.mean(av)]
        assert np.allclose([a,sd,b,delta],expected,atol=1e-12,rtol=0)
        assert sig==bool(abs(delta)>2*sd)
        assert (q is None and sd==0) or abs(q-abs(delta)/sd)<1e-10
    # Независимо воссоздать генератор набора из описания, включая шум всех сцен.
    yy,xx=np.mgrid[0:32,0:32]/31
    rng=np.random.default_rng(7);f=rng.random((5120,3));x=np.zeros((5120,3,32,32),np.float32)
    x[:]=(0.10+0.25*(xx+yy)/2)[None,None]
    for i,fi in enumerate(f):
        mask=(xx-(.25+.5*fi[0]))**2+(yy-(.25+.5*fi[1]))**2<=(.12+.10*fi[2])**2
        for k,c in enumerate((1.,.76,.29)):x[i,k][mask]=c
    clean=x[4096:4104].copy()
    x=np.clip(x+.05*rng.standard_normal(x.shape).astype(np.float32),0,1)
    assert np.array_equal(x[4096:4104],z['original'])
    assert np.max(abs(clean-z['clean_first8']))<1e-6
    baseline_mse=((x[4096:]-x[:4096].mean(0,keepdims=True))**2).mean(axis=(1,2,3))
    assert np.max(abs(baseline_mse-z['mean_test_mse']))<1e-6
    baseline=float((-10*np.log10(baseline_mse.astype('float64'))).mean())
    assert abs(baseline-read('baseline.json')['psnr'])<1e-4
    grid=np.concatenate([np.concatenate(list(z[n].transpose(0,2,3,1)),axis=1) for n in ['original','A_reconstruction','B_reconstruction']],axis=0)
    expected=Image.fromarray((grid*255).round().astype('uint8')).resize((1024,384),Image.Resampling.NEAREST)
    assert np.array_equal(np.asarray(expected),np.asarray(Image.open(A/'comparison.png')))
    diag=read('diagnostic.json');assert [v['d'] for v in diag]==[1,2,3,4,8]
    for v in diag:
        assert abs(np.mean(v['psnr'])-v['mean'])<1e-12
        assert abs(np.std(v['psnr'],ddof=1)-v['sd'])<1e-12
    for name,h in pas['source_hashes'].items():assert hashlib.sha256((R/'src/original'/name).read_bytes()).hexdigest()==h
    result=dict(max_psnr_recalculation_error=max(errors),baseline_numpy=baseline,baseline_error=abs(baseline-read('baseline.json')['psnr']),primary_values=10,images_per_metric=1024,comparison_exact=True,data_generation_exact=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':check()
