"""Независимый пересчёт ЛР08 без PyTorch и без повторного обучения."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image


def load(root, name):
    return json.loads((root/'artifacts'/name).read_text(encoding='utf-8'))


def psnr(mse):
    return float(np.mean(-10*np.log10(np.maximum(np.asarray(mse,dtype=np.float64),1e-10))))


def generated_dataset():
    """Формулы disc_noise реализованы отдельно от учебной библиотеки."""
    rng=np.random.default_rng(7)
    factors=rng.random((5120,3))
    yy,xx=np.mgrid[0:32,0:32]/31
    clean=np.empty((5120,3,32,32),dtype=np.float32)
    for i,(fx,fy,fr) in enumerate(factors):
        clean[i]=(0.10+0.25*(xx+yy)/2)[None]
        mask=(xx-(0.25+0.5*fx))**2+(yy-(0.25+0.5*fy))**2<=(0.12+0.10*fr)**2
        for channel,color in enumerate((1.0,0.76,0.29)):
            clean[i,channel][mask]=color
    noisy=np.clip(clean+0.05*rng.standard_normal(clean.shape).astype(np.float32),0,1)
    return factors.astype(np.float32),clean,noisy


def verify(root):
    raw=load(root,'raw_metrics.json')
    summary=load(root,'summary.json')
    passport=load(root,'passport.json')
    plan=load(root,'expectations_before_run.json')
    times=load(root,'training_times.json')
    diagnostic=load(root,'diagnostic.json')
    assert passport['variant']==summary['variant']==plan['variant']==13
    assert plan['recorded_at'] < min(x['started_at'] for x in times)
    assert len(raw)==10 and {(x['mode'],x['seed']) for x in raw}=={(m,s) for m in ['A','B'] for s in range(101,106)}
    assert [k for k in passport['configs']['A'] if passport['configs']['A'][k]!=passport['configs']['B'][k]]==['beta']
    max_errors={m:0. for m in ('recon','interp','sample','r2')}
    factor_r2=[]
    for row in raw:
        mode,seed=row['mode'],row['seed']
        with np.load(root/'artifacts'/f'measurements_{mode}_{seed}.npz') as z:
            assert z['recon_mse'].shape==(1024,)
            assert z['interp_mse'].shape==(3,200)
            assert z['sample_mse'].shape==(256,)
            # Гребневая регрессия совпадает с определением, float32 — как в библиотеке.
            design=np.column_stack([z['mu_train'],np.ones(4096,dtype=np.float32)]).astype(np.float64)
            weights=np.linalg.solve(design.T@design+1e-6*np.eye(9),design.T@z['factors_train'].astype(np.float64)).astype(np.float32)
            prediction=np.column_stack([z['mu_test'],np.ones(1024,dtype=np.float32)])@weights
            f=z['factors_test']
            r2=1-((prediction-f)**2).sum(0)/((f-f.mean(0))**2).sum(0)
            computed=dict(recon=psnr(z['recon_mse']),interp=psnr(z['interp_mse']),sample=psnr(z['sample_mse']),r2=float(r2.mean()))
            for metric,value in computed.items():
                err=abs(value-row['metrics'][metric]); max_errors[metric]=max(max_errors[metric],err)
                assert err < (5e-5 if metric!='r2' else 3e-5),(mode,seed,metric,err)
            assert abs(psnr(z['sample_direct_mse'])-computed['sample'])<0.003
            kl=(-0.5*(1+z['lv_test']-z['mu_test']**2-np.exp(z['lv_test']))).mean(0)
            diag=next(d for d in diagnostic if d['mode']==mode and d['seed']==seed)
            np.testing.assert_allclose(kl,diag['kl_axis'],atol=3e-6)
            np.testing.assert_allclose(z['mu_test'].std(0,ddof=1),diag['mu_std'],atol=3e-6)
            assert int((kl>0.05).sum())==diag['active_axes']
            factor_r2.append(dict(mode=mode,seed=seed,r2_factors=r2.tolist()))
    for metric,ma,sd,mb,delta,ratio,significant in summary['rows']:
        a=np.array([x['metrics'][metric] for x in raw if x['mode']=='A'])
        b=np.array([x['metrics'][metric] for x in raw if x['mode']=='B'])
        expected=[a.mean(),a.std(ddof=1),b.mean(),b.mean()-a.mean(),abs(b.mean()-a.mean())/a.std(ddof=1)]
        np.testing.assert_allclose(expected,[ma,sd,mb,delta,ratio],atol=1e-10)
        assert bool(abs(delta)>2*sd)==significant
    for item in load(root,'classification.json'):
        delta=item['delta'];tol=plan['tolerances'][item['metric']]
        expected='улучшение' if delta>tol else 'ухудшение' if delta < -tol else 'нейтрально'
        assert item['classification']==expected
    factors,clean,noisy=generated_dataset()
    with np.load(root/'artifacts/dataset_evidence.npz') as z:
        np.testing.assert_array_equal(z['factors_train'],factors[:4096])
        np.testing.assert_array_equal(z['factors_test'],factors[4096:])
        np.testing.assert_array_equal(z['original'],noisy[4096:4104])
        # PyTorch и NumPy могут иметь разный порядок суммирования float32.
        np.testing.assert_allclose(z['mean_train'],noisy[:4096].mean(0),atol=2e-6)
        baseline_mse=((noisy[:4096].mean(0)-noisy[4096:])**2).mean(axis=(1,2,3))
        assert abs(psnr(baseline_mse)-load(root,'baseline.json')['psnr'])<1e-4
    images={}
    for mode in ['A','B','AE']:
        images[mode]=dict(np.load(root/'artifacts'/f'measurements_{mode}_101.npz'))
    rows=[images['AE']['visual_samples'],images['A']['visual_samples'],images['A']['path'],images['B']['path']]
    expected_grid=np.concatenate([np.concatenate(list(x.transpose(0,2,3,1)),axis=1) for x in rows],axis=0)
    expected_img=Image.fromarray((expected_grid*255).round().astype('uint8')).resize((1024,512),Image.Resampling.NEAREST)
    np.testing.assert_array_equal(np.array(expected_img),np.array(Image.open(root/'artifacts/comparison.png')))
    noise={}
    clean8=clean[4096:4104]
    for mode in ['A','B']:
        residual=images[mode]['reconstruction']-clean8
        noise[mode]=dict(background_residual_std=float(residual[:,:,:4,:].std()),clean_psnr_first8=psnr((residual**2).mean(axis=(1,2,3))))
    noise['input_background_residual_std']=float((noisy[4096:4104]-clean8)[:,:,:4,:].std())
    report=dict(status='passed',raw_records=10,metrics_max_abs_error=max_errors,
                regenerated_scenes=5120,first8_inputs_exact=True,comparison_png_exact=True,
                baseline_psnr=psnr(baseline_mse),r2_by_factor=factor_r2,noise_check=noise,
                scope='No training repeated by this check; seed101 A/B retraining is stored separately in repeat_check.json.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=verify(root)
    if args.write:args.write.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
