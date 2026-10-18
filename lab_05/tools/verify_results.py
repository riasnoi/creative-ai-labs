"""Независимый пересчёт метрик по сохранённым RGB-пикселям (без PyTorch/GPU)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def verify(root):
    out=root/'artifacts'
    metrics=json.loads((out/'metrics.json').read_text())
    images={}
    for name in ['A','B','error_first','error_second','corrected']:
        d=out/name; b=(d/'result.png').read_bytes()
        m=json.loads((d/'manifest.json').read_text())
        assert hashlib.sha256(b).hexdigest()==m['sha256'],name
        assert (m['variant'],m['seed'],m['steps'],m['guidance_scale'],m['size'])==(13,1013,1,0.0,512)
        assert m['model_id']=='stabilityai/sd-turbo'
        assert m['revision']=='b261bac6fd2cf515557d5d0707481eafa0485ec2'
        assert m['generator_device']=='cpu' and m['dtype']=='float16'
        image=Image.open(d/'result.png'); assert image.mode=='RGB' and image.size==(512,512)
        images[name]=np.asarray(image).astype(np.float64)
    checks={}
    for key,a,b in [('repeat','A','B'),('shared_generator','error_first','error_second'),('error_first_vs_A','A','error_first'),('corrected_vs_A','A','corrected')]:
        delta=images[a]-images[b]
        mse=float(np.mean(delta**2)); psnr='+Infinity' if mse==0 else 10*math.log10(255**2/mse)
        actual=dict(pixel_equal=bool(np.array_equal(images[a],images[b])),max_diff=int(np.max(abs(delta))),mse=mse,psnr=psnr,
            verdict='identical' if mse==0 else ('close' if psnr>=40 else 'different'))
        for k,v in actual.items():
            stored=metrics[key][k]
            assert math.isclose(v,stored,rel_tol=1e-12,abs_tol=1e-12) if isinstance(v,float) else v==stored,(key,k,v,stored)
        checks[key]=actual
    plan=json.loads((out/'expectations_before_run.json').read_text())
    assert plan['tolerance_db']==40
    assert all(plan['recorded_at']<json.loads((out/n/'manifest.json').read_text())['created_at']<=metrics['finished_at'] for n in images)
    summary=metrics['variant_summary']
    assert all(summary[k]==metrics['repeat'][k] for k in ['sha_a','sha_b','pixel_equal','max_diff','psnr','verdict'])
    assert len(set(metrics['generator_state_sha256']))==3
    return dict(status='PASS',method='Пересчёт из PNG с float64; без генерации модели',checks=checks)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--output',type=Path)
    args=p.parse_args(); r=verify(args.root)
    data=json.dumps(r,ensure_ascii=False,indent=2)
    if args.output: args.output.write_text(data,encoding='utf-8')
    print(data)
