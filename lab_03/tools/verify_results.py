"""Независимая проверка метрик по точным Float64, выгруженным из Engee."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np

def metrics(a):
    # Библиотека относит границу k/32 к левому интервалу, через ceil.
    bins = np.clip(np.ceil(a * 32).astype(int), 1, 32) - 1
    counts = np.bincount(bins, minlength=32)
    p = counts[counts > 0] / a.size
    return {"mean": float(a.mean()), "contrast": float(a.std(ddof=1)),
            "entropy": float(-(p * np.log2(p)).sum())}, counts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    r = tomllib.loads((args.directory / "results.toml").read_text())
    assert r["variant"] == 13 and r["seeds"] == list(range(101,106))
    dtype = '<f8' if r['environment']['endian']=='little' else '>f8'
    before, after, psnrs = [], [], []
    for item in r["per_seed"]:
        seed = item["seed"]
        raw = (args.directory / f"before_{seed}.f64").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        a = np.frombuffer(raw,dtype=dtype)
        b = np.fromfile(args.directory / f"after_{seed}.f64",dtype=dtype)
        assert a.size == b.size == 128**2
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert a.min() >= 0 and a.max() <= 1
        assert np.array_equal(b, np.rint(a*15)/15)
        ma, ca = metrics(a); mb, cb = metrics(b)
        for label, calc in [('before',ma),('after',mb)]:
            for key,value in calc.items():
                assert np.isclose(value,item[label][key],atol=1e-12,rtol=1e-12)
        mse = float(((a-b)**2).mean()); psnr = float(10*np.log10(1/mse))
        assert np.isclose(mse,item['mse'],atol=1e-15,rtol=1e-12)
        assert np.isclose(psnr,item['psnr_db'],atol=1e-12,rtol=1e-12)
        before.append(ma); after.append(mb); psnrs.append(psnr)
        if seed==101:
            assert item['sha256']==r['luma']['sha256_first']==r['luma']['sha256_repeat']
            table=np.genfromtxt(args.directory/'histogram.csv',delimiter=',',skip_header=1)
            assert np.array_equal(table[:,3],ca) and np.array_equal(table[:,4],cb)
            assert len(np.unique(b))==r['luma']['occupied_levels_after']
    checked=[]
    for row in r['rows']:
        name=row['metric']; base=np.mean([x[name] for x in before]); post=np.mean([x[name] for x in after])
        s=np.std([x[name] for x in before],ddof=1); delta=post-base
        values={'base':base,'after':post,'s':s,'delta':delta,'ratio':abs(delta)/s,'relative_percent':100*abs(delta)/abs(base)}
        for key,value in values.items():
            assert np.isclose(value,row[key],atol=1e-11,rtol=1e-9),(name,key,value,row[key])
        assert row['statistical']==bool(abs(delta)>2*s)
        assert row['practical']==bool(values['relative_percent']>5)
        checked.append(dict(metric=name,**{k:float(v) for k,v in values.items()}))
    assert np.isclose(np.mean(psnrs),r['psnr_mean_db'],atol=1e-12,rtol=1e-12)
    assert r['luma']['maxdiff']<=1e-7 and r['luma']['equal_at_tolerance']
    result={'status':'passed','arrays_checked':10,'hashes_checked':5,'histogram_bins':32,'metrics':checked,'mean_psnr_db':float(np.mean(psnrs))}
    if args.output: args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
