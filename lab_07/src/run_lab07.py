"""ЛР07, вариант 13: первичные измерения и диагностика; запуск в новой папке."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import contextlib
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import numpy as np
import torch
from PIL import Image
import lab07_lib as L

OUT = Path('artifacts')
OUT.mkdir(exist_ok=False)
now = lambda: datetime.now(timezone(timedelta(hours=3))).isoformat()
def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = dict(recorded_at=now(), variant=13, kind='disc_noise', factor='latent_8_to_2', seeds=list(range(101,106)),
    hypothesis='Уменьшение d с 8 до 2 снизит PSNR более чем на 1 дБ; изменение разрыва останется в пределах 0,1 дБ. Число параметров уменьшится, существенное ускорение не ожидается. Непредсказуемый шум датчика не будет воспроизводиться в обоих режимах.',
    tolerances=dict(psnr_absolute=1.0, gap_absolute=0.1, seconds_relative=0.2),
    rules='Граница допуска нейтральна; PSNR выше — лучше; разрыв и время ниже — лучше. Параметры оцениваются точно без 2s и без допуска; экономия полезна при приемлемом PSNR.',
    diagnostic=dict(d=[1,2,3,4,8], seeds=[101,102,103], reuse='Основные A/B повторно используются при d=8/2.'),
    demonstration='Сначала исходный lab07_demo.py выполняется целиком для N=1; далее отдельный опыт N=13.')
save('expectations_before_run.json',plan)
print('ПЛАН ДО ЗАПУСКА',json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
assert torch.cuda.is_available(), 'Нужна CUDA/T4'
env=dict(recorded_at=now(),python=sys.version, platform=platform.platform(), torch=torch.__version__,numpy=np.__version__,pillow=__import__('PIL').__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
 nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),pip_list=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True),
 library_sha256=sha('lab07_lib.py'),demo_sha256=sha('lab07_demo.py'))
save('environment.json',env)
class Tee(io.StringIO):
    def write(self,s):
        sys.__stdout__.write(s);sys.__stdout__.flush();return super().write(s)
log=Tee()
with contextlib.redirect_stdout(log):
    print('НАЧАЛО',now())
    print('ДЕМОНСТРАЦИЯ N=1 — исходный код без изменений',flush=True)
    exec(compile(Path('lab07_demo.py').read_text(), 'lab07_demo.py','exec'),{'__name__':'__main__'})
    shutil.move('comparison.png',OUT/'comparison_demo01.png')
    print('ОСНОВНОЙ ОПЫТ N=13',now(),flush=True)
    a,b=L.settings(L.FACTORS[0]);print('A',a,'B',b)
    assert [k for k in a if a[k]!=b[k]]==['d']
    records=[];arrays={};diagnostic=[];original_run=L.run
    def tracked(kind,cfg,seed):
        started=now();metrics,model=original_run(kind,cfg,seed)
        if seed in range(101,106):
            mode='A' if cfg['d']==8 else 'B'
            Xtr,Xte,_,_=L.dataset(kind)
            r=L.reconstruct(model,Xte);rtr=L.reconstruct(model,Xtr[:1024])
            arrays[f'{mode}_{seed}_test_mse']=((r-Xte)**2).flatten(1).mean(1).cpu().numpy()
            arrays[f'{mode}_{seed}_train_mse']=((rtr-Xtr[:1024])**2).flatten(1).mean(1).cpu().numpy()
            records.append(dict(mode=mode,seed=seed,started_at=started,finished_at=now(),metrics=metrics))
            if seed==101:
                arrays[f'{mode}_reconstruction']=r[:8].cpu().numpy()
                torch.save(model.state_dict(), OUT/f'model_{mode}_seed101.pt')
            print(mode,seed,metrics,flush=True)
        return metrics,model
    L.run=tracked
    result=L.variant_summary(13)
    L.run=original_run
    # JSON имеет null вместо нечислового отношения при s=0.
    result['rows']=[list(row[:5])+[row[5] if np.isfinite(row[5]) else None,row[6]] for row in result['rows']]
    save('summary.json',result);save('raw_metrics.json',records)
    Xtr,Xte,Ftr,Fte=L.dataset('disc_noise')
    mean=Xtr.mean(0,keepdim=True).expand_as(Xte)
    baseline=L.psnr_images(mean,Xte)
    arrays['mean_test_mse']=((mean-Xte)**2).flatten(1).mean(1).cpu().numpy()
    arrays['original']=Xte[:8].cpu().numpy()
    arrays['factors_test_first8']=Fte[:8].cpu().numpy()
    # Чистые сцены тех же факторов нужны только для дополнительной проверки шума.
    clean=np.stack([L.render('disc',f) for f in Fte[:8].cpu().numpy()])
    arrays['clean_first8']=clean
    save('baseline.json',dict(psnr=baseline, target='noisy test images',n_test=1024))
    for d in (1,2,3,4,8):
        vals=[]
        for seed in (101,102,103):
            if d in (2,8):
                mode='A' if d==8 else 'B'
                v=next(x['metrics']['psnr'] for x in records if x['seed']==seed and x['mode']==mode)
                origin='main series'
            else:
                v=original_run('disc_noise',dict(L.BASE,d=d),seed)[0]['psnr'];origin='new training'
            vals.append(v)
        diagnostic.append(dict(d=d,seeds=[101,102,103],psnr=vals,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),origin=origin))
        print('ДИАГНОСТИКА',diagnostic[-1],flush=True)
    save('diagnostic.json',diagnostic)
    np.savez_compressed(OUT/'measurements.npz',**arrays)
    grid=np.concatenate([np.concatenate(list(arr.transpose(0,2,3,1)),axis=1) for arr in [arrays['original'],arrays['A_reconstruction'],arrays['B_reconstruction']]],axis=0)
    Image.fromarray((grid*255).round().astype('uint8')).resize((1024,384),Image.Resampling.NEAREST).save(OUT/'comparison.png')
    noise_check={}
    for mode in ['original','A_reconstruction','B_reconstruction']:
        residual=arrays[mode]-clean
        # Фон за пределами любого диска: первые четыре строки.
        noise_check[mode]=dict(background_residual_std=float(residual[:,:,:4,:].std()),clean_psnr_first8=float(np.mean(-10*np.log10(np.maximum((residual**2).mean(axis=(1,2,3)),1e-10)))))
    save('noise_check.json',noise_check)
    repeat=[]
    for mode,cfg in [('A',a),('B',b)]:
        m,_=original_run('disc_noise',cfg,101)
        first=next(x['metrics']['psnr'] for x in records if x['mode']==mode and x['seed']==101)
        repeat.append(dict(mode=mode,seed=101,psnr_initial=first,psnr_repeat=m['psnr'],difference=m['psnr']-first))
    save('repeat_check.json',repeat)
    save('passport.json',dict(variant=13,kind='disc_noise',configs=dict(A=a,B=b),data_seed=7,n_train=4096,n_test=1024,train_eval=1024,epochs=20,batch=128,optimizer='Adam',lr=0.002,updates=640,seeds=list(range(101,106)),dtype='float32',target='noisy images, sigma=0.05; no extra training/eval corruption',baseline_psnr=baseline,plan_time=plan['recorded_at'],main_started_at=records[0]['started_at'],finished_at=now(),source_hashes={n:sha(n) for n in ['lab07_lib.py','lab07_demo.py']},limitations='CUDA convolution algorithms are not forced deterministic; runtime varies. Diagnostic reuses main-series d=2/8 for seeds 101..103.'))
    print('ИТОГ',result,'ЭТАЛОН',baseline,'ПОВТОР',repeat,now(),flush=True)
(OUT/'run.log').write_text(log.getvalue(),encoding='utf-8')
print('ГОТОВО: artifacts/passport.json')
