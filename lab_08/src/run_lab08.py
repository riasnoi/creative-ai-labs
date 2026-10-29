"""ЛР08, вариант 13. Запускать в новой папке рядом с учебными файлами."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import contextlib
import hashlib
import json
import platform
import runpy
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from PIL import Image
import lab07_lib as L
import lab08_lib as V

OUT = Path('artifacts')
NOW = lambda: datetime.now(timezone(timedelta(hours=3))).isoformat()
METRICS = ('recon', 'interp', 'sample', 'r2')


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def sheet(rows, filename):
    strips = [np.concatenate(list(x.transpose(0, 2, 3, 1)), axis=1) for x in rows]
    grid = (np.concatenate(strips, axis=0) * 255).round().astype('uint8')
    Image.fromarray(grid).resize((1024, len(rows) * 128), Image.Resampling.NEAREST).save(OUT / filename)


@torch.no_grad()
def primary(model, kind, cfg, seed):
    """Сохранение ошибок до усреднения, латентов и поосевой диагностики."""
    Xtr, Xte, Ftr, Fte = L.dataset(kind)
    model.eval()
    mu_tr = model.encode(Xtr)
    mu, lv = model.encode_dist(Xte)
    reconstruction = model.decode(mu)
    ia, ib, targets = V.interp_targets(kind)
    interp_mse = []
    for t, target in targets.items():
        z = (1 - t) * mu[ia] + t * mu[ib]
        interp_mse.append(((model.decode(z) - target) ** 2).flatten(1).mean(1).cpu().numpy())
    g = torch.Generator(device='cpu').manual_seed(1000 + seed)
    z = torch.randn((V.N_SAMPLES, cfg['d']), generator=g).to(L.DEVICE)
    samples = model.decode(z)
    distances = torch.cdist(samples.flatten(1), Xte.flatten(1)).pow(2) / (3 * 32 * 32)
    # Прямой пересчёт расстояний до выбранного соседа не использует cdist.
    nearest = distances.argmin(1)
    sample_direct_mse = ((samples - Xte[nearest]) ** 2).flatten(1).mean(1)
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
    clean = np.stack([L.render('disc', f) for f in Fte[:8].cpu().numpy()])
    t = torch.linspace(0, 1, 8, device=L.DEVICE).unsqueeze(1)
    path = model.decode((1 - t) * mu[0:1] + t * mu[2:3])
    sg = torch.Generator(device='cpu').manual_seed(5)
    visual_z = torch.randn((8, cfg['d']), generator=sg).to(L.DEVICE)
    visual_samples = model.decode(visual_z)
    arrays = dict(recon_mse=((reconstruction-Xte)**2).flatten(1).mean(1).cpu().numpy(),
                  interp_mse=np.stack(interp_mse), sample_mse=distances.min(1).values.cpu().numpy(),
                  sample_direct_mse=sample_direct_mse.cpu().numpy(), nearest_index=nearest.cpu().numpy(),
                  mu_train=mu_tr.cpu().numpy(), mu_test=mu.cpu().numpy(), lv_test=lv.cpu().numpy(),
                  reconstruction=reconstruction[:8].cpu().numpy(), path=path.cpu().numpy(),
                  visual_samples=visual_samples.cpu().numpy(), original=Xte[:8].cpu().numpy(),
                  clean=clean, factors_train=Ftr.cpu().numpy(), factors_test=Fte.cpu().numpy())
    diag = dict(mu_std=mu.std(0).cpu().tolist(), kl_axis=kl.cpu().tolist(),
                active_axes=int((kl > 0.05).sum()) if cfg['beta'] > 0 else None,
                threshold_kl_nat=0.05,
                background_residual_std=float((reconstruction[:8].cpu().numpy()-clean)[:, :, :4, :].std()),
                sample_background_std=float(visual_samples.cpu().numpy()[:, :, :4, :].std(axis=0).mean()))
    return arrays, diag


def main():
    OUT.mkdir(exist_ok=False)
    plan = dict(recorded_at=NOW(), variant=13, kind='disc_noise', factor='beta_1_to_8',
                seeds=list(L.SEEDS),
                hypothesis='Рост beta с 1 до 8 ухудшит реконструкцию более чем на 1 дБ и R2 более чем на 0,05; число активных осей снизится. Предполагается улучшение интерполяции и сэмплов более чем на 1 дБ за счёт регуляризации; это предположение проверяется независимо. Сэмплы не должны воспроизводить шум датчика.',
                tolerances=dict(recon=1.0, interp=1.0, sample=1.0, r2=0.05),
                tolerance_reason='1 дБ соответствует изменению MSE примерно на 26%; 0,05 R2 — пяти процентным пунктам объяснённой дисперсии.',
                decision_rule='Все метрики выше — лучше; внутри допуска и на границе — нейтрально. Критерий |delta|>2s_A — учебная мера относительно разброса seed, не формальный статистический тест.',
                diagnostic_seed=101, active_threshold_nat=0.05,
                demonstration='Исходный lab08_demo.py целиком для N=1, затем отдельная серия N=13.',
                repeat='После основной серии повтор A/B seed 101 в отдельном каталоге repeat_check.')
    save('expectations_before_run.json', plan)
    env = dict(recorded_at=NOW(), python=sys.version, platform=platform.platform(),
               torch=torch.__version__, numpy=np.__version__, pillow=__import__('PIL').__version__,
               device=L.DEVICE, cuda_available=torch.cuda.is_available(),
               threads=torch.get_num_threads(),
               source_hashes={n: sha(n) for n in ('lab07_lib.py','lab08_lib.py','lab08_demo.py','run_lab08.py')})
    save('environment.json', env)
    with (OUT/'run.log').open('w', encoding='utf-8') as logfile:
        tee = Tee(sys.stdout, logfile)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print('ПЛАН ДО ЗАПУСКА', json.dumps(plan, ensure_ascii=False), flush=True)
            print('СРЕДА', json.dumps(env, ensure_ascii=False), flush=True)
            timings = []
            original_train = V.train_vae

            def timed_train(model, Xtr, beta, epochs=30, lr=2e-3, seed=0, batch=128, warmup=10):
                started, timer = NOW(), time.perf_counter()
                m = original_train(model, Xtr, beta, epochs, lr, seed, batch, warmup)
                record = dict(kind='disc' if phase[0]=='demo' else 'disc_noise', phase=phase[0],
                              beta=beta, seed=seed, epochs=epochs, warmup=warmup,
                              started_at=started, finished_at=NOW(), seconds=time.perf_counter()-timer)
                timings.append(record)
                save('training_times.json', timings)
                print('ОБУЧЕНИЕ', json.dumps(record, ensure_ascii=False), flush=True)
                return m

            phase = ['demo']
            V.train_vae = timed_train
            print('ДЕМОНСТРАЦИЯ N=1, исходный файл без изменений', NOW(), flush=True)
            runpy.run_path('lab08_demo.py', run_name='__main__')
            shutil.move('comparison.png', OUT/'comparison_demo01.png')
            demo_summary = V.variant_summary(1)
            save('demo_summary.json', demo_summary)
            demo_diag = []
            for beta in (0.0,1.0,8.0):
                cfg = dict(V.BASE, beta=beta)
                m = V.get_model('disc',cfg,101)
                _, diag = primary(m,'disc',cfg,101)
                demo_diag.append(dict(beta=beta,**diag, metrics=V.metrics(m,'disc','lin',1.0,101)))
            save('demo_diagnostic.json',demo_diag)
            # Кэш демонстрации не используется для другого набора данных.
            V._MODELS.clear()
            phase[0]='main'
            print('ОСНОВНОЙ ОПЫТ N=13', NOW(), flush=True)
            a,b=V.settings(V.FACTORS[0])
            assert [k for k in a if a[k]!=b[k]]==['beta']
            print('A:',a,'B:',b,flush=True)
            raw,diagnostics,visuals=[],[],{}
            original_run=V.run

            def tracked(kind,cfg,seed):
                started=NOW()
                metrics=original_run(kind,cfg,seed)
                mode='A' if cfg['beta']==1.0 else 'B'
                record=dict(mode=mode,seed=seed,started_at=started,finished_at=NOW(),metrics=metrics)
                raw.append(record)
                model=V.get_model(kind,cfg,seed)
                arrays,diag=primary(model,kind,cfg,seed)
                np.savez_compressed(OUT/f'measurements_{mode}_{seed}.npz',**arrays)
                diagnostics.append(dict(mode=mode,seed=seed,**diag))
                save('raw_metrics.json',raw)
                save('diagnostic.json',diagnostics)
                if seed==101:
                    torch.save(model.state_dict(),OUT/f'model_{mode}_seed101.pt')
                    visuals[mode]=arrays
                print('ИЗМЕРЕНИЕ',json.dumps(record,ensure_ascii=False),flush=True)
                return metrics

            V.run=tracked
            summary=V.variant_summary(13)
            V.run=original_run
            save('summary.json',summary)
            classifications=[]
            for name,ma,sd,mb,delta,ratio,significant in summary['rows']:
                tol=plan['tolerances'][name]
                label='улучшение' if delta>tol else 'ухудшение' if delta < -tol else 'нейтрально'
                classifications.append(dict(metric=name,tolerance=tol,delta=delta,classification=label,
                                            above_2s=significant))
                print(f'{name:7s} A={ma:.6f} s={sd:.6f} B={mb:.6f} delta={delta:+.6f} |delta|/s={ratio:.3f} 2s={significant} {label}',flush=True)
            save('classification.json',classifications)
            Xtr,Xte,Ftr,Fte=L.dataset('disc_noise')
            baseline=L.psnr_images(Xtr.mean(0,keepdim=True).expand_as(Xte),Xte)
            np.savez_compressed(OUT/'dataset_evidence.npz',factors_train=Ftr.cpu().numpy(),factors_test=Fte.cpu().numpy(),
                                original=Xte[:8].cpu().numpy(),mean_train=Xtr.mean(0).cpu().numpy(),
                                baseline_mse=((Xtr.mean(0,keepdim=True)-Xte)**2).flatten(1).mean(1).cpu().numpy())
            save('baseline.json',dict(psnr=baseline,target='noisy test',n_test=1024))
            ae=V.get_model('disc_noise',dict(V.BASE,beta=0.0),101)
            ae_arrays,ae_diag=primary(ae,'disc_noise',dict(V.BASE,beta=0.0),101)
            save('ae_diagnostic.json',dict(beta=0.0,seed=101,**ae_diag,metrics=V.metrics(ae,'disc_noise','lin',1.0,101)))
            np.savez_compressed(OUT/'measurements_AE_101.npz',**ae_arrays)
            torch.save(ae.state_dict(),OUT/'model_AE_seed101.pt')
            # Четыре строки, как в демонстрации; вторая — VAE режима A.
            sheet([ae_arrays['visual_samples'],visuals['A']['visual_samples'],visuals['A']['path'],visuals['B']['path']], 'comparison.png')
            sheet([visuals['A']['original'],visuals['A']['reconstruction'],visuals['B']['reconstruction'],
                   visuals['A']['visual_samples'],visuals['B']['visual_samples']], 'noise_comparison.png')
            # Линейная и сферическая интерполяции одной модели: дополнительная иллюстрация.
            model=V.get_model('disc_noise',a,101)
            with torch.no_grad():
                za,zb=model.encode(Xte[0:1]),model.encode(Xte[2:3])
                t=torch.linspace(0,1,8,device=L.DEVICE).unsqueeze(1)
                sph=model.decode(V.slerp(za,zb,t)).cpu().numpy()
            sheet([visuals['A']['path'],sph], 'interpolation_methods.png')
            save('interpolation_control.json',dict(seed=101,kind='disc_noise',linear=V.metrics(model,'disc_noise','lin',1.0,101),
                 spherical=V.metrics(model,'disc_noise','slerp',1.0,101),scope='Additional same-model check, outside the main beta series.'))
            phase[0]='repeat'
            repeat_dir=Path('repeat_check')
            repeat_dir.mkdir(exist_ok=False)
            repeats=[]
            for mode,cfg in [('A',a),('B',b)]:
                torch.manual_seed(101)
                m=V.train_vae(V.ConvVAE(8).to(L.DEVICE),Xtr,cfg['beta'],30,seed=101)
                ms=V.metrics(m,'disc_noise','lin',1.0,101)
                first=next(x['metrics'] for x in raw if x['mode']==mode and x['seed']==101)
                repeats.append(dict(mode=mode,seed=101,initial=first,repeat=ms,differences={k:ms[k]-first[k] for k in METRICS}))
            (repeat_dir/'repeat_check.json').write_text(json.dumps(repeats,ensure_ascii=False,indent=2),encoding='utf-8')
            shutil.copy2(repeat_dir/'repeat_check.json',OUT/'repeat_check.json')
            save('passport.json',dict(variant=13,kind='disc_noise',factor='beta_1_to_8',configs=dict(A=a,B=b),
                 data_seed=7,training_seeds=list(L.SEEDS),pairs_seed=11,sampling_seed_rule='1000+training_seed',
                 visual_sampling_seed=5,n_train=4096,n_test=1024,n_pairs=200,n_samples=256,
                 interp_t=[0.25,0.5,0.75],optimizer='Adam',lr=0.002,batch=128,epochs=30,warmup_epochs=10,
                 updates_per_model=960,dtype='float32',noise_sigma=0.05,target_recon_sample='noisy test images',
                 target_interp='clean render of interpolated factors',active_threshold_kl_nat=0.05,
                 plan_recorded_at=plan['recorded_at'],main_started_at=raw[0]['started_at'],finished_at=NOW(),
                 baseline_psnr=baseline,source_hashes=env['source_hashes'],
                 limitations='CPU/GPU kernels and libraries can alter last digits. Five training seeds share one fixed synthetic dataset. Nearest-neighbor PSNR does not assess sample diversity.'))
            print('ИТОГ',json.dumps(summary,ensure_ascii=False),'ЭТАЛОН',baseline,flush=True)
            print('ПОВТОР',json.dumps(repeats,ensure_ascii=False),flush=True)
            print('ГОТОВО',NOW(),flush=True)
    if Path('engee_environment.log').exists():
        (OUT/'engee').mkdir(exist_ok=True)
        shutil.copy2('engee_environment.log', OUT/'engee/environment.log')
    with zipfile.ZipFile('lab08_13_artifacts.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): archive.write(p,p.as_posix())
    print('АРХИВ lab08_13_artifacts.zip',flush=True)


if __name__=='__main__':
    main()
