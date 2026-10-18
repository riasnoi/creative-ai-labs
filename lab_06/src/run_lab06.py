"""ЛР06, вариант 13. Запуск в новой папке Colab с GPU T4."""
import contextlib
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import pathlib
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image
import lab05_lib as L
import lab06_lib as M

OUT = pathlib.Path('artifacts')
TZ = dt.timezone(dt.timedelta(hours=3))
def now(): return dt.datetime.now(TZ).isoformat()
def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    assert torch.cuda.is_available(), 'Требуется GPU T4'
    assert not OUT.exists(), 'Используйте новую папку: первичные результаты нельзя заменять'
    OUT.mkdir()
    a_cfg,b_cfg=M.settings(M.FACTORS[0],2)
    plan=dict(recorded_at=now(),variant=13,context=3,factor='steps_1_to_4',seeds=list(M.SEEDS),
        hypothesis='Четыре шага увеличат энергию границ, контраст и время; сходство CLIP останется в пределах ±1 балла.',
        tolerances=dict(clip_absolute=1.0,edge_relative=0.10,contrast_relative=0.10,seconds_relative=0.20),
        classification='CLIP: рост улучшение, падение ухудшение; время: снижение улучшение. Рост edge/contrast условно улучшение выраженности признака, не качества; внутри допуска нейтрально. Граница включена в нейтральную область.',
        benefit_rule='Рост времени окупается по критериям опыта, только если CLIP растёт более чем на 1 балл и узлы/линии остаются различимы; одна резкость не обосновывает пользу.',
        A=a_cfg,B=b_cfg,model=L.MODEL_ID,revision=L.REVISION,clip_model=M.CLIP_ID,clip_revision=M.CLIP_REVISION,
        dtype='float16',generator_device='cpu',batch=1)
    save('expectations_before_run.json',plan)
    print('ПЛАН ДО ЗАПУСКА',json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    env=dict(recorded_at=now(),python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        packages={p:md.version(p) for p in ['torch','diffusers','transformers','Pillow','numpy','scipy','accelerate','safetensors']},
        libraries={pathlib.Path(m.__file__).name:sha(m.__file__) for m in [L,M]},
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),pip_list=subprocess.check_output([sys.executable,'-m','pip','list'],text=True))
    save('environment.json',env)
    print('GPU',env['gpu'],'ВЕРСИИ',env['packages'],flush=True)
    t=time.perf_counter(); pipe=L.load_pipe(); torch.cuda.synchronize(); load_pipe_s=time.perf_counter()-t
    t=time.perf_counter(); clip=M.load_clip(); torch.cuda.synchronize(); load_clip_s=time.perf_counter()-t
    # variant_summary вызывается без изменения учебной библиотеки. Обёртка сохраняет каждое измерение и PNG.
    raw=[]; original_measure=M.measure
    def recorded_measure(pipe_arg,clip_arg,cfg,c,seed):
        metrics,img=original_measure(pipe_arg,clip_arg,cfg,c,seed)
        label='A' if cfg['steps']==1 else 'B'
        folder=OUT/label; folder.mkdir(exist_ok=True)
        file=folder/f'seed_{seed}.png'; file.write_bytes(L.png_bytes(img))
        row=dict(mode=label,seed=seed,settings=cfg,metrics=metrics,image=str(file.relative_to(OUT)),sha256=sha(file),recorded_at=now())
        raw.append(row); save('raw_metrics.json',raw)
        print(label,seed,metrics,flush=True)
        return metrics,img
    M.measure=recorded_measure
    try: summary=M.variant_summary(pipe,clip,13)
    finally: M.measure=original_measure
    save('summary.json',summary)
    images=[Image.open(OUT/x/'seed_101.png').resize((384,384)) for x in ['A','B']]
    sheet=Image.new('RGB',(768,384)); sheet.paste(images[0],(0,0)); sheet.paste(images[1],(384,0)); sheet.save(OUT/'comparison.png')
    # Ошибка демонстрационного варианта воспроизводится на исходном брифе 1.
    err=[]; ims={}
    for g in [0.0,1.0,3.0]:
        img=L.generate(pipe,L.BRIEFS[0],101,guidance=g)[0]; ims[g]=img
        file=OUT/f'guidance_{int(g)}.png'; file.write_bytes(L.png_bytes(img))
        err.append(dict(guidance=g,cfg_active=bool(pipe.do_classifier_free_guidance),sha256=sha(file),image=file.name))
    error=dict(brief=L.BRIEFS[0],seed=101,runs=err,equal_sha_0_1=err[0]['sha256']==err[1]['sha256'],psnr_0_3=L.psnr_db(ims[0.0],ims[3.0]))
    save('guidance_error.json',error)
    # Дополнительный повтор seed 101 подтверждает сохранённые изображения, время в основную серию не входит.
    repeats=[]
    for label,cfg in [('A',a_cfg),('B',b_cfg)]:
        img=L.generate(pipe,cfg['prompt'],101,steps=cfg['steps'])[0]
        repeats.append(dict(mode=label,seed=101,sha256=L.sha(L.png_bytes(img)),matches_saved=L.sha(L.png_bytes(img))==sha(OUT/label/'seed_101.png')))
    save('repeat_check.json',repeats)
    passport=dict(started_at=plan['recorded_at'],finished_at=now(),variant=13,plan=plan,environment='environment.json',
        load_seconds=dict(sd_turbo=load_pipe_s,clip=load_clip_s),measurement='M.measure: perf_counter вокруг L.generate, без загрузки и CLIP, после прогрева обоих режимов seed 0',
        reference_text=L.BRIEFS[2],metrics='clip=100*cos; edge=Sobel/8; contrast=std(luma); s_A ddof=1; delta=mean_B-mean_A',
        summary=summary,error=error,repeat=repeats)
    save('passport.json',passport)
    print('ИТОГ',json.dumps(passport,ensure_ascii=False,indent=2),flush=True)
    from IPython.display import display
    display(sheet)

if __name__=='__main__':
    # Журнал хранится рядом с artifacts, чтобы проверка существования каталога защищала первичные данные.
    with open('run.log','w',encoding='utf-8') as log:
        original=sys.stdout
        class Tee:
            def write(self,s): original.write(s); log.write(s)
            def flush(self): original.flush(); log.flush()
        with contextlib.redirect_stdout(Tee()): main()
    import shutil
    shutil.copy('run.log',OUT/'run.log')
