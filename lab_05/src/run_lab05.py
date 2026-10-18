"""ЛР05, вариант 13: запускать в Google Colab с GPU T4 из каталога с lab05_lib.py."""
import contextlib
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import math
import pathlib
import platform
import subprocess
import sys
import time
import numpy as np
import torch
import lab05_lib as L

OUT = pathlib.Path('artifacts')
OUT.mkdir(exist_ok=True)
PROMPT, SEED = L.BRIEFS[2], 1013
TZ = dt.timezone(dt.timedelta(hours=3))
def now():
    return dt.datetime.now(TZ).isoformat()
def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return '+Infinity' if obj > 0 else '-Infinity'
    return obj

def main():
    assert torch.cuda.is_available(), 'Требуется Colab с GPU T4'
    hypothesis = dict(recorded_at=now(), variant=13, seed=SEED, prompt=PROMPT,
        hypothesis='Свежие CPU-генераторы с seed 1013 в одном сеансе дадут побитово одинаковые изображения. Общий генератор изменит состояние и второе изображение.',
        tolerance_db=40.0, decision_rule='identical: одинаковые пиксели; close: PSNR >= 40 dB; different: PSNR < 40 dB',
        fixed=dict(model=L.MODEL_ID, revision=L.REVISION, size=512, steps=1, guidance_scale=0.0, dtype='float16', generator_device='cpu', batch=1))
    save_json(OUT/'expectations_before_run.json', hypothesis)
    print('Гипотеза ДО загрузки модели:', json.dumps(hypothesis,ensure_ascii=False,indent=2), flush=True)
    env = dict(recorded_at=now(), python=sys.version, platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), packages={p:md.version(p) for p in ['torch','diffusers','transformers','Pillow','numpy','accelerate','safetensors']},
        library_sha256=hashlib.sha256(pathlib.Path(L.__file__).read_bytes()).hexdigest(),
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
    save_json(OUT/'environment.json',env)
    print(json.dumps(env,ensure_ascii=False,indent=2),flush=True)
    t0=time.perf_counter(); pipe=L.load_pipe(); torch.cuda.synchronize()
    load_s=time.perf_counter()-t0
    L.generate(pipe,PROMPT,1); torch.cuda.synchronize()
    def run_fresh():
        t=time.perf_counter(); img=L.generate(pipe,PROMPT,SEED)[0]; torch.cuda.synchronize()
        return img,time.perf_counter()-t
    def store(img,label,seconds,generator_policy):
        dest=OUT/label; dest.mkdir(exist_ok=True)
        data=L.png_bytes(img); (dest/'result.png').write_bytes(data)
        manifest=dict(created_at=now(),variant=13,condition=label,model_id=L.MODEL_ID,revision=L.REVISION,
            prompt=PROMPT,seed=SEED,steps=1,guidance_scale=0.0,size=512,batch=1,
            generator_device='cpu',generator_policy=generator_policy,dtype='float16',
            device=env['gpu'],platform=env['platform'],python=sys.version.split()[0],packages=env['packages'],
            load_seconds=load_s,generation_seconds=seconds,image_size=list(img.size),image_mode=img.mode,sha256=L.sha(data))
        save_json(dest/'manifest.json',manifest)
        return manifest
    a,sa=run_fresh(); ma=store(a,'A',sa,'fresh')
    b,sb=run_fresh(); mb=store(b,'B',sb,'fresh')
    def compare(x,y):
        mse=float(np.mean((L.to_arr(x)-L.to_arr(y))**2)); p=L.psnr_db(x,y)
        return clean(dict(sha_a=L.sha(L.png_bytes(x)),sha_b=L.sha(L.png_bytes(y)),
            pixel_equal=bool(np.array_equal(L.to_arr(x),L.to_arr(y))),max_diff=L.max_abs_diff(x,y),mse=mse,psnr=p,verdict=L.classify(p)))
    summary=clean(L.variant_summary(pipe,13))
    result=dict(repeat=compare(a,b),variant_summary=summary)
    g=torch.Generator(device='cpu').manual_seed(SEED)
    state0=L.sha(g.get_state().numpy().tobytes())
    def wrong():
        t=time.perf_counter()
        image=pipe(prompt=[PROMPT],num_inference_steps=1,guidance_scale=0.0,height=512,width=512,generator=g).images[0]
        torch.cuda.synchronize()
        return image,time.perf_counter()-t
    e1,s1=wrong(); state1=L.sha(g.get_state().numpy().tobytes())
    e2,s2=wrong(); state2=L.sha(g.get_state().numpy().tobytes())
    store(e1,'error_first',s1,'shared:first'); store(e2,'error_second',s2,'shared:second')
    fixed,sf=run_fresh(); store(fixed,'corrected',sf,'fresh')
    result.update(shared_generator=compare(e1,e2),error_first_vs_A=compare(a,e1),corrected_vs_A=compare(a,fixed),
        generator_state_sha256=[state0,state1,state2],finished_at=now())
    save_json(OUT/'metrics.json',result)
    print('РЕЗУЛЬТАТЫ:',json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    from IPython.display import display
    display(a); display(b); display(e2)

if __name__=='__main__':
    with (OUT/'run.log').open('w',encoding='utf-8') as log:
        class Tee:
            def write(self,s):
                sys.__stdout__.write(s); log.write(s)
            def flush(self):
                sys.__stdout__.flush(); log.flush()
        # В блокноте stdout должен оставаться IPython-потоком, поэтому журнал ведётся через Tee.
        original=sys.stdout
        class NotebookTee:
            def write(self,s):
                original.write(s); log.write(s)
            def flush(self):
                original.flush(); log.flush()
        with contextlib.redirect_stdout(NotebookTee()):
            main()
