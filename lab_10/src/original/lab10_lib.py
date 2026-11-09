# ЛР 10. Сравнительное исследование моделей text-to-image. Библиотека учебных функций.
# Среда получения результатов (20.09.2026): Engee 26.8.2-H3 на процессоре; Python 3.11; torch 2.4.1+cu121; diffusers 0.32.2; transformers 4.57.6.
# Установка пакетов, один раз (как в ЛР 9):  pip install "diffusers==0.32.2" "transformers<5"
import gc, os, time
import numpy as np
import torch
from scipy import ndimage
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

torch.set_num_threads(min(4, torch.get_num_threads()))
SEEDS = range(101, 106)
CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
TEXT_PROBE = "written text and letters"

# Конфигурация модели = веса + планировщик + число шагов + масштаб управления + размер кадра. Всё это входит в «модель» при сравнении.
MODELS = {
    "turbo": dict(repo="stabilityai/sd-turbo", revision="b261bac6fd2cf515557d5d0707481eafa0485ec2", steps=1, guidance=0.0, size=512,
                  note="дистиллированная SD 2.1, 1 шаг, без управления"),
    "lcm": dict(repo="SimianLuo/LCM_Dreamshaper_v7", revision="a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5", steps=4, guidance=8.0, size=512,
                note="латентная согласованность, 4 шага"),
    "bk": dict(repo="nota-ai/bk-sdm-tiny", revision="0364108e53b7f7f4d2585e817a0b7a83dc261cfa",
steps=10, guidance=7.5, size=512,
               scheduler="dpm", note="облегчённая SD 1.x, DPM-Solver++, 10 шагов"),
    "amused": dict(repo="amused/amused-256", revision="09b6259bf96dbe6d70a852b70812420fe02df55e", steps=12, guidance=10.0, size=256,
                   note="маскированные токены (не диффузия), 256×256, 12 шагов"),
}
# Пять брифов (английские, как в ЛР 5): одинаковый запрос для всех моделей
BRIEFS = [
    "editorial poster concept for a robotics festival, glowing mechanical arm silhouette, deep blue and amber palette, clean composition, no text, no logo, no people",
    "square podcast cover about the city environment, abstract layered skyline of geometric shapes, teal and orange palette, minimal flat style, no text, no people",
    "poster concept for a science exhibition, abstract constellation of connected nodes, dark background with white and cyan lines, balanced composition, no text, no logo",
    "visual for an ecology lecture series, stylized forest canopy seen from above, layered green shapes, soft morning light, no text, no people",
    "cover art for an electronic music release, brushed metal surface with iridescent reflections, macro photography style, strong contrast, no text, no logo, no people",
]
# Шесть сравнений: пары четырёх моделей (фактор варианта)

PAIRS = [("turbo", "lcm"), ("turbo", "bk"), ("turbo", "amused"), ("lcm", "bk"), ("lcm", "amused"), ("bk", "amused")]
METRICS = ("clip_brief", "clip_text", "edge", "contrast", "seconds")


def load_model(key):
    c = MODELS[key]
    pipe = DiffusionPipeline.from_pretrained(c["repo"], revision=c["revision"], torch_dtype=torch.float32, safety_checker=None,
                                             requires_safety_checker=False, use_safetensors=True)
    if c.get("scheduler") == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_clip():
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION, use_safetensors=True).eval(
)
    return model, CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION)


def generate(pipe, key, prompt, seed, cfg=None):
    """cfg — необязательные замены параметров конфигурации (например, dict(steps=1)); нужны
только для учебных ошибок."""
    c = {**MODELS[key], **(cfg or {})}
    g = torch.Generator("cpu").manual_seed(seed)
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=c["steps"], guidance_scale=c["guidance"], height=c["size"],
 width=c["size"], generator=g).images[0]
    return img, time.perf_counter() - t0


@torch.no_grad()
def clip_sims(clip, img, texts):
    """Косинусное сходство изображения с текстами в CLIP, умноженное на 100 (не CLIPScore)."""
    model, proc = clip
    inp = proc(text=list(texts), images=img, return_tensors="pt", padding=True, truncation=True)
    out = model(**inp)
    a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return [float(v) * 100 for v in (a @ b.T)[0]]


def luma256(img):
    """Яркость изображения, приведённого к 256×256 (чтобы размер кадра не влиял на энергию границ)."""
    x = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float64) / 255
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]



def edge_energy(img):
    y = luma256(img)
    return float(np.mean(np.hypot(ndimage.sobel(y, axis=0), ndimage.sobel(y, axis=1)) / 8))


def contrast(img):
    return float(np.std(luma256(img)))


def measure(pipe, clip, key, ctx, seed, cfg=None):
    """Одно изображение: метрики и время генерации, с (без загрузки модели)."""
    img, dt = generate(pipe, key, BRIEFS[ctx], seed, cfg)
    s = clip_sims(clip, img, [BRIEFS[ctx], TEXT_PROBE])
    return dict(clip_brief=s[0], clip_text=s[1], edge=edge_energy(img), contrast=contrast(img), seconds=dt), img


def compare(a_list, b_list):
    """Сводка: для каждой метрики среднее A, s (по seed режима A), среднее B, разность, |d|/s, признак |d| > 2s."""
    rows = []
    for m in METRICS:
        A = [r[m] for r in a_list]; B = [r[m] for r in b_list]
        d = float(np.mean(B) - np.mean(A)); s = float(np.std(A, ddof=1))
        rows.append((m, float(np.mean(A)), s, float(np.mean(B)), d, abs(d) / s if s > 0 else float("inf"), bool(abs(d) > 2 * s)))
    return rows


def variant_summary(n, seeds=SEEDS, save_dir=None, cfg=None):
    """Вариант n = 1..30: бриф c = (n-1)//6, пара k = (n-1)%6. Модели загружаются по очереди (память).
    save_dir — папка для изображений (файлы <модель>_s<seed>.jpg); cfg — замены конфигурации
по моделям (только для учебных ошибок)."""
    c, k = (n - 1) // 6, (n - 1) % 6
    clip = load_clip()
    res = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for key in PAIRS[k]:
        pipe = load_model(key)
        generate(pipe, key, BRIEFS[c], 0, (cfg or {}).get(key))      # прогрев
        res[key] = []
        for s in seeds:
            r, img = measure(pipe, clip, key, c, s, (cfg or {}).get(key))
            if save_dir:
                img.save(os.path.join(save_dir, f"{key}_s{s}.jpg"), quality=90)

            res[key].append(r)
        del pipe; gc.collect()
    return dict(variant=n, ctx=c + 1, pair=PAIRS[k], rows=compare(res[PAIRS[k][0]], res[PAIRS[k][1]])
)


def blind_sheets(n, folder, seeds=SEEDS):
    """Листы для слепого сравнения: для каждого seed — пара изображений A и B, порядок слева и
справа случаен (фиксируется номером варианта).
    Пишет folder/blind_<seed>.png и folder/blind_key.json (какая модель слева); ключ откройте только после выбора."""
    import json, random
    from PIL import Image
    a, b = PAIRS[(n - 1) % 6]
    rnd = random.Random(n)
    key = {}
    for s in seeds:
        left, right = (a, b) if rnd.random() < 0.5 else (b, a)
        key[str(s)] = left
        im_l = Image.open(os.path.join(folder, f"{left}_s{s}.jpg")).convert("RGB").resize((384, 384))
        im_r = Image.open(os.path.join(folder, f"{right}_s{s}.jpg")).convert("RGB").resize((384, 384))
        sheet = Image.new("RGB", (384 * 2 + 12, 384), "white")
        sheet.paste(im_l, (0, 0))
        sheet.paste(im_r, (384 + 12, 0))
        sheet.save(os.path.join(folder, f"blind_{s}.png"))
    json.dump(key, open(os.path.join(folder, "blind_key.json"), "w"), ensure_ascii=False)
    return key
