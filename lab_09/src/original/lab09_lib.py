# ЛР 9. Промпт-инжиниринг, часть 1: контролируемое сравнение запросов для SD Turbo. Библиотека учебных функций.
# Проверено 20.09.2026: Engee 26.8.2-H3, процессор, Python 3.11, torch 2.4.1+cu121, diffusers 0.32.2, transformers 4.57.6.
# Установка (один раз):  pip install "diffusers==0.32.2" "transformers<5"   (новые версии не работают с torch 2.4.1)
import time
import numpy as np
import torch
from scipy import ndimage
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

torch.set_num_threads(min(4, torch.get_num_threads()))
MODEL_ID = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
# CLIP ViT-B/32 из ЛР 6 не годится: его веса лежат только в pickle-файле, а transformers запрещает torch.load при torch < 2.6 (CVE-2025-32434).
# Берём CLIP ViT-L/14 того же автора (OpenAI): у него есть model.safetensors. Числа ЛР 9 несопоставимы с ЛР 6 по абсолютной величине.
CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SEEDS = range(101, 106)          # серия seed
STEPS, GUIDANCE, SIZE = 1, 0.0, 512   # рекомендованный режим SD Turbo (лекция 4)

# Пять контекстов: базовый запрос A — только предмет (как в ЛР 6).
SHORT = ["robotic arm poster", "podcast cover about the city", "science exhibition poster",
         "ecology lecture visual, forest", "electronic music cover art"]
FACTORS = ["style", "light", "palette", "composition", "boilerplate", "negation"]
BOILER = "masterpiece, best quality, highly detailed, 8k"
# ELEMENTS[c][k] = (добавляемая фраза, текст-цель для CLIP). Запрос B = A + ", " + фраза.
ELEMENTS = [
    [("flat vector illustration",) * 2, ("dramatic rim light",) * 2, ("deep blue and amber colors",) * 2,
     ("centered composition with empty space around",) * 2, (BOILER, "highly detailed, sharp, high quality image"), ("no people, no text", "a person")],
    [("minimal flat geometric style",) * 2, ("warm sunset light",) * 2, ("teal and orange colors",) * 2,
     ("symmetrical layered horizon",) * 2, (BOILER, "highly detailed, sharp, high quality image"), ("no people, no text", "a person")],
    [("isometric 3D render",) * 2, ("soft glow",) * 2, ("dark background with white and cyan lines",) * 2,
     ("balanced constellation of connected nodes",) * 2, (BOILER, "highly detailed, sharp, high quality image"), ("no logo, no text", "written text and letters")],
    [("watercolor illustration",) * 2, ("soft morning light",) * 2, ("green and gold colors",) * 2,
     ("top-down view from above",) * 2, (BOILER, "highly detailed, sharp, high quality image"), ("no people, no text", "a person")],
    [("macro photography",) * 2, ("neon rim light",) * 2, ("iridescent purple and cyan colors",) * 2,
     ("close-up with strong contrast",) * 2, (BOILER, "highly detailed, sharp, high quality image"), ("no logo, no text", "written text and letters")],
]
METRICS = ("clip_target", "clip_subject", "edge", "contrast")


def load_pipe():
    """Закреплённая ревизия, float32, процессор. Предупреждение о safety_checker ожидаемо: у SD Turbo его нет."""
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_clip():
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION, use_safetensors=True).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
    return model, proc


def generate(pipe, prompt, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=SIZE, width=SIZE, generator=g).images[0]


@torch.no_grad()
def clip_sims(clip, img, texts):
    """Косинусное сходство изображения с каждым текстом в CLIP, умноженное на 100 (не CLIPScore)."""
    model, proc = clip
    inp = proc(text=list(texts), images=img, return_tensors="pt", padding=True, truncation=True)
    out = model(**inp)
    a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return [float(v) * 100 for v in (a @ b.T)[0]]


def luma(img):
    x = np.asarray(img.convert("RGB"), dtype=np.float64) / 255
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def edge_energy(img):
    """Средняя длина градиента яркости (Собель, нормировка 1/8), как в ЛР 4 и ЛР 6."""
    y = luma(img)
    return float(np.mean(np.hypot(ndimage.sobel(y, axis=0), ndimage.sobel(y, axis=1)) / 8))


def contrast(img):
    return float(np.std(luma(img)))


def texts_for(c):
    """Тексты для CLIP по контексту c: предмет и шесть целей факторов."""
    return [SHORT[c]] + [ELEMENTS[c][k][1] for k in range(6)]


def prompt_a(c):
    return SHORT[c]


def prompt_b(c, k):
    return SHORT[c] + ", " + ELEMENTS[c][k][0]


def measure(pipe, clip, c, prompt, seed):
    """Одно изображение: метрики (сходство с предметом и со всеми шестью целями, энергия границ, контраст) и время, с."""
    t0 = time.perf_counter()
    img = generate(pipe, prompt, seed)
    dt = time.perf_counter() - t0
    sims = clip_sims(clip, img, texts_for(c))
    return dict(sims=sims, edge=edge_energy(img), contrast=contrast(img), seconds=dt), img


def compare(a_list, b_list, k):
    """Сводка по четырём метрикам: A, s (по seed режима A), B, разность, |d|/s, признак |d| > 2s. a_list, b_list — результаты measure()."""
    rows = []
    for m in METRICS:
        pick = {"clip_target": lambda r: r["sims"][1 + k], "clip_subject": lambda r: r["sims"][0],
                "edge": lambda r: r["edge"], "contrast": lambda r: r["contrast"]}[m]
        A = [pick(r) for r in a_list]; B = [pick(r) for r in b_list]
        d = float(np.mean(B) - np.mean(A)); s = float(np.std(A, ddof=1))
        rows.append((m, float(np.mean(A)), s, float(np.mean(B)), d, abs(d) / s if s > 0 else float("inf"), bool(abs(d) > 2 * s)))
    return rows


def variant_summary(pipe, clip, n, seeds=SEEDS):
    """Вариант n = 1..30: контекст c = (n-1)//6, фактор k = (n-1)%6. Режим A — только предмет, режим B — предмет плюс одна фраза."""
    c, k = (n - 1) // 6, (n - 1) % 6
    generate(pipe, prompt_a(c), 0)      # прогрев
    a = [measure(pipe, clip, c, prompt_a(c), s)[0] for s in seeds]
    b = [measure(pipe, clip, c, prompt_b(c, k), s)[0] for s in seeds]
    return dict(variant=n, ctx=c + 1, factor=FACTORS[k], rows=compare(a, b, k))
