# ЛР 8. Демонстрационный вариант 1: набор «диск», фактор «вес KL-члена beta: 1 → 8».
import numpy as np
import torch
import lab07_lib as L
import lab08_lib as V

N = 1
c, k = (N - 1) // 6, (N - 1) % 6
kind = L.KINDS[c]
a_cfg, b_cfg = V.settings(V.FACTORS[k])
print("A:", a_cfg)
print("B:", b_cfg)

# Серия из пяти seed: четыре показателя режимов A и B
result = V.variant_summary(N)
for name, a, s, b, d, q, sig in result["rows"]:
    print(f"{name:7s} A={a:7.3f} s={s:6.3f} B={b:7.3f} d={d:+7.3f} |d|/s={q:7.2f} 2s={'да' if sig else 'нет'}")

# Диагностика: сэмплы из N(0, I) у обычного автокодировщика (beta = 0) и у вариационных (beta = 1, 8); seed 101
Xte = L.dataset(kind)[1]
for name, beta in (("AE, beta=0", 0.0), ("VAE, beta=1", 1.0), ("VAE, beta=8", 8.0)):
    model = V.get_model(kind, dict(V.BASE, beta=beta), 101)
    with torch.no_grad():
        mu, lv = model.encode_dist(Xte)
        kl_axis = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)          # KL по осям, нат
    m = V.metrics(model, kind, "lin", 1.0, 101)
    active = int((kl_axis > 0.05).sum()) if beta > 0 else "-"
    print(f"{name}: разброс mu по осям {mu.std(0).cpu().numpy().round(2)}; активных осей {active}; "
          f"сэмплы PSNR {m['sample']:.2f}; интерполяция PSNR {m['interp']:.2f}; реконструкция PSNR {m['recon']:.2f}")

# Схлопывание: обучение с beta = 1 сразу и с разогревом KL-члена; PSNR реконструкции по seed 101-105
Xtr = L.dataset(kind)[0]
for name, wu in (("без разогрева", 1e-9), ("разогрев 10 эпох", 10)):
    vals = []
    for s in L.SEEDS:
        torch.manual_seed(s)
        m = V.train_vae(V.ConvVAE(8).to(L.DEVICE), Xtr, 1.0, 30, seed=s, warmup=wu)
        vals.append(round(V.metrics(m, kind, "lin", 1.0, s)["recon"], 1))
    print(f"{name}: {vals}")

# Лист сравнения comparison.png: 8 сэмплов из N(0, I) (AE, VAE beta=1) и линейный путь между тестовыми сценами 0 и 2 (VAE beta=1, 8)
from PIL import Image
def strip(imgs):
    return np.concatenate(list(imgs.permute(0, 2, 3, 1).cpu().numpy()), axis=1)
g = torch.Generator(device="cpu").manual_seed(5)
z = torch.randn((8, 8), generator=g).to(L.DEVICE)
rows = []
with torch.no_grad():
    for beta in (0.0, 1.0):
        rows.append(strip(V.get_model(kind, dict(V.BASE, beta=beta), 101).decode(z)))
    for beta in (1.0, 8.0):
        model = V.get_model(kind, dict(V.BASE, beta=beta), 101)
        za, zb = model.encode(Xte[0:1]), model.encode(Xte[2:3])
        t = torch.linspace(0, 1, 8, device=L.DEVICE).unsqueeze(1)
        rows.append(strip(model.decode((1 - t) * za + t * zb)))
sheet = Image.fromarray((np.concatenate(rows, axis=0) * 255).round().astype("uint8")).resize((8 * 32 * 4, 4 * 32 * 4), Image.NEAREST)
sheet.save("comparison.png")
sheet
