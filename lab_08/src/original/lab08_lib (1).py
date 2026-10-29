# ЛР 8. Генерация и интерполяция в латентном пространстве. Библиотека учебных функций.
# Требуется файл lab07_lib.py из ЛР 7 в той же папке (из него берутся сцены, устройство и базовые слои).
# Проверено 19.09.2026: Google Colab (GPU T4, torch 2.11.0+cu128) и Engee 26.8.2-H3 (процессор, Python 3.11, torch 2.4.1).
import numpy as np
import torch
import torch.nn as nn
import lab07_lib as L

torch.set_num_threads(min(4, torch.get_num_threads()))   # в Engee по умолчанию видно 112 потоков при 5 ядрах — это замедляет обучение в разы
DEVICE, SEEDS, KINDS = L.DEVICE, L.SEEDS, L.KINDS
N_PAIRS = 200      # пар изображений для проверки интерполяции
N_SAMPLES = 256    # число случайных сэмплов из априорного распределения


class ConvVAE(L.ConvAE):
    """Тот же свёрточный автокодировщик, но кодировщик выдаёт среднее и логарифм дисперсии латентного вектора."""
    def __init__(self, d):
        super().__init__(d)
        self.enc = nn.Sequential(*list(self.enc.children())[:-1])   # без последнего линейного слоя
        self.fc_mu, self.fc_lv = nn.Linear(64 * 16, d), nn.Linear(64 * 16, d)

    def encode_dist(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def encode(self, x):
        return self.encode_dist(x)[0]

    def forward(self, x):
        mu, lv = self.encode_dist(x)
        return self.decode(mu), mu, lv


def train_vae(model, Xtr, beta, epochs=30, lr=2e-3, seed=0, batch=128, warmup=10):
    """Потери: сумма квадратов ошибок по пикселям + beta * KL (на изображение). beta = 0 даёт обычный автокодировщик (используется среднее).
    Вес KL-члена растёт линейно от 0 до beta за первые warmup эпох: без «разогрева» часть запусков схлопывается (все изображения кодируются одинаково)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, batch):
            x = Xtr[perm[i:i + batch]]
            mu, lv = model.encode_dist(x)
            z = mu if beta == 0 else mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, generator=g).to(DEVICE)
            rec = ((model.decode(z) - x) ** 2).flatten(1).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            w = beta * min(1.0, (epoch + i / n) / warmup)
            loss = rec + w * kl if beta > 0 else rec
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def slerp(za, zb, t):
    """Сферическая интерполяция; для почти коллинеарных векторов — линейная."""
    na, nb = za.norm(dim=-1, keepdim=True), zb.norm(dim=-1, keepdim=True)
    cos = ((za * zb).sum(-1, keepdim=True) / (na * nb).clamp_min(1e-8)).clamp(-1, 1)
    om = torch.acos(cos)
    so = torch.sin(om)
    w = torch.where(so.abs() < 1e-4, torch.full_like(so, 1.0), so)
    lin = (1 - t) * za + t * zb
    sph = (torch.sin((1 - t) * om) / w) * za + (torch.sin(t * om) / w) * zb
    return torch.where(so.abs() < 1e-4, lin, sph)


_TARGETS = {}


def interp_targets(kind):
    """Эталонные сцены для середины интерполяции: сцена, построенная по средним значениям факторов пары (истинная «середина»)."""
    if kind not in _TARGETS:
        _, _, _, Fte = L.dataset(kind)
        rng = np.random.default_rng(11)
        ia, ib = rng.integers(0, Fte.shape[0], N_PAIRS), rng.integers(0, Fte.shape[0], N_PAIRS)
        f = Fte.cpu().numpy()
        tg = {t: torch.tensor(np.stack([L.render(kind, (1 - t) * f[a] + t * f[b]) for a, b in zip(ia, ib)]), device=DEVICE) for t in (0.25, 0.5, 0.75)}
        _TARGETS[kind] = (torch.tensor(ia, device=DEVICE), torch.tensor(ib, device=DEVICE), tg)
    return _TARGETS[kind]


@torch.no_grad()
def metrics(model, kind, interp="lin", temp=1.0, seed=0):
    """Четыре показателя: PSNR реконструкции; PSNR интерполяции (по трём точкам пути) относительно истинной сцены;
    PSNR ближайшего соседа для случайных сэмплов из N(0, temp^2 I) в тестовой выборке; средний R^2 линейного восстановления факторов по среднему."""
    model.eval()
    Xtr, Xte, Ftr, Fte = L.dataset(kind)
    mu_te = model.encode(Xte)
    recon = L.psnr_images(model.decode(mu_te), Xte)
    ia, ib, tg = interp_targets(kind)
    za, zb = mu_te[ia], mu_te[ib]
    vals = []
    for t, target in tg.items():
        z = slerp(za, zb, t) if interp == "slerp" else (1 - t) * za + t * zb
        vals.append(L.psnr_images(model.decode(z), target))
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    zs = temp * torch.randn((N_SAMPLES, mu_te.shape[1]), generator=g).to(DEVICE)
    xs = model.decode(zs).flatten(1)
    mse = torch.cdist(xs, Xte.flatten(1)).pow(2) / xs.shape[1]
    nn_psnr = float((10 * torch.log10(1.0 / mse.min(1).values.clamp_min(1e-10))).mean())
    mu_tr = model.encode(Xtr)
    A = torch.cat([mu_tr, torch.ones(mu_tr.shape[0], 1, device=DEVICE)], 1).double()
    W = torch.linalg.solve(A.T @ A + 1e-6 * torch.eye(A.shape[1], device=DEVICE, dtype=A.dtype), A.T @ Ftr.double()).float()
    pred = torch.cat([mu_te, torch.ones(mu_te.shape[0], 1, device=DEVICE)], 1) @ W
    r2 = 1 - ((pred - Fte) ** 2).sum(0) / ((Fte - Fte.mean(0)) ** 2).sum(0)
    return dict(recon=recon, interp=float(np.mean(vals)), sample=nn_psnr, r2=float(r2.mean()))


# Шесть факторов; базовый режим A: VAE, d = 8, beta = 1, линейная интерполяция, сэмплы из N(0, I).
FACTORS = ["beta_1_to_8", "ae_to_vae", "latent_8_to_2", "latent_8_to_32", "interp_linear_to_slerp", "sample_temp_1_to_2"]
BASE = dict(d=8, beta=1.0, interp="lin", temp=1.0, epochs=30)


def settings(factor):
    a, b = dict(BASE), dict(BASE)
    if factor == "beta_1_to_8":
        b["beta"] = 8.0
    elif factor == "ae_to_vae":
        a["beta"] = 0.0
    elif factor == "latent_8_to_2":
        b["d"] = 2
    elif factor == "latent_8_to_32":
        b["d"] = 32
    elif factor == "interp_linear_to_slerp":
        b["interp"] = "slerp"
    elif factor == "sample_temp_1_to_2":
        b["temp"] = 2.0
    else:
        raise ValueError(factor)
    return a, b


_MODELS = {}


def get_model(kind, cfg, seed):
    """Модель обучается один раз для каждой комбинации (набор данных, d, beta, число эпох, seed): факторы интерполяции и температуры используют одну модель."""
    key = (kind, cfg["d"], cfg["beta"], cfg["epochs"], seed)
    if key not in _MODELS:
        Xtr = L.dataset(kind)[0]
        torch.manual_seed(seed)
        _MODELS[key] = train_vae(ConvVAE(cfg["d"]).to(DEVICE), Xtr, cfg["beta"], cfg["epochs"], seed=seed)
    return _MODELS[key]


def run(kind, cfg, seed):
    return metrics(get_model(kind, cfg, seed), kind, cfg["interp"], cfg["temp"], seed)


def variant_summary(n, seeds=SEEDS):
    """Вариант n = 1..30: набор данных c = (n-1)//6, фактор k = (n-1)%6. Для каждой метрики: A, s, B, разность, |d|/s, признак |d| > 2s."""
    c, k = (n - 1) // 6, (n - 1) % 6
    a_cfg, b_cfg = settings(FACTORS[k])
    A = {m: [] for m in ("recon", "interp", "sample", "r2")}
    B = {m: [] for m in A}
    for s in seeds:
        ma, mb = run(KINDS[c], a_cfg, s), run(KINDS[c], b_cfg, s)
        for m in A:
            A[m].append(ma[m]); B[m].append(mb[m])
    rows = []
    for m in A:
        d = np.mean(B[m]) - np.mean(A[m]); sa = np.std(A[m], ddof=1)
        rows.append((m, float(np.mean(A[m])), float(sa), float(np.mean(B[m])), float(d), float(abs(d) / sa) if sa > 0 else float("inf"), bool(abs(d) > 2 * sa)))
    return dict(variant=n, ctx=c + 1, factor=FACTORS[k], rows=rows)
