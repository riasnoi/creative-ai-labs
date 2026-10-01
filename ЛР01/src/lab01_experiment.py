#!/usr/bin/env python3
"""ЛР 1. Эксперимент: протокол воспроизводимости, разброс по seed, влияние фактора, паспорт эксперимента.

Пример:
    python lab01_experiment.py --variant 1 --out ../demo_journal/variant01
    python lab01_experiment.py --context clouds --factor size --seed 101 --n 5 --out out
Порядок «контекст × фактор» фиксирован: вариант N = (номер контекста − 1) * 6 + номер фактора.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import lab01_generator as gen

CONTEXTS = {
    "clouds": ("облачное небо (фон афиши)", {"cell": 32, "octaves": 4, "persistence": 0.50}),
    "paper": ("бумажная текстура (подложка макета)", {"cell": 4, "octaves": 3, "persistence": 0.60}),
    "marble": ("мраморный узор (обложка)", {"cell": 24, "octaves": 5, "persistence": 0.55}),
    "water": ("водная рябь (фон видеозаставки)", {"cell": 8, "octaves": 4, "persistence": 0.45}),
    "film": ("зернистая плёнка (постобработка)", {"cell": 2, "octaves": 2, "persistence": 0.70}),
}
FACTORS = {
    "size": ("размер изображения ×1,5", lambda p: {"size": int(p["size"] * 1.5)}),
    "octaves": ("число октав +1", lambda p: {"octaves": p["octaves"] + 1}),
    "cell": ("шаг решётки ×2", lambda p: {"cell": p["cell"] * 2}),
    "persistence": ("затухание амплитуд −0,15", lambda p: {"persistence": round(p["persistence"] - 0.15, 2)}),
    "quantize": ("квантование до 4 бит", lambda p: {"quantize": 4}),
    "interp": ("линейная интерполяция вместо сглаженной", lambda p: {"interp": "linear"}),
}
METRICS = ("mean", "contrast", "edge", "entropy")
SIGNIFICANCE_K = 2.0  # критерий заранее: |Δ| > K·s_base


def variant_to_pair(n: int) -> tuple[str, str]:
    c, f = divmod(n - 1, len(FACTORS))
    return list(CONTEXTS)[c], list(FACTORS)[f]


def stats(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, help="номер варианта 1–30 (задаёт контекст и фактор)")
    ap.add_argument("--context", choices=list(CONTEXTS))
    ap.add_argument("--factor", choices=list(FACTORS))
    ap.add_argument("--seed", type=int, default=101, help="базовый seed S")
    ap.add_argument("--n", type=int, default=5, help="число seed в серии (>=2)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    context, factor = (variant_to_pair(a.variant) if a.variant else (a.context, a.factor))
    if not context or not factor:
        ap.error("укажите --variant или пару --context и --factor")
    if a.n < 2:
        ap.error("--n должно быть не меньше 2 (для выборочного s нужно n>1)")

    ctx_title, ctx_params = CONTEXTS[context]
    fac_title, fac_fn = FACTORS[factor]
    base = {**gen.DEFAULT_PARAMS, **ctx_params}
    pert = {**base, **fac_fn(base)}
    art = a.out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    S = a.seed

    # A. Протокол воспроизводимости: S, S, S+1000, S
    seeds_a = [S, S, S + 1000, S]
    runs_a = [gen.run_once(base, s, art / f"repro_run{i + 1}_seed{s}.png") for i, s in enumerate(seeds_a)]
    h = [r["sha256_pixels"] for r in runs_a]
    eq = {"EQ_1_2": h[0] == h[1], "EQ_1_4": h[0] == h[3], "EQ_1_3": h[0] == h[2], "EQ_2_4": h[1] == h[3]}
    eq_png = {"EQ_1_2_png": runs_a[0]["sha256_png"] == runs_a[1]["sha256_png"]}

    # B и C. Серия seed для базовых и изменённых параметров
    seeds = [S + i for i in range(a.n)]
    base_runs = [gen.run_once(base, s, art / f"base_seed{s}.png" if s == S else None) for s in seeds]
    pert_runs = [gen.run_once(pert, s, art / f"perturbed_seed{s}.png" if s == S else None) for s in seeds]
    table = {}
    for m in METRICS:
        mb, sb = stats([r[m] for r in base_runs])
        mp, sp = stats([r[m] for r in pert_runs])
        d = mp - mb
        table[m] = {"mean_base": mb, "s_base": sb, "mean_pert": mp, "s_pert": sp, "delta": d,
                    "delta_over_s": (abs(d) / sb) if sb > 0 else float("inf"),
                    "significant": bool(sb > 0 and abs(d) > SIGNIFICANCE_K * sb)}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    script_hash = {p.name: gen.sha256(p.read_bytes()) for p in (Path(gen.__file__), Path(__file__))}
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(), "zlib": zlib.ZLIB_VERSION,
           "generator_version": gen.SCRIPT_VERSION, "third_party_packages": "не используются (стандартная библиотека)"}
    result = {"date": now, "variant": a.variant, "context": context, "context_title": ctx_title, "factor": factor,
              "factor_title": fac_title, "base_seed": S, "n": a.n, "seeds_repro": seeds_a, "eq": eq, "eq_png": eq_png,
              "params_base": base, "params_perturbed": pert, "runs_repro": runs_a, "metrics_table": table,
              "significance_rule": f"|Δ| > {SIGNIFICANCE_K}·s_base", "env": env, "script_sha256": script_hash}
    (a.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    passport = {"цель_и_критерий": f"Проверить воспроизводимость генератора и оценить, превышает ли эффект фактора «{fac_title}» "
                                  f"разброс между seed; критерий заранее: {result['significance_rule']} при n={a.n}",
                "данные": "входных данных нет: изображение порождается генератором (синтетические данные, лицензии не требуются)",
                "модель": f"lab01_generator.py v{gen.SCRIPT_VERSION}; sha256 скриптов: {script_hash}",
                "параметры": {"базовые": base, "изменённые": pert, "seed": S, "число_seed": a.n},
                "среда": env, "дата": now,
                "процедура": f"python lab01_experiment.py --context {context} --factor {factor} --seed {S} --n {a.n} --out <папка>",
                "результаты": {"eq": eq, "sha256_pixels_run1": h[0], "метрики": table},
                "ограничения": "n мало для точной оценки s; одна модель-заменитель; метрики описывают статистику яркости, а не эстетику"}
    (a.out / "passport.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Паспорт эксперимента ЛР 1 — вариант {a.variant or '—'}", "", f"Дата: {now}", "",
             f"**Контекст:** {ctx_title}. **Фактор:** {fac_title}.", "", "## Протокол воспроизводимости (полные массивы пикселей)", "",
             "| Запуск | seed | sha256 массива (первые 16 символов) |", "|---:|---:|---|"]
    lines += [f"| {i + 1} | {r['seed']} | `{r['sha256_pixels'][:16]}` |" for i, r in enumerate(runs_a)]
    lines += ["", "```text"] + [f"{k} = {str(v).lower()}" for k, v in eq.items()] + ["```", "",
              f"## Разброс и влияние фактора (n = {a.n}, seed {S}…{S + a.n - 1})", "",
              f"Критерий: {result['significance_rule']} (задан до эксперимента).", "",
              "| Метрика | среднее (база) | s (база) | среднее (фактор) | Δ | |Δ|/s | существенно |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for m, t in table.items():
        lines.append(f"| {m} | {t['mean_base']:.4f} | {t['s_base']:.4f} | {t['mean_pert']:.4f} | {t['delta']:+.4f} | "
                     f"{t['delta_over_s']:.2f} | {'да' if t['significant'] else 'нет'} |")
    lines += ["", "## Среда", "", *(f"* {k}: {v}" for k, v in env.items()), "",
              f"Скрипты (sha256): {json.dumps(script_hash, ensure_ascii=False)}"]
    (a.out / "passport.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"контекст={context} фактор={factor} EQ={eq} значимо={[m for m, t in table.items() if t['significant']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
