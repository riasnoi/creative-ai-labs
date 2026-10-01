#!/usr/bin/env python3
"""ЛР 1. Учебный генератор процедурных текстур (только стандартная библиотека Python 3.10+).

Зачем он нужен. Настоящие генеративные модели (ЛР 5–10) тяжёлые и требуют ресурсов. Для отработки практики
воспроизводимости достаточно любого случайного генератора: он тоже порождает объект x ~ p(x | условие) из
случайного начального состояния (seed). Здесь «условие» — параметры текстуры, «объект» — изображение 8 бит.

Использование (один запуск):
    python lab01_generator.py --seed 101 --out out/run1
Без --seed скрипт берёт seed из системного времени и предупреждает об этом (типичная ошибка, см. МУ).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
import zlib
from pathlib import Path

SCRIPT_VERSION = "1.0"
DEFAULT_PARAMS = {"size": 96, "cell": 32, "octaves": 4, "persistence": 0.5, "quantize": 8, "interp": "smooth"}


def _fade(t: float, interp: str) -> float:
    return t * t * (3.0 - 2.0 * t) if interp == "smooth" else t


def _octave(size: int, cell: int, rng: random.Random, interp: str) -> list[list[float]]:
    grid = size // cell + 2
    lattice = [[rng.random() for _ in range(grid)] for _ in range(grid)]
    field = []
    for y in range(size):
        gy = y / cell
        y0 = int(gy)
        ty = _fade(gy - y0, interp)
        row = []
        for x in range(size):
            gx = x / cell
            x0 = int(gx)
            tx = _fade(gx - x0, interp)
            a, b = lattice[y0][x0], lattice[y0][x0 + 1]
            c, d = lattice[y0 + 1][x0], lattice[y0 + 1][x0 + 1]
            top = a + (b - a) * tx
            bottom = c + (d - c) * tx
            row.append(top + (bottom - top) * ty)
        field.append(row)
    return field


def generate(params: dict, seed: int) -> bytes:
    """Возвращает size*size байт яркости. Весь источник случайности — локальный random.Random(seed)."""
    p = {**DEFAULT_PARAMS, **params}
    rng = random.Random(seed)
    size, octaves = int(p["size"]), int(p["octaves"])
    total = 0.0
    acc = [[0.0] * size for _ in range(size)]
    for o in range(octaves):
        amp = float(p["persistence"]) ** o
        total += amp
        field = _octave(size, max(1, int(p["cell"]) >> o), rng, p["interp"])
        for y in range(size):
            row, frow = acc[y], field[y]
            for x in range(size):
                row[x] += amp * frow[x]
    levels = (1 << int(p["quantize"])) - 1
    pixels = bytearray()
    for y in range(size):
        for x in range(size):
            v = min(1.0, max(0.0, acc[y][x] / total))
            q = round(v * levels)
            pixels.append(round(q * 255 / levels))
    return bytes(pixels)


def metrics(pixels: bytes, size: int) -> dict:
    n = len(pixels)
    mean = sum(pixels) / n
    var = sum((p - mean) ** 2 for p in pixels) / n
    edge = sum(abs(pixels[y * size + x + 1] - pixels[y * size + x]) for y in range(size) for x in range(size - 1)) / (size * (size - 1))
    counts = [0] * 256
    for p in pixels:
        counts[p] += 1
    entropy = -sum((c / n) * math.log2(c / n) for c in counts if c)  # бит на пиксель, максимум 8
    return {"mean": mean / 255, "contrast": math.sqrt(var) / 255, "edge": edge / 255, "entropy": entropy}


def png_bytes(pixels: bytes, size: int) -> bytes:
    raw = b"".join(b"\x00" + pixels[y * size:(y + 1) * size] for y in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_once(params: dict, seed: int, out_png: Path | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    pixels = generate(p, seed)
    png = png_bytes(pixels, int(p["size"]))
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(png)
    return {"seed": seed, "params": p, "sha256_pixels": sha256(pixels), "sha256_png": sha256(png), **metrics(pixels, int(p["size"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="начальное значение генератора; без него запуск невоспроизводим")
    for k, v in DEFAULT_PARAMS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    seed = a.seed
    if seed is None:
        seed = int(time.time_ns() % 2**31)
        print(f"ПРЕДУПРЕЖДЕНИЕ: seed не задан, использован {seed} — результат не воспроизводится", file=sys.stderr)
    params = {k: getattr(a, k) for k in DEFAULT_PARAMS}
    result = run_once(params, seed, a.out / f"texture_seed{seed}.png")
    (a.out / f"result_seed{seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
