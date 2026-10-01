#!/usr/bin/env python3
"""Проверить сохранённые результаты и повторить вариант 13, не меняя сдаваемые файлы."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "ЛР01"
METRICS = ("mean", "contrast", "edge", "entropy")
EXPECTED_EQ = {"EQ_1_2": True, "EQ_1_4": True, "EQ_1_3": False, "EQ_2_4": True}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_numbers(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        check(set(actual) == set(expected), f"{label}: различаются поля")
        for key in expected:
            same_numbers(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, float):
        check(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
              f"{label}: {actual} != {expected}")
    else:
        check(actual == expected, f"{label}: {actual!r} != {expected!r}")


def png_pixels(path: Path) -> tuple[bytes, int]:
    """Read the generator's 8-bit grayscale PNG independently of its writer."""
    data = path.read_bytes()
    check(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: неверная сигнатура PNG")
    pos, compressed, size, ended = 8, bytearray(), None, False
    while pos < len(data):
        check(pos + 12 <= len(data), f"{path}: обрезанный PNG")
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        end = pos + 12 + length
        check(end <= len(data), f"{path}: обрезанный блок PNG")
        crc = struct.unpack(">I", data[end - 4:end])[0]
        check(zlib.crc32(tag + payload) & 0xFFFFFFFF == crc, f"{path}: неверная CRC")
        if tag == b"IHDR":
            w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            check(w == h and depth == 8 and color == compression == filtering == interlace == 0,
                  f"{path}: неподдерживаемый формат PNG")
            size = w
        elif tag == b"IDAT":
            compressed.extend(payload)
        elif tag == b"IEND":
            ended = True
            check(end == len(data), f"{path}: данные после IEND")
            break
        pos = end
    check(size is not None and ended, f"{path}: неполный PNG")
    raw = zlib.decompress(compressed)
    check(len(raw) == size * (size + 1), f"{path}: неверное число пикселей")
    check(all(raw[y * (size + 1)] == 0 for y in range(size)), f"{path}: неожиданный фильтр")
    pixels = b"".join(raw[y * (size + 1) + 1:(y + 1) * (size + 1)] for y in range(size))
    return pixels, size


def verify_png(path: Path, expected_run: dict) -> None:
    pixels, size = png_pixels(path)
    check(size == expected_run["params"]["size"], f"{path}: размер не совпадает")
    check(hashlib.sha256(pixels).hexdigest() == expected_run["sha256_pixels"], f"{path}: пиксели изменены")
    check(hashlib.sha256(path.read_bytes()).hexdigest() == expected_run["sha256_png"], f"{path}: PNG изменён")


def verify_manifest() -> None:
    manifest = REPO / "MANIFEST.sha256"
    if not manifest.exists():
        return
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = REPO / name
        check(path.is_file(), f"Отсутствует файл из MANIFEST: {name}")
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Изменён файл из MANIFEST: {name}")
        count += 1
    print(f"OK: контрольные суммы комплекта ({count} файлов)", flush=True)


def verify_series(n: int, temporary: Path) -> None:
    folder = LAB / "artifacts" / ("v13" if n == 5 else "v13_n50")
    saved = load(folder / "results.json")
    check(saved["variant"] == 13 and saved["n"] == n and saved["base_seed"] == 101,
          f"{folder}: неверные условия опыта")
    check(saved["context"] == "marble" and saved["factor"] == "size", "Неверный вариант")
    changed = [k for k in saved["params_base"] if saved["params_base"][k] != saved["params_perturbed"][k]]
    check(changed == ["size"], "Варьируется больше одного фактора")
    check(saved["params_base"]["size"] == 96 and saved["params_perturbed"]["size"] == 144, "Неверные размеры")
    check(saved["eq"] == EXPECTED_EQ, "Нарушен протокол seed")
    check(saved["seeds_repro"] == [101, 101, 1101, 101], "Неверная последовательность seed")
    for name, digest in saved["script_sha256"].items():
        check(hashlib.sha256((LAB / "src" / name).read_bytes()).hexdigest() == digest,
              f"Исходный учебный скрипт {name} изменён")
    passport = load(folder / "passport.json")
    same_numbers(passport["результаты"]["метрики"], saved["metrics_table"], "паспорт")
    check(passport["результаты"]["eq"] == saved["eq"], "Паспорт и результаты расходятся")
    for i, run in enumerate(saved["runs_repro"], 1):
        verify_png(folder / "artifacts" / f"repro_run{i}_seed{run['seed']}.png", run)
    for metric in METRICS:
        row = saved["metrics_table"][metric]
        check(row["s_base"] > 0, f"{metric}: нулевой разброс")
        check(row["significant"] == (abs(row["delta"]) > 2 * row["s_base"]), "Неверный вывод по критерию")
    output = temporary / f"n{n}"
    subprocess.run([sys.executable, str(LAB / "src/lab01_experiment.py"), "--variant", "13",
                    "--seed", "101", "--n", str(n), "--out", str(output)],
                   check=True, cwd=LAB, capture_output=True, text=True)
    fresh = load(output / "results.json")
    for key in ("params_base", "params_perturbed", "eq", "eq_png", "metrics_table"):
        same_numbers(fresh[key], saved[key], key)
    for current, historical in zip(fresh["runs_repro"], saved["runs_repro"]):
        check(current["sha256_pixels"] == historical["sha256_pixels"], "Повтор не совпал по пикселям")
    for name in ("base_seed101.png", "perturbed_seed101.png"):
        old, _ = png_pixels(folder / "artifacts" / name)
        new, _ = png_pixels(output / "artifacts" / name)
        check(old == new, f"{name}: повтор не совпал по пикселям")
    # PNG bytes may differ on another zlib; compare pixel arrays across machines.
    png_same = all((folder / "artifacts" / p.name).read_bytes() == p.read_bytes()
                   for p in (output / "artifacts").glob("*.png"))
    print(f"OK: n={n}, метрики и пиксели воспроизведены; PNG байт-в-байт: {png_same}", flush=True)


def verify_seed_demo() -> None:
    hashes = []
    for name in ("noseed_run1", "noseed_run2", "seed101_run1", "seed101_run2"):
        folder = LAB / "artifacts/error_demo_v13" / name
        saved = load(next(folder.glob("result_seed*.json")))
        verify_png(folder / f"texture_seed{saved['seed']}.png", saved)
        hashes.append(saved["sha256_pixels"])
    check(hashes[0] != hashes[1] and hashes[2] == hashes[3], "Демонстрация seed не подтверждается")
    print("OK: сохранённая диагностика отсутствующего seed", flush=True)


def verify_engee() -> None:
    """Check the downloaded evidence; this does not execute a remote Julia kernel."""
    notebook = load(LAB / "reports/LR01_environment.ngscript")
    cells = notebook["cells"]
    check(notebook["nbformat"] == 4 and len(cells) == 3, "Engee: неверный формат экспорта")
    sources = ["".join(cell["source"]) for cell in cells]
    check(sources == ["VERSION\nversioninfo()", "VERSION", "seed = 101"],
          "Engee: изменены код или порядок ячеек")
    check(all(cell["cell_type"] == "code" for cell in cells), "Engee: ожидаются ячейки кода")
    stdout = "".join(output["text"] for output in cells[0]["outputs"]
                     if output["output_type"] == "stream" and output["name"] == "stdout")
    raw = (LAB / "artifacts/engee/versioninfo.log").read_text(encoding="utf-8")
    check(stdout == raw and raw.startswith("Julia Version 1.12.4\n"),
          "Engee: журнал versioninfo не совпал с экспортом")
    for cell, expected in zip(cells[1:], ['v"1.12.4"', "101"]):
        results = [output["data"]["text/plain"] for output in cell["outputs"]
                   if output["output_type"] == "execute_result"]
        check(results == [expected], "Engee: не совпал результат VERSION или seed")
    for name in ("passport.md", "state_checks.log"):
        check((LAB / "artifacts/engee" / name).is_file(), f"Engee: отсутствует {name}")
    print("OK: экспорт Engee, VERSION, seed и полный вывод Julia 1.12.4 согласованы (без удалённого запуска)",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="повторить только обязательную серию n=5")
    args = parser.parse_args()
    try:
        verify_manifest()
        verify_seed_demo()
        verify_engee()
        with tempfile.TemporaryDirectory(prefix="lab01-verify-") as temp:
            for n in ([5] if args.quick else [5, 50]):
                verify_series(n, Path(temp))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, zlib.error) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        return 1
    print("Проверка пройдена. Сохранённые результаты не перезаписывались.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
