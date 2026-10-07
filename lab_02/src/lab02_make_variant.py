#!/usr/bin/env python3
"""ЛР 2. Создаёт входные данные варианта: manifest.csv (12 синтетических объектов), policy.json, variant.json.

Данные синтетические: домены example.org, вымышленные правообладатели, контрольные суммы — от строк, не от файлов.
Вариант N = (номер сценария − 1) * 6 + номер профиля дефектов. Набор детерминирован: то же N — тот же набор.

    python lab02_make_variant.py --variant 1 --out ..\\artifacts\\v01
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import lab02_common as C

CLEAN_LICENSES = ["CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0"]
TYPES = [("image", "png"), ("audio", "wav"), ("text", "txt")]


def build(n: int) -> tuple[list[dict], dict, dict]:
    scen, prof = C.variant_to_pair(n)
    rng = random.Random(1000 + n)
    rows = []
    for i in range(1, 13):
        mt, ext = TYPES[i % 3]
        item = f"item{i:02d}"
        rows.append({"id": item, "filename": f"{item}.{ext}", "media_type": mt,
                     "source": f"https://example.org/archive/{item}",
                     "date_acquired": f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}",
                     "license": rng.choice(CLEAN_LICENSES), "rights_holder": f"Автор-{rng.randint(1, 40):02d}",
                     "personal_data": "no", "consent": "n-a",
                     "sha256": hashlib.sha256(f"variant{n}-{item}".encode()).hexdigest()})
    k = 3 if prof == "nd" else 4
    idx = rng.sample(range(len(rows)), k + 1)
    defect, noise = idx[:k], idx[k]
    for j, pos in enumerate(defect):
        r = rows[pos]
        if prof == "nc":
            r["license"] = ["CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"][j % 2]
        elif prof == "sa":
            r["license"] = "CC-BY-SA-4.0"
        elif prof == "nd":
            r["license"] = "CC-BY-ND-4.0"
        elif prof == "license":
            r["license"] = ["", "free-to-use", "CC BY 4", "Public domain"][j % 4]
        elif prof == "privacy":
            r["personal_data"] = "yes"
            r["consent"] = ["none", "document", "n-a", "document"][j % 4]
        elif prof == "origin":
            if j == 0:
                r["source"] = ""
            elif j == 1:
                r["date_acquired"] = ""
            elif j == 2:
                r["sha256"] = rows[(pos + 1) % len(rows)]["sha256"]
            else:
                r["source"], r["date_acquired"] = "", ""
    # фоновая несущественная проблема в каждом варианте (набор не бывает «идеальным»)
    kind = rng.choice(["rights_holder", "date", "duplicate"])
    r = rows[noise]
    if kind == "rights_holder":
        r["rights_holder"] = ""
    elif kind == "date":
        r["date_acquired"] = ""
    else:
        r["sha256"] = rows[(noise + 3) % len(rows)]["sha256"]
    title, policy = C.SCENARIOS[scen]
    meta = {"variant": n, "scenario": scen, "scenario_title": title, "profile": prof, "profile_title": C.PROFILES[prof]}
    return rows, {"scenario": title, **policy}, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not 1 <= a.variant <= 30:
        ap.error("--variant должен быть от 1 до 30")
    rows, policy, meta = build(a.variant)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (a.out / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "variant.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"вариант {a.variant}: {meta['scenario_title']} | {meta['profile_title']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
