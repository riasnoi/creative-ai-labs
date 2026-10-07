#!/usr/bin/env python3
"""ЛР 2. Реестр рисков: проверка и ранжирование (категории — 12 групп рисков NIST AI 600-1).

    python lab02_risks.py --template ..\\artifacts\\risk_register.csv       # пустой шаблон с примером строки
    python lab02_risks.py --csv ..\\artifacts\\risk_register.csv --out ..\\artifacts\\risk_ranking.md

Колонки: risk_id, category, description, likelihood, impact, measure, owner, trigger.
Оценка = likelihood × impact (оба 1–5). Проверки: категория из перечня NIST; оценки 1–5; заполнены описание, мера,
ответственный и признак наступления; не менее 5 рисков и не менее 4 разных категорий.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import lab02_common as C

FIELDS = ["risk_id", "category", "description", "likelihood", "impact", "measure", "owner", "trigger"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.template:
        a.template.parent.mkdir(parents=True, exist_ok=True)
        with a.template.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerow({"risk_id": "R01", "category": "Intellectual Property", "description": "пример: в набор попал материал без подтверждённых прав",
                        "likelihood": 3, "impact": 5, "measure": "проверка лицензий при приёме, карантин", "owner": "ответственный за данные",
                        "trigger": "запись без лицензии в манифесте"})
        print(f"шаблон записан: {a.template}")
        return 0
    if not a.csv:
        ap.error("укажите --csv или --template")
    with a.csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    errors = []
    if rows and set(FIELDS) - set(rows[0]):
        errors.append(f"нет колонок: {sorted(set(FIELDS) - set(rows[0]))}")
    scored = []
    for r in rows:
        rid = r.get("risk_id", "?")
        if r.get("category") not in C.NIST_RISKS:
            errors.append(f"{rid}: категория «{r.get('category')}» не из перечня NIST AI 600-1")
        valid_scale = True
        try:
            l, i = int(r["likelihood"]), int(r["impact"])
            if not (1 <= l <= 5 and 1 <= i <= 5):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            errors.append(f"{rid}: likelihood и impact должны быть целыми от 1 до 5")
            valid_scale = False
        for f in ("description", "measure", "owner", "trigger"):
            if not (r.get(f) or "").strip():
                errors.append(f"{rid}: не заполнено поле «{f}»")
        if valid_scale:
            scored.append((l * i, r))
    if len(rows) < 5:
        errors.append(f"рисков {len(rows)}, требуется не менее 5")
    if len({r.get("category") for r in rows}) < 4:
        errors.append("нужно не менее 4 разных категорий")
    scored.sort(key=lambda x: -x[0])
    lines = ["# Реестр рисков (ранжирование)", "", "| Место | Оценка | Категория | Описание | Мера | Ответственный | Признак |", "|---:|---:|---|---|---|---|---|"]
    for k, (s, r) in enumerate(scored, 1):
        lines.append(f"| {k} | {s} | {r['category']} | {r['description']} | {r['measure']} | {r['owner']} | {r['trigger']} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    for n, e in enumerate(errors, 1):
        print(f"FAIL {n}: {e}")
    print("RESULT: PASS" if not errors else f"RESULT: FAIL ({len(errors)})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
