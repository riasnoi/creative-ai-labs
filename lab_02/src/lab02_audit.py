#!/usr/bin/env python3
"""ЛР 2. Аудит манифеста набора данных по политике использования; заготовка паспорта набора данных.

    python lab02_audit.py --manifest ..\\artifacts\\v01\\manifest.csv --policy ..\\artifacts\\v01\\policy.json --out ..\\artifacts\\v01

Правила (учебная упрощённая модель, не юридическая консультация):
  R1 лицензия не указана (ошибка)                   R5 условие ShareAlike при производных и неприемлемом SA (предупреждение)
  R2 идентификатор не из каталога SPDX (ошибка)     R6 персональные данные без документа-согласия (ошибка)
  R3 условие NonCommercial при коммерческом использовании (ошибка)
  R4 условие NoDerivatives при производных (ошибка) R7 согласие есть, но набор публикуется: проверить охват (предупреждение)
  R8 не указан источник или дата (предупреждение)   R9 совпадающая контрольная сумма — дубликат (предупреждение)
  R10 не указан правообладатель (предупреждение)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import lab02_common as C

POLICY_RU = {"commercial_use": "коммерческое использование", "redistribution": "публикация или передача набора",
             "derivatives": "создание производных работ", "share_alike_acceptable": "допустимо передавать производные на тех же условиях"}


def audit(rows: list[dict], policy: dict) -> list[dict]:
    out = []

    def add(rule, sev, item, msg):
        out.append({"rule": rule, "severity": sev, "item": item, "message": msg})

    seen: dict[str, str] = {}
    for r in rows:
        i, lic = r["id"], r["license"].strip()
        if not lic:
            add("R1", "error", i, "лицензия не указана")
        elif lic not in C.LICENSES:
            add("R2", "error", i, f"идентификатор «{lic}» отсутствует в каталоге SPDX учебной работы")
        else:
            cond = C.LICENSES[lic]
            if cond["nc"] and policy["commercial_use"]:
                add("R3", "error", i, f"{lic}: условие NonCommercial противоречит коммерческому использованию")
            if cond["nd"] and policy["derivatives"]:
                add("R4", "error", i, f"{lic}: условие NoDerivatives запрещает производные, а сценарий их создаёт")
            if cond["sa"] and policy["derivatives"] and not policy["share_alike_acceptable"]:
                add("R5", "warning", i, f"{lic}: ShareAlike требует передавать производные на тех же условиях; сценарий этого не допускает — нужно решение")
        if r["personal_data"].strip().lower() == "yes":
            if r["consent"].strip().lower() != "document":
                add("R6", "error", i, f"персональные данные; согласие: «{r['consent']}» — документа-согласия нет")
            elif policy["redistribution"]:
                add("R7", "warning", i, "есть документ-согласие, но набор публикуется: проверьте, что согласие охватывает публикацию и срок")
        no_src, no_date = not r["source"].strip(), not r["date_acquired"].strip()
        if no_src or no_date:
            add("R8", "warning", i, "не указаны источник и дата получения" if no_src and no_date
                else ("не указан источник" if no_src else "не указана дата получения"))
        if not r["rights_holder"].strip():
            add("R10", "warning", i, "не указан правообладатель")
        h = r["sha256"]
        if h in seen:
            add("R9", "warning", i, f"контрольная сумма совпадает с {seen[h]} — дубликат")
        else:
            seen[h] = i
    return out


def data_card(rows: list[dict], policy: dict, findings: list[dict]) -> str:
    n = len(rows)
    lic = collections.Counter((r["license"] or "не указана") for r in rows)
    types = collections.Counter(r["media_type"] for r in rows)
    dates = sorted(r["date_acquired"] for r in rows if r["date_acquired"])
    pd = sum(1 for r in rows if r["personal_data"].lower() == "yes")
    attr = sum(1 for r in rows if C.LICENSES.get(r["license"], {}).get("attribution"))
    errs = sum(1 for f in findings if f["severity"] == "error")
    warns = len(findings) - errs
    lines = [f"# Паспорт набора данных (заготовка, разделы по Gebru и др., 2018)", "",
             f"Сценарий использования: {policy['scenario']}.", "",
             "## 1. Motivation (мотивация)", "*Заполняет студент: для чего создан набор и для кого.*", "",
             "## 2. Composition (состав)", "",
             f"* Число объектов: {n}; по типам: {', '.join(f'{k} — {v}' for k, v in types.items())}.",
             "* Распределение лицензий: " + ", ".join(f"{k} — {v}" for k, v in lic.most_common()) + ".",
             f"* Объектов с персональными данными: {pd}.",
             f"* Объектов, для которых лицензия требует указания авторства: {attr} (список авторов нужно публиковать вместе с набором).",
             f"* Результаты аудита: ошибок — {errs}, предупреждений — {warns}.", "",
             "## 3. Collection Process (процесс сбора)",
             f"* Даты получения: {dates[0] if dates else '—'} … {dates[-1] if dates else '—'}.",
             "* Источники: " + ", ".join(sorted({r['source'].split('/')[2] for r in rows if r['source'].startswith('http')})) + ".",
             "*Студент дополняет: кто и как собирал материалы, согласия.*", "",
             "## 4. Preprocessing/cleaning/labeling (обработка и разметка)", "*Заполняет студент по итогам решений аудита: что исключено, что исправлено.*", "",
             "## 5. Uses (использование)",
             f"* Предполагаемое: {policy['scenario']}; коммерческое использование — {'да' if policy['commercial_use'] else 'нет'}; производные — {'да' if policy['derivatives'] else 'нет'}.",
             "*Студент дополняет: для каких задач набор не подходит.*", "",
             "## 6. Distribution (распространение)",
             f"* Публикация набора — {'предполагается' if policy['redistribution'] else 'не предполагается'}; условия зависят от лицензий объектов (раздел 2).",
             "", "## 7. Maintenance (сопровождение)", "*Заполняет студент: кто отвечает, как принимаются запросы об удалении.*"]
    return "\n".join(lines) + "\n"


def report_md(meta: dict, policy: dict, findings: list[dict]) -> str:
    errs = sum(1 for f in findings if f["severity"] == "error")
    head = [f"# Отчёт аудита набора данных", "", f"Сценарий: {policy['scenario']}.", "",
            "Условия: " + "; ".join(f"{POLICY_RU.get(k, k)} — {'да' if v else 'нет'}" for k, v in policy.items() if isinstance(v, bool)) + ".", "",
            f"Найдено: ошибок — {errs}, предупреждений — {len(findings) - errs}.", "",
            "| № | Правило | Уровень | Объект | Описание | Решение студента |", "|---:|---|---|---|---|---|"]
    for k, f in enumerate(findings, 1):
        head.append(f"| {k} | {f['rule']} | {'ошибка' if f['severity'] == 'error' else 'предупреждение'} | {f['item']} | {f['message']} | |")
    head += ["", "*Учебная упрощённая модель условий лицензий; не юридическая консультация.*"]
    return "\n".join(head) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    policy = json.loads(a.policy.read_text(encoding="utf-8"))
    findings = audit(rows, policy)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audit_report.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "audit_report.md").write_text(report_md({}, policy, findings), encoding="utf-8")
    (a.out / "data_card_draft.md").write_text(data_card(rows, policy, findings), encoding="utf-8")
    errs = sum(1 for f in findings if f["severity"] == "error")
    print(f"объектов {len(rows)}; ошибок {errs}; предупреждений {len(findings) - errs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
