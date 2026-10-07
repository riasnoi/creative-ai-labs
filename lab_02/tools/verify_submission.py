#!/usr/bin/env python3
"""Проверить ЛР02 и повторить вычисления в новой временной папке."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

LAB = Path(__file__).resolve().parents[1]
ART = LAB / "artifacts"


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script, *args):
    result = subprocess.run([sys.executable, str(LAB / "src" / script), *map(str, args)],
                            cwd=LAB, text=True, capture_output=True)
    require(result.returncode == 0, f"Не прошла команда {script}: {result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-manifest", action="store_true",
                        help="только для подготовки комплекта до создания манифеста")
    args = parser.parse_args()
    if not args.skip_manifest:
        entries = {}
        for line in (LAB / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            require(name not in entries, "Повтор пути в манифесте")
            entries[name] = expected
            require(digest(LAB / name) == expected, f"SHA-256 не совпал: {name}")
        actual = {p.relative_to(LAB).as_posix() for p in LAB.rglob("*") if p.is_file()
                  and p.name != "MANIFEST.sha256" and "__pycache__" not in p.parts
                  and not p.name.startswith(".")}
        require(actual == set(entries), "Манифест покрывает не весь комплект")
        print(f"OK: SHA-256 и покрытие {len(entries)} файлов")

    origin = load(ART / "provenance.json")
    for name, expected in origin["scripts"].items():
        require(digest(LAB / "src" / name) == expected, f"Изменён учебный скрипт {name}")
    passport = LAB / "docs/input/lab01_passport.json"
    require(digest(passport) == origin["lab01_passport_sha256"], "Изменён паспорт ЛР01")
    original_passport = LAB.parent / "ЛР01/artifacts/v13/passport.json"
    if original_passport.exists():
        require(passport.read_bytes() == original_passport.read_bytes(), "Паспорт не свой")
        results = load(LAB.parent / "ЛР01/artifacts/v13/results.json")
        require(load(passport)["результаты"]["метрики"] == results["metrics_table"],
                "Числа карточки не соответствуют результатам ЛР01")
        for name, expected in results["script_sha256"].items():
            require(digest(LAB.parent / "ЛР01/src" / name) == expected,
                    "Происхождение генератора ЛР01 изменилось")

    expected = load(ART / "expectations_before_audit.json")
    audit_time = (ART / "audit_run.log").read_text().splitlines()[0]
    require(datetime.fromisoformat(expected["recorded_at_msk"]) <= datetime.fromisoformat(audit_time),
            "Ожидания не предшествуют аудиту")
    chronology = load(ART / "chronology_evidence.json")
    require(chronology["expectations_file_mtime_ns"] < chronology["audit_log_mtime_ns"]
            and expected["before_audit"], "Нарушен зафиксированный порядок действий")
    original = rows(ART / "v13/manifest.csv")
    findings = load(ART / "v13/audit_report.json")
    signature = [{k: f[k] for k in ("rule", "item", "severity")} for f in findings]
    require(signature == expected["expected_findings"], "Аудит расходится с ожиданиями")
    decisions = load(ART / "decisions.json")
    require({d["item"] for d in decisions} == {r["id"] for r in original}, "Не все объекты разобраны")
    by_item = {d["item"]: d for d in decisions}
    for finding in findings:
        d = by_item[finding["item"]]
        require(finding["rule"] in d["rules"] and d["decision"] and d["rationale"],
                "Находка не покрыта решением")
    nc = {r["id"] for r in original if r["license"].startswith("CC-BY-NC")}
    require(nc == {"item02", "item05", "item06", "item10"}, "Неверный список NC")
    require(all(by_item[i]["release_candidate"] for i in nc), "Решения по NC расходятся")
    release = rows(ART / "manifest_release_candidate.csv")
    require(release == [r for r in original if r["id"] != "item07"], "Кандидат выпуска изменён")
    require(rows(ART / "quarantine.csv") == [r for r in original if r["id"] == "item07"],
            "Неверный карантин")
    require(load(ART / "release_audit/audit_report.json") == [], "Кандидат имеет находки")

    for filename, count in (("data_card.md", 7), ("model_card.md", 9)):
        text = (ART / filename).read_text()
        sections = re.split(r"(?m)^## ", text)[1:]
        require(len(sections) == count and all(len(s.split("\n", 1)[1].strip()) > 50 for s in sections),
                f"Неполные разделы {filename}")
        require("Заполняет студент" not in text, "В финальном документе осталась заготовка")
    card = (ART / "model_card.md").read_text()
    for values in load(passport)["результаты"]["метрики"].values():
        for k in ("mean_base", "s_base", "mean_pert"):
            require(f"{values[k]:.4f}" in card, "Метрика карточки не соответствует паспорту")
    riskrows = rows(ART / "risk_register.csv")
    require({"Data Privacy", "Harmful Bias or Homogenization"} <= {r["category"] for r in riskrows},
            "Нет обязательных категорий варианта")
    require(len({r["risk_id"] for r in riskrows}) == len(riskrows), "Повтор ID риска")
    require("RESULT: PASS" in (ART / "risk_validation.log").read_text(), "Нет сохранённого PASS")

    with tempfile.TemporaryDirectory(prefix="lr02-repeat-") as name:
        temporary = Path(name)
        generated = temporary / "v13"
        run("lab02_make_variant.py", "--variant", 13, "--out", generated)
        for file in ("manifest.csv", "policy.json", "variant.json"):
            require((generated / file).read_bytes() == (ART / "v13" / file).read_bytes(),
                    f"Повтор отличается: {file}")
        run("lab02_audit.py", "--manifest", generated / "manifest.csv", "--policy",
            generated / "policy.json", "--out", generated)
        require(load(generated / "audit_report.json") == findings, "Повтор исходного аудита расходится")
        run("lab02_audit.py", "--manifest", ART / "manifest_release_candidate.csv", "--policy",
            generated / "policy.json", "--out", temporary / "release")
        require(load(temporary / "release/audit_report.json") == [], "Повтор кандидата расходится")
        stdout = run("lab02_risks.py", "--csv", ART / "risk_register.csv", "--out", temporary / "ranking.md")
        require("RESULT: PASS" in stdout, "Повтор реестра не прошёл")
        require((temporary / "ranking.md").read_bytes() == (ART / "risk_ranking.md").read_bytes(),
                "Ранжирование изменилось")
        run("lab02_model_card.py", "--passport", passport, "--out", temporary / "card.md", "--owner",
            "Рясной Владимир Олегович — составитель карточки и ответственный за учебный эксперимент")
        require((temporary / "card.md").read_bytes() == (LAB / "docs/generated/model_card_original.md").read_bytes(),
                "Автозаготовка карточки не воспроизведена")
    print("OK: варианты, решения, 7/9 разделов, метрики ЛР01 и повтор вычислений")

    with zipfile.ZipFile(ART / "ЛР02_Вариант_13_Артефакты.zip") as archive:
        require(archive.testzip() is None, "Повреждён архив")
        for line in archive.read("ARCHIVE_MANIFEST.sha256").decode().splitlines():
            expected_hash, path = line.split("  ", 1)
            require(hashlib.sha256(archive.read(path)).hexdigest() == expected_hash,
                    f"Повреждён файл архива: {path}")
            require(archive.read(path) == (LAB / path).read_bytes(), f"Архив устарел: {path}")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(LAB / "reports/Рясной_Вариант_13_Лаб02.docx") as docx:
        require(not any("comments" in n.lower() for n in docx.namelist()), "Комментарии Word")
        root = ET.fromstring(docx.read("word/document.xml"))
        require(not root.findall(".//w:ins", ns) and not root.findall(".//w:del", ns), "Track changes")
        text = " ".join(root.itertext())
        require("№2" in text and "Вариант 13" in text, "Неверный титул или вариант")
        require("не является юридической консультацией" in text, "Нет оговорки")
    require((LAB / "reports/Рясной_Вариант_13_Лаб02.pdf").read_bytes().startswith(b"%PDF-"), "Нет PDF")
    print("OK: архив соответствует файлам; DOCX без comments/track changes; PDF существует")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
