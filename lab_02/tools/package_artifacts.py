#!/usr/bin/env python3
"""Упаковать первичные артефакты ЛР02 с внутренним манифестом SHA-256."""
from pathlib import Path
import hashlib
import zipfile

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "artifacts/ЛР02_Вариант_13_Артефакты.zip"
paths = sorted([p for folder in ("src", "artifacts", "docs/input")
                for p in (LAB / folder).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".zip"]
               + [LAB / "experiment_journal.md"])
lines = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        name = path.relative_to(LAB).as_posix()
        data = path.read_bytes()
        lines.append(hashlib.sha256(data).hexdigest() + "  " + name)
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, data)
    archive.writestr("ARCHIVE_MANIFEST.sha256", "\n".join(lines) + "\n")
print(f"Архив первичных артефактов: {len(paths)} файлов")
