#!/usr/bin/env python3
"""Собрать чистый ZIP для преподавателя и при необходимости обновить манифест."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", ".gitignore", ".gitattributes", "requirements-report.txt")
FOLDERS = ("docs", "tools", "ЛР01")
IGNORED_PARTS = {".venv", "venv", "__pycache__", ".git", ".DS_Store"}


def submission_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for folder in FOLDERS:
        for path in (ROOT / folder).rglob("*"):
            relative = path.relative_to(ROOT)
            if any(part in IGNORED_PARTS or part.startswith("~$") for part in relative.parts):
                continue
            if path.is_file() and not path.is_symlink() and path.suffix not in {".pyc", ".zip", ".tmp"}:
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-manifest", action="store_true", help="обновить SHA-256 после намеренных правок")
    args = parser.parse_args()
    files = submission_files()
    manifest = ROOT / "MANIFEST.sha256"
    expected = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n"
                       for p in files)
    if args.update_manifest:
        manifest.write_text(expected, encoding="utf-8")
        print(f"Обновлён MANIFEST.sha256: {len(files)} файлов")
    elif not manifest.exists() or manifest.read_text(encoding="utf-8") != expected:
        parser.error("манифест отсутствует или файлы изменены; проверьте изменения и выполните --update-manifest")
    destination = ROOT / "dist/lab01_variant13_submission.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in [*files, manifest]:
            archive.write(path, "lab01_variant13/" + path.relative_to(ROOT).as_posix())
    with ZipFile(destination) as archive:
        error = archive.testzip()
        if error is not None:
            raise ValueError(f"Ошибка ZIP: {error}")
    print(f"Архив: {destination} ({destination.stat().st_size:,} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
