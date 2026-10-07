"""Распаковать транспорт TOML из Engee с проверкой исходных байтов."""
import argparse
import base64
import hashlib
from pathlib import Path
import tomllib

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    files = tomllib.loads(args.bundle.read_text(encoding="utf-8"))["files"]
    decoded = {}
    for item in files:
        name = item["name"]
        if Path(name).name != name or name in decoded or name in (".", ".."):
            raise ValueError(f"Недопустимое имя: {name}")
        payload = base64.b64decode(item["base64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 не совпал: {name}")
        decoded[name] = payload
    args.destination.mkdir(parents=True, exist_ok=True)
    if any(args.destination.iterdir()):
        raise ValueError("Каталог назначения должен быть пустым")
    for name, payload in decoded.items():
        (args.destination / name).write_bytes(payload)
    print(f"Проверено и распаковано файлов: {len(decoded)}")

if __name__ == "__main__":
    main()
