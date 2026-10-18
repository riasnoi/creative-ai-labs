"""Проверить SHA-256 и извлечь транспортный TOML, выгруженный из Engee."""
import argparse
import base64
import hashlib
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    bundle = tomllib.loads(args.export.read_text(encoding='utf-8'))
    checked = {}
    for item in bundle['files']:
        name = item['name']
        if not name or Path(name).name != name or name in {'.', '..'} or name in checked:
            raise ValueError(f'Недопустимое или повторное имя: {name!r}')
        data = base64.b64decode(item['base64'], validate=True)
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Не совпадает SHA-256: {name}')
        checked[name] = data
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Каталог назначения должен быть пустым')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in checked.items():
        (args.output / name).write_bytes(data)
    print(f'Извлечено и проверено файлов: {len(checked)}')


if __name__ == '__main__':
    main()
