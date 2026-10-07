"""Проверка комплекта ЛР03 без перезаписи сохранённых результатов."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    expected = {}
    for line in (ROOT / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        expected[name] = digest
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Хеш: {name}')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts}
    check(actual==set(expected), 'Покрытие MANIFEST')
    r = tomllib.loads((ROOT/'artifacts/engee/results.toml').read_text())
    pre = json.loads((ROOT/'artifacts/expectations_before_run.json').read_text())
    check(pre['recorded_at_msk'] < r['started_at_msk'], 'Время регистрации')
    check(pre['variant']==13 and pre['seeds']==r['seeds'], 'Условия')
    passport = json.loads((ROOT/'artifacts/passport.json').read_text())
    check(passport['summary']==r['rows'] and passport['luma']==r['luma'], 'Паспорт и результаты')
    check(passport['run_started_msk']==r['started_at_msk'], 'Дата в паспорте')
    versions = json.loads((ROOT/'artifacts/engee_versions.json').read_text())
    for key in ('engee','cabinet'):
        check(passport['environment'][key]==versions[key], 'Версии')
    library = ROOT/'src/original/lab03_lib.jl'
    check(library.stat().st_size==5027, 'Размер оригинальной библиотеки')
    check(hashlib.sha256(library.read_bytes()).hexdigest()==passport['library_sha256'], 'Исходная библиотека')
    nb = json.loads((ROOT/'src/LR03_13.ngscript').read_text())
    check(len(nb['cells'])==5, 'Ячейки экспорта Engee')
    check(''.join(nb['cells'][0]['source']).replace('\r\n','\n')==library.read_text(), 'Текст учебной библиотеки')
    outputs=[o for cell in nb['cells'] for o in cell.get('outputs',[])]
    check(all(cell.get('outputs') for cell in nb['cells']), 'Невыполненные ячейки')
    check(not any(o.get('output_type')=='error' for o in outputs), 'Ошибка в экспорте')
    check(r['luma']['sha256_first'] in json.dumps(outputs), 'Хеш в выводе Engee')
    check(r['demo']['psnr_db']==34.594244830067446 and r['demo']['sha256'].startswith('50567c6b9e5e4ac1'), 'Воспроизведение демо')
    docx=ROOT/'reports/Рясной_Вариант_13_Лаб03.docx'
    with ZipFile(docx) as z:
        xml=z.read('word/document.xml');tree=ET.fromstring(xml)
        for tag in ('commentRangeStart','commentRangeEnd','commentReference','ins','del'):
            check(not any(e.tag.endswith('}'+tag) for e in tree.iter()), 'Комментарии/правки Word')
        text=''.join(tree.itertext())
        check('ЛАБОРАТОРНАЯ РАБОТА №3' in text and 'Вариант 13' in text,'Номер работы')
        check(len([n for n in z.namelist() if n.startswith('word/media/')])>=4,'Рисунки Word')
    with ZipFile(ROOT/'artifacts/LR03_13_artifacts.zip') as z:
        check(z.testzip() is None,'Целостность архива')
        for name in z.namelist():
            if name=='ARCHIVE_README.txt':continue
            check(hashlib.sha256(z.read(name)).hexdigest()==expected[name],'Байты архива: '+name)
    subprocess.run([sys.executable,str(ROOT/'tools/verify_results.py'),str(ROOT/'artifacts/engee')],check=True)
    print(f'OK: комплект, паспорт, экспорт, архив и SHA-256 ({len(expected)} файлов)')

if __name__=='__main__':
    main()
