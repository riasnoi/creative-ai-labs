"""Проверить комплект, манифест, происхождение, notebook и повтор Engee."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import tomllib
import zipfile
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    lines = (root / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    expected = {}
    for line in lines:
        digest, name = line.split('  ', 1)
        assert name not in expected and not Path(name).is_absolute() and '..' not in Path(name).parts
        expected[name] = digest
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts}
    assert actual == set(expected), (actual - set(expected), set(expected) - actual)
    provenance = json.loads((root / 'artifacts/provenance.json').read_text())
    for item in provenance['files']:
        data = (root / item['path']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    results = tomllib.loads((root / 'artifacts/engee/results.toml').read_text())
    plan = tomllib.loads((root / 'artifacts/engee/expectations.toml').read_text())
    initial = json.loads((root / 'artifacts/expectations_before_run.json').read_text())
    assert initial['recorded_at_msk'] < plan['recorded_at_msk'] <= results['started_at_msk']
    assert initial['threshold_db'] == plan['practical_threshold_db'] == results['threshold_db'] == 1.0
    repeat = tomllib.loads((root / 'artifacts/engee/recheck.toml').read_text())
    assert all(repeat[k] for k in ['repeat_rows_exact', 'repeat_reference_exact', 'repeat_array_hashes_exact'])
    for name, digest in repeat['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    passport = json.loads((root / 'artifacts/passport.json').read_text())
    assert passport['experiment'] == results and passport['repeat_verification'] == repeat
    for name, digest in passport['source_files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    nb = json.loads((root / 'src/LR04_13.ngscript').read_text())
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 6 and all(c['outputs'] for c in cells)
    for cell in cells:
        assert all(o['output_type'] != 'error' for o in cell['outputs'])
    report = root / 'reports/Рясной_Вариант_13_Лаб04.docx'
    with zipfile.ZipFile(report) as z:
        assert z.testzip() is None
        assert not any('comments' in n.lower() for n in z.namelist())
        xml = z.read('word/document.xml')
        tree = ET.fromstring(xml)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text = ''.join(n.text or '' for n in tree.findall('.//w:t', ns))
        assert all(x in text for x in ['№4', 'Вариант 13', '20,1212', '27,9529', '7,8317', '26.10.1'])
        assert all(tree.find('.//w:' + x, ns) is None for x in ['ins', 'del', 'commentRangeStart', 'commentReference'])
        assert len(tree.findall('.//w:sectPr', ns)) == 2
        assert b'LR04_13_artifacts.zip' in z.read('word/_rels/document.xml.rels')
        assert 'мрамор' not in text.lower()
    pdf = root / 'reports/Рясной_Вариант_13_Лаб04.pdf'
    assert pdf.read_bytes().startswith(b'%PDF-')
    with zipfile.ZipFile(root / 'artifacts/LR04_13_artifacts.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert z.read(name) == (root / name).read_bytes(), name
    print(f'PASS: {len(expected)} файлов, исходники, паспорт, 6 выполненных ячеек, отчёты, архив и повтор Engee')


if __name__ == '__main__':
    main()
