"""Проверка полноты комплекта ЛР08, отчётов и контрольных сумм."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from verify_results import verify

ROOT=Path(__file__).resolve().parents[1]
report=verify(ROOT)
manifest={}
for line in (ROOT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
    digest,path=line.split('  ',1)
    assert path not in manifest
    assert (ROOT/path).is_file(),path
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    manifest[path]=digest
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
        and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.name!='.DS_Store'}
assert set(manifest)==actual,(set(manifest)^actual)
for p in ROOT.rglob('*'):
    assert not p.is_symlink()
    assert p.name not in ['.git','SKILL.md','AGENTS.md','Artifact.md','.env']
docx=ROOT/'reports/Рясной_Вариант_13_Лаб08.docx'
pdf=docx.with_suffix('.pdf')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(docx) as z:
    assert z.testzip() is None
    assert not any('comments' in n.lower() for n in z.namelist())
    doc=ET.fromstring(z.read('word/document.xml'))
    for name in ['ins','del','commentRangeStart','commentReference']:
        assert not doc.findall('.//w:'+name,ns),name
    text=' '.join(doc.itertext())
    for term in ['ЛАБОРАТОРНАЯ РАБОТА №8','Рясной Владимир Олегович','МИК11','2602149','Вариант 13','Генерация и интерполяция']:
        assert term in text,term
    assert not re.search(r'(мрамор|96 x 96|144 x 144)',text,re.I)
    assert len(doc.findall('.//w:sectPr',ns))==2
    # Явные отступы нужны в таблицах результатов; титульная таблица — безрамочная.
    for table in doc.findall('.//w:tbl',ns)[1:]:
        for side in ['top','bottom','left','right']:
            assert table.find('.//w:tblCellMar/w:'+side,ns) is not None,side
reader=PdfReader(pdf)
assert len(reader.pages)>=7
pdf_text='\n'.join(p.extract_text() or '' for p in reader.pages)
for term in ['Рясной','МИК11','2602149','Контрольные вопросы']:
    assert term in pdf_text,term
with zipfile.ZipFile(ROOT/'artifacts/lab08_13_artifacts.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert '..' not in Path(name).parts
        assert (ROOT/name).is_file(),name
        assert z.read(name)==(ROOT/name).read_bytes(),name
env=json.loads((ROOT/'artifacts/environment.json').read_text())
for name,digest in env['source_hashes'].items():
    p=ROOT/'src/original'/name
    if not p.exists():p=ROOT/'src'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(status='passed',manifest_files=len(manifest),pdf_pages=len(reader.pages),
                     raw_records=report['raw_records'],data_scenes=report['regenerated_scenes'],
                     docx_comments=False,docx_tracked_changes=False,archive_exact=True),ensure_ascii=False,indent=2))
