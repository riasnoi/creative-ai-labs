"""Проверка сдаваемого комплекта ЛР06; запуск без генерации и GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as E
from pypdf import PdfReader
from verify_results import main as verify_results

R=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    verify_results()
    nb=json.loads((R/'src/lab06.ipynb').read_text())
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(code)==4
    for c in code:
        assert c['execution_count'] is not None and c['outputs']
        assert not any(o['output_type']=='error' for o in c['outputs'])
    for i,n in enumerate(['lab05_lib.py','lab06_lib.py']):
        source=''.join(code[i]['source']);file=(R/'src/original'/n).read_text()
        assert source.split('\n',1)[1].rstrip()==file.rstrip()
    assert ''.join(code[2]['source']).rstrip()==(R/'src/run_lab06.py').read_text().rstrip()
    printed='\n'.join(''.join(o.get('text',[])) for o in code[2]['outputs'])
    assert '2026-10-18T23:21:21.867318+03:00' in printed and 'ИТОГ' in printed
    for n in ['raw_metrics.json','summary.json','passport.json','expectations_before_run.json','environment.json','guidance_error.json','repeat_check.json']:
        assert json.loads((R/'artifacts'/n).read_text())
    with zipfile.ZipFile(R/'artifacts/lab06_13_artifacts.zip') as z:
        for name in z.namelist():
            if not name.endswith('/'):
                assert z.read(name)==(R/'artifacts'/name).read_bytes(),name
    path=R/'reports/Рясной_Вариант_13_Лаб06.docx'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert not any('comments' in n.lower() for n in z.namelist())
        xml=z.read('word/document.xml');root=E.fromstring(xml)
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        for tag in ['ins','del','commentRangeStart','commentReference','moveFrom','moveTo']:
            assert not root.findall('.//w:'+tag,ns),tag
        text=''.join(root.itertext())
        assert 'ЛАБОРАТОРНАЯ РАБОТА №6' in text
        assert 'Рясной Владимир Олегович' in text and '2602149' in text and 'МИК11' in text
        assert 'codex' not in text.lower()
    pdf=PdfReader(R/'reports/Рясной_Вариант_13_Лаб06.pdf')
    assert len(pdf.pages)==9
    txt='\n'.join(p.extract_text() for p in pdf.pages)
    assert '29,8085' in txt and '11,6051' in txt and '122,13' in txt
    assert '768×768' in txt and 'Источники' in txt
    assert 'codex' not in txt.lower()
    manifest=R/'MANIFEST.sha256'
    entries={}
    for line in manifest.read_text().splitlines():
        h,n=line.split('  ',1);entries[n]=h
        assert sha((R/n).read_bytes())==h,n
    actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts}
    assert actual==set(entries),(actual-set(entries),set(entries)-actual)
    print('PASS: notebook outputs/source, original export, 9-page PDF, DOCX structure, manifest coverage and hashes')
if __name__=='__main__':main()
