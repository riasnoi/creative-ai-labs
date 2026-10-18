"""Проверить комплект ЛР07 без GPU и повторного обучения."""
import hashlib
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as E
from pypdf import PdfReader
from verify_results import check

R=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    result=check()
    assert result==json.loads((R/'artifacts/independent_verification.json').read_text())
    nb=json.loads((R/'src/lab07.ipynb').read_text())
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(code)==4
    for i,c in enumerate(code):
        assert c['execution_count']==i+1 and c['outputs']
        assert not any(o['output_type']=='error' for o in c['outputs'])
    for i,n in enumerate(['lab07_lib.py','lab07_demo.py']):
        assert ''.join(code[i]['source']).split('\n',1)[1].rstrip()==(R/'src/original'/n).read_text().rstrip()
    assert ''.join(code[2]['source']).rstrip()==(R/'src/run_lab07.py').read_text().rstrip()
    text='\n'.join(''.join(o.get('text',[])) for c in code for o in c['outputs'])
    assert '2026-10-18T23:40:41.025725+03:00' in text and 'ИТОГ' in text and 'ГОТОВО' in text
    with zipfile.ZipFile(R/'artifacts/lab07_13_artifacts.zip') as z:
        for n in z.namelist():
            if not n.endswith('/'): assert z.read(n)==(R/'artifacts'/n).read_bytes(),n
    for mode,count in [('A',101035),('B',88741)]:
        with zipfile.ZipFile(R/f'artifacts/model_{mode}_seed101.pt') as z:
            assert sum(z.getinfo(n).file_size for n in z.namelist() if '/data/' in n)==count*4
    with zipfile.ZipFile(R/'reports/Рясной_Вариант_13_Лаб07.docx') as z:
        assert z.testzip() is None
        assert not any('comments' in n.lower() for n in z.namelist())
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        root=E.fromstring(z.read('word/document.xml'))
        for t in ['ins','del','moveFrom','moveTo','commentRangeStart','commentReference']:
            assert not root.findall('.//w:'+t,ns)
        assert len(root.findall('.//w:sectPr',ns))==2
        text=''.join(root.itertext())
        for required in ['ЛАБОРАТОРНАЯ РАБОТА №7','Рясной Владимир Олегович','МИК11','2602149','23,0824','21,4431','16,7632']:
            assert required in text,required
    p=PdfReader(R/'reports/Рясной_Вариант_13_Лаб07.pdf');assert len(p.pages)==9
    pt=' '.join(('\n'.join(page.extract_text() for page in p.pages)).split())
    for v in ['23,0824','21,4431','16,7632','Источники','Нужны исходный код','Число факторов']:assert v in pt,v
    manifest=R/'MANIFEST.sha256';entries={}
    for line in manifest.read_text().splitlines():
        h,n=line.split('  ',1);entries[n]=h;assert sha((R/n).read_bytes())==h,n
    actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts}
    assert actual==set(entries),(actual-set(entries),set(entries)-actual)
    print('PASS: notebook sources/outputs, raw ZIP, model parameter counts, 9-page PDF, DOCX structure and manifest')
if __name__=='__main__':main()
