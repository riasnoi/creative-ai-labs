"""Проверка комплекта ЛР05: байты, происхождение, результаты и выполненный блокнот."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from verify_results import verify

ROOT=Path(__file__).resolve().parents[1]
def main():
    print(json.dumps(verify(ROOT),ensure_ascii=False,indent=2))
    expected={}
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert name not in expected and not Path(name).is_absolute() and '..' not in Path(name).parts
        expected[name]=digest
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts}
    assert actual==set(expected),(actual-set(expected),set(expected)-actual)
    provenance=json.loads((ROOT/'artifacts/provenance.json').read_text())
    for f in provenance['files']:
        b=(ROOT/f['path']).read_bytes();assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
    env=json.loads((ROOT/'artifacts/environment.json').read_text())
    assert env['library_sha256']==hashlib.sha256((ROOT/'src/original/lab05_lib.py').read_bytes()).hexdigest()
    nb=json.loads((ROOT/'src/lab05.ipynb').read_text()); cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==4 and [c['execution_count'] for c in cells]==[1,2,3,4]
    assert all(c['outputs'] and all(o['output_type']!='error' for o in c['outputs']) for c in cells)
    def source(c):return ''.join(c['source'])
    assert source(cells[1]).split('\n',1)[1]==(ROOT/'src/original/lab05_lib.py').read_text()
    assert source(cells[2])==(ROOT/'src/run_lab05.py').read_text()
    strings=[]
    for o in cells[2]['outputs']:
        if 'text' in o:strings.append(''.join(o['text']))
    stdout=''.join(strings)
    metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())
    assert metrics['repeat']['sha_a'] in stdout and str(metrics['shared_generator']['psnr']) in stdout
    passport=json.loads((ROOT/'artifacts/passport.json').read_text())
    assert passport['environment']==env and passport['results']==metrics
    with zipfile.ZipFile(ROOT/'artifacts/lab05_13_artifacts.zip') as z:
        for name in z.namelist():
            if not name.endswith('/'): assert z.read(name)==(ROOT/'artifacts'/name).read_bytes(),name
    with zipfile.ZipFile(ROOT/'reports/Рясной_Вариант_13_Лаб05.docx') as z:
        assert z.testzip() is None
        assert not any('comments' in p.lower() for p in z.namelist())
        tree=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text=''.join(n.text or '' for n in tree.findall('.//w:t',ns))
        for word in ['№5','Вариант 13','1013','11,564944',metrics['repeat']['sha_a'],metrics['shared_generator']['sha_b']]:assert word in text,word
        for tag in ['ins','del','commentRangeStart','commentReference']:assert tree.find('.//w:'+tag,ns) is None
        assert len(tree.findall('.//w:sectPr',ns))==2
    assert (ROOT/'reports/Рясной_Вариант_13_Лаб05.pdf').read_bytes().startswith(b'%PDF')
    print('PASS: manifest, provenance, notebook, outputs, metrics, archive, passport, DOCX/PDF')

if __name__=='__main__':main()
