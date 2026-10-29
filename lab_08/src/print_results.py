"""Краткий вывод сохранённых результатов ЛР08 без повторного обучения."""
from pathlib import Path
import json

ART=Path('artifacts')
if not ART.is_dir():
    ART=Path(__file__).resolve().parents[1]/'artifacts'
read=lambda name:json.loads((ART/name).read_text(encoding='utf-8'))
s=read('summary.json');env=read('environment.json');p=read('passport.json')
print('LR08 | Variant 13 | beta 1 -> 8')
print('Python '+env['python'].split()[0]+' | torch '+env['torch'])
print(f"{env['device'].upper()} | {env['threads']} threads | {p['epochs']} epochs | warmup {p['warmup_epochs']}")
print('')
print('METRIC          A        B       DELTA')
for m,a,sd,b,delta,ratio,sig in s['rows']:
    print(f'{m:8s} {a:8.4f} {b:8.4f} {delta:+9.4f}')
print('')
print('METRIC        s_A    |delta|/s    >2s')
for m,a,sd,b,delta,ratio,sig in s['rows']:
    print(f'{m:8s} {sd:8.4f} {ratio:10.3f}    {str(sig):5s}'.rstrip())
print('')
print('Tolerance: PSNR +/-1 dB; R2 +/-0.05')
labels={'улучшение':'better','ухудшение':'worse','нейтрально':'neutral'}
print('; '.join(f"{r['metric']}: {labels[r['classification']]}" for r in read('classification.json')))
diag=read('diagnostic.json')
active=[next(r['active_axes'] for r in diag if r['seed']==101 and r['mode']==mode) for mode in ['A','B']]
print(f'Active axes seed101: A={active[0]}, B={active[1]}')
print('Seeds: 101 102 103 104 105; data seed: 7')
max_repeat=max(abs(v) for r in read('repeat_check.json') for v in r['differences'].values())
print(f'Repeat seed101 max difference: {max_repeat:.6f}')
print('Finished MSK: '+p['finished_at'][:19])
