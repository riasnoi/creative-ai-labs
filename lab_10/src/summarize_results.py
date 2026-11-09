import json,statistics,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; p=R/'artifacts/main'
r=json.loads((p/'results.json').read_text()); h=json.loads((R/'docs/hypotheses_before_run.json').read_text())
summary=[]
for row in r['rows']:
 m,a,s,b,d,ratio,flag=row
 rel=d/a
 practical=abs(d)>h['practical_thresholds'][m] if m.startswith('clip') else (abs(rel)>.1 if m!='seconds' else max(b/a,a/b)>1.5)
 klass='2s и практический допуск' if flag and practical else ('только 2s' if flag else ('только практический допуск' if practical else 'ни один порог'))
 summary.append(dict(metric=m,A=a,s_A=s,B=b,delta=d,ratio=ratio,rule_2s=flag,relative_change=rel,practical=practical,classification=klass))
paired=[]
for i,seed in enumerate(range(101,106)):
 a=r['per_seed']['turbo'][i];b=r['per_seed']['lcm'][i]
 paired.append(dict(seed=seed,clip_brief_delta=b['clip_brief']-a['clip_brief'],seconds_ratio=b['seconds']/a['seconds']))
out=dict(source_completed_at=r['completed_at'],summary=summary,paired=paired,
     time_ratio=summary[-1]['B']/summary[-1]['A'],p_extreme_5_pairs=2*(.5**5),
     formal_test_performed=False,thresholds=h['practical_thresholds'])
(R/'artifacts/analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
