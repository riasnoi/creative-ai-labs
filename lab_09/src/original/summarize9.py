# Сводка результатов ЛР 9 (часть 1): читает res9.jsonl, пишет summary9.txt — по строке на вариант, значения округлены.
# Формат строки: №|контекст|фактор|метрика:A/s/B/d/q/флаг ...   (q = |d|/s, флаг 1 — |d| > 2s)
import json

names = {"clip_target": "ct", "clip_subject": "cs", "edge": "ed", "contrast": "co"}
lines, sig = [], {m: 0 for m in names}
rows = [json.loads(l) for l in open("res9.jsonl", encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["variant"])
for r in rows:
    parts = []
    for m, a, s, b, d, q, f in r["rows"]:
        nd = 4 if m in ("edge", "contrast") else 2
        parts.append(f"{names[m]}:{a:.{nd}f}/{s:.{nd}f}/{b:.{nd}f}/{d:+.{nd}f}/{q:.1f}/{int(f)}")
        sig[m] += int(f)
    lines.append(f"{r['variant']}|{r['ctx']}|{r['factor'][:4]}|" + " ".join(parts) + f"|t{r['seconds_mean']:.0f}")
v1 = [r for r in rows if r["variant"] == 1]
if v1:
    a, b = v1[0]["per_seed_target"]
    lines.append("V1 A=" + ",".join(f"{x:.2f}" for x in a) + " B=" + ",".join(f"{x:.2f}" for x in b))
lines.append("SIG " + " ".join(f"{names[m]}={v}" for m, v in sig.items()) + f" n={len(rows)}")
open("summary9.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("готово", len(rows))
