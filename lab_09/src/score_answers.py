"""Сводка K1–K4 по полным неизменённым ответам контекста 3."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "original"))
import lab09_llm as L

parser = argparse.ArgumentParser()
parser.add_argument("--answers", default="artifacts/answers")
args = parser.parse_args()
out = {}
for approach in ("zero", "few", "step"):
    paths = [Path(args.answers) / f"ctx3_{approach}_r{i}.txt" for i in (1, 2, 3)]
    scores = [L.score_file(path, 3) for path in paths]
    out[approach] = {k: statistics.fmean(s[k] for s in scores) for k in ("k1", "k2", "k3")}
    out[approach].update(k4=L.stability(paths), per_reply=scores)
print(json.dumps(out, ensure_ascii=False, indent=2))
