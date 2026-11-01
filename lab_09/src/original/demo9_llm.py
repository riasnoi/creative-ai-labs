# Демонстрация ЛР 9, часть 2: проверка ответов языковой модели по трём приёмам (контекст 1) и показ намеренной ошибки.
# Файлы ответов: ctx1_<zero|few|step>_r<1..3>.txt (лежат рядом или в папке answers/). Результат: res9_llm.json и печать сводки.
import json, os
import numpy as np
import lab09_llm as L

D = "answers/" if os.path.isdir("answers") else ""
CTX = 1
out = {}
for a in ("zero", "few", "step"):
    paths = [f"{D}ctx{CTX}_{a}_r{i}.txt" for i in (1, 2, 3)]
    rs = [L.score_file(p, CTX) for p in paths]
    out[a] = dict(k1=float(np.mean([r["k1"] for r in rs])), k2=float(np.mean([r["k2"] for r in rs])),
                  k3=float(np.mean([r["k3"] for r in rs])), k4=L.stability(paths),
                  k1_each=[r["k1"] for r in rs], k2_each=[r["k2"] for r in rs], errors=[r["errors"] for r in rs])
err = L.score_file(f"{D}demo_error_old_prompt.txt", CTX)
out["error_demo"] = dict(k1=err["k1"], k2=err["k2"], k3=err["k3"], errors=err["errors"])
json.dump(out, open("res9_llm.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for a in ("zero", "few", "step"):
    r = out[a]
    print(f"{a}: K1={r['k1']:.2f} K2={r['k2']:.3f} K3={r['k3']:.2f} K4={r['k4']:.3f}")
print("error_demo:", err["k1"], round(err["k2"], 3), err["k3"], err["errors"])
