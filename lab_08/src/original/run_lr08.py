# Запуск шести вариантов одного набора сцен (для проверки выполнимости в Engee): python run_lr08.py <номер набора 0..4>
# Возобновляемый: уже посчитанные варианты (строки в res_c<номер>.jsonl) пропускаются.
import sys, os, json, time
import lab07_lib as L
import lab08_lib as V
import torch

torch.set_num_threads(1)   # пять процессов по одному потоку быстрее, чем один процесс на четырёх потоках
c = int(sys.argv[1])
path = f"res_c{c}.jsonl"
done = set()
if os.path.exists(path):
    done = {json.loads(line)["variant"] for line in open(path, encoding="utf-8") if line.strip()}
with open(path, "a", encoding="utf-8") as out:
    for n in range(6 * c + 1, 6 * c + 7):
        if n in done:
            continue
        t = time.time()
        r = V.variant_summary(n)
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
        print(n, round(time.time() - t), flush=True)
