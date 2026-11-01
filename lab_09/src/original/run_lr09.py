# Запуск всех 30 вариантов ЛР 9 (часть 1) одним процессом; возобновляемый: уже посчитанное пропускается.
# Режим A каждого контекста (только предмет) не зависит от фактора, поэтому считается один раз на контекст.
# Использование:  python run_lr09.py        Результаты: res9.jsonl (по строке на вариант), a9_c<N>.json (режим A), img9/*.jpg (seed 101), log9.txt
import json, os, sys, time
import lab09_lib as L

OUT = "res9.jsonl"
os.makedirs("img9", exist_ok=True)


def log(*a):
    with open("log9.txt", "a", encoding="utf-8") as f:
        print(time.strftime("%H:%M:%S"), *a, file=f, flush=True)


def done():
    if not os.path.exists(OUT):
        return set()
    return {json.loads(l)["variant"] for l in open(OUT, encoding="utf-8") if l.strip()}


def main():
    t_all = time.time()
    log("старт: загрузка модели"); pipe = L.load_pipe(); clip = L.load_clip(); log("модель загружена", round(time.time() - t_all, 1), "с")
    L.generate(pipe, L.prompt_a(0), 0)
    for c in range(5):
        todo = [n for n in range(c * 6 + 1, c * 6 + 7) if n not in done()]
        if not todo:
            continue
        fa = f"a9_c{c + 1}.json"
        if os.path.exists(fa):
            a = json.load(open(fa, encoding="utf-8"))
        else:
            a = []
            for s in L.SEEDS:
                r, img = L.measure(pipe, clip, c, L.prompt_a(c), s)
                a.append(r)
                if s == 101:
                    img.save(f"img9/c{c + 1}_A.jpg", quality=88)
            json.dump(a, open(fa, "w", encoding="utf-8"))
            log("контекст", c + 1, "режим A готов")
        for n in todo:
            k = (n - 1) % 6
            b = []
            for s in L.SEEDS:
                r, img = L.measure(pipe, clip, c, L.prompt_b(c, k), s)
                b.append(r)
                if s == 101:
                    img.save(f"img9/v{n:02d}_B.jpg", quality=88)
            rec = dict(variant=n, ctx=c + 1, factor=L.FACTORS[k], prompt_b=L.prompt_b(c, k), rows=L.compare(a, b, k),
                       per_seed_target=[[x["sims"][1 + k] for x in a], [x["sims"][1 + k] for x in b]],
                       seconds_mean=sum(x["seconds"] for x in b) / len(b))
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("вариант", n, "готов", [round(r[4], 2) for r in rec["rows"]])
    log("конец, всего", round(time.time() - t_all, 1), "с")


if __name__ == "__main__":
    main()
