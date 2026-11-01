"""Вариант 13: сохранение каждого изображения и первичных измерений."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "original"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/part1")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise SystemExit("Результат уже существует; используйте другую папку для повтора")
    import lab09_lib as L
    import torch, diffusers, transformers, numpy, scipy
    stamp = lambda: datetime.now(timezone(timedelta(hours=3))).isoformat()
    passport = dict(started_at=stamp(), python=platform.python_version(),
                    torch=torch.__version__, diffusers=diffusers.__version__,
                    transformers=transformers.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
                    device="cpu", dtype="float32", threads=torch.get_num_threads(),
                    model=L.MODEL_ID, revision=L.REVISION, clip=L.CLIP_ID,
                    clip_revision=L.CLIP_REVISION, seeds=list(L.SEEDS),
                    steps=L.STEPS, guidance_scale=L.GUIDANCE, size=L.SIZE,
                    prompt_a=L.prompt_a(2), prompt_b=L.prompt_b(2, 0), target=L.ELEMENTS[2][0][1])
    (out / "passport.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2) + "\n")
    print("START", passport["started_at"], flush=True)
    pipe, clip = L.load_pipe(), L.load_clip()
    L.generate(pipe, L.prompt_a(2), 0)
    records = {"A": [], "B": []}
    for mode, prompt in [("A", L.prompt_a(2)), ("B", L.prompt_b(2, 0))]:
        for seed in L.SEEDS:
            row, img = L.measure(pipe, clip, 2, prompt, seed)
            name = f"{mode}_{seed}.png"
            img.save(out / name)
            row.update(seed=seed, image=name, sha256=hashlib.sha256((out / name).read_bytes()).hexdigest())
            records[mode].append(row)
            with (out / "measurements.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(mode=mode, measured_at=stamp(), **row)) + "\n")
            print(mode, seed, row, flush=True)
    result = dict(variant=13, context=3, factor="style", completed_at=stamp(),
                  per_seed=records, rows=L.compare(records["A"], records["B"], 0))
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("COMPLETE", result["completed_at"], flush=True)


if __name__ == "__main__":
    main()
