"""Независимая проверка первичных PNG, сводок, ответов и происхождения кода."""
import hashlib
import itertools
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src/original"))
import lab09_llm as L


def read(path):
    return json.loads((BASE / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read("artifacts/preregistration.json")
    passport = read("artifacts/part1/passport.json")
    results = read("artifacts/part1/results.json")
    assert plan["variant"] == results["variant"] == 13
    assert passport["prompt_b"] == passport["prompt_a"] + ", isometric 3D render"
    assert passport["seeds"] == list(range(101, 106))
    assert datetime.fromisoformat(plan["recorded_at"]) < datetime.fromisoformat(passport["started_at"])
    assert (passport["steps"], passport["guidance_scale"], passport["size"]) == (1, 0.0, 512)
    for mode, rows in results["per_seed"].items():
        assert [r["seed"] for r in rows] == passport["seeds"]
        for row in rows:
            path = BASE / "artifacts/part1" / row["image"]
            assert sha(path) == row["sha256"]
            image = Image.open(path).convert("RGB")
            assert image.size == (512, 512)
            x = np.asarray(image, dtype=np.float64) / 255
            y = 0.299*x[..., 0] + 0.587*x[..., 1] + 0.114*x[..., 2]
            p = np.pad(y, 1, mode="symmetric")
            gx = (p[:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:]
                  - p[:-2, :-2] - 2*p[1:-1, :-2] - p[2:, :-2])
            gy = (p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:]
                  - p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:])
            edge = float(np.hypot(gx, gy).mean()/8)
            assert math.isclose(edge, row["edge"], abs_tol=1e-14)
            assert math.isclose(float(y.std()), row["contrast"], abs_tol=1e-14)
    for name, avg_a, sd, avg_b, delta, ratio, flag in results["rows"]:
        pick = (lambda r: r["sims"][1]) if name == "clip_target" else (
            (lambda r: r["sims"][0]) if name == "clip_subject" else lambda r: r[name])
        a, b = [[pick(r) for r in results["per_seed"][m]] for m in ["A", "B"]]
        values = [statistics.fmean(a), statistics.stdev(a), statistics.fmean(b)]
        values += [values[2]-values[0]]
        values += [abs(values[3])/values[1]]
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(values, [avg_a, sd, avg_b, delta, ratio]))
        assert flag == (abs(delta) > 2*sd)
    repeat = read("artifacts/repeat_seed101/result.json")
    for mode, h in repeat["images"].items():
        assert sha(BASE / f"artifacts/repeat_seed101/{mode}_101.png") == h
        assert sha(BASE / f"artifacts/part1/{mode}_101.png") == h
    summary = read("artifacts/llm_results.json")
    for approach in ["zero", "few", "step"]:
        paths = [BASE / f"artifacts/answers/ctx3_{approach}_r{i}.txt" for i in [1, 2, 3]]
        scores = [L.score_file(p, 3) for p in paths]
        assert scores == summary[approach]["per_reply"]
        arrays = [L.extract_array(p.read_text())[0] for p in paths]
        comparisons = [a[i][f] == b[i][f] for a, b in itertools.combinations(arrays, 2)
                       for i in range(5) for f in ["style", "composition"]]
        assert len(comparisons) == 30
        assert sum(comparisons)/30 == summary[approach]["k4"]
        for k in ["k1", "k2", "k3"]:
            assert statistics.fmean(s[k] for s in scores) == summary[approach][k]
    for source in read("docs/SOURCE_PROVENANCE.json")["provided_sources"]:
        assert sha(BASE / source["file"]) == source["sha256"]
    for approach in ["zero", "few", "step"]:
        assert (BASE / f"artifacts/prompts/ctx3_{approach}.txt").read_text() == L.build_prompt(approach, 3) + "\n"
    manifest = BASE / "MANIFEST.sha256"
    if manifest.exists():
        expected = set()
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert sha(BASE / name) == digest, name
            expected.add(name)
        actual = {str(p.relative_to(BASE)) for p in BASE.rglob("*") if p.is_file()
                  and p != manifest and "__pycache__" not in p.parts and p.name != ".DS_Store"}
        assert actual == expected, actual ^ expected
    print("PASS: 10 primary PNG, 2 repeated PNG, 4 metrics, 9 replies, 90 K4 comparisons, source bytes and prompts")


if __name__ == "__main__":
    main()
