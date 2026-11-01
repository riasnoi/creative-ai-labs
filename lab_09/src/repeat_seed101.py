"""Повтор двух изображений без изменения первичного запуска."""
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "original"))
import lab09_lib as L

out = Path("artifacts/repeat_seed101")
out.mkdir(parents=True, exist_ok=False)
stamp = lambda: datetime.now(timezone(timedelta(hours=3))).isoformat()
result = {"started_at": stamp(), "seed": 101, "images": {}}
pipe = L.load_pipe()
for mode, prompt in [("A", L.prompt_a(2)), ("B", L.prompt_b(2, 0))]:
    image = L.generate(pipe, prompt, 101)
    path = out / f"{mode}_101.png"
    image.save(path)
    result["images"][mode] = hashlib.sha256(path.read_bytes()).hexdigest()
result["completed_at"] = stamp()
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
