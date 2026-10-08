#!/usr/bin/env python3
"""Progress of the main experiment: completed calls/conversations per block, errors, lost students, stalls.

    python3 status.py [model]      (default gemma4:26b)
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from run_batch import PROFILES, make_design

ROOT = Path(__file__).resolve().parent
model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:26b"
safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
key = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
ALL = list(texts)
SEED = 20260926

blocks = [  # name, file prefix, designs, splits, reps
    ("A1 independent (batch fmt)", "batch", ["single_code"], 1, 3),
    ("B  batch random 5/10/20/30", "batch", ["random5", "random10", "random20", "random30"], 3, 3),
    ("Db batch profiles", "batch", list(PROFILES), 6, 3),
    ("A2 independent (Gem fmt)", "seq", ["single_code"], 1, 3),
    ("C  sequential random 5/10/20/30", "seq", ["random5", "random10", "random20", "random30"], 3, 3),
    ("Ds sequential profiles", "seq", list(PROFILES), 6, 3),
]
files = {p: ROOT / "results" / f"{p}_main_{safe}.jsonl" for p in ("batch", "seq")}
recs = {p: [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else [] for p, f in files.items()}

print(f"{model} - main experiment")
total_done = total_all = 0
for name, prefix, designs, splits, reps in blocks:
    ids = {b["batch_id"] for b in make_design(ALL, designs, splits, reps, texts, key, SEED)}
    expected = 2 * len(ids)  # with and without criteria
    ok = [r for r in recs[prefix] if r["batch_id"] in ids and r.get("error") is None]
    done = len({r["job_key"] for r in ok})
    total_done += done
    total_all += expected
    lost = 0
    for r in ok:
        if prefix == "batch":
            lost += len((r.get("diagnostics") or {}).get("missing", []))
        else:
            lost += sum(1 for t in r["turns"] if not t.get("grade"))
    failed = sum(1 for r in recs[prefix] if r["batch_id"] in ids and r.get("error") is not None)
    bar = "#" * int(20 * done / expected) + "." * (20 - int(20 * done / expected))
    print(f"  {name:<31} [{bar}] {done:>5}/{expected:<5}" + (f"  lost students {lost}" if lost else "")
          + (f"  failed calls {failed}" if failed else ""))
print(f"  overall {total_done}/{total_all} ({total_done / total_all:.0%})")

last = [f.stat().st_mtime for f in files.values() if f.exists()]
if last:
    age = time.time() - max(last)
    print(f"  last result written {datetime.fromtimestamp(max(last)):%a %H:%M:%S} ({age / 60:.1f} min ago)"
          + ("  <-- possible stall" if age > 600 else ""))
