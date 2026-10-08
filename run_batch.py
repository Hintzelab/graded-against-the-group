#!/usr/bin/env python3
"""Batch grading: several student texts in one prompt, labelled with random anonymisation codes.

Designs (per assignment, 60 texts = 10 per intended grade):
  random20     random split into 3 classes of 20 (repeated --splits times)
  all60        all 60 texts in one prompt (repeated --splits times)
  profile_*    classes of 20 drawn with a discretised normal grade distribution (SD 1 step):
               weak (mu 1.5)  F3 E7 D7 C3 B0 A0 | middle (mu 2.5) F0 E3 D7 C7 B3 A0 | strong (mu 3.5) F0 E0 D3 C7 B7 A3
  single_code  one text per prompt, labelled with a code (reference for the batch conditions)

Each batch is graded with and without criteria using the identical composition, order and codes.
Intended grades are used only to compose the profile classes, never as an answer key.

    python3 run_batch.py --tag pilot --assignments G6-1 G9-1 --designs random20 all60 profiles --splits 1
    python3 run_batch.py --tag pilot --assignments G6-1 G9-1 --designs single_code --reps 3 --num-ctx 8192

One JSON line per call in results/batch_<tag>_<model>.jsonl, with every student's grade, position and code,
plus completeness diagnostics (missing, duplicate or unknown codes, order changes). Resumable.
"""
import argparse
import json
import queue
import random
import re
import sys
import threading
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from prompts import GRADES, build_messages, output_schema, prompt_hash
from run_ollama import HEADERS, call_ollama, job_seed, model_digest

ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "v1-codes"
PROFILES = {
    "profile_weak":   {"F": 3, "E": 7, "D": 7, "C": 3, "B": 0, "A": 0},
    "profile_middle": {"F": 0, "E": 3, "D": 7, "C": 7, "B": 3, "A": 0},
    "profile_strong": {"F": 0, "E": 0, "D": 3, "C": 7, "B": 7, "A": 3},
}


def make_design(assignment_ids, designs, splits, reps, texts, key, seed, version=2):
    """Deterministic list of batches: dict(batch_id, assignment_id, design, text_ids, codes).

    version 2 (default): every class draw has its own RNG seeded by (seed, assignment, design, split), so any
    subset of assignments/designs reproduces exactly. Designs: random<k> (k divides 60: 5, 10, 20, 30, 60),
    profile_weak/middle/strong (splits = number of classes per profile), single_code (reps per text).
    version 1: the pilot's original single-RNG draw (kept to reproduce the pilot classes).
    """
    if version == 1:
        return _make_design_v1(assignment_ids, designs, splits, reps, texts, key, seed)
    batches = []
    for aid in assignment_ids:
        ids = [t["text_id"] for t in texts[aid]]
        by_grade = {g: [t for t in ids if key[t]["intended_grade"] == g] for g in GRADES}
        for d in designs:
            m = re.fullmatch(r"random(\d+)", d)
            if m and 60 % int(m.group(1)) == 0:
                k = int(m.group(1))
                for s in range(1, splits + 1):
                    rng = random.Random(f"{seed}|{aid}|{d}|{s}")
                    order = ids[:]
                    rng.shuffle(order)
                    for c in range(60 // k):
                        batches.append({"batch_id": f"{aid}|{d}|s{s}|c{c + 1}", "design": d,
                                        "text_ids": order[c * k:(c + 1) * k]})
            elif d in PROFILES:
                for s in range(1, splits + 1):
                    rng = random.Random(f"{seed}|{aid}|{d}|{s}")
                    tids = [t for g, n in PROFILES[d].items() for t in rng.sample(by_grade[g], n)]
                    rng.shuffle(tids)
                    batches.append({"batch_id": f"{aid}|{d}|s{s}", "design": d, "text_ids": tids})
            elif d == "single_code":
                for t in ids:
                    for r in range(1, reps + 1):
                        batches.append({"batch_id": f"{aid}|single_code|{t}|r{r}", "design": d, "text_ids": [t]})
            else:
                sys.exit(f"unknown design {d}")
    for b in batches:
        b["assignment_id"] = b["batch_id"].split("|")[0]
        rng = random.Random(f"{seed}|codes|{b['batch_id']}")
        b["codes"] = [f"Elev {c}" for c in rng.sample(range(1000, 10000), len(b["text_ids"]))]
    return batches


def _make_design_v1(assignment_ids, designs, splits, reps, texts, key, seed):
    rng = random.Random(seed)
    batches = []

    def codes(n):
        return [f"Elev {c}" for c in rng.sample(range(1000, 10000), n)]

    for aid in assignment_ids:
        ids = [t["text_id"] for t in texts[aid]]
        by_grade = {g: [t for t in ids if key[t]["intended_grade"] == g] for g in GRADES}
        for d in designs:
            if d == "random20":
                for s in range(1, splits + 1):
                    order = ids[:]
                    rng.shuffle(order)
                    for c in range(3):
                        tids = order[c * 20:(c + 1) * 20]
                        batches.append({"batch_id": f"{aid}|random20|s{s}|c{c + 1}", "design": d, "text_ids": tids})
            elif d == "all60":
                for s in range(1, splits + 1):
                    order = ids[:]
                    rng.shuffle(order)
                    batches.append({"batch_id": f"{aid}|all60|s{s}", "design": d, "text_ids": order})
            elif d in PROFILES:
                for s in range(1, splits + 1):
                    tids = [t for g, n in PROFILES[d].items() for t in rng.sample(by_grade[g], n)]
                    rng.shuffle(tids)
                    batches.append({"batch_id": f"{aid}|{d}|s{s}", "design": d, "text_ids": tids})
            elif d == "single_code":
                for t in ids:
                    for r in range(1, reps + 1):
                        batches.append({"batch_id": f"{aid}|single_code|{t}|r{r}", "design": d, "text_ids": [t]})
            else:
                sys.exit(f"unknown design {d}")
        for b in batches:
            if "codes" not in b:
                b["assignment_id"] = aid
                b["codes"] = codes(len(b["text_ids"]))
    return batches


def parse_batch(content, codes):
    """Match returned items to codes. Returns (per-code results, diagnostics) or (None, reason)."""
    try:
        items = json.loads(content)["bedomningar"]
    except (json.JSONDecodeError, KeyError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.S)
        try:
            items = json.loads(m.group())["bedomningar"] if m else None
        except (json.JSONDecodeError, KeyError, TypeError):
            items = None
    if not isinstance(items, list):
        return None, "unparseable"
    results, returned = {}, []
    for it in items:
        if not isinstance(it, dict):
            continue
        label = str(it.get("elev", "")).strip()
        digits = re.findall(r"\d{4}", label)
        label = f"Elev {digits[0]}" if digits else label
        returned.append(label)
        if label in codes and label not in results and it.get("betyg") in GRADES:
            results[label] = {"grade": it["betyg"], "motivering": it.get("motivering"), "aterkoppling": it.get("aterkoppling")}
    known = [c for c in returned if c in codes]
    first_seen = list(dict.fromkeys(known))
    diag = {
        "n_expected": len(codes), "n_returned": len(returned), "n_graded": len(results),
        "missing": [c for c in codes if c not in results],
        "duplicates": sorted({c for c in known if known.count(c) > 1}),
        "unknown": [c for c in returned if c not in codes],
        "order_preserved": first_seen == [c for c in codes if c in first_seen],
    }
    return results, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of this run, e.g. pilot")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--hosts", nargs="+", default=["http://localhost:11434"])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--api-key-file", default="~/.ollama_gateway_key")
    ap.add_argument("--assignments", nargs="+", required=True)
    ap.add_argument("--designs", nargs="+", required=True, help="random20 all60 profiles profile_weak ... single_code")
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3, help="repetitions for single_code")
    ap.add_argument("--criteria", choices=["with", "without", "both"], default="both")
    ap.add_argument("--think", choices=["on", "off", "low", "medium", "high"], default="off")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--design-version", type=int, default=2, help="1 = reproduce the pilot's class draws")
    args = ap.parse_args()

    HEADERS["Authorization"] = "Bearer " + Path(args.api_key_file).expanduser().read_text().strip()
    a = json.load(open(ROOT / "assignments.json", encoding="utf-8"))
    scale, assignments = a["meta"]["betygsskala"], {x["id"]: x for x in a["assignments"]}
    texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
    text_of = {t["text_id"]: t["text"] for v in texts.values() for t in v}
    key = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
    designs = [d for x in args.designs for d in (PROFILES if x == "profiles" else [x])]
    think = {"on": True, "off": False}.get(args.think, args.think)
    crit_conds = {"with": [True], "without": [False], "both": [True, False]}[args.criteria]

    batches = make_design(args.assignments, designs, args.splits, args.reps, texts, key, args.seed, args.design_version)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out = ROOT / "results" / f"batch_{args.tag}_{safe_model}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done, tried = set(), {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["job_key"])
            else:
                tried[r["job_key"]] = tried.get(r["job_key"], 0) + r.get("attempts", 0)

    jobs = [(f"{args.tag}|{args.model}|{PROMPT_VERSION}|{b['batch_id']}|crit={int(c)}", b, c)
            for b in batches for c in crit_conds]
    jobs = [j for j in jobs if j[0] not in done]
    jobs.sort(key=lambda j: -len(j[1]["text_ids"]))  # longest prompts first so they don't straggle at the end

    digest = model_digest(args.hosts[0], args.model)
    if not digest:
        sys.exit(f"model {args.model} not found")
    print(f"{args.model} ({digest[:12]}): {len(jobs)} calls to run ({len(done)} done) -> {out}", flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    f = open(out, "a", encoding="utf-8")
    counter = {"n": 0}
    t_start = time.time()

    def request(host, messages, schema, seed, num_predict):
        waited, pause = 0, 5
        while True:
            try:
                return call_ollama(host, args.model, messages, schema, seed, None, args.num_ctx, think,
                                   args.timeout, num_predict), None
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429:
                    return None, f"HTTPError {e.code}"
                err = f"HTTPError {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                err = f"{type(e).__name__}: {e}"
            if waited >= 1800:
                return None, err
            time.sleep(pause)
            waited += pause
            pause = min(pause * 2, 60)

    def worker(host):
        while True:
            try:
                jk, b, crit = q.get_nowait()
            except queue.Empty:
                return
            tids, codes = b["text_ids"], b["codes"]
            messages, _ = build_messages(assignments[b["assignment_id"]], scale, [text_of[t] for t in tids],
                                         criteria=crit, labels=codes)
            num_predict = 500 + 450 * len(tids)
            t0 = time.time()
            resp, results, diag, err, attempts = None, None, None, None, 0
            for attempt in range(3):
                attempts = attempt + 1
                seed = job_seed(jk) + tried.get(jk, 0) + attempt
                resp, err = request(host, messages, output_schema(len(tids)), seed, num_predict)
                if resp is None:
                    break
                results, diag = parse_batch(resp.get("message", {}).get("content", ""), codes)
                err = diag if results is None else None
                if results is not None:
                    break
            prompt_tokens = (resp or {}).get("prompt_eval_count")
            rec = {
                "job_key": jk, "tag": args.tag, "model": args.model, "model_digest": digest,
                "prompt_version": PROMPT_VERSION, "batch_id": b["batch_id"], "assignment_id": b["assignment_id"],
                "design": b["design"], "batch_size": len(tids), "criteria": crit, "think": think,
                "num_ctx": args.num_ctx, "num_predict": num_predict, "seed": seed, "attempts": attempts,
                "prompt_hash": prompt_hash(messages), "error": err,
                "context_truncated": bool(prompt_tokens and prompt_tokens >= args.num_ctx - 10),
                "diagnostics": diag if results is not None else None,
                "students": [
                    {"position": i + 1, "code": c, "text_id": t, **(results or {}).get(c, {"grade": None})}
                    for i, (c, t) in enumerate(zip(codes, tids))
                ],
                "raw": (resp or {}).get("message", {}).get("content"),
                "prompt_tokens": prompt_tokens, "output_tokens": (resp or {}).get("eval_count"),
                "done_reason": (resp or {}).get("done_reason"),
                "seconds": round(time.time() - t0, 1), "host": host,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                counter["n"] += 1
                d = rec["diagnostics"] or {}
                print(f"  {counter['n']}/{len(jobs)} {b['batch_id']} crit={int(crit)}: "
                      f"{d.get('n_graded', 0)}/{len(tids)} graded, {rec['seconds']:.0f}s, "
                      f"in {prompt_tokens} out {rec['output_tokens']} tok"
                      + (f"  ERROR {err}" if err else "")
                      + ("  CONTEXT TRUNCATED" if rec["context_truncated"] else "")
                      + (f"  missing {len(d['missing'])}" if d.get("missing") else "")
                      + ("  order changed" if d and not d.get("order_preserved", True) else ""), flush=True)

    threads = [threading.Thread(target=worker, args=(h,), daemon=True) for h in args.hosts for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    f.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
