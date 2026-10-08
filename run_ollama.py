#!/usr/bin/env python3
"""Anonymous grading round against a local Ollama server.

Every text is graded on its own (batch size 1), with no name and no background,
with and/or without the task criteria, repeated --reps times to measure run-to-run noise.

    python3 run_ollama.py --model gemma4:26b                        # full round: 720 texts x 2 criteria x 3 reps
    python3 run_ollama.py --model gemma4:26b --pilot 2              # 2 texts per grade per assignment, 1 rep
    python3 run_ollama.py --model qwen3:14b --criteria with --reps 5
    python3 run_ollama.py --model gemma4:26b --hosts http://localhost:11434 --workers 8 \
        --api-key-file ~/.ollama_gateway_key

With several --hosts, one worker per host pulls from a shared job queue. Before starting, every host
is checked for the model, and all hosts must have the identical model digest (same weights and
quantization), otherwise the run stops. Host and digest are recorded for every call.

Results are appended to results/anon_<model>.jsonl, one line per call. The run is resumable:
jobs already in the file with a valid grade are skipped. Jobs run in a shuffled (seeded) order so
time-of-run effects are not confounded with assignment or grade. Standard library only.
"""
import argparse
import hashlib
import json
import random
import re
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prompts import GRADES, PROMPT_VERSION, build_messages, output_schema, prompt_hash

ROOT = Path(__file__).resolve().parent
ORDER_SEED = 20260925


def load():
    a = json.load(open(ROOT / "assignments.json", encoding="utf-8"))
    texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
    return a["meta"]["betygsskala"], {x["id"]: x for x in a["assignments"]}, texts


def pilot_ids(n_per_grade):
    """Pick n texts per intended grade per assignment (needs the key; only used to choose the pilot sample)."""
    key = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
    rng = random.Random(ORDER_SEED)
    chosen = set()
    by = {}
    for tid, k in key.items():
        by.setdefault((k["assignment_id"], k["intended_grade"]), []).append(tid)
    for ids in by.values():
        chosen.update(rng.sample(sorted(ids), n_per_grade))
    return chosen


HEADERS = {"Content-Type": "application/json"}


def model_digest(host, model, timeout=10):
    req = urllib.request.Request(f"{host}/api/tags", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == model or m["model"] == model:
                return m["digest"]
    return None


def job_seed(job_key):
    return int(hashlib.sha256(job_key.encode()).hexdigest()[:8], 16)


def call_ollama(host, model, messages, schema, seed, temperature, num_ctx, think, timeout, num_predict=None):
    options = {"seed": seed, "num_ctx": num_ctx}
    if num_predict:
        options["num_predict"] = num_predict
    if temperature is not None:
        options["temperature"] = temperature
    body = {"model": model, "messages": messages, "format": schema, "stream": False, "options": options}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse(content, labels):
    try:
        items = json.loads(content)["bedomningar"]
    except (json.JSONDecodeError, KeyError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.S)
        if not m:
            return None
        try:
            items = json.loads(m.group())["bedomningar"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    if not isinstance(items, list) or len(items) != len(labels):
        return None
    if any(not isinstance(it, dict) or it.get("betyg") not in GRADES for it in items):
        return None
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hosts", nargs="+", default=["http://localhost:11434"],
                    help="one or more Ollama base URLs; the same URL twice = two parallel workers on it")
    ap.add_argument("--api-key-file", help="file containing a bearer token for the Ollama gateway")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers per host (e.g. 8 for inferno)")
    ap.add_argument("--criteria", choices=["with", "without", "both"], default="both")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--assignments", nargs="*", help="e.g. G6-1 G9-3 (default: all)")
    ap.add_argument("--pilot", type=int, metavar="N", help="only N texts per grade per assignment, 1 rep")
    ap.add_argument("--temperature", type=float, help="default: the model's own default")
    ap.add_argument("--think", choices=["on", "off", "low", "medium", "high"],
                    help="for thinking models (default: model default); gpt-oss only accepts low/medium/high")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=1024,
                    help="max output tokens; a normal answer needs ~200, this stops repetition loops early (0 = no cap)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", help="output file (default results/anon_<model>[_pilot].jsonl)")
    args = ap.parse_args()

    if args.api_key_file:
        HEADERS["Authorization"] = "Bearer " + Path(args.api_key_file).expanduser().read_text().strip()
    scale, assignments, texts = load()
    aids = args.assignments or list(assignments)
    crit_conds = {"with": [True], "without": [False], "both": [True, False]}[args.criteria]
    reps = 1 if args.pilot else args.reps
    keep = pilot_ids(args.pilot) if args.pilot else None
    think = {"on": True, "off": False, None: None}.get(args.think, args.think)

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out = Path(args.out) if args.out else ROOT / "results" / f"anon_{safe_model}{'_pilot' if args.pilot else ''}.jsonl"
    out.parent.mkdir(exist_ok=True)

    done, tried = set(), {}  # tried: attempts already used by failed rows, so a rerun continues with fresh seeds
    if out.exists():
        for line in open(out, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("grade") in GRADES:
                done.add(rec["job_key"])
            elif rec.get("error") == "unparseable":
                tried[rec["job_key"]] = tried.get(rec["job_key"], 0) + rec.get("attempts", 0)

    jobs = []
    for aid in aids:
        for item in texts[aid]:
            if keep is not None and item["text_id"] not in keep:
                continue
            for crit in crit_conds:
                for rep in range(1, reps + 1):
                    key = f"anon|{args.model}|{PROMPT_VERSION}|{item['text_id']}|crit={int(crit)}|rep={rep}"
                    if key not in done:
                        jobs.append((key, aid, item, crit, rep))
    random.Random(ORDER_SEED).shuffle(jobs)
    total = len(jobs) + len(done)
    print(f"{args.model}: {len(jobs)} calls to run ({len(done)} already done) -> {out}", flush=True)

    digests = {}
    for h in dict.fromkeys(args.hosts):
        try:
            digests[h] = model_digest(h, args.model)
        except (urllib.error.URLError, TimeoutError) as e:
            sys.exit(f"host {h} not reachable: {e}")
        if not digests[h]:
            sys.exit(f"model {args.model} not found on {h}")
    if len(set(digests.values())) > 1:
        sys.exit("model digests differ between hosts:\n" + "\n".join(f"  {h}: {d}" for h, d in digests.items()))
    digest = next(iter(digests.values()))
    print(f"{len(args.hosts) * args.workers} worker(s), model digest {digest[:12]} identical on all hosts", flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    counter = {"n": 0}
    t_start = time.time()
    f = open(out, "a", encoding="utf-8")

    def request_with_backoff(host, messages, schema, seed, max_wait=1800):
        """Retry connection failures and 5xx/429 with growing pauses (5 s ... 60 s) for up to max_wait seconds."""
        waited, pause = 0, 5
        while True:
            try:
                return call_ollama(host, args.model, messages, schema, seed, args.temperature,
                                   args.num_ctx, think, args.timeout, args.num_predict), None
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429:
                    return None, f"HTTPError {e.code}: {e.reason}"
                err = f"HTTPError {e.code}: {e.reason}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                err = f"{type(e).__name__}: {e}"
            if waited >= max_wait:
                return None, err
            with lock:
                print(f"  [{host}] {err} - waiting {pause}s", flush=True)
            time.sleep(pause)
            waited += pause
            pause = min(pause * 2, 60)

    def run_job(host, key, aid, item, crit, rep):
        messages, labels = build_messages(assignments[aid], scale, [item["text"]], criteria=crit)
        schema = output_schema(1)
        rec = {
            "job_key": key, "round": "anon", "model": args.model, "model_digest": digest, "host": host,
            "prompt_version": PROMPT_VERSION,
            "assignment_id": aid, "text_id": item["text_id"], "criteria": crit, "rep": rep,
            "batch_size": 1, "position": 1, "identity": None,
            "temperature": args.temperature, "think": think, "num_ctx": args.num_ctx, "num_predict": args.num_predict,
            "prompt_hash": prompt_hash(messages),
        }
        items, resp, err, attempts, seed = None, None, None, 0, None
        t0 = time.time()
        for attempt in range(3):  # new seed only for unparseable output; server outages are waited out below
            attempts = attempt + 1
            seed = job_seed(key) + tried.get(key, 0) + attempt
            resp, err = request_with_backoff(host, messages, schema, seed)
            if resp is None:
                break  # server unreachable for too long: leave the job ungraded so a rerun picks it up
            items = parse(resp.get("message", {}).get("content", ""), labels)
            err = None if items else "unparseable"
            if items:
                break
        it = items[0] if items else {}
        rec.update({
            "seed": seed, "attempts": attempts, "error": err,
            "grade": it.get("betyg"), "motivering": it.get("motivering"), "aterkoppling": it.get("aterkoppling"),
            "raw": (resp or {}).get("message", {}).get("content"),
            "thinking": (resp or {}).get("message", {}).get("thinking"),
            "prompt_tokens": (resp or {}).get("prompt_eval_count"), "output_tokens": (resp or {}).get("eval_count"),
            "seconds": round(time.time() - t0, 2),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return rec

    def worker(host):
        while True:
            try:
                key, aid, item, crit, rep = q.get_nowait()
            except queue.Empty:
                return
            rec = run_job(host, key, aid, item, crit, rep)
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                counter["n"] += 1
                n = counter["n"]
                if n % 20 == 0 or n == len(jobs) or rec["error"]:
                    el = time.time() - t_start
                    print(f"  {n}/{len(jobs)}  {el/n:.2f}s/call overall  eta {el/n*(len(jobs)-n)/60:.0f} min"
                          + (f"  [{host} {item['text_id']}: {rec['error']}]" if rec["error"] else ""), flush=True)

    threads = [threading.Thread(target=worker, args=(h,), daemon=True) for h in args.hosts for _ in range(args.workers)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("interrupted; finished calls are saved, rerun the same command to resume", flush=True)
    finally:
        f.close()
    print(f"done: {total} calls in file", flush=True)


if __name__ == "__main__":
    sys.exit(main())
