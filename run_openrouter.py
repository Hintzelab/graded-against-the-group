#!/usr/bin/env python3
"""Batch-grading conditions on commercial models via OpenRouter, with a hard spending cap.

Same prompts, codes and class draws as run_batch.py (design version 2, same seed), so results are directly
comparable with the local gemma4 / qwen3 runs. Designs: single_code (reference), random20, profiles.
Every call logs OpenRouter's reported cost; the run stops before exceeding --max-usd.

    # small test run: 1 text per assignment alone, 1 class of each kind, with/without criteria
    python3 run_openrouter.py --tag ortest --model google/gemini-3.1-flash-lite --max-usd 0.10 \\
        --designs single_code random20 profiles --texts-per-assignment 1 --max-classes 1

    # full focused run
    python3 run_openrouter.py --tag ormain --model openai/gpt-6-luna --max-usd 3 \\
        --designs single_code random20 profiles --reps 2 --splits 6

One JSON line per call in results/or_<tag>_<model>.jsonl (same student-level format as run_batch.py). Resumable.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prompts import GRADES, build_messages, prompt_hash
from run_batch import PROFILES, make_design, parse_batch
from run_ollama import job_seed

ROOT = Path(__file__).resolve().parent
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "v1-codes"


def schema(n):
    """Strict JSON schema (OpenAI strict mode: all properties required, no additional properties)."""
    item = {"type": "object", "additionalProperties": False,
            "properties": {"elev": {"type": "string"}, "betyg": {"type": "string", "enum": GRADES},
                           "motivering": {"type": "string"}, "aterkoppling": {"type": "string"}},
            "required": ["elev", "betyg", "motivering", "aterkoppling"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"bedomningar": {"type": "array", "items": item}}, "required": ["bedomningar"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-usd", type=float, required=True, help="hard cap on total spend for this model and tag")
    ap.add_argument("--key-file", default="~/.openrouter_key")
    ap.add_argument("--assignments", nargs="*")
    ap.add_argument("--designs", nargs="+", required=True, help="single_code random20 profiles")
    ap.add_argument("--reps", type=int, default=2, help="repetitions for single_code")
    ap.add_argument("--splits", type=int, default=1, help="random20 splits / classes per profile")
    ap.add_argument("--texts-per-assignment", type=int, help="single_code: only the first N texts per assignment")
    ap.add_argument("--max-classes", type=int, help="only the first N classes per design and assignment")
    ap.add_argument("--criteria", choices=["with", "without", "both"], default="both")
    ap.add_argument("--reasoning", default='{"enabled": false}', help="OpenRouter reasoning object (JSON)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    a = json.load(open(ROOT / "assignments.json", encoding="utf-8"))
    scale, assignments = a["meta"]["betygsskala"], {x["id"]: x for x in a["assignments"]}
    texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
    text_of = {t["text_id"]: t["text"] for v in texts.values() for t in v}
    tkey = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
    aids = args.assignments or list(assignments)
    designs = [d for x in args.designs for d in (PROFILES if x == "profiles" else [x])]
    crit_conds = {"with": [True], "without": [False], "both": [True, False]}[args.criteria]
    reasoning = json.loads(args.reasoning) if args.reasoning else None

    batches = make_design(aids, designs, args.splits, args.reps, texts, tkey, args.seed)
    if args.texts_per_assignment:
        keep = {t["text_id"] for aid in aids for t in texts[aid][:args.texts_per_assignment]}
        batches = [b for b in batches if b["design"] != "single_code" or b["text_ids"][0] in keep]
    if args.max_classes:
        seen, kept = {}, []
        for b in batches:
            k = (b["assignment_id"], b["design"])
            if b["design"] == "single_code" or seen.get(k, 0) < args.max_classes:
                seen[k] = seen.get(k, 0) + 1
                kept.append(b)
        batches = kept

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out = ROOT / "results" / f"or_{args.tag}_{safe}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done, spent = set(), 0.0
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            spent += r.get("cost") or 0
            if r.get("error") is None:
                done.add(r["job_key"])
    jobs = [(f"{args.tag}|{args.model}|{PROMPT_VERSION}|{b['batch_id']}|crit={int(c)}", b, c)
            for b in batches for c in crit_conds]
    jobs = [j for j in jobs if j[0] not in done]
    random.Random(args.seed).shuffle(jobs)
    print(f"{args.model}: {len(jobs)} calls to run ({len(done)} done, ${spent:.4f} spent so far, cap ${args.max_usd}) -> {out}",
          flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    state = {"spent": spent, "n": 0, "stop": False}
    f = open(out, "a", encoding="utf-8")

    def call(messages, n, seed):
        body = {"model": args.model, "messages": messages, "seed": seed, "max_tokens": 500 + 450 * n,
                "response_format": {"type": "json_schema", "json_schema": {"name": "bedomning", "strict": True, "schema": schema(n)}},
                "provider": {"require_parameters": True}, "usage": {"include": True}}
        if reasoning is not None:
            body["reasoning"] = reasoning
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "grading-independence-study"})
        waited, pause = 0, 5
        while True:
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read()), None
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code not in (408, 429, 500, 502, 503, 504):
                    return None, f"HTTPError {e.code}: {detail}"
                err = f"HTTPError {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                err = f"{type(e).__name__}: {e}"
            if waited >= 900:
                return None, err
            time.sleep(pause)
            waited += pause
            pause = min(pause * 2, 60)

    def worker():
        while True:
            with lock:
                if state["stop"]:
                    return
            try:
                jk, b, crit = q.get_nowait()
            except queue.Empty:
                return
            tids, codes = b["text_ids"], b["codes"]
            messages, _ = build_messages(assignments[b["assignment_id"]], scale, [text_of[t] for t in tids],
                                         criteria=crit, labels=codes)
            t0 = time.time()
            resp, err = call(messages, len(tids), job_seed(jk) % 2**31)
            content = ((resp or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
            usage = (resp or {}).get("usage") or {}
            results, diag = (None, None)
            if resp is not None:
                results, diag = parse_batch(content, codes)
                if results is None:
                    err = "unparseable"
                elif len(tids) == 1 and not results:  # single text: a garbled code is still unambiguous
                    try:
                        item = json.loads(content)["bedomningar"][0]
                        if item.get("betyg") in GRADES:
                            results = {codes[0]: {"grade": item["betyg"], "motivering": item.get("motivering"),
                                                  "aterkoppling": item.get("aterkoppling")}}
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        pass
            rec = {
                "job_key": jk, "tag": args.tag, "model": args.model, "prompt_version": PROMPT_VERSION,
                "batch_id": b["batch_id"], "assignment_id": b["assignment_id"], "design": b["design"],
                "batch_size": len(tids), "criteria": crit, "reasoning_setting": reasoning,
                "prompt_hash": prompt_hash(messages), "error": err, "diagnostics": diag if results is not None else None,
                "students": [{"position": i + 1, "code": c, "text_id": t, **(results or {}).get(c, {"grade": None})}
                             for i, (c, t) in enumerate(zip(codes, tids))],
                "raw": content, "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": usage.get("cost"), "provider": (resp or {}).get("provider"),
                "seconds": round(time.time() - t0, 1), "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                state["spent"] += rec["cost"] or 0
                state["n"] += 1
                d = rec["diagnostics"] or {}
                if state["n"] % 25 == 0 or err or len(tids) > 1:
                    print(f"  {state['n']}/{len(jobs)} {b['batch_id']} crit={int(crit)}: {d.get('n_graded', 0)}/{len(tids)} graded, "
                          f"in {rec['prompt_tokens']} out {rec['output_tokens']} (reasoning {rec['reasoning_tokens']}) "
                          f"${rec['cost'] or 0:.5f} | total ${state['spent']:.4f}" + (f"  ERROR {err}" if err else ""), flush=True)
                if state["spent"] >= args.max_usd:
                    state["stop"] = True
                    print(f"SPENDING CAP REACHED (${state['spent']:.4f} >= ${args.max_usd}); stopping", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    f.close()
    print(f"done: {state['n']} calls, total spent ${state['spent']:.4f}", flush=True)


if __name__ == "__main__":
    main()
