#!/usr/bin/env python3
"""Accumulating-context grading: one conversation per class, one student per user turn.

This mimics a teacher using a Gemini Gem / custom GPT and pasting students one after another into the
same chat: when student k is graded, the model sees students 1..k-1 and its own earlier answers.
Classes (composition, order, codes) are identical to run_batch.py for the same --seed, so the same class
can be compared across independent / batch / sequential grading.

    python3 run_sequential.py --tag seqpilot --assignments G6-1 G9-1 --designs random20 all60 profiles \
        --only-assignments G9-1 --only-designs random20 all60 --design-version 1

One JSON line per conversation in results/seq_<tag>_<model>.jsonl with every turn's grade, timing and tokens.
"""
import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from prompts import build_gem_system, build_turn, output_schema, prompt_hash
from run_batch import PROFILES, make_design, parse_batch
from run_ollama import HEADERS, call_ollama, job_seed, model_digest

ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "v1-gem"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--hosts", nargs="+", default=["http://localhost:11434"])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--api-key-file", default="~/.ollama_gateway_key")
    ap.add_argument("--assignments", nargs="+", required=True)
    ap.add_argument("--designs", nargs="+", required=True, help="random10 random20 random30 profiles single_code ...")
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3, help="repetitions for single_code (one-turn conversations)")
    ap.add_argument("--only-assignments", nargs="*", help="run only these assignments of the generated design")
    ap.add_argument("--only-designs", nargs="*", help="run only these designs of the generated design")
    ap.add_argument("--criteria", choices=["with", "without", "both"], default="both")
    ap.add_argument("--think", choices=["on", "off", "low", "medium", "high"], default="off")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--num-predict", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=900)
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
    if args.only_assignments:
        batches = [b for b in batches if b["assignment_id"] in args.only_assignments]
    if args.only_designs:
        batches = [b for b in batches if b["design"] in args.only_designs]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out = ROOT / "results" / f"seq_{args.tag}_{safe_model}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["job_key"] for l in open(out, encoding="utf-8")}
    jobs = [(f"{args.tag}|{args.model}|{PROMPT_VERSION}|{b['batch_id']}|crit={int(c)}", b, c)
            for b in batches for c in crit_conds]
    jobs = [j for j in jobs if j[0] not in done]
    jobs.sort(key=lambda j: -len(j[1]["text_ids"]))

    digest = model_digest(args.hosts[0], args.model)
    if not digest:
        sys.exit(f"model {args.model} not found")
    n_turns = sum(len(b["text_ids"]) for _, b, _ in jobs)
    print(f"{args.model} ({digest[:12]}): {len(jobs)} conversations, {n_turns} turns ({len(done)} done) -> {out}", flush=True)

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    f = open(out, "a", encoding="utf-8")
    counter = {"n": 0}
    t_start = time.time()
    schema = output_schema(1)

    def request(host, messages, seed):
        waited, pause = 0, 5
        while True:
            try:
                return call_ollama(host, args.model, messages, schema, seed, None, args.num_ctx, think,
                                   args.timeout, args.num_predict), None
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
            system = build_gem_system(assignments[b["assignment_id"]], scale, criteria=crit)
            messages = [{"role": "system", "content": system}]
            turns, t_conv = [], time.time()
            for pos, (code, tid) in enumerate(zip(b["codes"], b["text_ids"]), start=1):
                messages.append({"role": "user", "content": build_turn(code, text_of[tid])})
                t0 = time.time()
                resp, result, err, attempts, content = None, None, None, 0, ""
                for attempt in range(3):
                    attempts = attempt + 1
                    resp, err = request(host, messages, job_seed(f"{jk}|{pos}") + attempt)
                    if resp is None:
                        break
                    content = resp.get("message", {}).get("content", "")
                    res, diag = parse_batch(content, [code])
                    if res is None:
                        err = "unparseable"
                        continue
                    result = res.get(code)
                    if result is None:  # one student per turn: a garbled code is still unambiguous
                        try:
                            item = json.loads(content)["bedomningar"][0]
                            result = {"grade": item.get("betyg"), "motivering": item.get("motivering"),
                                      "aterkoppling": item.get("aterkoppling")}
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            result = None
                    err = None if result and result.get("grade") else "no grade"
                    if not err:
                        break
                # the model's answer stays in the conversation, as it would in a chat
                messages.append({"role": "assistant", "content": content})
                turns.append({
                    "position": pos, "code": code, "text_id": tid, "error": err, "attempts": attempts,
                    "code_copied": bool(result) and code.split()[-1] in content,  # model often writes "7946" without "Elev"
                    "raw": content,
                    **(result or {"grade": None}),
                    "prompt_tokens": (resp or {}).get("prompt_eval_count"),
                    "prompt_ms": round((resp or {}).get("prompt_eval_duration", 0) / 1e6),
                    "output_tokens": (resp or {}).get("eval_count"),
                    "seconds": round(time.time() - t0, 1),
                })
                if resp is None:
                    break  # server gone for too long; keep what we have
            rec = {
                "job_key": jk, "tag": args.tag, "model": args.model, "model_digest": digest,
                "prompt_version": PROMPT_VERSION, "batch_id": b["batch_id"], "assignment_id": b["assignment_id"],
                "design": b["design"], "mode": "sequential", "batch_size": len(b["text_ids"]), "criteria": crit,
                "think": think, "num_ctx": args.num_ctx, "num_predict": args.num_predict,
                "system_hash": prompt_hash([{"role": "system", "content": system}]),
                "n_graded": sum(1 for t in turns if t["grade"]), "turns": turns,
                "context_truncated": any((t["prompt_tokens"] or 0) >= args.num_ctx - 10 for t in turns),
                "seconds": round(time.time() - t_conv, 1),
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                counter["n"] += 1
                last = turns[-1] if turns else {}
                print(f"  {counter['n']}/{len(jobs)} {b['batch_id']} crit={int(crit)}: {rec['n_graded']}/{len(b['text_ids'])} graded,"
                      f" {rec['seconds']:.0f}s total, last turn {last.get('seconds')}s with {last.get('prompt_tokens')} prompt tok"
                      + ("  CONTEXT TRUNCATED" if rec["context_truncated"] else ""), flush=True)

    threads = [threading.Thread(target=worker, args=(h,), daemon=True) for h in args.hosts for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    f.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
