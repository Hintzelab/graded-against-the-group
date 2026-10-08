#!/usr/bin/env python3
"""Verbatim prompts, output schemas and tasks for the Supplementary Material.

1. Rebuilds the prompt of every grading request used in the paper with prompts.py and compares its hash with the
   hash stored in the result record (prompt_hash for single-text and batch calls, system_hash for chat
   conversations). Any mismatch stops the script.
2. Writes example prompts taken from real, hash-verified requests to paper/si/*.txt, together with the JSON output
   schemas and the 12 tasks (from assignments.json), for \\VerbatimInput in supplementary.tex.
3. Reads every written file back and compares it byte for byte with the strings that produced the verified hashes.

    python3 make_si_prompts.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import run_openrouter
from prompts import SYSTEM, build_gem_system, build_messages, build_turn, output_schema, prompt_hash

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "si"
OUT.mkdir(parents=True, exist_ok=True)
a = json.load(open(ROOT / "assignments.json", encoding="utf-8"))
scale, assignments = a["meta"]["betygsskala"], {x["id"]: x for x in a["assignments"]}
texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
text_of = {t["text_id"]: t["text"] for v in texts.values() for t in v}

SOURCES = {  # file: kind
    "anon_inferno_gemma4-26b_nothink.jsonl": "anon",
    "anon_inferno_qwen3-30b-a3b_think-off.jsonl": "anon",
    "batch_main_gemma4-26b.jsonl": "batch",
    "batch_main_qwen3-30b-a3b.jsonl": "batch",
    "or_ormain_openai-gpt-6-luna.jsonl": "batch",
    "or_ormain_google-gemini-3.1-flash-lite.jsonl": "batch",
    "seq_main_gemma4-26b.jsonl": "chat",
    "seq_main_qwen3-30b-a3b.jsonl": "chat",
}


def rebuild(rec, kind):
    """The messages (anon/batch) or the system prompt (chat) that the runner built for this record."""
    aid, crit = rec["assignment_id"], rec["criteria"]
    if kind == "anon":
        messages, _ = build_messages(assignments[aid], scale, [text_of[rec["text_id"]]], criteria=crit)
        return messages
    if kind == "batch":
        students = sorted(rec["students"], key=lambda s: s["position"])
        messages, _ = build_messages(assignments[aid], scale, [text_of[s["text_id"]] for s in students],
                                     criteria=crit, labels=[s["code"] for s in students])
        return messages
    return [{"role": "system", "content": build_gem_system(assignments[aid], scale, criteria=crit)}]


# 1) hash check over all records
stats, examples = Counter(), {}
for fname, kind in SOURCES.items():
    for line in open(ROOT / "results" / fname, encoding="utf-8"):
        rec = json.loads(line)
        if kind == "batch" and not rec.get("students"):
            stats[(fname, "no students stored (failed call)")] += 1
            continue
        stored = rec["system_hash"] if kind == "chat" else rec["prompt_hash"]
        messages = rebuild(rec, kind)
        ok = prompt_hash(messages) == stored
        stats[(fname, "match" if ok else "MISMATCH")] += 1
        if not ok:
            print("MISMATCH", fname, rec.get("job_key"))
        examples.setdefault((fname, rec.get("design", "single_code"), rec["criteria"], rec["assignment_id"]), (rec, messages))

print("hash check (rebuilt prompt vs. hash stored at run time):")
for (fname, status), n in sorted(stats.items()):
    print(f"  {fname:<48} {status:<35} {n:>6}")
if any(s == "MISMATCH" for _, s in stats):
    sys.exit("prompt mismatch -- do not use these prompts in the SI")

# 2) example prompts from real, verified requests (task G8-1)
TASK = "G8-1"
files = {}
rec, msg = examples[("anon_inferno_gemma4-26b_nothink.jsonl", "single_code", True, TASK)]
files["prompt_system.txt"] = msg[0]["content"]
files["prompt_alone_withCriteria.txt"] = msg[1]["content"]
rec, msg = examples[("anon_inferno_gemma4-26b_nothink.jsonl", "single_code", False, TASK)]
files["prompt_alone_withoutCriteria.txt"] = msg[1]["content"]
rec, msg = examples[("batch_main_gemma4-26b.jsonl", "random5", True, TASK)]
files["prompt_batch_withCriteria.txt"] = msg[1]["content"]
batch_id = rec["batch_id"]
chat = next(json.loads(l) for l in open(ROOT / "results" / "seq_main_gemma4-26b.jsonl", encoding="utf-8")
            if f'"{batch_id}"' in l and json.loads(l)["batch_id"] == batch_id and json.loads(l)["criteria"])
files["prompt_chat_system_withCriteria.txt"] = rebuild(chat, "chat")[0]["content"]
rec, msg = examples[("seq_main_gemma4-26b.jsonl", "random5", False, TASK)]
files["prompt_chat_system_withoutCriteria.txt"] = msg[0]["content"]
first = chat["turns"][0]
files["prompt_chat_turn1.txt"] = build_turn(first["code"], text_of[first["text_id"]])
files["prompt_chat_answer1.txt"] = first["raw"]
files["schema_ollama.json"] = json.dumps(output_schema(5), ensure_ascii=False, indent=2)
files["schema_openrouter.json"] = json.dumps(run_openrouter.schema(5), ensure_ascii=False, indent=2)

# tasks: all fields of assignments.json, unchanged
for t in a["assignments"]:
    lines = [f"{t['id']}  (årskurs {t['grade']}, {t['cefr']})", f"Titel: {t['title_de']}", "",
             f"Instruktion (sv): {t['instruction_sv']}", f"Instruktion (de): {t['instruction_de']}", "",
             "Innehållspunkter:"] + [f"  - {p}" for p in t["content_points"]] + \
            ["Språkligt fokus:"] + [f"  - {p}" for p in t["language_focus"]] + [""] + \
            [f"{g}: {t['criteria'][g]}" for g in ("E", "C", "A")]
    files[f"task_{t['id']}.txt"] = "\n".join(lines)
files["scale.txt"] = "\n".join(f"{g}: {scale[g]}" for g in ["A", "B", "C", "D", "E", "F"])

# text generation: prompt (instance for the example task), rule set unchanged, one example text per intended grade
gen_template = (ROOT / "texts" / "generation_prompt.md").read_text(encoding="utf-8").split("---\n\n", 1)[1].rstrip("\n")
files["generation_prompt.txt"] = gen_template.replace("<TASK>", TASK).replace(
    f"Project folder: {ROOT}", "Project folder: [project folder]")          # no local path in the SI
files["generation_spec.md"] = (ROOT / "texts" / "generation_spec.md").read_text(encoding="utf-8").rstrip("\n")
key = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
raw = json.load(open(ROOT / "texts" / "raw" / f"{TASK}.json", encoding="utf-8"))["texts"]
blocks = []
for g in ["F", "E", "D", "C", "B", "A"]:
    t = next(x for x in raw if x["intended_grade"] == g)
    tid = next(k for k, v in key.items() if v["assignment_id"] == TASK and v["raw_n"] == t["n"])
    assert text_of[tid] == t["text"] and key[tid]["intended_grade"] == g        # same text the models graded
    head = f"Intended grade {g}" + (f" (F type: {t['f_type'].replace('_', ' ')})" if t.get("f_type") else "")
    blocks.append(f"{head}, text {tid}\n\"\"\"\n{t['text']}\n\"\"\"")
files[f"examples_{TASK}.txt"] = "\n\n".join(blocks)

# texts the expert marked as unrealistic, with intended and expert grade
expert = json.load(open(ROOT / "ratings" / "backups" / "leonie_FINAL_20260927_1527.json", encoding="utf-8"))["ratings"]
flagged = sorted(t for t, v in expert.items() if v.get("flag"))
files["expert_flagged.txt"] = "\n\n".join(
    f"Text {t} (task {key[t]['assignment_id']}): intended grade {key[t]['intended_grade']} "
    f"(F type: {(key[t]['f_type'] or '-').replace('_', ' ')}), expert grade {expert[t]['grade']}\n\"\"\"\n{text_of[t]}\n\"\"\""
    for t in flagged)

for name, content in files.items():
    (OUT / name).write_text(content + "\n", encoding="utf-8")

# 3) read back and compare byte for byte; tasks: every field of assignments.json must appear verbatim
for name, content in files.items():
    assert (OUT / name).read_text(encoding="utf-8") == content + "\n", name
for t in a["assignments"]:
    body = (OUT / f"task_{t['id']}.txt").read_text(encoding="utf-8")
    fields = [t["title_de"], t["instruction_sv"], t["instruction_de"], *t["content_points"], *t["language_focus"],
              *t["criteria"].values()]
    assert all(f in body for f in fields), t["id"]
# the task texts that reached the models are covered by the hash check: criteria, instruction_sv, title and scale
# are part of every verified prompt
print(f"wrote {len(files)} files to {OUT}; all read back identical")
print("example batch:", batch_id, "| chat conversation:", chat["job_key"])
nonascii = Counter(ch for c in files.values() for ch in c if ord(ch) > 127)
print("non-ASCII characters in the SI files:", "".join(sorted(nonascii)))
