#!/usr/bin/env python3
"""Blind rating server for the synthetic learner texts.

Run:   python3 review_server.py                      -> http://localhost:8765
       python3 review_server.py --host 0.0.0.0       -> also reachable from the local network

Standard library only. Every click is saved immediately to ratings/<rater>.json.
Each rater gets their own random text order per assignment, created on first login and
stored in their ratings file, so moving back and forth stays consistent across sessions.
The intended grades (texts/texts_key.json) are never read by this server.
"""
import argparse
import csv
import io
import json
import os
import random
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
RATINGS_DIR = ROOT / "ratings"
GRADES = ["F", "E", "D", "C", "B", "A"]
LOCK = threading.Lock()


def load_data():
    a = json.load(open(ROOT / "assignments.json", encoding="utf-8"))
    texts = json.load(open(ROOT / "texts" / "texts.json", encoding="utf-8"))
    fields = ("id", "grade", "cefr", "title_de", "instruction_sv", "instruction_de", "content_points", "criteria")
    return {
        "scale": a["meta"]["betygsskala"],
        "assignments": [{k: x[k] for k in fields} | {"texts": texts[x["id"]]} for x in a["assignments"]],
    }


DATA = load_data()
TEXT_TO_ASSIGNMENT = {t["text_id"]: a["id"] for a in DATA["assignments"] for t in a["texts"]}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_rater(name):
    name = (name or "").strip().lower()
    return name if re.fullmatch(r"[a-z0-9_-]{1,40}", name) else None


def save_state(state):
    RATINGS_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=RATINGS_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RATINGS_DIR / f"{state['rater']}.json")


SUBSET = None  # {assignment_id: [text_ids]} when the server runs with --per-grade


def make_subset(per_grade):
    """Stratified, fixed sample: per_grade texts per intended grade per assignment (seeded, saved to disk).
    The intended grades are only used here to balance the sample; they are never sent to the browser."""
    path = ROOT / "texts" / f"review_subset_{per_grade}pergrade.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    key = json.load(open(ROOT / "texts" / "texts_key.json", encoding="utf-8"))
    rng = random.Random(f"review-subset|{per_grade}")
    subset = {}
    for a in DATA["assignments"]:
        ids = [t["text_id"] for t in a["texts"]]
        subset[a["id"]] = sorted(t for g in GRADES for t in rng.sample(sorted(i for i in ids if key[i]["intended_grade"] == g), per_grade))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=1)
    return subset


def load_state(rater):
    path = RATINGS_DIR / f"{rater}.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    rng = random.SystemRandom()
    order = {}
    for a in DATA["assignments"]:
        ids = SUBSET[a["id"]][:] if SUBSET else [t["text_id"] for t in a["texts"]]
        rng.shuffle(ids)
        order[a["id"]] = ids
    state = {"rater": rater, "created": now(), "order": order, "ratings": {},
             "subset": bool(SUBSET)}
    save_state(state)
    return state


def export_csv(state):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rater", "assignment_id", "position", "text_id", "grade", "flag", "comment", "updated", "n_changes"])
    for aid, ids in state["order"].items():
        for pos, tid in enumerate(ids, start=1):
            r = state["ratings"].get(tid, {})
            w.writerow([state["rater"], aid, pos, tid, r.get("grade") or "", int(bool(r.get("flag"))),
                        r.get("comment", ""), r.get("updated", ""), max(len(r.get("history", [])) - 1, 0)])
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _rater(self, qs):
        return clean_rater(qs.get("rater", [""])[0])

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if url.path == "/api/data":
            return self._json(DATA)
        if url.path == "/api/state":
            rater = self._rater(qs)
            if not rater:
                return self._json({"error": "invalid rater name"}, 400)
            with LOCK:
                return self._json(load_state(rater))
        if url.path == "/api/export.csv":
            rater = self._rater(qs)
            if not rater or not (RATINGS_DIR / f"{rater}.json").exists():
                return self._json({"error": "unknown rater"}, 404)
            with LOCK:
                body = export_csv(load_state(rater))
            return self._send(200, "﻿" + body, "text/csv; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="ratings_{rater}.csv"'})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/rate":
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)
        rater = clean_rater(body.get("rater"))
        tid = body.get("text_id")
        grade = body.get("grade")
        if not rater or tid not in TEXT_TO_ASSIGNMENT or grade not in GRADES + [None]:
            return self._json({"error": "invalid request"}, 400)
        with LOCK:
            state = load_state(rater)
            r = state["ratings"].setdefault(tid, {"history": []})
            if grade != r.get("grade"):
                r["history"].append({"grade": grade, "time": now()})
            r["grade"] = grade
            r["comment"] = str(body.get("comment", ""))[:2000]
            r["flag"] = bool(body.get("flag"))
            r["updated"] = now()
            save_state(state)
        self._json({"ok": True, "rating": r})

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/state"):
            super().log_message(fmt, *args)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text Rating</title>
<style>
:root {
  --bg: #f6f5f1; --card: #ffffff; --ink: #1f2328; --muted: #6b6f76; --line: #e3e1da;
  --accent: #2f5d8a; --accent-soft: #e6eef6; --done: #3c7a4a; --flag: #b3541e; --warn: #b42318;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #16181c; --card: #1f2227; --ink: #e8e6e1; --muted: #9ca0a8; --line: #33373e;
          --accent: #7fb0e0; --accent-soft: #24313f; --done: #7cc28c; --flag: #e39163; --warn: #f07b6f; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card); border-bottom: 1px solid var(--line);
         padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
select, button, input, textarea { font: inherit; color: inherit; }
select { padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--card); }
.who { color: var(--muted); font-size: 13px; }
.who a { color: var(--accent); }
.bar { flex: 1 1 160px; min-width: 120px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.bar > div { height: 100%; background: var(--done); width: 0; transition: width .2s; }
.overall { font-size: 13px; color: var(--muted); white-space: nowrap; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; display: grid; gap: 16px;
       grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); align-items: start; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.task h2 { margin: 0 0 2px; font-size: 18px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 10px; }
.instr { margin: 6px 0; }
.instr.de { color: var(--muted); font-style: italic; }
.label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 14px 0 4px; }
ul { margin: 0; padding-left: 20px; }
.crit { border-left: 3px solid var(--line); padding: 4px 0 4px 10px; margin: 8px 0; font-size: 14px; }
.crit b { display: inline-block; width: 18px; }
details { margin-top: 12px; font-size: 14px; }
summary { cursor: pointer; color: var(--accent); }
.pos { display: flex; justify-content: space-between; align-items: baseline; color: var(--muted); font-size: 13px; }
.text { white-space: pre-wrap; font-size: 19px; line-height: 1.6; margin: 14px 0 18px; padding: 16px;
        background: var(--bg); border-radius: 8px; min-height: 120px; font-family: Georgia, "Times New Roman", serif; }
.grades { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
.grades button { padding: 14px 0; font-size: 20px; font-weight: 600; border: 1px solid var(--line);
                 border-radius: 8px; background: var(--card); cursor: pointer; }
.grades button:hover { border-color: var(--accent); }
.grades button.sel { background: var(--accent); color: var(--card); border-color: var(--accent); }
.hint { color: var(--muted); font-size: 12px; margin: 6px 0 14px; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
textarea { width: 100%; min-height: 54px; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
           background: var(--card); resize: vertical; }
label.flag { display: flex; gap: 6px; align-items: center; font-size: 14px; cursor: pointer; }
.nav { display: flex; justify-content: space-between; gap: 8px; margin-top: 14px; }
.nav button, .btn { padding: 8px 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--card); cursor: pointer; }
.nav button:disabled { opacity: .4; cursor: default; }
.status { font-size: 13px; color: var(--muted); min-height: 20px; margin-top: 8px; }
.status.err { color: var(--warn); font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(30px, 1fr)); gap: 4px; margin-top: 16px; }
.grid button { height: 30px; padding: 0; font-size: 12px; border: 1px solid var(--line); border-radius: 4px;
               background: var(--card); cursor: pointer; color: var(--muted); }
.grid button.g { background: var(--accent-soft); color: var(--ink); font-weight: 600; }
.grid button.f { box-shadow: inset 0 -3px 0 var(--flag); }
.grid button.cur { outline: 2px solid var(--accent); outline-offset: 1px; }
.done-msg { padding: 10px 12px; border-radius: 8px; background: var(--accent-soft); margin-bottom: 12px; font-size: 14px; }
#login { position: fixed; inset: 0; background: var(--bg); display: none; place-items: center; z-index: 10; padding: 16px; }
#login .card { max-width: 380px; width: 100%; }
#login input { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--card); margin: 10px 0; }
</style>
</head>
<body>
<div id="login"><form class="card" id="loginForm">
  <h2 style="margin-top:0">Text rating</h2>
  <div>Enter your rater name (lowercase letters, digits, - or _), e.g. <code>leonie</code>. Your progress is saved under this name.</div>
  <input id="raterInput" autocomplete="off" pattern="[a-z0-9_\-]{1,40}" required>
  <button class="btn" type="submit">Start</button>
</form></div>

<header>
  <h1>Text rating</h1>
  <select id="assignSel" aria-label="Assignment"></select>
  <div class="bar" title="Overall progress"><div id="barFill"></div></div>
  <span class="overall" id="overall"></span>
  <span class="who">Rater: <b id="raterName"></b> · <a href="#" id="exportLink">export CSV</a> · <a href="#" id="switchRater">switch</a></span>
</header>

<main>
  <section class="card task" id="task"></section>
  <section class="card">
    <div id="doneMsg"></div>
    <div class="pos"><span id="posLabel"></span><span id="idLabel"></span></div>
    <div class="text" id="text"></div>
    <div class="grades" id="grades"></div>
    <div class="hint">Click a grade (or press F / E / D / C / B / A) to save it and go to the next text. ← → to move without grading.</div>
    <div class="row" style="justify-content:space-between">
      <label class="flag"><input type="checkbox" id="flag"> Problem with this text (unrealistic / not neutral)</label>
      <button class="btn" id="clearBtn" type="button">Clear grade</button>
    </div>
    <div class="label">Comment (optional, saved when you leave the field)</div>
    <textarea id="comment"></textarea>
    <div class="nav">
      <button id="prevBtn" type="button">← Previous</button>
      <button id="nextUngradedBtn" type="button">Next ungraded</button>
      <button id="nextBtn" type="button">Next →</button>
    </div>
    <div class="status" id="status"></div>
    <div class="grid" id="grid"></div>
  </section>
</main>

<script>
const GR = ["F","E","D","C","B","A"];
let DATA, STATE, rater = null, aid = null, idx = 0;
const A = {}, TEXT = {};
const $ = id => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  del(k) { try { localStorage.removeItem(k); } catch (e) {} },
};
const valid = r => /^[a-z0-9_-]{1,40}$/.test(r || "");

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of kids) e.append(c);
  return e;
}

async function start() {
  DATA = await (await fetch("/api/data")).json();
  for (const a of DATA.assignments) { A[a.id] = a; for (const t of a.texts) TEXT[t.text_id] = t.text; }
  const fromUrl = new URLSearchParams(location.search).get("rater");
  rater = valid(fromUrl) ? fromUrl : store.get("rater");
  if (!valid(rater)) return showLogin();
  await loadRater();
}

function showLogin() { $("login").style.display = "grid"; $("raterInput").focus(); }
$("loginForm").addEventListener("submit", async e => {
  e.preventDefault();
  const r = $("raterInput").value.trim().toLowerCase();
  if (!valid(r)) return;
  rater = r; $("login").style.display = "none"; await loadRater();
});
$("switchRater").addEventListener("click", e => { e.preventDefault(); store.del("rater"); $("raterInput").value = ""; showLogin(); });
$("exportLink").addEventListener("click", e => { e.preventDefault(); location.href = "/api/export.csv?rater=" + encodeURIComponent(rater); });

async function loadRater() {
  const res = await fetch("/api/state?rater=" + encodeURIComponent(rater));
  if (!res.ok) { store.del("rater"); return showLogin(); }
  STATE = await res.json();
  store.set("rater", rater);
  $("raterName").textContent = rater;
  const last = store.get("assignment:" + rater);
  aid = A[last] ? last : DATA.assignments[0].id;
  buildSelect();
  goFirstUngraded();
}

const order = () => STATE.order[aid];
const rating = tid => STATE.ratings[tid] || {};
const nGraded = id => STATE.order[id].filter(t => rating(t).grade).length;

function buildSelect() {
  const sel = $("assignSel"); sel.textContent = "";
  for (const a of DATA.assignments) {
    const n = nGraded(a.id);
    sel.append(el("option", { value: a.id }, `${a.id} · åk ${a.grade} · ${a.title_de} (${n}/${STATE.order[a.id].length})${n === STATE.order[a.id].length ? " ✓" : ""}`));
  }
  sel.value = aid;
}
$("assignSel").addEventListener("change", async e => { await saveComment(); aid = e.target.value; store.set("assignment:" + rater, aid); goFirstUngraded(); });

function goFirstUngraded() {
  const i = order().findIndex(t => !rating(t).grade);
  idx = i === -1 ? 0 : i;
  renderTask(); render();
}

function renderTask() {
  const a = A[aid], box = $("task"); box.textContent = "";
  box.append(el("h2", {}, a.title_de), el("div", { class: "meta" }, `${a.id} · årskurs ${a.grade} · ${a.cefr}`));
  box.append(el("div", { class: "label" }, "Uppgift"), el("div", { class: "instr" }, a.instruction_sv), el("div", { class: "instr de" }, a.instruction_de));
  const ul = el("ul"); a.content_points.forEach(p => ul.append(el("li", {}, p)));
  box.append(el("div", { class: "label" }, "Innehållspunkter"), ul);
  box.append(el("div", { class: "label" }, "Bedömningskriterier"));
  for (const g of ["E", "C", "A"]) box.append(el("div", { class: "crit" }, el("b", {}, g), a.criteria[g]));
  const d = el("details", {}, el("summary", {}, "Betygsskala F–A"));
  for (const g of ["A", "B", "C", "D", "E", "F"]) d.append(el("div", { class: "crit" }, el("b", {}, g), DATA.scale[g]));
  box.append(d);
}

function render() {
  const ids = order(), tid = ids[idx], r = rating(tid), n = nGraded(aid);
  $("posLabel").textContent = `Text ${idx + 1} of ${ids.length} · ${n}/${ids.length} graded`;
  $("idLabel").textContent = tid;
  $("text").textContent = TEXT[tid];
  const g = $("grades"); g.textContent = "";
  for (const x of GR) g.append(el("button", { type: "button", class: r.grade === x ? "sel" : "", onclick: () => rate(x) }, x));
  $("flag").checked = !!r.flag;
  $("comment").value = r.comment || "";
  $("prevBtn").disabled = idx === 0;
  $("nextBtn").disabled = idx === ids.length - 1;
  $("doneMsg").textContent = "";
  if (n === ids.length) $("doneMsg").append(el("div", { class: "done-msg" }, "All texts in this assignment are graded. You can still go back and change any of them, or choose the next assignment at the top."));
  const grid = $("grid"); grid.textContent = "";
  ids.forEach((t, i) => {
    const rr = rating(t);
    grid.append(el("button", { type: "button", title: `Text ${i + 1}`, class: [rr.grade ? "g" : "", rr.flag ? "f" : "", i === idx ? "cur" : ""].join(" "),
      onclick: async () => { await saveComment(); idx = i; render(); } }, rr.grade || String(i + 1)));
  });
  const total = DATA.assignments.reduce((s, a) => s + nGraded(a.id), 0);
  const all = DATA.assignments.reduce((s, a) => s + STATE.order[a.id].length, 0);
  $("overall").textContent = `${total} / ${all}`;
  $("barFill").style.width = (100 * total / all) + "%";
}

async function post(tid, patch) {
  const r = rating(tid);
  const body = { rater, text_id: tid, grade: r.grade ?? null, comment: r.comment || "", flag: !!r.flag, ...patch };
  const st = $("status");
  try {
    const res = await fetch("/api/rate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(res.status);
    STATE.ratings[tid] = (await res.json()).rating;
    st.className = "status"; st.textContent = "Saved ✓ " + new Date().toLocaleTimeString();
    return true;
  } catch (e) {
    st.className = "status err"; st.textContent = "Not saved: the server is not reachable. Your last click was not stored.";
    return false;
  }
}

let busy = false;  // ignore further grade input until the current save has finished
async function rate(g) {
  if (busy) return;
  busy = true;
  try {
    const tid = order()[idx];
    if (!(await post(tid, { grade: g, comment: $("comment").value, flag: $("flag").checked }))) return;
    if (order()[idx] === tid && idx < order().length - 1) idx++;
    buildSelect(); render();
  } finally {
    busy = false;
  }
}

async function saveComment() {
  if (!STATE) return;
  const tid = order()[idx], c = $("comment").value;
  if (c !== (rating(tid).comment || "")) await post(tid, { comment: c });
}

$("comment").addEventListener("blur", saveComment);
$("flag").addEventListener("change", async e => { await post(order()[idx], { flag: e.target.checked, comment: $("comment").value }); render(); });
$("clearBtn").addEventListener("click", async () => { if (await post(order()[idx], { grade: null })) { buildSelect(); render(); } });
$("prevBtn").addEventListener("click", async () => { await saveComment(); if (idx > 0) { idx--; render(); } });
$("nextBtn").addEventListener("click", async () => { await saveComment(); if (idx < order().length - 1) { idx++; render(); } });
$("nextUngradedBtn").addEventListener("click", async () => {
  await saveComment();
  const ids = order(), n = ids.length;
  for (let k = 1; k <= n; k++) { const j = (idx + k) % n; if (!rating(ids[j]).grade) { idx = j; return render(); } }
});
document.addEventListener("keydown", e => {
  if (!STATE || e.repeat || e.metaKey || e.ctrlKey || e.altKey || ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  const k = e.key.toUpperCase();
  if (GR.includes(k)) { e.preventDefault(); rate(k); }
  else if (e.key === "ArrowLeft") $("prevBtn").click();
  else if (e.key === "ArrowRight") $("nextBtn").click();
});

start();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--per-grade", type=int, help="grade only a stratified sample: N texts per intended grade "
                    "per assignment (e.g. 2 -> 12 per assignment, 144 in total); new raters only")
    args = ap.parse_args()
    global SUBSET
    if args.per_grade:
        SUBSET = make_subset(args.per_grade)
        print(f"Sample mode: {sum(map(len, SUBSET.values()))} texts ({args.per_grade} per grade per assignment)")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Rating server on http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}  (Ctrl+C to stop)")
    print(f"Ratings are saved in {RATINGS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
