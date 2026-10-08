# Graded against the group

Data, code and analysis for the paper *Graded against the group: language models make a student's grade depend on
the classmates* (PNAS Nexus, submitted).

Large language models graded 720 synthetic student texts (German as a foreign language, Swedish school years 6–9)
alone, in one prompt with the whole group (batch), and one student after another in the same conversation (chat).
The same texts received different grades depending on their classmates.

## Contents

| Path | What it is |
|---|---|
| `assignments.json` | The 12 tasks: instructions (Swedish, German), content points, language focus, E/C/A criteria, A–F grade scale |
| `texts/texts.json` | The 720 texts under blind identifiers, as the models received them |
| `texts/texts_key.json` | Answer key: intended grade (and F type) of every text |
| `texts/generation_prompt.md`, `texts/generation_spec.md` | Prompt and rule set used to generate the texts (Claude Opus 5.5) |
| `texts/raw/` | The generated texts per task before blinding |
| `texts/review_subset_2pergrade.json` | The 144 texts of the expert rating |
| `ratings/backups/leonie_FINAL_*.{json,csv}` | Blind grades of the expert rater (co-author) |
| `results/*.jsonl` | Every model request and answer (see below) |
| `analysis.ipynb` | All figures and tables of the paper and the Supplementary Material |
| `prompts.py` | The only place where prompts are built |
| `run_*.py`, `run_*.sh` | Experiment runners (Ollama, OpenRouter) |
| `make_si_prompts.py` | Rebuilds every prompt, checks it against the hash stored at run time, writes the verbatim prompts of the SI |
| `build_texts.py`, `review_server.py`, `status.py` | Text blinding, expert rating tool, progress check |

### Result files

| File | Content |
|---|---|
| `anon_inferno_<model>.jsonl` | Local models, each text graded alone, 3 runs, with/without criteria |
| `batch_main_<model>.jsonl` | Local models, batch workflow: alone (3 runs) and groups of 5/10/20/30 and weak/middle/strong groups of 20 |
| `seq_main_<model>.jsonl` | Local models, chat workflow: same design, one conversation per group |
| `or_ormain_<model>.jsonl` | Commercial models (OpenRouter), batch workflow: alone (2 runs), groups of 20, weak/middle/strong groups |
| `batch_pilot_gemma4-26b.jsonl`, `seq_seqpilot_gemma4-26b.jsonl` | Pilot including 60 students in one prompt / conversation |

Each record holds the model, the model digest (local models), seed, criteria condition, design, the anonymisation
codes, the raw answer and the parsed grades. `prompt_hash` / `system_hash` identify the exact prompt; run
`python make_si_prompts.py` to rebuild and check all of them.

## Reproduce the figures and tables

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
.venv/bin/python make_si_prompts.py
```

The notebook writes figures to `paper/figures/` (PDF, EPS) and tables to `paper/tables/` (LaTeX); `make_si_prompts.py`
writes the verbatim prompts to `paper/si/`.

## Rerun the experiments

The local models ran with Ollama (`gemma4:26b`, `qwen3:30b-a3b`) behind an authenticated gateway;
pass your own host with `--hosts` and a key file with `--api-key-file`. The commercial models ran through OpenRouter
(`--key-file`, spending cap `--max-usd`). `run_main.sh <model>` runs the full local design and `run_or_main.sh` the
commercial one; every runner is resumable. Grades can differ from ours if the model files differ: compare the
`model_digest` in the result records.

## License

Code (`*.py`, `*.sh`, `analysis.ipynb`): MIT License, see [`LICENSE`](LICENSE).
Data (`assignments.json`, `texts/`, `ratings/`, `results/`): CC BY 4.0, see
[`LICENSE-DATA`](LICENSE-DATA).
