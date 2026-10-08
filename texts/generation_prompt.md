# Text-generation prompt

Prompt given to each of 12 Claude Opus 5.5 agents (Claude Code, 2026-09-25), one per task;
<TASK> stands for the task id (G6-1 ... G9-3). Each agent also read texts/generation_spec.md.

---

You are generating synthetic German learner texts for a research study on bias in LLM grading.

Project folder: [project folder]

1. Read `texts/generation_spec.md` in full and follow it exactly.
2. Read `assignments.json`: the meta section (grade scale) and the entry for assignment **<TASK>**.
3. Write 60 answers to assignment <TASK> (10 each at F, E, D, C, B, A) to `texts/raw/<TASK>.json` in the format the spec gives.
4. Run the self-check from the spec with Python, fix any problems, and rerun until it passes.

The most important things: learner language that is realistic for a Swedish student at this grade; a clear progression across the grade bands; the neutrality constraints (no names, no gender, origin or religion cues, no signature); and varied content. Write only that one file. Report back briefly: the file path, whether the check passed, and one example text per grade.
