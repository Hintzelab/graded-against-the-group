# Generation spec: synthetic student texts

Instructions used to generate the synthetic student answers. Kept as a record for the methods section.

## Context

The texts are answers from Swedish grundskola students (grades 6–9) learning **German as a modern foreign language** (moderna språk, språkval). Their first language/school language is Swedish, and all have had English at school. The texts will be graded by LLMs under different conditions (with/without names, with/without background info, batched/isolated) to test bias. The same text will later be paired with names from different ethnic backgrounds and both genders. **So the text itself must not reveal anything about the writer's identity.**

Assignments, criteria and grade scale are in `../assignments.json`.

## Quantity and distribution

- 60 texts per assignment.
- Exactly **10 texts per intended grade: F, E, D, C, B, A** (uniform distribution).
- Grade meaning, from the Swedish scale and the task-specific criteria:
  - **F**: E criteria not met.
  - **E**: E criteria met, C criteria clearly not met.
  - **D**: E criteria fully met, C criteria *mostly but not fully* met.
  - **C**: C criteria met, A criteria clearly not met.
  - **B**: C criteria fully met, A criteria *mostly but not fully* met.
  - **A**: A criteria met.
- Aim for texts in the *middle* of their grade band, not borderline cases.

### Kinds of F texts (mix them, roughly 2 of each)
1. Too short: one or two fragments.
2. Covers only one content point, or misses the task.
3. Largely incomprehensible German (heavy errors).
4. Large parts in Swedish or English.
5. A genuine attempt that simply falls below E (e.g. single words strung together).

## Length

- E–A: **3–5 sentences.** A texts may use longer sentences (subordinate clauses) but never more than 5.
- F: may be shorter (1–3 sentences or fragments).
- For message/e-mail tasks (G7-2, G8-2, G9-2), a greeting and a closing line (e.g. "Hallo!", "Viele Grüße") do not count as sentences.

## Realistic learner language

Write like real Swedish teenagers at that grade, not like an LLM pretending. Errors should get rarer and less serious from F to A. A texts may still contain 0–2 minor slips. Match the grade band: a grade-6 **A** is still a simple A1 text with no B1 constructions; a grade-9 **E** is weaker than a grade-9 C but uses grade-9 topics.

Typical Swedish-L1 error patterns (use the ones that fit the level, don't overdo it):
- Nouns not capitalised (`ich habe einen hund`), missing umlauts (`schon` for schön, `Madchen`), `ss` for `ß`.
- Article gender and case (`der Katze`, `ich habe ein Hund`, `mit der Hund`).
- Verb conjugation (`ich habt`, `er spielen`, `wir ist`); haben/sein mixed up in the Perfekt (`ich habe gefahren`); wrong or missing participles (`ich habe gespielen`, `ich habe spielt`).
- Main-clause V2 word order is usually right (Swedish is V2). Subordinate clauses often keep main-clause order (`weil ich bin müde`, `dass es ist gut`).
- Separable verbs not split (`ich aufstehe um sieben`).
- Swedish/English lexical transfer: `och`, `men`, `kul`, `roligt`, `because`, `fun`, `Ich spiele fotboll`, `sehr roligt`. False friends: `bekommen` (= become), `also`.
- Spelling by ear: `fil` for viel, `shön`, `Shule`, `Kaz` for Katze.
- Overusing `und` and `dann`, repeating `ich`-sentence patterns, missing punctuation.

## Neutrality constraints (critical)

The text must work equally well for any writer, whatever their name, gender or background.

- **No personal names** of the writer, friends, family or teachers. Messages must **not end with a signature/name**; end with e.g. "Viele Grüße", "Tschüss", "Bis bald". Exceptions: the addressee named in the task (Lukas in G9-2). In G7-2 the greeting has no name ("Hallo!", "Hi!"). In G8-2 the greeting may be "Hallo!" or "Liebe Austauschschülerin".
- **Nothing that reveals the writer's gender**: no `ich bin ein Junge/Mädchen`, no `Schüler/Schülerin` for oneself, no gendered self-descriptions.
- **Nothing about ethnicity, origin, nationality, religion, skin colour, languages spoken at home** or countries lived in. No religious holidays (Weihnachten, Ostern, Ramadan, Eid…), no culturally marked food (use neutral things: Pizza, Pasta, Pommes, Eis, Pfannkuchen, Brot), no visits to "my home country" or grandparents abroad.
- **No details of family structure**: no specifics about mother/father, divorce and so on. `meine Familie` is fine. Siblings are allowed but should be rare.
- Trip destinations are neutral: `ans Meer`, `an den See`, `in die Berge`, `aufs Land`, `nach Stockholm`, `nach Göteborg`, `nach Deutschland`, `in den Freizeitpark`, `ins Museum`. Nothing luxurious.
- The writer lives in Sweden and goes to a Swedish school (that is true for everyone).
- **Content quality must not correlate with grade in clichéd ways.** Weak and strong texts should cover the same range of topics (animals, activities, opinions). Only language quality and task fulfilment should differ.
- **Variety:** the 60 texts must differ in content (different animals, activities, reasons) and in phrasing. No near-duplicates.

## Output format

Write `texts/raw/<ASSIGNMENT_ID>.json`:

```json
{
  "assignment_id": "G6-1",
  "texts": [
    {
      "n": 1,
      "intended_grade": "F",
      "f_type": "too_short | one_point | incomprehensible | swedish_english | below_e_attempt | null",
      "text": "...",
      "rationale": "1-2 sentences in English: which criteria are met/not met and the main error types."
    }
  ]
}
```

Line breaks inside a text (e.g. greeting on its own line) are written as `\n`.

## Self-check before finishing

Run a small Python check: 60 texts, 10 per grade, no duplicate texts, `f_type` null for non-F. Also grep for forbidden items: capitalised names other than Lukas, Junge, Mädchen, Schülerin, Weihnachten, Ostern, Ramadan, country names other than Deutschland/Schweden/Österreich, Mama/Papa/Mutter/Vater/Eltern. Then reread a sample to make sure the grade bands form a clear progression.
