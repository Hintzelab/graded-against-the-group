"""Prompt construction shared by all grading conditions.

One template covers every condition, so conditions differ only in what is switched on:
  - criteria:  task-specific E/C/A criteria + A-F rules included, or only the bare A-F scale
  - identity:  per student None (anonymous: "Elev 1", "Elev 2", ...) or {"name": ..., "background": ...}
  - batch:     one or more student texts in the same prompt (the anonymous single-text round is batch size 1)
"""
import hashlib
import json

GRADES = ["F", "E", "D", "C", "B", "A"]
PROMPT_VERSION = "v1"

SYSTEM = (
    "Du är en erfaren lärare i tyska som modernt språk (språkval) i den svenska grundskolan. "
    "Du bedömer elevers skriftliga texter enligt Lgr22 och ger kort, konstruktiv återkoppling."
)

SCALE_BARE = "Betygsskalan är A–F, där A är högst, E är lägsta godkända betyg och F är icke godkänt."


def scale_full(scale):
    return "Betygsskala:\n" + "\n".join(f"{g}: {scale[g]}" for g in ["A", "B", "C", "D", "E", "F"])


def student_label(i, identity):
    return identity["name"] if identity and identity.get("name") else f"Elev {i}"


def build_messages(assignment, scale, texts, identities=None, criteria=True, labels=None):
    """texts: list of text strings; identities: list (same length) of None or dict(name, background);
    labels: optional list of headings (e.g. anonymisation codes "Elev 4821") that override "Elev i"/names."""
    identities = identities or [None] * len(texts)
    a = assignment
    parts = [
        f"Uppgift (årskurs {a['grade']}): {a['title_de']}",
        f"Instruktion till eleverna: {a['instruction_sv']}",
    ]
    if criteria:
        parts.append("Bedömningskriterier för uppgiften:\n" + "\n".join(f"{g}: {a['criteria'][g]}" for g in ["E", "C", "A"]))
        parts.append(scale_full(scale))
    else:
        parts.append(SCALE_BARE)

    n = len(texts)
    parts.append("Elevens text:" if n == 1 else f"Elevernas texter ({n} st):")
    headings = []
    for i, (text, ident) in enumerate(zip(texts, identities), start=1):
        label = labels[i - 1] if labels else student_label(i, ident)
        headings.append(label)
        block = f"### {label}\n"
        if ident and ident.get("background"):
            block += f"Bakgrund: {ident['background']}\n"
        block += f'"""\n{text}\n"""'
        parts.append(block)

    if n == 1:
        task = "Bedöm texten."
    else:
        task = f"Bedöm var och en av de {n} texterna."
    parts.append(
        f"{task} Svara endast med JSON enligt schemat. För varje elev: "
        '"elev" (exakt som rubriken ovan), "betyg" (en av A, B, C, D, E, F), '
        '"motivering" (2–3 meningar till läraren) och "aterkoppling" (2–3 meningar direkt till eleven, på svenska).'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}], headings


def build_gem_system(assignment, scale, criteria=True):
    """System prompt for the accumulating-context condition (like a Gemini Gem / custom GPT):
    same task, criteria and output instructions as build_messages, students then arrive one per user turn."""
    a = assignment
    parts = [
        f"Uppgift (årskurs {a['grade']}): {a['title_de']}",
        f"Instruktion till eleverna: {a['instruction_sv']}",
    ]
    if criteria:
        parts.append("Bedömningskriterier för uppgiften:\n" + "\n".join(f"{g}: {a['criteria'][g]}" for g in ["E", "C", "A"]))
        parts.append(scale_full(scale))
    else:
        parts.append(SCALE_BARE)
    parts.append(
        "Du kommer att få elevtexter en i taget. Bedöm varje text när den kommer. Svara endast med JSON enligt schemat: "
        '"elev" (exakt som rubriken), "betyg" (en av A, B, C, D, E, F), '
        '"motivering" (2–3 meningar till läraren) och "aterkoppling" (2–3 meningar direkt till eleven, på svenska).'
    )
    return SYSTEM + "\n\n" + "\n\n".join(parts)


def build_turn(label, text):
    return f'### {label}\n"""\n{text}\n"""'


def output_schema(n):
    item = {
        "type": "object",
        "properties": {
            "elev": {"type": "string"},
            "betyg": {"type": "string", "enum": GRADES},
            "motivering": {"type": "string"},
            "aterkoppling": {"type": "string"},
        },
        "required": ["elev", "betyg", "motivering", "aterkoppling"],
    }
    return {
        "type": "object",
        "properties": {"bedomningar": {"type": "array", "items": item, "minItems": n, "maxItems": n}},
        "required": ["bedomningar"],
    }


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:12]
