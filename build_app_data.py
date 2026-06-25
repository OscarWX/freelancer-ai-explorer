"""Build the data bundle for the interactive freelancer-AI explorer.

Inputs:
  - data/study_complete_prolific_anonymized.xlsx : the anonymized study export

Outputs (in the repo root):
  - app_data.js     : `window.APP_DATA = {...}` consumed by index.html
  - llm_cache.json  : cached LLM inference (so reruns are instant / offline)

You only need this script to REGENERATE the data. To just view the app, open
index.html directly (it reads the committed app_data.js).

Run:  python build_app_data.py
Set OPENAI_API_KEY in a local .env.local (copy .env.example). The model is auto-picked
as the newest available from a priority list; override with OPENAI_MODEL. With no key,
cached results are reused, so the bundle still rebuilds from llm_cache.json.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent          # repo root
DATA = HERE / "data"
XLSX = DATA / "study_complete_prolific_anonymized.xlsx"
CACHE = HERE / "llm_cache.json"

# --------------------------------------------------------------------------- env
def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

ENV = load_env(HERE / ".env.local")
API_KEY = ENV.get("OPENAI_API_KEY", "").strip()

# --------------------------------------------------------------------- readable labels
SURVEY_GROUPS = [
    ("Confidence (before)", [
        ("pre_conf_can_learn", "I can learn what I need to keep up"),
        ("pre_conf_problem_solve", "I can solve problems that come up in my work"),
        ("pre_conf_effort_results", "My effort leads to the results I want"),
    ]),
    ("Understanding of AI (before)", [
        ("pre_understand_capabilities", "I understand what AI can and can't do"),
        ("pre_understand_keeps_up", "I keep up with new AI tools"),
    ]),
    ("Views on AI (before)", [
        ("pre_views_improve_life", "AI will improve my life"),
        ("pre_views_improve_work", "AI will improve my work"),
        ("pre_views_will_use", "I will use AI tools"),
        ("pre_views_positive_humanity", "AI is positive for humanity"),
    ]),
    ("Professional identity (before)", [
        ("pre_identity_skills_central", "My skills are central to who I am"),
        ("pre_identity_valued_skills_matter", "The skills I'm valued for matter to me"),
        ("pre_identity_recognition", "I get recognition for my skills"),
    ]),
    ("Career outlook (before)", [
        ("pre_career_excited", "I feel excited about my career"),
        ("pre_career_eager", "I'm eager about where my career is going"),
        ("pre_career_unsure", "I feel unsure about my career (reverse-scored)"),
        ("pre_career_right_decisions", "I'm making the right career decisions"),
    ]),
    ("Views on AI (after interview)", [
        ("post_views2_improve_life", "AI will improve my life"),
        ("post_views2_improve_work", "AI will improve my work"),
        ("post_views2_will_use", "I will use AI tools"),
        ("post_views2_positive_humanity", "AI is positive for humanity"),
    ]),
    ("Ownership of AI-assisted work (after)", [
        ("post_ownership_feels_mine", "Work I do with AI still feels like mine"),
        ("post_ownership_name_on_it", "I'm happy to put my name on AI-assisted work"),
        ("post_ownership_expression", "My work is an expression of me"),
        ("post_ownership_style_voice", "My work reflects my own style and voice"),
    ]),
    ("AI and meaningful work (after)", [
        ("post_ai_tedious_vs_meaningful", "AI frees me for meaningful (vs tedious) work"),
    ]),
    ("Worry about the future (after)", [
        ("post_future_could_be_replaced", "My work could be replaced by AI"),
        ("post_future_worried_replaced", "I'm worried about being replaced by AI"),
        ("post_future_worried_field", "I'm worried about the future of my field"),
    ]),
    ("How others see my work (after)", [
        ("post_others_value_less", "Others value my work less because of AI"),
        ("post_others_disclosure_risky", "Disclosing that I use AI feels risky"),
    ]),
    ("Intent (after)", [
        ("post_intent_keep_using", "I intend to keep using AI tools"),
    ]),
]

DEMO_FIELDS = [
    ("demo_freelance_type", "Main job"),
    ("demo_freelance_category", "Field"),
    ("demo_main_platform", "Main platform"),
    ("demo_platforms", "Platforms used"),
    ("demo_freelance_years", "Years freelancing"),
    ("demo_ai_usage_freq", "AI usage frequency"),
    ("demo_age", "Age"),
    ("demo_gender", "Gender"),
    ("demo_us_state", "Location"),
]

# --------------------------------------------------------------------- theme codebook
CODEBOOK = {
    "AI improves productivity": ["faster", "efficien", "productiv", "save time", "saves time", "saving time", "speed", "streamline", "quicker", "automate", "automating", "time-saving", "boost", "helps me", "makes it easier", "more done"],
    "Job displacement concern": ["replace", "replaced", "replacing", "lose my job", "lose work", "losing work", "out of a job", "out of work", "obsolete", "take over", "taking over", "no longer need", "don't need me", "redundant", "put out"],
    "Need to learn AI skills": ["learn", "learning", "prompt", "prompting", "upskill", "train", "training", "keep up", "adapt", "adapting", "figure out how", "get better at"],
    "Entry-level jobs at risk": ["entry-level", "entry level", "junior", "beginner", "newcomer", "new freelancer", "starting out", "break in", "breaking in", "get started", "first job"],
    "AI as a tool/assistant": ["a tool", "as a tool", "assistant", "collaborat", "supplement", "helper", "aid", "sidekick", "starting point", "brainstorm", "co-pilot", "copilot"],
    "Quality concerns about AI": ["quality", "mediocre", "generic", "average", "accuracy", "inaccurate", "errors", "mistakes", "hallucinat", "not as good", "soulless", "bland", "good enough", "low quality"],
    "Client devaluation / pricing pressure": ["client", "rate", "pricing", "price", "cheap", "devalue", "value less", "undercut", "pushback", "push back", "budget", "pay less", "lower pay", "too high", "cut costs"],
    "Human creativity / authenticity": ["creativ", "human touch", "authentic", "unique", "original", "soul", "personal touch", "my voice", "my style", "art", "craft", "genuine"],
    "Disclosure / stigma / ethics": ["disclose", "disclosure", "stigma", "hide", "admit", "transparen", "ethic", "cheat", "plagiar", "honest about", "secret"],
    "Emotional impact (worry/sadness)": ["worried", "worry", "anxious", "anxiety", "sad", "scared", "fear", "afraid", "cynical", "stress", "uncertain", "nervous", "frustrat", "depress"],
}
CODE_RE = {t: re.compile("(" + "|".join(re.escape(k) for k in kws) + ")", re.IGNORECASE) for t, kws in CODEBOOK.items()}
THEMES = list(CODEBOOK.keys())

SECTION_KEYWORDS = {
    "Financial": ["earn", "income", "money", "pay", "rate", "price", "pricing", "financial", "afford", "cost", "budget", "charge", "secure"],
    "Identity": ["feel", "your own", "identity", "skills", "role", "ownership", "pride", "meaningful", "yourself", "prefer to do"],
    "Relational": ["input", "feedback", "turn to", "support", "another person", "community", "mentor", "resources", "advice", "help"],
    "Reputational": ["client", "quality of your work", "reputation", "trust", "stand out", "credibility", "value of your work", "judge"],
    "Closing": ["looking back", "changed the most", "building something", "four areas", "focus on first"],
}

TURN_RE = re.compile(r"(INTERVIEWER|USER):\s*", re.IGNORECASE)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")


def classify_section(prompt: str) -> str:
    if not prompt:
        return "Opening"
    low = prompt.lower()
    scores = {s: sum(low.count(k) for k in kws) for s, kws in SECTION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Opening"


def parse_turns(transcript: str):
    parts = TURN_RE.split(str(transcript))
    turns = []
    for i in range(1, len(parts) - 1, 2):
        spk = parts[i].upper()
        txt = WS_RE.sub(" ", parts[i + 1]).strip()
        if txt:
            turns.append((spk, txt))
    return turns


def qa_pairs(turns):
    """Pair each interviewer prompt with the user's following answer."""
    pairs, pending = [], None
    for spk, txt in turns:
        if spk == "INTERVIEWER":
            pending = txt
        else:
            pairs.append({"section": classify_section(pending or ""),
                          "question": pending or "", "answer": txt})
            pending = None
    return pairs


def user_chunks(turns):
    chunks = []
    for spk, txt in turns:
        if spk != "USER":
            continue
        sents = [s.strip() for s in SENT_RE.split(txt) if s.strip()]
        buf = ""
        for s in sents:
            buf = (buf + " " + s).strip() if buf else s
            if len(buf.split()) >= 6:
                chunks.append(buf); buf = ""
        if buf:
            if chunks:
                chunks[-1] = (chunks[-1] + " " + buf).strip()
            else:
                chunks.append(buf)
    return chunks

# --------------------------------------------------------------------- LLM
def pick_model(client) -> str:
    if ENV.get("OPENAI_MODEL_FORCE"):
        return ENV["OPENAI_MODEL_FORCE"]
    priority = ["gpt-5.5", "gpt-5.5-mini", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o",
                ENV.get("OPENAI_MODEL", "gpt-4o-mini"), "gpt-4o-mini"]
    try:
        available = {m.id for m in client.models.list()}
        for m in priority:
            if m in available:
                return m
    except Exception as e:
        print("  (could not list models:", e, ")")
    return ENV.get("OPENAI_MODEL", "gpt-4o-mini")


SYS_PROMPT = (
    "You are a qualitative research assistant. You read one freelancer's interview transcript "
    "about how AI affects their work, and you answer four questions. Ground every answer ONLY "
    "in this transcript; never invent facts. For each question give a concise 1-3 sentence "
    "summary and 1-3 short VERBATIM quotes taken from the participant's own (USER) words. "
    "Also propose 1-3 concrete, specific interventions or tools that would genuinely help THIS "
    "person, based on what they said. Reply with strict JSON only, no markdown, matching:\n"
    "{\n"
    '  "feel": {"summary": str, "quotes": [str]},\n'
    '  "use": {"summary": str, "quotes": [str]},\n'
    '  "job_impact": {"summary": str, "quotes": [str]},\n'
    '  "tools_wanted": {"summary": str, "quotes": [str]},\n'
    '  "interventions": [str]\n'
    "}\n"
    "Question meanings: feel = how the person feels about AI; use = how the person uses AI; "
    "job_impact = how their job/income is affected by AI; tools_wanted = what AI tools they "
    "find useful or wish existed."
)


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def run_llm(client, model, transcript: str):
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": "TRANSCRIPT:\n" + transcript},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"})
    except Exception:
        resp = client.chat.completions.create(model=model, messages=messages)
    return parse_json(resp.choices[0].message.content)


# ----------------------------------------------------- wise-intervention synthesis
# Brockner & Sherman (2020), "Wise interventions in organizations" — three
# fundamental needs (need to understand, need for self-integrity, need to belong).
WISE_SYS_PROMPT = (
    "You are a research assistant applying the 'wise interventions' framework from "
    "Brockner & Sherman (2020), 'Wise interventions in organizations'. Wise interventions "
    "are small, theory-based shifts in how people construe themselves, others, and their "
    "situation. They target three fundamental needs:\n"
    "1. NEED TO UNDERSTAND — threatened by role ambiguity, role conflict, unclear "
    "expectations, or 'flavor of the month' initiatives. People want a clear, stable, "
    "workable interpretation of what is happening and what is expected.\n"
    "2. NEED FOR SELF-INTEGRITY — threatened in work with frequent rejection or negative "
    "feedback (e.g., sales). People who construe rejection in less self-blaming ways "
    "persist more; self-affirmation protects a sense of being competent and good.\n"
    "3. NEED TO BELONG — threatened by non-inclusive environments, exclusion from formal "
    "decisions, or exclusion from informal social life. People want to feel accepted and "
    "connected.\n\n"
    "You are given evidence from freelancers describing how generative AI is affecting "
    "their work (per-person summaries plus verbatim transcript quotes). Map their "
    "experience onto the three needs and propose what WE could build to address each need.\n"
    "Rules: ground every claim ONLY in the supplied data; never invent facts. Quotes must "
    "be VERBATIM participant words and each tagged with the correct participant id. For each "
    "need give 3-5 evidence items and 2-4 concrete product/intervention proposals. Each "
    "proposal must name the wise MECHANISM (the specific construal it changes). Reply with "
    "strict JSON only, no markdown, matching:\n"
    "{\n"
    '  "overview": str,\n'
    '  "needs": [\n'
    "    {\n"
    '      "key": "understand" | "self_integrity" | "belong",\n'
    '      "name": str,\n'
    '      "definition": str,\n'
    '      "how_it_shows_up": str,\n'
    '      "evidence": [ {"point": str, "pid": int, "quote": str} ],\n'
    '      "products": [ {"name": str, "what": str, "how_it_helps": str, "mechanism": str} ]\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Return the three needs in this order: understand, self_integrity, belong."
)


def build_wise_corpus(people) -> str:
    """Compact but grounded corpus: per-person summary data + transcript answers."""
    blocks = []
    for p in people:
        lines = [f"### Participant {p['id']} — {p['job']} ({p['field']})"]
        s = p.get("ai_summary") or {}
        for k, label in [("feel", "Feels"), ("use", "Uses AI"),
                         ("job_impact", "Job impact"), ("tools_wanted", "Wants")]:
            obj = s.get(k) or {}
            if obj.get("summary"):
                lines.append(f"- {label}: {obj['summary']}")
        themes = ", ".join(f"{t['theme']} ({t['count']})" for t in p.get("themes", []))
        if themes:
            lines.append(f"- Themes: {themes}")
        # transcript: the participant's own answers (grounding for verbatim quotes)
        answers = [qa["answer"] for qa in p.get("transcript", [])]
        if answers:
            lines.append("- Transcript answers:")
            for a in answers:
                lines.append(f"    * {a}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run_wise(client, model, people):
    corpus = build_wise_corpus(people)
    messages = [
        {"role": "system", "content": WISE_SYS_PROMPT},
        {"role": "user", "content": "FREELANCER EVIDENCE:\n" + corpus},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"})
    except Exception:
        resp = client.chat.completions.create(model=model, messages=messages)
    return parse_json(resp.choices[0].message.content)


# --------------------------------------------------------------------- main build
def main():
    raw = pd.read_excel(XLSX, sheet_name="responses")
    raw = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")])
    raw = raw.dropna(subset=["chat_transcript"]).reset_index(drop=True)
    print(f"Participants: {len(raw)}")

    # ---- LLM client + cache ----
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    client = model = None
    if API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        model = pick_model(client)
        print("Using model:", model)
    else:
        print("No OPENAI_API_KEY found - LLM summaries will be skipped.")

    people = []
    all_theme_counts = {t: 0 for t in THEMES}
    theme_participant = {t: set() for t in THEMES}
    person_theme_matrix = {}
    person_total = {}

    for _, row in raw.iterrows():
        pid = int(row["id"])
        turns = parse_turns(row["chat_transcript"])
        pairs = qa_pairs(turns)
        chunks = user_chunks(turns)
        person_total[pid] = len(chunks)

        # themes for this person
        theme_counts = {}
        theme_quotes = {}
        for t in THEMES:
            hits = [c for c in chunks if CODE_RE[t].search(c)]
            theme_counts[t] = len(hits)
            if hits:
                theme_quotes[t] = hits[0]
                all_theme_counts[t] += len(hits)
                theme_participant[t].add(pid)
        person_theme_matrix[pid] = theme_counts

        # readable survey
        survey_groups = []
        for gname, items in SURVEY_GROUPS:
            rows = []
            for col, label in items:
                val = row.get(col)
                rows.append({"label": label,
                             "value": None if pd.isna(val) else int(val)})
            survey_groups.append({"group": gname, "items": rows})

        # demographics
        demo = []
        for col, label in DEMO_FIELDS:
            val = row.get(col)
            if pd.isna(val):
                continue
            demo.append({"label": label, "value": (int(val) if isinstance(val, (int, np.integer)) else str(val))})

        # LLM
        key = str(pid)
        if client and key not in cache:
            print(f"  LLM for participant {pid} ...")
            try:
                cache[key] = run_llm(client, model, str(row["chat_transcript"]))
                cache[key]["_model"] = model
            except Exception as e:
                print(f"    failed: {e}")
                cache[key] = None
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        llm = cache.get(key)

        people.append({
            "id": pid,
            "job": str(row.get("demo_freelance_type", "")),
            "field": str(row.get("demo_freelance_category", "")),
            "demographics": demo,
            "themes": [{"theme": t, "count": theme_counts[t], "quote": theme_quotes.get(t, "")}
                       for t in THEMES if theme_counts[t] > 0],
            "survey": survey_groups,
            "transcript": pairs,
            "ai_summary": llm,
        })

    # ---- chart data for the overview tab (rendered natively in the UI) ----
    charts = build_charts(raw, theme_participant, person_theme_matrix)
    charts["correlation"] = build_correlation(raw, person_theme_matrix, person_total)

    # ---- interventions roster ----
    interventions = []
    for p in people:
        ivs = (p["ai_summary"] or {}).get("interventions") or []
        interventions.append({"id": p["id"], "job": p["job"], "interventions": ivs})

    # ---- wise-intervention synthesis (3 fundamental needs) ----
    if client and "_wise" not in cache:
        print("  LLM for wise-intervention synthesis (3 needs) ...")
        try:
            cache["_wise"] = run_wise(client, model, people)
            cache["_wise"]["_model"] = model
        except Exception as e:
            print(f"    wise synthesis failed: {e}")
            cache["_wise"] = None
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    wise = cache.get("_wise")

    data = {
        "n_participants": len(people),
        "model": model or "none",
        "themes": THEMES,
        "people": people,
        "interventions": interventions,
        "charts": charts,
        "wise": wise,
    }
    out = HERE / "app_data.js"
    out.write_text("window.APP_DATA = " + json.dumps(data, ensure_ascii=False) + ";",
                   encoding="utf-8")
    print("Wrote", out)


def build_charts(raw, theme_participant, person_theme_matrix):
    def counts(col, order=None):
        vc = raw[col].dropna().astype(str).str.strip().value_counts()
        if order:
            idx = [o for o in order if o in vc.index] + [i for i in vc.index if i not in order]
            vc = vc.reindex(idx)
        return {"labels": list(vc.index), "values": [int(x) for x in vc.values]}

    years_order = ["Less than 1 year", "1 to 3 years", "4 to 7 years", "8 to 15 years", "More than 15 years"]
    usage_order = ["Never", "Rarely", "Sometimes", "Often", "Almost always", "Always"]

    # age binned by decade
    ages = pd.to_numeric(raw["demo_age"], errors="coerce").dropna()
    bins = [20, 30, 40, 50, 60, 70]
    age_labels = ["20-29", "30-39", "40-49", "50-59", "60-69"]
    agecut = pd.cut(ages, bins=bins, right=False, labels=age_labels).value_counts().reindex(age_labels, fill_value=0)
    age_chart = {"labels": age_labels, "values": [int(x) for x in agecut.values]}

    # themes mentioned = participants per theme (sorted desc)
    tp = sorted(((t, len(s)) for t, s in theme_participant.items()), key=lambda kv: kv[1], reverse=True)
    themes_chart = {"labels": [t for t, _ in tp], "values": [n for _, n in tp]}

    # AI views before vs after
    pairs = [("improve life", "pre_views_improve_life", "post_views2_improve_life"),
             ("improve work", "pre_views_improve_work", "post_views2_improve_work"),
             ("will use", "pre_views_will_use", "post_views2_will_use"),
             ("positive humanity", "pre_views_positive_humanity", "post_views2_positive_humanity")]
    ai_views = {
        "labels": [p[0] for p in pairs],
        "pre": [round(float(pd.to_numeric(raw[p[1]], errors="coerce").mean()), 2) for p in pairs],
        "post": [round(float(pd.to_numeric(raw[p[2]], errors="coerce").mean()), 2) for p in pairs],
    }

    # participant x theme heatmap (rows = themes, cols = participant ids)
    pids = sorted(person_theme_matrix.keys())
    matrix = [[int(person_theme_matrix[pid].get(t, 0)) for pid in pids] for t in THEMES]

    return {
        "main_platform": counts("demo_main_platform"),
        "freelance_years": counts("demo_freelance_years", years_order),
        "age": age_chart,
        "ai_usage": counts("demo_ai_usage_freq", usage_order),
        "gender": counts("demo_gender"),
        "themes_participants": themes_chart,
        "ai_views": ai_views,
        "heatmap": {"themes": THEMES, "participants": pids, "matrix": matrix},
    }


def build_correlation(raw, person_theme_matrix, person_total):
    """Spearman rho between transcript theme shares and survey composites
    (mirrors analysis/survey_theme_correlation.ipynb)."""
    survey = raw.copy()
    likert = [c for _, items in SURVEY_GROUPS for c, _ in items]
    for c in likert:
        survey[c] = pd.to_numeric(survey[c], errors="coerce")

    AI_FREQ_MAP = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4, "Always": 5}
    YEARS_MAP = {"Less than 1 year": 0, "1 to 3 years": 1, "4 to 7 years": 2, "8 to 15 years": 3, "More than 15 years": 4}

    def mean_of(cols):
        return survey[cols].mean(axis=1)

    comp = pd.DataFrame({"id": survey["id"].astype(int)})
    comp["age"] = pd.to_numeric(survey["demo_age"], errors="coerce")
    comp["ai_usage_freq"] = survey["demo_ai_usage_freq"].map(AI_FREQ_MAP)
    comp["freelance_years"] = survey["demo_freelance_years"].map(YEARS_MAP)
    comp["confidence"] = mean_of(["pre_conf_can_learn", "pre_conf_problem_solve", "pre_conf_effort_results"])
    comp["understanding_ai"] = mean_of(["pre_understand_capabilities", "pre_understand_keeps_up"])
    comp["ai_views_pre"] = mean_of(["pre_views_improve_life", "pre_views_improve_work", "pre_views_will_use", "pre_views_positive_humanity"])
    comp["ai_views_post"] = mean_of(["post_views2_improve_life", "post_views2_improve_work", "post_views2_will_use", "post_views2_positive_humanity"])
    comp["ai_views_change"] = comp["ai_views_post"] - comp["ai_views_pre"]
    comp["identity_centrality"] = mean_of(["pre_identity_skills_central", "pre_identity_valued_skills_matter", "pre_identity_recognition"])
    comp["career_optimism"] = (
        survey[["pre_career_excited", "pre_career_eager", "pre_career_right_decisions"]].sum(axis=1)
        + (8 - survey["pre_career_unsure"])) / 4
    comp["ownership_post"] = mean_of(["post_ownership_feels_mine", "post_ownership_name_on_it", "post_ownership_expression", "post_ownership_style_voice"])
    comp["ai_meaningful"] = survey["post_ai_tedious_vs_meaningful"]
    comp["future_threat"] = mean_of(["post_future_could_be_replaced", "post_future_worried_replaced", "post_future_worried_field"])
    comp["others_devalue"] = mean_of(["post_others_value_less", "post_others_disclosure_risky"])
    comp["intent_keep_using"] = survey["post_intent_keep_using"]
    comp = comp.set_index("id")

    pids = sorted(person_theme_matrix.keys())
    share = pd.DataFrame(
        {t: {pid: (person_theme_matrix[pid].get(t, 0) / person_total[pid] if person_total[pid] else 0.0)
             for pid in pids} for t in THEMES})
    merged = comp.join(share, how="inner")

    survey_cols = [c for c in comp.columns if merged[c].nunique() >= 2]
    usable_themes = [t for t in THEMES if (merged[t] > 0).sum() >= 3 and merged[t].nunique() >= 2]

    rho_rows, p_rows = [], []
    for t in usable_themes:
        rr, pp = [], []
        for s in survey_cols:
            x, y = merged[t], merged[s]
            m = x.notna() & y.notna()
            if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
                rr.append(None); pp.append(None)
            else:
                r, p = spearmanr(x[m], y[m])
                rr.append(round(float(r), 2)); pp.append(round(float(p), 3))
        rho_rows.append(rr); p_rows.append(pp)

    return {"rows": usable_themes, "cols": survey_cols, "rho": rho_rows, "p": p_rows}


if __name__ == "__main__":
    main()
