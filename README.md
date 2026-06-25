# Freelancer × AI — Study Explorer

An interactive, dependency-free explorer for a small interview + survey study on how
**generative AI is affecting freelancers' work** (12 participants, mostly writers/creatives).
It turns raw interview transcripts and survey responses into a browsable summary, and maps the
findings onto a psychological framework of three fundamental needs.

> ⚠️ The data here is **anonymized**. All AI-generated summaries are grounded in the transcripts,
> and participant quotes are verbatim.

## Quick start (just look at it)

No server, no build, no API key needed.

1. Download / clone this repo.
2. Open **`index.html`** in any browser (double-click it).

That's it — the page reads the committed `app_data.js`.

## What's inside

The explorer has three tabs:

- **People** — one panel per participant: demographics, detected themes, an AI summary of how they
  *feel about / use / are affected by* AI and *what tools they want* (with verbatim quotes), their
  survey scores, and the full interview transcript.
- **Overview & interventions** — distribution charts (platform, years, age, AI usage, gender,
  themes), AI views before vs after the interview, a participant × theme heatmap, a Spearman
  correlation heatmap (theme share × survey measures), and a per-participant intervention roster.
- **Wise interventions** — maps the freelancers' own words onto the three fundamental needs from
  Brockner & Sherman (2020), *Wise interventions in organizations* — the **need to understand**,
  the **need for self-integrity**, and the **need to belong** — with evidence (quotes) and concrete
  product/intervention ideas for each.

## Headline findings (exploratory, n = 12)

- The dominant theme is **client devaluation / pricing pressure** (12/12 participants): the worry is
  less "AI replaces me" and more "clients now think my work is worth less."
- **Human creativity / authenticity** (11/12) is where people anchor their value; most use AI as an
  *editing assistant*, not a drafter.
- More worry/sadness language correlates with *lower* understanding of AI and *higher* feelings of
  being devalued and threatened (directional only, small sample).

## Repository layout

```
.
├── index.html          # the app (open this)
├── app_data.js         # generated data bundle the app reads (committed)
├── llm_cache.json      # cached LLM outputs so rebuilds are instant/offline (committed)
├── build_app_data.py   # regenerates app_data.js from the xlsx
├── data/
│   └── study_complete_prolific_anonymized.xlsx   # anonymized study export
├── requirements.txt
├── .env.example
└── .gitignore
```

## Regenerate the data (optional)

You only need this if you change the data, prompts, or codebook.

```bash
pip install -r requirements.txt
cp .env.example .env.local      # then put your OpenAI key in .env.local
python build_app_data.py
```

- LLM results are cached in `llm_cache.json`, so reruns don't re-call the API.
- To refresh just the wise-intervention synthesis, delete the `"_wise"` entry in
  `llm_cache.json` and rerun. Delete the whole file to refresh everything.
- With **no** API key, the script still rebuilds `app_data.js` from the existing cache.

## Method notes

- Themes are detected with a transparent keyword codebook (auditable, deterministic).
- Per-person summaries and the wise-intervention synthesis are produced by an LLM, constrained to
  the transcript and required to quote verbatim.
- The sample is 12 participants — treat everything as **exploratory** and directional, not as
  statistical evidence.

## References

- Brockner, J., & Sherman, D. K. (2020). *Wise interventions in organizations.* Research in
  Organizational Behavior, 39, 100125.
