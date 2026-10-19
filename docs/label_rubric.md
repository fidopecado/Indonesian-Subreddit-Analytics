# Sentiment Labeling Rubric — r/indonesia comments

Label **each** comment with exactly one of: `positive`, `neutral`, `negative`.

## Definitions

- **positive** — expresses approval, praise, gratitude, amusement/celebration, hope, or support. Includes genuine compliments and warm agreement.
- **negative** — expresses criticism, anger, contempt, mockery, frustration, despair, insult, or hostility. Includes sarcasm that attacks something.
- **neutral** — factual statements, questions, procedural answers, speculation, descriptions, and jokes with no clear valence. Also use when the sentiment is genuinely ambiguous or mixed and no side dominates.

## Rules

1. **Judge the comment itself**, not the topic. A calm report about bad news is `neutral`; an outraged report is `negative`.
2. **Sarcasm/irony**: label the *intended* meaning. `"Wah hebat ya, kerja 10 tahun gak kelar"` = `negative`.
3. **Questions** are `neutral` unless clearly hostile or praising.
4. **Indonesian slang** is normal, not negative: `gw`, `gue`, `lu`, `gak`, `ga`, `nggak`, `sih`, `deh`, `dong`, `banget`, `wkwk`, `anjir`, `bjir`, `cok`.
5. **`wkwk` / `haha` / emoji**: treat as light amusement. Amusement alone = `neutral`. Amusement used to mock someone = `negative`.
6. **Code-mixed English** is common; label by meaning, not language.
7. **Mixed sentiment**: pick the dominant one. If truly equal, use `neutral`.
8. **Very short comments**: `"iya"`, `"setuju"`, `"mantap"` → `positive`. `"gak"`, `"nggak"`, `"salah"` alone → `negative`. Bare acknowledgements like `"oke"`, `"sip"` → `neutral`.
9. Do not infer sentiment from the author, score, or thread topic — only the text.

## Required output format

Return **only** a JSON array. One object per input comment, in the same order:

```json
[
  {"comment_id": "<exact id from input>", "label": "positive|neutral|negative"},
  ...
]
```

Rules for output:
- Copy `comment_id` **exactly** as given. Never invent or modify an ID.
- Emit **exactly one** object per input comment — no more, no fewer.
- Do not add notes, explanations, markdown fences, or commentary.
