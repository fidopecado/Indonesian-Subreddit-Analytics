# Labeling every comment with an AI agent

A spec for using a general-purpose LLM as a sentiment rater at corpus scale — no
GPU, no training, resumable across turns. Copy the prompts in §7 as-is; the rest
is the reasoning behind their shape.

**In scope:** one independent sentiment label per comment.
**Out of scope:** comparing against another model, tuning thresholds, updating
the dashboard. Those happen later, by a different actor. Keeping them out of the
labeler's context is deliberate (§8.1).

---

## 1. The unit

One comment body → one label. The body is what the author wrote: no thread
title, no parent comment, no author history, no score. This matches the existing
Hugging Face pass (bodies only), so the two remain comparable later.

- Bodies are flattened to one line before batching (tabs, `\r`, `\n` → space;
  runs collapsed). **One line = one comment.** That is what makes line-based
  answering possible.
- Comments under 2 words or 2 characters are out of scope; they carry no
  recoverable stance.
- Emoji, `[gif]` and link syntax stay in the text — they are part of the stance.

## 2. Label space

Four characters. No intensity, no multi-label, no confidence:

| char | name | means |
|---|---|---|
| `+` | positive | approval, praise, gratitude, amusement, hope, satisfaction, support |
| `-` | negative | disapproval, criticism, anger, insult, disgust, disappointment, worry, mockery, complaint |
| `0` | neutral | no evaluative stance: questions, facts, links, procedure, speculation, valence-free self-description |
| `?` | undecidable | stance is not recoverable: removed/deleted stub, or irony unresolvable from the text alone |

`?` is a **failure flag, not a third polarity**. Target under 1% of any batch. A
labeler returning 10% `?` is being timid: most of those are `0`.

## 3. Decision procedure

Apply top-down; the first rule that fires wins. This is the whole rubric.

1. **Is there a stance at all?** A question, a fact, a link, a quote, bot text,
   a description of someone else's feelings → `0`. Stop.
2. **Whose stance?** The commenter's. Not the topic's, not the person quoted,
   not the subreddit's mood. "He was furious" → `0`; "I'm furious" → `-`.
3. **Favourable or unfavourable?** → `+` / `-`.
4. **Irony:** label the stance the author intends, not the words used. "Great,
   another traffic jam" → `-`. Use `?` only when intent is genuinely unrecoverable.
5. **Insults, slurs, name-calling** → `-`, including when framed as a joke.
   Hostility is the stance.
6. **Questions:** `0` unless the question itself judges ("kenapa sih semuanya
   sampah?" → `-`).
7. **Mixed comments:** net stance toward the main subject; if the sides truly
   cancel → `0`.
8. **Social formulas:** thanks, congratulations, hope, good luck → `+`.
   Condolence, sympathy, worry for someone → `-` (the affect is sadness or fear
   even when the intent is kind).
9. **Humour without a target:** amusement → `+`. A joke at someone's expense →
   `-` (rule 5).
10. **No outside knowledge.** No author reputation, no thread context, no
    "this is probably sarcastic because r/indonesia". Text only.

Code-mixing has no special rule: judge intent, not language. Indonesian, English
and slang in one sentence is normal and does not change the label.

## 4. Anchors

Include these verbatim in the prompt. They cover the cases labelers actually get
wrong, and they double as a 12-item calibration test (§9).

| text | label | why |
|---|---|---|
| Makasih infonya, sangat membantu 🙏 | `+` | gratitude |
| Ini kenapa sampah banget sih pelayanannya | `-` | complaint in question form (rule 6) |
| Ada yang tau harga tiketnya berapa? | `0` | question, no stance |
| Keren, tambah satu lagi masalah yang harus diurus | `-` | ironic praise (rule 4) |
| Dasar tolol, becanda doang nggak ngerti | `-` | insult, even as a joke (rule 5) |
| Turut berduka cita ya, semoga keluarga diberi kekuatan | `-` | condolence (rule 8) |
| Semoga lancar terus ya! | `+` | hope |
| Dia marah banget waktu itu | `0` | reported emotion, not the commenter's (rule 2) |
| Ternyata begini cara kerjanya, menarik | `+` | mild positive evaluation |
| Menurut saya kebijakannya kurang tepat | `-` | criticism |
| [ Removed by Reddit ] | `?` | no content |
| ![gif](giphy\|abc) | `0` | artifact, no stance |

Two traps worth naming: **reported** emotion is `0`, and **condolence is `-`**
even though it is polite.

## 5. Output contract

- Exactly **N characters** for N comments, in input order.
- Alphabet `+ - 0 ?` only. No spaces, ids, numbering, punctuation or prose.
- **One character per line** is preferred: misalignment becomes visible and the
  count is trivial.
- The answer sits alone in a single fenced code block. Anything outside it is a
  protocol violation.

Validation is mechanical and happens before anything is written: strip
whitespace, check the alphabet, check the length, reject otherwise. Never repair
a wrong-length answer by truncating or padding — re-ask (§7.3).

## 6. The loop

Both modes use the same contract; only who runs the commands differs.

### 6.1 Agent with a shell

```powershell
python src\llm_label.py status            # target size, progress, class mix
python src\llm_label.py next --n 500      # -> data/derived/llm_labels/batches/batch_0003.txt
# the labeler reads batch_0003.txt and writes 500 characters to a labels file
python src\llm_label.py commit --batch 3 --labels-file labels\batch_0003.txt
```

`next` hands out the first N **unlabeled** comments in `md5(comment_id)` order.
That buys three properties:

- the order is stable across machines and cache rebuilds;
- the cursor cannot drift, because it is derived from the labels file rather
  than from a stored offset;
- any prefix of the run is a **uniform random sample** of the corpus, so
  stopping early still yields usable statistics.

`commit` validates, merges by `comment_id` (re-labeling overwrites), writes
atomically, then advances the batch counter. If anything fails, nothing is
written and the batch is simply re-issued.

**Give the labeler `batch_NNNN.txt` only.** The `.csv` exists for the join; it
carries ids and nothing the rater needs.

### 6.2 Chat-only agent (web UI, no tools)

1. `next --n 200`, then paste the contents of `batch_NNNN.txt` with the system
   prompt from §7.
2. The model returns 200 characters.
3. Save them to `labels\batch_NNNN.txt`, one per line.
4. `commit --batch NNNN --labels-file labels\batch_NNNN.txt`.

The checkpoint is the labels file, so a lost tab, a rate limit or a closed
session costs one batch — never the run.

## 7. The prompts

### 7.1 System prompt (verbatim)

```text
You are a sentiment rater for Indonesian Reddit comments from r/indonesia.
The comments mix Indonesian, English, and slang; this is normal and does not
change the task.

You receive N comment bodies, one per line, in a fixed order. Assign exactly
one sentiment character to each comment, in the same order.

Labels:
+  positive  approval, praise, gratitude, amusement, hope, satisfaction, support
-  negative  disapproval, criticism, anger, insult, disgust, disappointment,
             worry, mockery, complaint
0  neutral   no evaluative stance: questions, facts, links, procedure,
             speculation, valence-free self-description
?  undecidable  stance not recoverable: removed/deleted stub, or irony that
                cannot be resolved from the text alone. Target under 1%.

Rules, applied in order, first match wins:
1. No stance at all (question, fact, link, quotation, bot text, description of
   someone else's feelings) -> 0.
2. Label the commenter's own stance, not the topic's, not the person quoted.
3. Favourable -> +, unfavourable -> -.
4. Irony: label the stance the author intends, not the literal words.
5. Insults, slurs and name-calling -> -, including when framed as a joke.
6. Questions are 0 unless the question itself judges or complains.
7. Mixed comments: net stance on the main subject; if truly balanced -> 0.
8. Thanks, congratulations, hope -> +. Condolence, sympathy, worry for someone -> -.
9. Amusement -> +. A joke at someone's expense -> -.
10. Use only the text. No outside knowledge, no assumptions about the thread,
    the author, or any other model's opinion.

Output contract:
- exactly N characters, alphabet + - 0 ?, input order preserved;
- one character per line;
- no ids, no numbering, no spaces, no explanation;
- put the whole answer in one fenced code block and nothing else;
- count the lines before you answer. If the count is not N, fix it before
  replying.
```

### 7.2 Batch message

Append the 12 anchors from §4, then:

```text
N = 500. Return exactly 500 characters, one per line, alphabet + - 0 ?.

<500 comment bodies, one per line>
```

If the batch is small (≤100), put the bodies as a numbered list for readability.
For larger batches keep them unnumbered: body lines are already unambiguous, and
numbering wastes tokens and invites the model to echo ids.

### 7.3 Repair prompt

Use when the length or alphabet is wrong. Do not patch by hand.

```text
That answer has <M> characters; there are <N> comments. Re-answer with exactly
<N> characters, one per line, alphabet + - 0 ?, input order preserved. No
explanation, no ids.
```

## 8. How to instruct a model to do this

### 8.1 Blind, and stay blind

Never include another model's label, an accuracy figure, or "the HF model says
X". Models anchor hard on a prior: show one and the run measures agreement
instead of classification. Also do not reuse a conversation that contains
earlier batches — start a fresh context per batch and repeat the rubric each
time. Blind is not a courtesy; it is what makes the labels independent.

### 8.2 Give the rubric verbatim

Paraphrasing a rubric is how you get a different rubric. Paste the same text
every batch. If you must shorten, cut anchors, never rules.

### 8.3 One character per line, no ids

Ids invite transcription errors and burn output tokens. Order *is* the contract;
the harness already knows the ids. One input line maps to one output line, so
position is unambiguous.

### 8.4 Make the count a self-check

State N twice — once in the prompt, once in the closing instruction. Models
cannot reliably count to 500, but the instruction shifts behaviour at the
margin, and the validator catches the rest.

### 8.5 Cap the batch

300–500 per turn is the sweet spot: efficient, and small enough that attention
does not collapse. Above ~800, quality drops, and very long bodies start
truncating the batch read itself. Split the job; never "just do 5,000".

### 8.6 Temperature 0

Same input, same label. Sampling is a defect here.

### 8.7 Validate, never trust

Length + alphabet, before writing. A model returning 301 for 300 is telling you
it drifted mid-batch: re-ask the whole batch rather than repairing one position.

### 8.8 Watch the mix, not the individual label

Per-batch class shares are the drift detector. A jump of more than 10 pp means
the rubric is slipping (often after a run of long political comments), not that
the corpus changed. Re-anchor with §4 and re-run the batch.

### 8.9 Re-label blind, occasionally

Re-run 2% of an old batch in a fresh context. Self-agreement below ~85% means
the rubric has an unresolved edge: fix the rubric, not the individual labels.

### 8.10 Keep the model out of the analysis

The labeler's job ends at the string. Comparison, kappa, precision/recall and
bias correction are separate work. Mixing them into the prompt contaminates the
labeler and makes the labels worthless as an independent second opinion.

## 9. Quality control without another model's labels

- **Anchor check:** run the 12 items in §4 before the first real batch. Under
  10/12 means the model is not following the rubric; fix the prompt first.
- **Self-agreement:** re-label 2% blind; expect ≥85%.
- **Mix stability:** per batch, expect ±5 pp; >10 pp is drift.
- **`?` rate:** expect <1%; higher means timidity.
- **Throughput:** ~300–500 comments per turn, ~20–35k input tokens, 1–2 minutes
  of model time.

## 10. Failure modes

| symptom | cause | fix |
|---|---|---|
| 301 labels for 300 | mid-batch drift, or hand-concatenated strings | re-ask with §7.3; one char per line |
| labels shifted by one | a merged or empty line in the batch | regenerate the batch; never patch by hand |
| class mix jumps between batches | rubric drift after a heavy topic | re-anchor, re-run that batch |
| `?` above 5% | timidity, over-thinking irony | restate: `?` is for deleted stubs and unrecoverable irony only |
| almost everything is `-` | labelling the topic, or anchoring on a prior | restate rules 1–2 and 8.1 |
| batch read truncated | batch too large or bodies too long | split; keep ≤500 |

## 11. For the analyst (never shown to the labeler)

Once a run exists, the comparison is a one-liner — but it belongs to a different
session, with a different prompt, after the labels are frozen:

```powershell
python src\llm_label.py report --drift
```

That prints agreement, Cohen's kappa, the confusion matrix, per-class
precision/recall of the model being audited, and the per-batch mix. Treat the
labeler as the reference and the audited model as the subject: "of the comments
the model called neutral, how many really were" is the number that matters, not
the headline agreement rate.

Do not feed any of it back into the labeling prompt.
