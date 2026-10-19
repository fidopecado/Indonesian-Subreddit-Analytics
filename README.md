# Indonesian Subreddit Analytics Dashboard

Interactive Streamlit dashboard + sentiment-labeling pipeline for the
**r/indonesia** dataset (~12,976 threads, ~728,522 comments).

## Repository layout

```text
.
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt           # dashboard / cache-build deps
├── requirements-gpu.txt       # torch + transformers deps for sentiment runs
├── docs/
│   ├── HEADER_MAP.md          # auto-generated schema reference (regenerate, don't edit)
│   ├── label_rubric.md        # sentiment labeling rubric
│   └── agent.md               # unrelated local agent-helper notes
├── src/                       # all pipeline scripts
│   ├── build_cache.py         # raw CSV -> data/cache parquet
│   ├── sample_10k.py          # deterministic 10K comment sample
│   ├── make_batches.py        # sample -> artifacts/label_batches/*.json
│   ├── assemble_llm_labels.py # artifacts/labels_out -> labeled parquet
│   ├── build_sentiment_cache.py
│   ├── sentiment_gpu.py       # Colab/Kaggle HF model run (resumable)
│   ├── sentiment_pilot.py     # local language-router pilot
│   ├── benchmark_sentiment.py
│   └── gen_header_map.py      # regenerates docs/HEADER_MAP.md
├── data/
│   ├── raw/                   # source CSVs (not versioned)
│   ├── derived/               # aggregates + model outputs (not versioned)
│   └── cache/                 # parquet the dashboard reads (not versioned)
└── artifacts/
    ├── label_batches/         # per-batch labeling inputs
    ├── labels_out/            # per-batch labeling outputs
    └── scratch/               # archived one-off review dumps (safe to delete)
```

Only code and docs are version-controlled; `data/` and `artifacts/` are
git-ignored (see `.gitignore`) because they are large or reproducible.

## Quick start (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\build_cache.py --force
streamlit run app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

### If the virtual environment does not exist

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\build_cache.py --force
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Refreshing data

The app reads the raw CSVs only through the cache layer. When the CSVs change:

```powershell
.\.venv\Scripts\python.exe src\build_cache.py --force
```

The builder writes compressed Parquet files under `data/cache/` and never
modifies the raw CSVs. Regenerate the schema reference after a rebuild:

```powershell
.\.venv\Scripts\python.exe src\gen_header_map.py
```

## Features

- Reactive global filters: WIB date range, flair, author UUID, title search, score, comment threshold, post type, and moderation/content flags.
- Overview: activity over time, flair distribution, score/comment relationships, and score distributions.
- Threads: ranked thread explorer, title/body search, CSV export, and live flair comparisons.
- Comments: comment activity, score and length distributions, top/bottom comments, and live thread-filter matching.
- Authors: live filtered engagement leaderboard plus the static source leaderboard.
- Flairs: live performance profile, normalized metric heatmap, and top-word/reference views.
- Timing: WIB day/hour heatmaps for threads and comments.
- Sentiment: full-corpus model labels, a 10,000-comment assistant-labeled reference, agreement/calibration analysis, flair and time breakdowns, and a disagreement explorer.

## Sentiment labeling

Three complementary artifacts:

1. `data/cache/comment_sentiment_llm_10k.parquet` — exactly 10,000 unique comments
   labeled with a fixed Indonesian Reddit rubric by LLM agents. Integrity checks
   confirm 10,000 unique IDs, no null labels, and only `positive`, `neutral`, or
   `negative` values.
2. `data/cache/sentiment_router_gold40.parquet` — the local language-router output
   on an independent 40-comment LLM gold set.
3. `src/sentiment_colab_kaggle.ipynb` + `src/sentiment_gpu.py` — GPU workflow for
   classifying 10K or the corpus on Colab/Kaggle with checkpoints and canonical
   probabilities.

The local language router achieved **50% agreement** against the 40-comment LLM
rubric benchmark while reporting **0.90 mean confidence**. It over-predicted
negative sentiment (29 predictions vs. 17 gold labels). This benchmark is small,
but it shows why model confidence must not be treated as ground truth.

### Pipeline steps

```powershell
python src\sample_10k.py            # cache -> data/cache/comments_sample_10k.parquet
python src\make_batches.py          # -> artifacts/label_batches/batch_NNNN.json
# ... run each batch through the labeling agent into artifacts/labels_out/ ...
python src\assemble_llm_labels.py   # -> data/cache/comment_sentiment_llm_10k.parquet
python src\build_sentiment_cache.py # -> dashboard-ready sentiment artifacts
```

### Full-corpus run and calibration

The corpus run (678,522 comments) was classified on a Colab GPU (mean confidence
0.84). The raw model distribution is **67.5% negative / 16.4% positive / 16.1%
neutral**, which contradicts the independent 10K LLM reference (**32.8% negative
/ 56.4% neutral / 10.8% positive**).

Comparing the model to the LLM reference on the same 10,000 comment IDs:

- Exact agreement: **45.5%**
- Cohen's kappa: **0.188**
- The model keeps a rubric-neutral comment as neutral only ~21% of the time,
  relabeling the rest negative — the mechanism behind the inflated negative share.

Applying a least-squares prevalence correction (using the 10K confusion matrix)
gives a bias-corrected full-corpus estimate of **30.2% negative / 59.6% neutral /
10.1% positive**, which lands close to the LLM reference. This is evidence the raw
distribution is a model artifact, not a property of the subreddit.

Rebuild the dashboard artifacts after new labels land:

```powershell
.\.venv\Scripts\python.exe src\build_sentiment_cache.py
```

### Run on Colab or Kaggle

1. Enable a T4/P100 GPU.
2. Upload or attach this project, including `data/cache/comments.parquet` and
   `data/cache/comments_text.parquet`.
3. Open `src/sentiment_colab_kaggle.ipynb` and update its `PROJECT` path.
4. Run the 10K pilot cells first.
5. After inspection, run the full-dataset command in the final notebook section.

```powershell
python src\sentiment_gpu.py --input-dir data/cache --limit 728522 --output data/derived/sentiment_full.parquet
```

The GPU script writes checkpoints and canonical probabilities in this order:

```text
p_negative, p_neutral, p_positive, label, confidence
```

This normalization matters because the Indonesian model and CardiffNLP models use
opposite class-index orders.

## Data notes

- `total_comments_actual` is preferred over Reddit's stale `num_comments` counter.
- `1970-01-01 00:00:00` in `edited` means "never edited" and is exposed as `was_edited`.
- Dashboard duplicate thread IDs are resolved by keeping the highest dataset version.
- The comments body text is stored separately (`comments_text.parquet`) so aggregate views remain fast.
- The pre-derived leaderboard, flair performance, flair top words, and hourly activity files are shown as static reference snapshots; the main dashboard views recompute from the cached source data.
- Sentiment labels are sample-only and are not ground truth; the two models intentionally disagree.

### Known coverage gaps (documented)

- **Sentiment "full corpus" is the filtered corpus.** The corpus run is labeled
  "full" but `src/sentiment_gpu.py` drops comments before classifying:
  duplicates by `body` and rows with `word_count < 2`. Starting from 726,031
  unique comments that yields **678,522** labeled rows (~93%), leaving ~47,500
  excluded. Rebuild with `--limit` raised and the filter relaxed to cover 100%.
- **Comment duplicates.** `data/raw/indonesia_subreddit_comments.csv` contains
  ~2,500 repeated `comment_id`s; the cache does not deduplicate the core table
  (the dashboard tolerates it, but joins should `drop_duplicates("comment_id")`).
- **Sentiment sample** (`sentiment_model_comparison_sample.csv`) is 50–96 rows —
  illustrative only, never ground truth.

## Theming

The dashboard uses the **e-ink** grayscale colorscheme
([`e-ink-colorscheme/e-ink.nvim`](https://github.com/e-ink-colorscheme/e-ink.nvim),
light mode) with its Everforest red as the single signal accent, rendered as
brutalist / engineering chrome: monospace type, 2px ink rules, hard offset
shadows and zero border-radius.

- The palette and stylesheet live in `app.py` (`EINK`, `EINK_RAMP`,
  `EINK_TOPICS`, `EINK_SEQ`, `CSS_TEMPLATE`) and every Plotly figure picks them
  up through the shared `style()` helper.
- **Flair (topic) colours** come from `EINK_TOPICS` via `flair_palette()`, a
  stable frequency-ordered map, so a topic keeps the same hue in the donut,
  the top-flair bars, the scatter and the top-words chart. Chart *chrome* stays
  monochrome; only topic-coded series use hue.
- **Contrast:** anything drawn on an ink surface (active tab, multiselect tags,
  hover/pressed buttons, highlighted dropdown options) flips to paper text, so
  there is no dark-on-dark text.
- Top-level section headings use a left rule instead of a full-width underline,
  so no horizontal line ever sits across the top of a card row.
- The base theme (paper, ink, monospace font) is set in `.streamlit/config.toml`.
- `docs/theme-preview.html` is a standalone swatch sheet of the palette and
  components.

## Schema reference

See **`docs/HEADER_MAP.md`** for the full, auto-generated column inventory, join
map, and gotchas. Do not edit it by hand — run `python src/gen_header_map.py`.
