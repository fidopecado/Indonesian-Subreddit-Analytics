# Indonesian Subreddit Analytics Dashboard

Interactive Streamlit dashboard + sentiment-labeling pipeline for the
**r/indonesia** dataset (~12,976 threads, ~728,522 comments).

## Live demo

**https://fidopecado-indonesian-subreddit-analytics-app-jsnenm.streamlit.app/**

Hosted on Streamlit Community Cloud from this repository (branch `master`, entry
point `app.py`). The app loads the committed parquet cache directly, so the
hosted instance needs no raw data and no pipeline run. The cache is versioned for
exactly this reason: `data/raw/` and `artifacts/` stay ignored, but the handful of
parquets `app.py` reads are committed (~143 MB, largest file 72 MB).

It is sized to fit Streamlit Community Cloud's 1 GB ceiling - the peak resident
set is ~0.71 GB, down from ~1.42 GB before the load path was reworked (see
*Memory* below).

## Repository layout

```text
.
├── app.py                     # Streamlit dashboard (entry point)
├── Dockerfile                 # container image (code + deps only, no data)
├── docker-compose.yml         # runs the image with ./data bind-mounted
├── .dockerignore
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
│   └── cache/                 # parquet the dashboard reads (versioned: the hosted demo has no raw data)
└── artifacts/
    ├── label_batches/         # per-batch labeling inputs
    ├── labels_out/            # per-batch labeling outputs
    └── scratch/               # archived one-off review dumps (safe to delete)
```

Code, docs and the dashboard cache are version-controlled. `data/raw/`,
`data/derived/` and `artifacts/` are git-ignored (see `.gitignore`) because they
are large or reproducible; `data/cache/` is tracked file-by-file, only the
parquets `app.py` loads.

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

## Run in Docker

```powershell
docker compose up --build
```

Then open `http://localhost:8501`. Build the cache first if you have not already
(see *Quick start*) — the container mounts `./data`, it does not ship the dataset.

The image carries code, dependencies and the in-app cache builders only. Data is
bind-mounted at `./data:/app/data` because the parquet cache is ~150 MB and the
raw CSVs ~400 MB: baking them in would make every data refresh an image rebuild.
Consequently:

- Refreshing the cache, or dropping in new raw CSVs, needs **no rebuild**.
- The sidebar's **Build cache now** button works inside the container, as long as
  `data/raw/` is populated on the host.
- The UI language default can be changed at runtime:
  `DASHBOARD_LANG=id docker compose up` (or set it in `docker-compose.yml`).
- The theme default likewise: `DASHBOARD_THEME=dark docker compose up`.
- The GPU pipeline is **not** in this image (`requirements-gpu.txt` is a separate,
  much larger install); the dashboard only needs the precomputed parquet files.

Equivalent without Compose:

```powershell
docker build -t r-indonesia-dashboard .
docker run --rm -p 127.0.0.1:8501:8501 -v "${PWD}\data:/app/data" r-indonesia-dashboard
```

The port binds to `127.0.0.1` on purpose: the dashboard has no authentication, so
publishing it on `0.0.0.0` would expose the dataset browser to your whole network.
Use `8501:8501` (or a reverse proxy) if you actually want that.

The container runs as an unprivileged user (uid `10001`) and exposes a
`/_stcore/health` healthcheck, so `docker ps` reports `healthy` once Streamlit is
serving. On Linux hosts, make sure the uid can read `./data` — or let the app's
own "cache missing" screen tell you the mount is wrong rather than a permissions
error.

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

- Reactive global filters: WIB date range, flair, author UUID, title search, score, comment threshold, post type, moderation/content flags, and an auto-posted-thread exclusion.
- Overview: activity over time, flair distribution, score/comment relationships, and score distributions.
- Threads: ranked thread explorer, title/body search, CSV export, and live flair comparisons.
- Comments: comment activity, score and length distributions, top/bottom comments, and live thread-filter matching.
- Authors: live filtered engagement leaderboard plus the static source leaderboard.
- Flairs: live performance profile, normalized metric heatmap, and top-word/reference views.
- Timing: WIB day/hour heatmaps for threads and comments, plus median replies and score per hour for organic threads.
- Sentiment: full-corpus model labels, a 10,000-comment assistant-labeled reference, agreement/calibration analysis, flair and time breakdowns, and a disagreement explorer.
- Methodology: an in-dashboard guide to the dataset, every page, three headline findings, and the interpretation traps.
- Bilingual interface (Indonesian / English) — switch with the selector at the top of the sidebar.
- Light and dark "e-ink" themes, switched live from the sidebar (no restart).
- The whole filter state lives in the URL, so any view can be bookmarked or shared; **Reset filters** returns to the defaults.

## Language / Bahasa

The whole interface is translated; the **data never is** (flair names, sentiment
values, author UUIDs and post titles stay as they are). English is the default —
set `DASHBOARD_LANG=id` (environment variable, handy in Docker) or change
`DEFAULT_LANG` in `app.py` to flip it.

Strings live in one place, the `STRINGS` dict in `app.py`, keyed by a dotted id
with an `(Indonesian, English)` pair. `t("key")` looks one up; a missing key
renders as the literal id rather than silently falling back, so gaps are visible
instead of wrong. The Methodology page text is in `METHOD_DOC`.

Some controls keep an internal id that is *not* translated and only localise the
label (`format_func`), so switching language never resets your filter selection.
The selected language is written to the URL (`?lang=id`), so a shared link opens
in the language it was copied in.

## Shareable views

Every filter is mirrored into the address bar as you change it, and only
*differences* from the defaults are encoded — a fresh dashboard has a clean URL.
A link therefore restores the whole view: date preset or custom range, topics,
score window, content flags, post type, minimum comments, search terms, language
and theme. Browser back/forward walks through filter changes instead of leaving
the app. **↺ Reset filters** (bottom of the sidebar) clears the URL and returns
every control to its default.

## Sentiment labeling

The rubric, the batch protocol and the prompt used by LLM raters are specified
in [`docs/agent-labeling.md`](docs/agent-labeling.md): label space, ordered
decision rules, anchors, output contract, validation, drift control, and how to
keep a labeler blind to every other model's output.

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

### Methodology notes

- **Auto-posted threads are excluded from the timing analysis.** `Daily Chat
  Thread`, `Weekend Chat Thread` and `Special Thread` are published by automod on
  a fixed clock (07:00 WIB). They are only ~2% of threads but ~22% of all
  comments, and *every* one of the top comment threads lands on the same hour —
  so including them makes "best hour to post" mean "when automod runs" rather
  than when an ordinary post earns replies. The sidebar **Exclude auto-posted
  threads** toggle applies the same exclusion everywhere.
- **Medians, not means, for per-thread outcomes.** Reply and score distributions
  are heavy-tailed (score skew ≈ 3.9, maximum 4,040 against a p99 of 957), so one
  viral thread moves a hourly *mean* by tens of points while the median barely
  budges. The Timing page therefore plots median replies and median score per
  hour, and overlays the mean as a marker so the tail stays visible.
- **Comments inherit their thread's date filter.** Filtering is applied to
  threads; comments are kept by `thread_id`. A comment posted *after* the
  selected window still counts as long as its parent thread is inside it.
- **The "Reference: WIB hourly activity file" chart is an all-time snapshot** and
  deliberately ignores the sidebar filters — it is a fixed baseline to compare
  the filtered live view against.

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

## Memory

Streamlit Community Cloud gives each app 1 GB of RAM, and the dashboard used to
peak at ~1.42 GB, so it could not be hosted. The figures below come from driving
a real server with headless Chrome and sampling the interpreter's working set
(not `DataFrame.memory_usage`, which understates what the process holds):

| | deep size of the frames | peak RSS |
| --- | --- | --- |
| before | 924 MB | 1423 MB |
| after | 310 MB | 712 MB |

What made the difference, in order of impact:

- **Arrow-backed text.** `pd.read_parquet(..., dtype_backend="pyarrow")` keeps
  strings in contiguous buffers instead of one Python object per value. At 728k
  rows an object string column costs ~50 bytes per value, so `comments` alone
  was 337 MB that way and is 104 MB now. Non-text columns are cast back to
  numpy so every existing pandas/plotly path sees the dtypes it expects.
- **`st.cache_resource` instead of `st.cache_data`.** `cache_data` keeps a
  pickled copy of the value next to the live object, doubling the footprint.
  Nothing mutates these frames in place, so sharing them is safe.
- **Comment text is never resident.** `comments_text` is 146 MB and the only
  rows ever displayed are the 15 highest and 15 lowest scoring comments, so
  those bodies are read on demand. The file is sorted by `comment_id` in
  2,000-row groups so the filter skips almost every row group: 48 ms, ~0 MB.
- **Unused columns are dropped at load time**, worth ~49 MB on the cached frame
  and the same again on the filtered copy.
- **Charts are pre-binned.** The score and word-count histograms hand plotly 60
  bars instead of 705k raw values, and the volume/heatmap aggregations build
  two-column stand-ins rather than copying all 16 columns to touch two of them.

## Theming

The dashboard uses the **e-ink** grayscale colorscheme
([`e-ink-colorscheme/e-ink.nvim`](https://github.com/e-ink-colorscheme/e-ink.nvim),
light mode) with its Everforest red as the single signal accent, rendered as
brutalist / engineering chrome: monospace type, 2px ink rules, hard offset
shadows and zero border-radius.

- The palette and stylesheet live in `app.py` (`THEMES`, `EINK`, `EINK_RAMP`,
  `EINK_TOPICS`, `EINK_SEQ`, `CSS_TEMPLATE`) and every Plotly figure picks them
  up through the shared `style()` helper.
- **Flair (topic) colours** come from `EINK_TOPICS` via `flair_palette()`, a
  stable frequency-ordered map, so a topic keeps the same hue in the donut,
  the top-flair bars, the scatter and the top-words chart. Chart *chrome* stays
  monochrome; only topic-coded series use hue.
- **Colour rule — hue must encode something:** single-series bars, lines and
  histograms stay **ink** (mapping one variable to hue adds nothing and implies
  categories that don't exist). Hue is reserved for *categories* (flairs) and
  *magnitude*:
  - `EINK_SEQ_BLUE` (blueprint) — thread activity, WIB reference, flair metric
    heatmap, continuous engagement colour.
  - `EINK_SEQ_EMBER` — the comment-activity heatmap.
  - `EINK_SEQ` (neutral greyscale, paper → `#0D0D0D`) — the confusion matrix.
  Every ramp starts at the theme's paper and is monotonic in lightness, so it
  stays on-theme and remains legible in greyscale print. In dark mode the ramps
  are inverted (dark paper → light), because "more" has to mean *brighter*.
- **Two inks, one role map.** `THEMES["light"]` is paper with dark ink,
  `THEMES["dark"]` is a dark board with light ink; `apply_theme()` swaps the
  palette *in place* at the top of every run, so charts, CSS and helpers that
  captured a reference all follow. Hues (accent red, flair palette) are shared,
  so a flair keeps its colour across a theme switch. The toggle sits under the
  language selector; `DASHBOARD_THEME=dark` sets the default. Streamlit paints a
  few canvas widgets (the data grid) from its server-side config, which cannot be
  switched at runtime, so the grid is inverted in dark mode to sit on the board
  instead of glowing white.
- **Contrast:** anything drawn on an ink surface (active tab, multiselect tags,
  hover/pressed buttons, highlighted dropdown options) flips to paper text, so
  there is no dark-on-dark text.
- Top-level section headings use a left rule instead of a full-width underline,
  so no horizontal line ever sits across the top of a card row.
- The base theme (paper, ink, monospace font) is set in `.streamlit/config.toml`.
  That file also sets `gatherUsageStats = false`: Streamlit 1.40 sends a
  PageProfile telemetry message after every run whose handler can fire before the
  session registers, popping a "Bad message format" modal on a fraction of new
  sessions. Disabling usage stats removes the message (and the telemetry).
- **Streamlit is pinned** (`requirements.txt`) because the stylesheet targets its
  widget DOM, which changes between minor releases: 1.66, for instance, replaces
  the BaseWeb tabs/selects with react-aria components and silently drops most of
  this theme. Bump the pin and the selectors in `CSS_TEMPLATE` together.
- `docs/theme-preview.html` is a standalone swatch sheet of the palette and
  components.

## Schema reference

See **`docs/HEADER_MAP.md`** for the full, auto-generated column inventory, join
map, and gotchas. Do not edit it by hand — run `python src/gen_header_map.py`.
