"""
Build the cached parquet layer for the Indonesian Subreddit dashboard.

Reads the raw CSVs once and writes small, fast parquet files into
`data/cache` so the Streamlit app can stay responsive and reactive.

Raw CSVs are never modified. Re-run this script whenever the CSVs change:

    python src/build_cache.py            # build only if cache is missing/stale
    python src/build_cache.py --force    # rebuild unconditionally

Layout: raw sources in `data/raw/`, static aggregates in `data/derived/`,
outputs in `data/cache/`.

Design notes (see docs/HEADER_MAP.md):
  * `1970-01-01 00:00:00` in `edited` is a sentinel meaning "never edited" ->
    exposed as a real boolean `was_edited`.
  * `num_comments` is Reddit's stale snapshot; `total_comments_actual` is the
    truth. Thread rows are enriched with the dashboard's actual count.
  * `dashboard_master_dataset.csv` has a few duplicate `thread_id`s across
    dataset `version`s; we keep the highest version (most recent scrape).
  * `all_comments_text` is dropped (concatenated blob, huge, not per-comment).
  * Comment bodies are split into a separate parquet so aggregate views never
    pay the cost of loading 250 MB of text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project layout: this script lives in <root>/src, data under <root>/data.
BASE = Path(__file__).resolve().parents[1]  # project root
DATA_RAW = BASE / "data" / "raw"
DATA_DERIVED = BASE / "data" / "derived"
CACHE = BASE / "data" / "cache"

SENTINEL = pd.Timestamp("1970-01-01")
WIB_OFFSET = pd.Timedelta(hours=7)
NO_FLAIR = "No Flair"
COMMENT_CHUNK = 150_000

THREADS_CSV = DATA_RAW / "indonesia_subreddit_threads.csv"
COMMENTS_CSV = DATA_RAW / "indonesia_subreddit_comments.csv"
DASHBOARD_CSV = DATA_RAW / "dashboard_master_dataset.csv"
LEADERBOARD_CSV = DATA_DERIVED / "author_leaderboard.csv"
FLAIR_PERF_CSV = DATA_DERIVED / "flair_performance.csv"
FLAIR_WORDS_CSV = DATA_DERIVED / "flair_top_words.csv"
HOURLY_CSV = DATA_DERIVED / "wib_hourly_activity.csv"
SENTIMENT_CSV = DATA_DERIVED / "sentiment_model_comparison_sample.csv"

# Aggregate/reference files copied straight into the cache.
PASSTHROUGH = [
    (LEADERBOARD_CSV, "author_leaderboard.parquet"),
    (FLAIR_PERF_CSV, "flair_performance.parquet"),
    (FLAIR_WORDS_CSV, "flair_top_words.parquet"),
    (HOURLY_CSV, "wib_hourly_activity.parquet"),
]


def log(msg: str) -> None:
    print(f"[build_cache] {msg}", flush=True)


def to_dt(series: pd.Series) -> pd.Series:
    """Parse a datetime column tolerantly, coercing junk to NaT."""
    return pd.to_datetime(series, errors="coerce")


def clean_flair(series: pd.Series) -> pd.Series:
    """Normalize the three names for the same concept into one label."""
    return (
        series.fillna(NO_FLAIR)
        .astype(str)
        .str.strip()
        .replace({"": NO_FLAIR, "nan": NO_FLAIR, "None": NO_FLAIR})
    )


def clean_distinguished(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def is_stale(force: bool) -> bool:
    """True when the cache is missing or older than any source CSV."""
    if force or not CACHE.exists():
        return True
    meta = CACHE / "meta.json"
    if not meta.exists():
        return True
    newest_source = max(
        src.stat().st_mtime for src, _ in PASSTHROUGH + [
            (THREADS_CSV, None), (COMMENTS_CSV, None), (DASHBOARD_CSV, None),
            (SENTIMENT_CSV, None),
        ]
    )
    oldest_cache = min(p.stat().st_mtime for p in CACHE.glob("*.parquet"))
    return newest_source > oldest_cache


def build_dashboard() -> pd.DataFrame:
    """Load the denormalized master table minus the giant text blob."""
    path = DASHBOARD_CSV
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    keep = [c for c in cols if c != "all_comments_text"]
    dm = pd.read_csv(path, usecols=keep, low_memory=False)

    before = len(dm)
    dm = dm.sort_values("version", ascending=False).drop_duplicates(
        "thread_id", keep="first"
    )
    dupes = before - len(dm)

    dm["created_utc"] = to_dt(dm["created_utc"])
    dm["created_wib"] = to_dt(dm["created_wib"])
    dm["edited_dt"] = to_dt(dm["edited"])
    dm["was_edited"] = dm["edited_dt"] > SENTINEL
    dm["link_flair_text"] = clean_flair(dm["link_flair_text"])
    dm["average_comment_score"] = pd.to_numeric(
        dm["average_comment_score"], errors="coerce"
    )
    dm["total_comments_actual"] = pd.to_numeric(
        dm["total_comments_actual"], errors="coerce"
    ).fillna(0).astype("int64")

    dm = dm.drop(columns=["edited"])
    log(f"dashboard: {len(dm):,} rows kept ({dupes:,} duplicate thread_ids dropped)")
    return dm


def build_threads(dm: pd.DataFrame) -> pd.DataFrame:
    """Threads enriched with the accurate comment count from the dashboard."""
    th = pd.read_csv(THREADS_CSV, low_memory=False)
    log(f"threads: {len(th):,} raw rows")

    th["created_utc"] = to_dt(th["created_utc"])
    th["created_wib"] = th["created_utc"] + WIB_OFFSET
    th["edited_dt"] = to_dt(th["edited"])
    th["was_edited"] = th["edited_dt"] > SENTINEL
    th = th.drop(columns=["edited"])

    th["distinguished"] = clean_distinguished(th["distinguished"])
    th["is_mod_post"] = th["distinguished"].ne("")
    th["link_flair_text"] = clean_flair(th["link_flair_text"])
    th["post_type"] = np.where(th["is_self"].fillna(False), "Text", "Link")
    th["title"] = th["title"].fillna("")
    th["selftext"] = th["selftext"].fillna("")
    th["title_len"] = th["title"].str.len()
    th["selftext_len"] = th["selftext"].str.len()
    th["external_url"] = th["external_url"].fillna("")

    # Prefer the real comment count over Reddit's stale counter.
    enrich = dm[["thread_id", "total_comments_actual", "average_comment_score"]]
    th = th.merge(enrich, on="thread_id", how="left")
    th["total_comments_actual"] = (
        pd.to_numeric(th["total_comments_actual"], errors="coerce").fillna(0).astype("int64")
    )
    th["average_comment_score"] = pd.to_numeric(
        th["average_comment_score"], errors="coerce"
    )
    th["has_comments_data"] = th["thread_id"].isin(set(dm["thread_id"]))

    for c in ["locked", "over_18", "spoiler", "stickied", "is_self", "is_original_content"]:
        th[c] = th[c].fillna(False).astype(bool)

    log(f"threads: {len(th):,} rows enriched, {int(th.has_comments_data.sum()):,} matched dashboard")
    return th


def build_comments() -> tuple[pd.DataFrame, int]:
    """Stream the comment CSV, returning metadata plus a separate text table."""
    path = COMMENTS_CSV
    core_cols = [
        "author", "created_utc", "distinguished", "edited", "comment_id",
        "is_submitter", "parent_id", "score", "stickied", "thread_id", "version",
    ]
    text_path = CACHE / "comments_text.parquet"
    writer: pq.ParquetWriter | None = None
    parts: list[pd.DataFrame] = []
    total = 0

    reader = pd.read_csv(path, usecols=core_cols + ["body"], chunksize=COMMENT_CHUNK, low_memory=False)
    for i, chunk in enumerate(reader, start=1):
        total += len(chunk)

        # --- text side table: written incrementally, never held in RAM ---
        text = chunk[["comment_id", "body"]].copy()
        text["body"] = text["body"].fillna("").astype(str)
        text["body_len"] = text["body"].str.len()
        text["word_count"] = text["body"].str.split().str.len().fillna(0).astype("int32")
        table = pa.Table.from_pandas(text, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(text_path, table.schema, compression="zstd")
        writer.write_table(table)

        # --- core side table: small, kept in memory ---
        core = chunk[core_cols].copy()
        core["created_utc"] = to_dt(core["created_utc"])
        core["created_wib"] = core["created_utc"] + WIB_OFFSET
        core["edited_dt"] = to_dt(core["edited"])
        core["was_edited"] = core["edited_dt"] > SENTINEL
        core["distinguished"] = clean_distinguished(core["distinguished"])
        core["is_mod"] = core["distinguished"].ne("")
        core["is_submitter"] = core["is_submitter"].fillna(False).astype(bool)
        core["stickied"] = core["stickied"].fillna(False).astype(bool)
        core["score"] = pd.to_numeric(core["score"], errors="coerce").fillna(0).astype("int64")
        core["body_len"] = text["body_len"].to_numpy()
        core["word_count"] = text["word_count"].to_numpy()
        parts.append(core.drop(columns=["edited"]))
        log(f"comments: chunk {i} done ({total:,} rows)")

    if writer is not None:
        writer.close()

    cm = pd.concat(parts, ignore_index=True)
    log(f"comments: {len(cm):,} rows, {text_path.name} written")
    return cm, total


def build_sentiment() -> pd.DataFrame:
    s = pd.read_csv(SENTIMENT_CSV, low_memory=False)
    s["created_utc"] = to_dt(s["created_utc"])
    s["created_wib"] = to_dt(s["created_wib"])
    s["thread_flair"] = clean_flair(s["thread_flair"])
    s["detected_lang"] = s["detected_lang"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    s["agree"] = s["App1_Label"].astype(str) == s["App2_Label"].astype(str)
    s["score"] = pd.to_numeric(s["score"], errors="coerce").fillna(0).astype("int64")
    s["body_len"] = s["body"].fillna("").astype(str).str.len()
    log(f"sentiment sample: {len(s):,} rows, agreement {s.agree.mean():.1%}")
    return s


def write_parquet(df: pd.DataFrame, name: str) -> None:
    out = CACHE / name
    df.to_parquet(out, index=False, compression="zstd")
    log(f"wrote {name} ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the dashboard cache.")
    ap.add_argument("--force", action="store_true", help="rebuild even if cache is fresh")
    args = ap.parse_args()

    t0 = time.time()
    missing = [
        f for f in [THREADS_CSV, COMMENTS_CSV, DASHBOARD_CSV, SENTIMENT_CSV]
        if not f.exists()
    ]
    if missing:
        log(f"ERROR: missing source files: {missing}")
        return 1

    if not is_stale(args.force):
        log("cache is up to date; use --force to rebuild")
        return 0

    CACHE.mkdir(exist_ok=True)
    meta: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(BASE),
        "sources": {},
        "notes": [],
    }

    dm = build_dashboard()
    write_parquet(dm, "dashboard.parquet")

    th = build_threads(dm)
    write_parquet(th, "threads.parquet")
    del dm

    cm, raw_total = build_comments()
    write_parquet(cm, "comments.parquet")

    sent = build_sentiment()
    write_parquet(sent, "sentiment.parquet")

    for src, dst in PASSTHROUGH:
        if src.exists():
            write_parquet(pd.read_csv(src), dst)

    meta["sources"] = {
        "threads": {
            "rows": int(len(th)),
            "min_wib": str(th.created_wib.min()),
            "max_wib": str(th.created_wib.max()),
            "flairs": int(th.link_flair_text.nunique()),
            "authors": int(th.author.nunique()),
        },
        "comments": {
            "rows": int(len(cm)),
            "min_wib": str(cm.created_wib.min()),
            "max_wib": str(cm.created_wib.max()),
            "authors": int(cm.author.nunique()),
            "threads_referenced": int(cm.thread_id.nunique()),
        },
        "sentiment": {"rows": int(len(sent))},
    }
    meta["notes"] = [
        "num_comments (stale Reddit counter) is shown alongside total_comments_actual (truth).",
        "edited == 1970-01-01 00:00:00 is a sentinel for 'never edited'; exposed as was_edited.",
        "Dashboard duplicate thread_ids resolved by keeping the highest dataset version.",
        "Derived files (leaderboard, flair_performance, flair_top_words, wib_hourly) are "
        "static snapshots and do NOT respond to filters; all other views are recomputed live.",
    ]
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log(f"done in {time.time() - t0:.1f}s -> {CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
