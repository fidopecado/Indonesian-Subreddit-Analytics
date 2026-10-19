"""
Sample exactly 10,000 unique, non-empty comments deterministically.

Selection rules:
  - Deduplicated by comment_id
  - Deduplicated by body text (prevents redundant inference and bot spam)
  - word_count >= 2 (drops empty or single-character artifacts)
  - Fixed seed (42) for exact reproducibility
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
CACHE = BASE / "data" / "cache"

core = pd.read_parquet(
    CACHE / "comments.parquet",
    columns=[
        "comment_id",
        "thread_id",
        "author",
        "score",
        "created_wib",
        "word_count",
        "body_len",
        "is_submitter",
        "is_mod",
    ],
).drop_duplicates("comment_id")

txt = pd.read_parquet(
    CACHE / "comments_text.parquet", columns=["comment_id", "body"]
).drop_duplicates("comment_id")

d = core.merge(txt, on="comment_id", how="left")
d["body"] = d["body"].fillna("").astype(str).str.strip()

# Meaningful comments only: drop empty, single-character, and identical bodies.
clean = d[(d["word_count"] >= 2) & (d["body"].str.len() >= 6)].copy()
clean = clean.drop_duplicates("body")

sample_10k = clean.sample(10_000, random_state=42).reset_index(drop=True)
sample_10k["sample_index"] = sample_10k.index

out_pq = CACHE / "comments_sample_10k.parquet"
sample_10k.to_parquet(out_pq, index=False, compression="zstd")

# Save a lightweight manifest for labeling tasks.
manifest = sample_10k[["sample_index", "comment_id", "body", "word_count", "score"]]
manifest.to_parquet(CACHE / "comments_sample_10k_manifest.parquet", index=False)

print(f"sampled: {len(sample_10k):,} rows -> {out_pq.name}")
print(f"median words: {sample_10k.word_count.median():.0f}")
print(f"unique authors: {sample_10k.author.nunique():,}")
print(f"unique threads: {sample_10k.thread_id.nunique():,}")
