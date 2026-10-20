"""Assistant-in-the-loop labeling queue: resumable, one batch per turn.

The idea is the same one that makes ``sentiment_gpu.py`` safe to interrupt: a
batch is only "done" once its labels are committed, and the cursor only moves
after an atomic write. If a turn dies mid-batch, the batch is re-issued.

A batch is handed out as ``batches/batch_NNNN.txt`` (one comment body per line,
newlines flattened) plus ``.csv`` (comment_id, body) for the join. Because the
order is fixed, the labeler only has to answer with one character per line:

    +  positive
    -  negative
    0  neutral
    ?  cannot tell (sarcasm, too short, pure link)

That keeps the answer to ~1 token per comment instead of repeating ids.

Usage:
    python src/llm_label.py status
    python src/llm_label.py next --n 120
    python src/llm_label.py commit --batch 1 --labels "+-0+..."
    python src/llm_label.py report            # agreement vs the HF pass

Eligibility matches the GPU pipeline: >=2 words and a body of >=2 characters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "derived" / "llm_labels"
BATCHES = OUT / "batches"
CURSOR = OUT / "cursor.json"
LABELS = OUT / "labels.parquet"
VALID = set("+-0?")


def log(msg: str) -> None:
    print(msg, flush=True)


def eligible() -> pd.DataFrame:
    """Comment ids in a stable, resumable order.

    The order is the md5 of the id, not row order: it is reproducible across
    machines and does not drift when the cache is rebuilt.
    """
    core = pd.read_parquet(CACHE / "comments.parquet", columns=["comment_id", "word_count"])
    text = pd.read_parquet(CACHE / "comments_text.parquet", columns=["comment_id", "body"])
    d = core.merge(text, on="comment_id", how="inner").drop_duplicates("comment_id")
    d["body"] = d["body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    d = d[(d["body"].str.len() >= 2) & (d["word_count"] >= 2)]
    d["order"] = d["comment_id"].map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
    return d.sort_values("order", kind="stable").reset_index(drop=True)


def load_cursor() -> dict:
    if CURSOR.exists():
        return json.loads(CURSOR.read_text(encoding="utf-8"))
    return {"offset": 0, "batch": 0}


def save_cursor(cursor: dict) -> None:
    tmp = CURSOR.with_suffix(".tmp")
    tmp.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
    os.replace(tmp, CURSOR)


def load_labels() -> pd.DataFrame:
    if LABELS.exists():
        return pd.read_parquet(LABELS)
    return pd.DataFrame(columns=["comment_id", "label", "batch", "labeler", "ts"])


def cmd_status(_: argparse.Namespace) -> None:
    total = len(eligible())
    lab = load_labels()
    cursor = load_cursor()
    done = lab["comment_id"].nunique()
    log(f"eligible comments : {total:,}")
    log(f"labeled           : {done:,}  ({done / max(total, 1):.2%})")
    log(f"cursor            : batch {cursor['batch']}, offset {cursor['offset']:,}")
    if done:
        mix = lab["label"].value_counts(normalize=True).sort_index()
        log("class mix         : " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items()))
    if done < total:
        log(f"remaining         : {total - done:,}")


def cmd_next(args: argparse.Namespace) -> None:
    total = len(eligible())
    cursor = load_cursor()
    start = cursor["offset"]
    if start >= total:
        log("nothing left to label")
        return
    d = eligible().iloc[start:start + args.n]
    BATCHES.mkdir(parents=True, exist_ok=True)
    num = cursor["batch"] + 1
    txt = BATCHES / f"batch_{num:04d}.txt"
    csv = BATCHES / f"batch_{num:04d}.csv"
    txt.write_text("\n".join(d["body"].tolist()), encoding="utf-8")
    d[["comment_id", "body"]].to_csv(csv, index=False, encoding="utf-8")
    log(f"batch {num}: {len(d)} comments  ({start:,} -> {start + len(d):,} of {total:,})")
    log(f"  bodies: {txt}")
    log(f"  ids   : {csv}")
    log(f"answer with exactly {len(d)} characters from + - 0 ?")


def cmd_commit(args: argparse.Namespace) -> None:
    csv = BATCHES / f"batch_{args.batch:04d}.csv"
    if not csv.exists():
        sys.exit(f"no such batch: {csv}")
    batch = pd.read_csv(csv)
    if args.labels_file:
        raw = Path(args.labels_file).read_text(encoding="utf-8")
    else:
        raw = args.labels or ""
    labels = raw.strip().replace(" ", "").replace("\n", "")
    bad = set(labels) - VALID
    if bad:
        sys.exit(f"invalid label characters: {sorted(bad)}")
    if len(labels) != len(batch):
        sys.exit(f"length mismatch: {len(labels)} labels for {len(batch)} comments")

    new = pd.DataFrame({
        "comment_id": batch["comment_id"].tolist(),
        "label": list(labels),
        "batch": args.batch,
        "labeler": args.labeler,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    old = load_labels()
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates("comment_id", keep="last")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = LABELS.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, LABELS)

    cursor = load_cursor()
    cursor["offset"] += len(batch)
    cursor["batch"] = args.batch
    save_cursor(cursor)
    log(f"committed batch {args.batch}: {len(new)} labels ({len(merged):,} total)")
    log("mix: " + ", ".join(f"{k} {v:.1%}" for k, v in new["label"].value_counts(normalize=True).sort_index().items()))


def cmd_report(_: argparse.Namespace) -> None:
    lab = load_labels()
    if lab.empty:
        sys.exit("no labels yet")
    sent = pd.read_parquet(CACHE / "sentiment_full_slim.parquet", columns=["comment_id", "label", "confidence"])
    m = lab.merge(sent, on="comment_id", how="left", suffixes=("", "_hf"))
    m = m[m["label_hf"].notna()]
    if m.empty:
        sys.exit("no overlap with the HF pass")
    short = {"positive": "+", "negative": "-", "neutral": "0"}
    # the label file may hold either short codes (+ - 0 ?) or full words
    a = m["label"].map({**short, "+": "+", "-": "-", "0": "0", "?": "?"})
    b = m["label_hf"].map(short)
    agree = (a == b).mean()
    log(f"overlap           : {len(m):,}")
    log(f"agreement         : {agree:.1%}")
    log("confusion (assistant rows x HF cols):")
    ct = pd.crosstab(a, b).reindex(index=list("+-0?"), columns=list("+-0"), fill_value=0)
    log(ct.to_string())
    log(f"assistant '?'     : {(a == '?').mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("next")
    p.add_argument("--n", type=int, default=120)
    p.set_defaults(func=cmd_next)
    c = sub.add_parser("commit")
    c.add_argument("--batch", type=int, required=True)
    c.add_argument("--labels", default="")
    c.add_argument("--labels-file", default="")
    c.add_argument("--labeler", default="assistant")
    c.set_defaults(func=cmd_commit)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
