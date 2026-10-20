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


def hf_ids() -> set:
    """The ids the HF pass actually scored. Labeling the same ids is what makes
    the two raters comparable; anything outside this set cannot be compared."""
    p = CACHE / "sentiment_full_slim.parquet"
    return set(pd.read_parquet(p, columns=["comment_id"])["comment_id"])


def eligible(hf_only: bool = True) -> pd.DataFrame:
    """Comment ids in a stable, resumable order.

    The order is the md5 of the id, not row order: it is reproducible across
    machines and does not drift when the cache is rebuilt. That matters more
    than it looks -- it means a prefix of the run is a *uniform random sample*
    of the corpus, so stopping early still yields usable statistics.
    """
    core = pd.read_parquet(CACHE / "comments.parquet", columns=["comment_id", "word_count"])
    text = pd.read_parquet(CACHE / "comments_text.parquet", columns=["comment_id", "body"])
    d = core.merge(text, on="comment_id", how="inner").drop_duplicates("comment_id")
    d["body"] = d["body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    d = d[(d["body"].str.len() >= 2) & (d["word_count"] >= 2)]
    if hf_only:
        d = d[d["comment_id"].isin(hf_ids())]
    d["order"] = d["comment_id"].map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
    return d.sort_values("order", kind="stable").reset_index(drop=True)


def load_cursor() -> dict:
    if CURSOR.exists():
        return json.loads(CURSOR.read_text(encoding="utf-8"))
    return {"batch": 0}


def save_cursor(cursor: dict) -> None:
    tmp = CURSOR.with_suffix(".tmp")
    tmp.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
    os.replace(tmp, CURSOR)


def load_labels() -> pd.DataFrame:
    if LABELS.exists():
        return pd.read_parquet(LABELS)
    return pd.DataFrame(columns=["comment_id", "label", "batch", "labeler", "ts"])


def cmd_status(_: argparse.Namespace) -> None:
    target = len(eligible(hf_only=True))
    everything = len(eligible(hf_only=False))
    lab = load_labels()
    cursor = load_cursor()
    done = len(set(lab["comment_id"]) & set(eligible(hf_only=True)["comment_id"]))
    log(f"HF-labeled target : {target:,}  (the set to compare against)")
    log(f"all eligible      : {everything:,}  (--all-eligible)")
    log(f"labeled           : {done:,}  ({done / max(target, 1):.2%} of target)")
    log(f"batch counter     : {cursor['batch']}")
    if done:
        mix = lab["label"].value_counts(normalize=True).sort_index()
        log("class mix         : " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items()))
    log(f"remaining         : {max(target - done, 0):,}")


def cmd_next(args: argparse.Namespace) -> None:
    cursor = load_cursor()
    d = eligible(hf_only=not args.all_eligible)
    done = set(load_labels()["comment_id"])
    todo = d[~d["comment_id"].isin(done)]
    if todo.empty:
        log("nothing left to label")
        return
    take = todo.iloc[:args.n]
    BATCHES.mkdir(parents=True, exist_ok=True)
    num = cursor["batch"] + 1
    txt = BATCHES / f"batch_{num:04d}.txt"
    csv = BATCHES / f"batch_{num:04d}.csv"
    txt.write_text("\n".join(take["body"].tolist()), encoding="utf-8")
    take[["comment_id", "body"]].to_csv(csv, index=False, encoding="utf-8")
    log(f"batch {num}: {len(take)} comments  ({len(done):,} labeled, {len(todo):,} left)")
    log(f"  bodies: {txt}")
    log(f"  ids   : {csv}")
    log(f"answer with exactly {len(take)} characters from + - 0 ?")


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
    cursor["batch"] = max(cursor["batch"], args.batch)
    save_cursor(cursor)
    log(f"committed batch {args.batch}: {len(new)} labels ({len(merged):,} total)")
    log("mix: " + ", ".join(f"{k} {v:.1%}" for k, v in new["label"].value_counts(normalize=True).sort_index().items()))


def cohens_kappa(a: pd.Series, b: pd.Series, labels: str = "+-0") -> float:
    pairs = pd.DataFrame({"a": a, "b": b}).dropna()
    pairs = pairs[pairs["a"].isin(list(labels)) & pairs["b"].isin(list(labels))]
    if pairs.empty:
        return float("nan")
    po = (pairs["a"] == pairs["b"]).mean()
    pe = sum((pairs["a"] == c).mean() * (pairs["b"] == c).mean() for c in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_report(args: argparse.Namespace) -> None:
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
    log(f"cohen's kappa     : {cohens_kappa(a, b):.3f}   (chance-corrected)")
    log("confusion (labeler rows x HF cols):")
    ct = pd.crosstab(a, b).reindex(index=list("+-0?"), columns=list("+-0"), fill_value=0)
    log(ct.to_string())
    log(f"labeler '?'       : {(a == '?').mean():.1%}")

    # Per-class precision/recall of the *HF* model, treating this labeler as
    # the reference. Precision = of the comments HF called X, how many really
    # are X. Recall = of the comments that really are X, how many HF found.
    log("HF per-class (labeler as reference):")
    for c in "+-0":
        tp = int(((a == c) & (b == c)).sum())
        fp = int(((a != c) & (b == c)).sum())
        fn = int(((a == c) & (b != c)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        log(f"  {c}  precision {prec:6.1%}   recall {rec:6.1%}   (n={tp + fn})")

    if args.drift:
        log("per-batch class mix (a jump >10pp between batches is drift, not news):")
        d = m.copy()
        d["k"] = d["label"].map({**short, "+": "+", "-": "-", "0": "0", "?": "?"})
        for batch, g in d.groupby("batch"):
            mix = g["k"].value_counts(normalize=True).reindex(list("+-0?")).fillna(0)
            log(f"  batch {int(batch):<3} n={len(g):<5} " + "  ".join(f"{c} {mix[c]:5.1%}" for c in "+-0?"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("next")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--all-eligible", action="store_true",
                   help="include comments the HF pass never scored (not comparable)")
    p.set_defaults(func=cmd_next)
    c = sub.add_parser("commit")
    c.add_argument("--batch", type=int, required=True)
    c.add_argument("--labels", default="")
    c.add_argument("--labels-file", default="")
    c.add_argument("--labeler", default="assistant")
    c.set_defaults(func=cmd_commit)
    r = sub.add_parser("report")
    r.add_argument("--drift", action="store_true", help="per-batch class mix")
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
