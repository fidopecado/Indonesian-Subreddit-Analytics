"""Verify all LLM batch outputs and assemble the 10K sentiment parquet.

Expected:
  artifacts/label_batches/batch_NNNN.json       input [{comment_id, body}]
  artifacts/labels_out/batch_NNNN.json          output [{comment_id, label}]

The script refuses to assemble if any batch is missing, malformed, duplicated,
or has IDs that do not exactly match its input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
VALID = {"positive", "neutral", "negative"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="artifacts/label_batches")
    ap.add_argument("--output-dir", default="artifacts/labels_out")
    ap.add_argument("--output", default="data/cache/comment_sentiment_llm_10k.parquet")
    args = ap.parse_args()

    inp = BASE / args.input_dir
    out = BASE / args.output_dir
    batches = sorted(inp.glob("batch_*.json"))
    if not batches:
        raise SystemExit("No input batches found")

    rows = []
    errors = []
    for bf in batches:
        of = out / bf.name
        try:
            # utf-8-sig accepts both plain UTF-8 and agent outputs that include a BOM.
            source = json.loads(bf.read_text(encoding="utf-8-sig"))
            if not of.exists():
                errors.append(f"missing output: {of.name}")
                continue
            pred = json.loads(of.read_text(encoding="utf-8-sig"))
            if not isinstance(source, list) or not isinstance(pred, list):
                errors.append(f"not arrays: {bf.name}")
                continue
            src_ids = [str(x.get("comment_id")) for x in source]
            pred_ids = [str(x.get("comment_id")) for x in pred]
            if len(pred) != len(source):
                errors.append(f"count mismatch {bf.name}: {len(pred)} != {len(source)}")
            if pred_ids != src_ids:
                errors.append(f"ID/order mismatch {bf.name}")
            bad = [x.get("label") for x in pred if x.get("label") not in VALID]
            if bad:
                errors.append(f"invalid labels {bf.name}: {sorted(set(bad))}")
            for s, p in zip(source, pred):
                rows.append({"comment_id": s["comment_id"], "body": s.get("body", ""), "label": p.get("label")})
        except Exception as e:
            errors.append(f"parse failure {bf.name}: {e}")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No valid labels assembled")
    dup = int(frame.comment_id.duplicated().sum())
    if dup:
        errors.append(f"duplicate IDs across outputs: {dup}")

    if errors:
        report = out / "assembly_errors.txt"
        report.write_text("\n".join(errors), encoding="utf-8")
        print(f"REFUSED assembly: {len(errors)} errors -> {report}")
        for e in errors[:30]: print(" -", e)
        return 2

    # Enrich labels with comment metadata from the cache.
    core = pd.read_parquet(BASE / "data" / "cache" / "comments.parquet")
    keep = [c for c in ["comment_id", "thread_id", "author", "score", "created_wib"] if c in core.columns]
    result = frame.merge(core[keep].drop_duplicates("comment_id"), on="comment_id", how="left")
    result["confidence"] = None
    result.to_parquet(BASE / args.output, index=False, compression="zstd")
    summary = {
        "batches": len(batches),
        "rows": len(result),
        "unique_comment_ids": int(result.comment_id.nunique()),
        "label_distribution": result.label.value_counts().to_dict(),
        "output": args.output,
    }
    (BASE / "data" / "cache" / "comment_sentiment_llm_10k.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
