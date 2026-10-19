"""Benchmark sentiment predictions against an independent LLM-labeled gold CSV.

Expected gold_llm.csv columns: comment_id,gold_llm,note
Expected prediction columns: comment_id,label,confidence,...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold", default=str(BASE / "data" / "derived" / "gold_llm.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred = pd.read_parquet(args.predictions)
    gold = pd.read_csv(args.gold)
    d = gold.merge(pred, on="comment_id", how="inner")
    if d.empty:
        raise SystemExit("No matching comment_id values between predictions and gold set")

    y = d["gold_llm"].astype(str).str.lower()
    p = d["label"].astype(str).str.lower()
    labels = ["negative", "neutral", "positive"]
    cm = pd.crosstab(y, p).reindex(index=labels, columns=labels, fill_value=0)

    metrics = {
        "matched_rows": int(len(d)),
        "gold_distribution": y.value_counts().to_dict(),
        "prediction_distribution": p.value_counts().to_dict(),
        "accuracy": float((y == p).mean()),
        "mean_confidence": float(d["confidence"].mean()) if "confidence" in d else None,
        "accuracy_high_confidence_ge_0_8": float((y[d.confidence >= 0.8] == p[d.confidence >= 0.8]).mean()) if "confidence" in d and (d.confidence >= 0.8).any() else None,
        "confusion_matrix": cm.to_dict(),
    }
    out = Path(args.out) if args.out else Path(args.predictions).with_suffix(".benchmark.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nRows with disagreements:")
    print(d.loc[y != p, ["comment_id", "body", "gold_llm", "label", "confidence"]].to_string(index=False) if (y != p).any() else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
