"""Split the 10,000-comment sample into batch files for LLM labeling.

Each batch file is a JSON array of {comment_id, body} objects, sized so a single
labeling agent can process it accurately in one pass.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
CACHE = BASE / "data" / "cache"
OUT = BASE / "artifacts" / "label_batches"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    sample = pd.read_parquet(CACHE / "comments_sample_10k.parquet")
    sample = sample.sort_values("sample_index").reset_index(drop=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    n = len(sample)
    n_batches = math.ceil(n / args.batch_size)
    index = []

    for b in range(n_batches):
        chunk = sample.iloc[b * args.batch_size:(b + 1) * args.batch_size]
        payload = [
            {"comment_id": r.comment_id, "body": r.body}
            for r in chunk.itertuples()
        ]
        name = f"batch_{b:04d}.json"
        (OUT / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        index.append({"batch": name, "count": len(payload)})

    (OUT / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    total = sum(i["count"] for i in index)
    print(f"batches: {n_batches}  batch_size: {args.batch_size}  total comments: {total}")
    print(f"output dir: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
