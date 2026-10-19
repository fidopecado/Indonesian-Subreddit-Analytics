"""Legacy one-off check for the pilot batches 0004/0005 (see build_labels.py)."""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # project root
BATCH_DIR = BASE / "artifacts" / "label_batches"
OUT_DIR = BASE / "artifacts" / "labels_out"

ok = True
for batch in ["batch_0004", "batch_0005"]:
    with open(BATCH_DIR / (batch + ".json"), encoding="utf-8") as f:
        src = json.load(f)
    with open(OUT_DIR / (batch + ".json"), encoding="utf-8") as f:
        out = json.load(f)
    src_ids = [c["comment_id"] for c in src]
    out_ids = [c["comment_id"] for c in out]
    same = src_ids == out_ids
    nodup = len(out_ids) == len(set(out_ids))
    valid_labels = all(c["label"] in ("positive", "neutral", "negative") for c in out)
    keys_ok = all(list(c.keys()) == ["comment_id", "label"] for c in out)
    print(f"{batch}: input={len(src)} output={len(out)} ids_match={same} no_dupes={nodup} labels_valid={valid_labels} keys_ok={keys_ok}")
    ok = ok and same and nodup and valid_labels and keys_ok and len(src) == len(out)
sys.exit(0 if ok else 1)
