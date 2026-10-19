"""GPU-friendly, resumable sentiment labeling for Colab/Kaggle.

Approach 2 router:
  id/unknown -> w11wo/indonesian-roberta-base-sentiment-classifier
  en         -> cardiffnlp/twitter-roberta-base-sentiment-latest

Canonical output columns:
  p_negative, p_neutral, p_positive, label, confidence

Examples:
  python sentiment_gpu.py --input-dir data/cache --limit 10000 --output data/derived/sentiment_10k.parquet
  python sentiment_gpu.py --input-dir data/cache --limit 728522 --output data/derived/sentiment_full.parquet

The script resumes from an existing output/checkpoint by comment_id. It imports
PyTorch before pandas to avoid the Windows OpenMP/DLL issue, but Colab/Kaggle
Linux does not need that workaround.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import numpy as np
import pandas as pd

CANONICAL = ["negative", "neutral", "positive"]
ID_MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"
EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 256
BATCH_SIZE = 64
CHUNK_SIZE = 10_000


def log(msg: str) -> None:
    print(f"[sentiment-gpu] {msg}", flush=True)


def load_data(input_dir: Path, limit: int | None, seed: int) -> pd.DataFrame:
    core_path = input_dir / "comments.parquet"
    text_path = input_dir / "comments_text.parquet"
    if not core_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            f"Expected {core_path} and {text_path}. Upload/copy both cache files."
        )

    core = pd.read_parquet(core_path).drop_duplicates("comment_id")
    text = pd.read_parquet(text_path, columns=["comment_id", "body"]).drop_duplicates("comment_id")
    d = core.merge(text, on="comment_id", how="inner")
    d["body"] = d["body"].fillna("").astype(str).str.strip()
    d = d[(d["body"].str.len() >= 2) & (d["word_count"] >= 2)].drop_duplicates("body")
    if limit is not None and limit < len(d):
        d = d.sample(limit, random_state=seed)
    return d.reset_index(drop=True)


def detect_routes(texts: list[str], mode: str) -> list[str]:
    if mode == "id_only":
        return ["id"] * len(texts)
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        out = []
        for text in texts:
            try:
                lang = detect(text[:512].replace("\n", " "))
            except Exception:
                lang = "id"
            # Short/code-mixed comments are safer through the Indonesian model.
            out.append("en" if lang == "en" and len(text) >= 48 else "id")
        return out
    except Exception:
        log("langdetect unavailable; routing all rows to Indonesian model")
        return ["id"] * len(texts)


class ModelRunner:
    def __init__(self, model_name: str, index_to_canonical: dict[int, int], device: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        log(f"loading {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        # canonical output [negative, neutral, positive]
        perm = np.zeros(3, dtype=np.int64)
        for source, canonical in index_to_canonical.items():
            perm[canonical] = source
        self.perm = perm

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        probs = torch.softmax(self.model(**enc).logits, dim=-1)
        return probs[:, self.perm].detach().cpu().numpy().astype("float32")


def classify_frame(d: pd.DataFrame, routing: str, batch_size: int, device: str) -> pd.DataFrame:
    id_runner = ModelRunner(ID_MODEL, {0: 2, 1: 1, 2: 0}, device)
    en_runner = None
    routes = detect_routes(d["body"].tolist(), routing)
    d = d.copy()
    d["detected_lang"] = routes
    d["route"] = routes
    probs = np.zeros((len(d), 3), dtype="float32")

    for route, runner in [("id", id_runner), ("en", en_runner)]:
        idx = np.flatnonzero(d["route"].to_numpy() == route)
        if not len(idx):
            continue
        if route == "en" and runner is None:
            runner = ModelRunner(EN_MODEL, {0: 0, 1: 1, 2: 2}, device)
            en_runner = runner
        log(f"route={route}: {len(idx):,} comments")
        for start in range(0, len(idx), batch_size):
            sel = idx[start:start + batch_size]
            probs[sel] = runner.predict(d.iloc[sel]["body"].tolist())
            if start == 0 or (start + len(sel)) % (batch_size * 10) == 0 or start + len(sel) == len(idx):
                log(f"  {min(start + len(sel), len(idx)):,}/{len(idx):,}")

    out = d[["comment_id", "thread_id", "author", "score", "created_wib", "body"]].copy()
    out["detected_lang"] = d["detected_lang"].to_numpy()
    out["route"] = d["route"].to_numpy()
    for i, lab in enumerate(CANONICAL):
        out[f"p_{lab}"] = probs[:, i]
    out["label"] = [CANONICAL[i] for i in probs.argmax(axis=1)]
    out["confidence"] = probs.max(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--routing", choices=["router", "id_only"], default="router")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}; torch={torch.__version__}")
    if device == "cuda":
        log(f"GPU={torch.cuda.get_device_name(0)}")
    d = load_data(Path(args.input_dir), args.limit, args.seed)
    log(f"prepared {len(d):,} unique comment bodies")

    output = Path(args.output)
    checkpoint = output.with_suffix(".checkpoint.parquet")
    done = pd.DataFrame()
    if checkpoint.exists():
        done = pd.read_parquet(checkpoint)
        done_ids = set(done.comment_id)
        d = d[~d.comment_id.isin(done_ids)].reset_index(drop=True)
        log(f"resuming: {len(done):,} already complete; {len(d):,} remaining")

    t0 = time.time()
    pieces = [done] if not done.empty else []
    for start in range(0, len(d), args.chunk_size):
        chunk = d.iloc[start:start + args.chunk_size].copy()
        result = classify_frame(chunk, args.routing, args.batch_size, device)
        pieces.append(result)
        all_done = pd.concat(pieces, ignore_index=True)
        all_done.to_parquet(checkpoint, index=False, compression="zstd")
        rate = len(all_done) / max(time.time() - t0, 1e-6)
        log(f"checkpoint {len(all_done):,}/{args.limit:,}; rate={rate:.1f}/s")

    result = pd.concat(pieces, ignore_index=True).drop_duplicates("comment_id")
    result.to_parquet(output, index=False, compression="zstd")
    if checkpoint.exists():
        checkpoint.unlink()
    summary = {
        "rows": int(len(result)),
        "unique_comment_ids": int(result.comment_id.nunique()),
        "label_distribution": result.label.value_counts().to_dict(),
        "route_distribution": result.route.value_counts().to_dict(),
        "mean_confidence": round(float(result.confidence.mean()), 4),
        "low_confidence_lt_0_6": round(float((result.confidence < 0.6).mean()), 4),
        "device": device,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
