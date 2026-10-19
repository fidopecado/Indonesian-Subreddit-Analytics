"""
Phase 1 pilot: language-routed sentiment classification.

Implements "Approach 2" (the language router) from the dataset design notes:

    detected lang = id  -> w11wo/indonesian-roberta-base-sentiment-classifier
    detected lang = en  -> cardiffnlp/twitter-roberta-base-sentiment-latest
    other / code-mixed  -> Indonesian model (Jakarta slang is routinely
                           misdetected as some other language)

Label-index normalization is critical: the two models order their classes
DIFFERENTLY. We always re-emit a canonical order:

    [p_negative, p_neutral, p_positive]

Usage:
    python sentiment_pilot.py --limit 5000
    python sentiment_pilot.py --limit 5000 --model app1     # comparison baseline
    python sentiment_pilot.py --ids gold_ids.txt            # label specific rows
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# IMPORTANT: torch must be imported BEFORE pandas.
# pandas loads an incompatible OpenMP runtime, after which torch's c10.dll
# fails with WinError 1114 ("DLL initialization routine failed").
# Keeping torch first is what makes this pipeline runnable on Windows.
import torch

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
CACHE = BASE / "data" / "cache"

# Redirect the Hugging Face cache into the workspace. The default location
# (%USERPROFILE%\.cache\huggingface) is outside the writable sandbox and
# raises PermissionError when transformers tries to write there.
HF_HOME = BASE / ".hf_cache"
HF_HOME.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CANONICAL = ["negative", "neutral", "positive"]

# Model registry. `index_map` maps the model's own class index -> canonical index.
#
# `use_fast=False` is required for the two cardiffnlp repos: neither ships a
# fast-tokenizer artefact (no tokenizer.json / tokenizer_config.json), only the
# slow vocab.json+merges.txt (RoBERTa) or sentencepiece.bpe.model (XLM-R).
MODELS = {
    "id": {
        "name": "w11wo/indonesian-roberta-base-sentiment-classifier",
        # model says: 0=positive, 1=neutral, 2=negative
        "index_map": {0: 2, 1: 1, 2: 0},
        "use_fast": True,
    },
    "en": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        # model says: 0=negative, 1=neutral, 2=positive
        "index_map": {0: 0, 1: 1, 2: 2},
        "use_fast": False,
    },
    # Approach 1 baseline: one multilingual model for everything.
    "app1": {
        "name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        # model says: 0=negative, 1=neutral, 2=positive
        "index_map": {0: 0, 1: 1, 2: 2},
        "use_fast": False,
    },
}

MAX_LEN = 256
BATCH = 32
CHECKPOINT_EVERY = 20_000


def log(msg: str) -> None:
    print(f"[sentiment] {msg}", flush=True)


def detect_lang(texts: list[str]) -> list[str]:
    """Language detection with a graceful fallback to Indonesian.

    Reddit Indonesian is code-mixed, so any detector will misfire. We only
    trust a confident 'en' verdict; everything else routes to Indonesian.
    """
    try:
        from ftlangdetect import detect  # fasttext-based, fast
        out = []
        for t in texts:
            try:
                r = detect(text=t.replace("\n", " ")[:512], low_memory=False)
                out.append((r.get("lang") or "id").lower())
            except Exception:
                out.append("id")
        return out
    except Exception:
        pass

    try:
        from langdetect import detect as ld, DetectorFactory  # type: ignore
        DetectorFactory.seed = 0
        out = []
        for t in texts:
            try:
                out.append(ld(t.replace("\n", " ")[:512]))
            except Exception:
                out.append("id")
        return out
    except Exception:
        log("no language detector available; routing everything to the Indonesian model")
        return ["id"] * len(texts)


def pick_route(langs: list[str], text_lens: list[int]) -> list[str]:
    """Map detected language -> model key.

    Short comments are unreliable for detection, so only trust 'en' when the
    text is long enough to be genuinely English.
    """
    routes = []
    for lg, n in zip(langs, text_lens):
        if lg == "en" and n >= 12:
            routes.append("en")
        else:
            routes.append("id")
    return routes


class Classifier:
    def __init__(self, model_name: str, index_map: dict[int, int], use_fast: bool = True):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log(f"loading {model_name} (use_fast={use_fast})")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.index_map = index_map
        n = len(index_map)
        # Build a permutation so outputs are always [neg, neu, pos].
        self.perm = np.zeros(n, dtype=int)
        for src, dst in index_map.items():
            self.perm[dst] = src

    @torch.no_grad()
    def probs(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(
            texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        logits = self.model(**enc).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        return p[:, self.perm]


def load_comments(limit: int | None, seed: int, ids_file: str | None) -> pd.DataFrame:
    core = pd.read_parquet(
        CACHE / "comments.parquet",
        columns=["comment_id", "thread_id", "author", "score", "created_wib",
                 "word_count", "body_len", "is_submitter", "is_mod"],
    ).drop_duplicates("comment_id")
    txt = pd.read_parquet(
        CACHE / "comments_text.parquet", columns=["comment_id", "body"]
    ).drop_duplicates("comment_id")

    d = core.merge(txt, on="comment_id", how="left")
    d["body"] = d["body"].fillna("").astype(str)

    if ids_file:
        ids = [l.strip() for l in Path(ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        d = d[d["comment_id"].isin(ids)]
        log(f"selected {len(d)} rows by id list")
        return d.reset_index(drop=True)

    d = d[d["word_count"] > 0]
    if limit and limit < len(d):
        d = d.sample(limit, random_state=seed)
    return d.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000, help="rows to classify")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["router", "app1"], default="router",
                    help="router = Approach 2; app1 = Approach 1 multilingual baseline")
    ap.add_argument("--ids", default=None, help="file of comment_ids to classify")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    d = load_comments(args.limit, args.seed, args.ids)
    log(f"classifying {len(d):,} comments")
    if d.empty:
        log("nothing to do")
        return 1

    t0 = time.time()

    # ---- route ----
    if args.model == "app1":
        routes = ["app1"] * len(d)
        d["detected_lang"] = "n/a"
    else:
        log("detecting language…")
        langs = detect_lang(d["body"].tolist())
        d["detected_lang"] = langs
        routes = pick_route(langs, d["body_len"].tolist())
        d["route"] = routes
        vc = pd.Series(routes).value_counts().to_dict()
        log(f"routing: {vc}")

    # ---- classify, grouped by model so each loads once ----
    probs = np.zeros((len(d), 3), dtype=np.float32)
    routes_arr = np.array(routes)
    for key in sorted(set(routes)):
        idx = np.where(routes_arr == key)[0]
        if len(idx) == 0:
            continue
        cfg = MODELS[key]
        clf = Classifier(cfg["name"], cfg["index_map"], cfg.get("use_fast", True))
        log(f"{key}: classifying {len(idx):,} rows with {cfg['name']}")
        for start in range(0, len(idx), BATCH):
            sel = idx[start:start + BATCH]
            probs[sel] = clf.probs(d["body"].iloc[sel].tolist())
            done = min(start + BATCH, len(idx))
            if done % (BATCH * 20) == 0 or done == len(idx):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(d) - done) / max(rate, 1e-6)
                log(f"  {done:,}/{len(idx):,}  {rate:.1f} rows/s  ETA {eta/60:.1f} min")
        del clf

    # ---- emit canonical columns ----
    out = d[["comment_id", "thread_id", "author", "score", "created_wib",
             "word_count", "detected_lang"]].copy()
    if "route" in d:
        out["route"] = d["route"].values
    for i, lab in enumerate(CANONICAL):
        out[f"p_{lab}"] = probs[:, i]
    out["label"] = [CANONICAL[i] for i in probs.argmax(axis=1)]
    out["confidence"] = probs.max(axis=1)

    # Keep the original body for human/LLM review.
    out["body"] = d["body"].values

    dest = Path(args.out) if args.out else CACHE / (
        f"sentiment_{args.model}_pilot.parquet"
    )
    out.to_parquet(dest, index=False, compression="zstd")

    elapsed = time.time() - t0
    rate = len(d) / max(elapsed, 1e-6)
    log(f"done: {len(d):,} rows in {elapsed/60:.1f} min ({rate:.1f} rows/s)")
    log(f"wrote {dest.name}")
    log(f"full-dataset ETA at this rate: {728522/rate/3600:.1f} hours")

    summary = {
        "rows": int(len(out)),
        "seconds": round(elapsed, 1),
        "rows_per_sec": round(rate, 2),
        "model": args.model,
        "label_dist": out["label"].value_counts().to_dict(),
        "mean_conf": round(float(out["confidence"].mean()), 3),
        "low_conf_pct": round(float((out["confidence"] < 0.6).mean()), 3),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
