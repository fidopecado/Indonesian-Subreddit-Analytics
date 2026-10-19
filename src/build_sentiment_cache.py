"""Prepare dashboard-ready sentiment artifacts from the labeling runs.

Inputs (data/derived unless noted):
    data/derived/sentiment_full.parquet         678,522 HF model labels (full corpus)
    data/derived/sentiment_10k.parquet          HF model labels on a 10K sample
    data/cache/comment_sentiment_llm_10k.parquet  assistant (LLM) labels, 10,000 rows

Outputs (data/cache/):
    sentiment_full_slim.parquet       per-comment full-corpus labels, no body text
    sentiment_hf_10k.parquet          the HF 10K sample, copied into cache
    sentiment_by_flair.parquet        flair x label counts
    sentiment_by_flair_month.parquet  flair x month x label counts (reactive)
    sentiment_monthly.parquet         month x label counts
    sentiment_overview.json           headline metrics, agreement, calibration

Run once after new labeling output lands:
    python src/build_sentiment_cache.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
DATA_DERIVED = BASE / "data" / "derived"
CACHE = BASE / "data" / "cache"

ORDER = ["negative", "neutral", "positive"]
NO_FLAIR = "No Flair"


def _read(name: str) -> pd.DataFrame:
    path = DATA_DERIVED / name
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_parquet(path)


def thread_flair_map() -> pd.Series:
    """thread_id -> link_flair_text, resolving duplicate thread versions."""
    th = pd.read_parquet(CACHE / "threads.parquet")
    cols = ["thread_id", "link_flair_text"]
    if "version" in th.columns:
        th = th.sort_values("version")
    th = th[cols].drop_duplicates("thread_id", keep="last")
    flair = th.set_index("thread_id")["link_flair_text"]
    return flair.fillna(NO_FLAIR).replace("", NO_FLAIR)


def calibrate(confusion: pd.DataFrame, observed: pd.Series) -> dict:
    """Bias-correct the full-corpus prevalence using the 10K confusion matrix.

    A[i][j] = P(model = j | reference = i). Because observed_j = sum_i true_i * A[i][j],
    the corrected prevalence solves A.T @ x = observed. We clip to the simplex.
    """
    a = confusion.reindex(index=ORDER, columns=ORDER).fillna(0).to_numpy(dtype=float)
    row_sums = a.sum(axis=1, keepdims=True)
    a = np.divide(a, row_sums, out=np.zeros_like(a), where=row_sums > 0)

    obs = observed.reindex(ORDER).fillna(0).to_numpy(dtype=float)
    obs = obs / obs.sum() if obs.sum() else obs

    x, *_ = np.linalg.lstsq(a.T, obs, rcond=None)
    x = np.clip(x, 0.0, None)
    x = x / x.sum() if x.sum() else x

    return {
        "method": "least-squares prevalence correction using the 10K confusion matrix",
        "observed": {lab: round(float(v), 6) for lab, v in zip(ORDER, obs)},
        "corrected": {lab: round(float(v), 6) for lab, v in zip(ORDER, x)},
        "transition_matrix_p_model_given_reference": {
            ref: {mod: round(float(v), 6) for mod, v in zip(ORDER, a[i])}
            for i, ref in enumerate(ORDER)
        },
    }


def main() -> int:
    CACHE.mkdir(exist_ok=True)
    full = _read("sentiment_full.parquet")
    hf10 = _read("sentiment_10k.parquet")
    llm = pd.read_parquet(CACHE / "comment_sentiment_llm_10k.parquet")

    for d in (full, hf10, llm):
        d["comment_id"] = d["comment_id"].astype(str)

    if "label" in llm.columns and "llm_label" not in llm.columns:
        llm = llm.rename(columns={"label": "llm_label"})

    # ---------------------------------------------------------------- slim full
    slim_cols = [
        c
        for c in [
            "comment_id", "thread_id", "label", "confidence", "route",
            "detected_lang", "p_negative", "p_neutral", "p_positive",
            "score", "created_wib",
        ]
        if c in full.columns
    ]
    slim = full[slim_cols].copy()
    slim.to_parquet(CACHE / "sentiment_full_slim.parquet", index=False, compression="zstd")

    hf10.to_parquet(CACHE / "sentiment_hf_10k.parquet", index=False, compression="zstd")

    # ------------------------------------------------------------- flair / time
    flair = thread_flair_map()
    enriched = slim.copy()
    enriched["flair"] = enriched["thread_id"].map(flair).fillna(NO_FLAIR)

    if "created_wib" in enriched.columns:
        ts = pd.to_datetime(enriched["created_wib"], errors="coerce")
        enriched["month"] = ts.dt.to_period("M").astype(str)
        enriched.loc[ts.isna(), "month"] = "Unknown"
    else:
        # Slim builds without a timestamp cannot be bucketed by month.
        enriched["month"] = "Unknown"

    by_flair = (
        enriched.groupby(["flair", "label"], observed=True)
        .agg(n=("comment_id", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )
    by_flair.to_parquet(CACHE / "sentiment_by_flair.parquet", index=False)

    by_flair_month = (
        enriched.groupby(["flair", "month", "label"], observed=True)
        .size()
        .reset_index(name="n")
    )
    by_flair_month.to_parquet(CACHE / "sentiment_by_flair_month.parquet", index=False)

    monthly = enriched.groupby(["month", "label"], observed=True).size().reset_index(name="n")
    monthly.to_parquet(CACHE / "sentiment_monthly.parquet", index=False)

    # ------------------------------------------------------------- 10K agreement
    compare = llm[["comment_id", "llm_label"]].merge(
        hf10[["comment_id", "label", "confidence", "route"]].rename(
            columns={"label": "hf_label", "confidence": "hf_confidence"}
        ),
        on="comment_id",
        how="inner",
        validate="one_to_one",
    )

    # The HF 10K sample does not share the LLM sample's draw, so the honest
    # same-comment comparison comes from the full run, which covers every LLM id.
    full_labels = slim[["comment_id", "label", "confidence", "route"]].rename(
        columns={"label": "hf_label", "confidence": "hf_confidence"}
    )
    compare_full = llm[["comment_id", "llm_label"]].merge(
        full_labels, on="comment_id", how="inner", validate="one_to_one"
    )
    compare_full["agree"] = compare_full["llm_label"] == compare_full["hf_label"]

    def agreement_block(c: pd.DataFrame) -> dict:
        if c.empty:
            return {}
        c = c.copy()
        c["agree"] = c["llm_label"] == c["hf_label"]
        cm = pd.crosstab(c["llm_label"], c["hf_label"]).reindex(
            index=ORDER, columns=ORDER, fill_value=0
        )
        po = float(c["agree"].mean())
        a = c["llm_label"].value_counts(normalize=True).reindex(ORDER, fill_value=0)
        b = c["hf_label"].value_counts(normalize=True).reindex(ORDER, fill_value=0)
        pe = float((a * b).sum())
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

        per_label = []
        for lab in ORDER:
            tp = int(cm.loc[lab, lab])
            fn = int(cm.loc[lab].sum() - tp)
            fp = int(cm[lab].sum() - tp)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
            per_label.append(
                {
                    "label": lab,
                    "support": int(cm.loc[lab].sum()),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                }
            )

        band = pd.cut(
            c["hf_confidence"],
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.00001],
            labels=["<0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
            include_lowest=True,
            right=False,
        )
        by_conf = (
            c.assign(band=band)
            .groupby("band", observed=False)
            .agg(n=("comment_id", "size"), agreement=("agree", "mean"))
            .reset_index()
        )
        by_route = (
            c.groupby("route")
            .agg(n=("comment_id", "size"), agreement=("agree", "mean"),
                 mean_confidence=("hf_confidence", "mean"))
            .reset_index()
        )

        return {
            "matched_rows": int(len(c)),
            "exact_agreement": round(po, 6),
            "cohen_kappa": round(float(kappa), 6),
            "disagreements": int((~c["agree"]).sum()),
            "reference_distribution": {k: int(v) for k, v in c["llm_label"].value_counts().items()},
            "model_distribution": {k: int(v) for k, v in c["hf_label"].value_counts().items()},
            "per_label_vs_reference": per_label,
            "confusion_matrix": {
                r: {col: int(cm.loc[r, col]) for col in ORDER} for r in ORDER
            },
            "agreement_by_confidence": {
                str(r.band): {"n": int(r.n), "agreement": round(float(r.agreement), 6)}
                for r in by_conf.itertuples()
            },
            "agreement_by_route": {
                str(r.route): {
                    "n": int(r.n),
                    "agreement": round(float(r.agreement), 6),
                    "mean_confidence": round(float(r.mean_confidence), 6),
                }
                for r in by_route.itertuples()
            },
        }

    block_full = agreement_block(compare_full)
    block_same = agreement_block(compare) if len(compare) else {}

    # Store the full-run comparison rows (with bodies) for the explorer.
    compare_full = compare_full.merge(
        llm[["comment_id", "body", "thread_id", "author", "score"]],
        on="comment_id",
        how="left",
    )
    compare_full["flair"] = compare_full["thread_id"].map(flair).fillna(NO_FLAIR)
    compare_full.to_parquet(
        CACHE / "sentiment_compare_10k.parquet", index=False, compression="zstd"
    )

    # ------------------------------------------------------------------ headline
    observed = slim["label"].value_counts(normalize=True)
    confusion = pd.crosstab(compare_full["llm_label"], compare_full["hf_label"])

    overview = {
        "generated_by": "build_sentiment_cache.py",
        "full_corpus": {
            "rows": int(len(slim)),
            "unique_comment_ids": int(slim["comment_id"].nunique()),
            "distribution": {k: int(v) for k, v in slim["label"].value_counts().items()},
            "distribution_pct": {
                k: round(float(v), 6) for k, v in observed.items()
            },
            "mean_confidence": round(float(slim["confidence"].mean()), 6),
            "low_confidence_lt_0_6": round(float((slim["confidence"] < 0.6).mean()), 6),
            "route_distribution": {
                str(k): int(v) for k, v in slim["route"].value_counts().items()
            },
            "flairs": int(enriched["flair"].nunique()),
            "month_min": str(monthly["month"].min()),
            "month_max": str(monthly["month"].max()),
        },
        "hf_10k_sample": {
            "rows": int(len(hf10)),
            "distribution": {k: int(v) for k, v in hf10["label"].value_counts().items()},
        },
        "llm_10k_sample": {
            "rows": int(len(llm)),
            "distribution": {k: int(v) for k, v in llm["llm_label"].value_counts().items()},
        },
        "comparison_full_run_vs_llm_10k": block_full,
        "comparison_hf10k_vs_llm_10k": block_same,
        "calibration": calibrate(confusion, observed),
        "gold40_benchmark_file": "sentiment_router_gold40.benchmark.json",
    }

    (CACHE / "sentiment_overview.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )

    print(json.dumps(
        {
            "full_rows": len(slim),
            "distribution_pct": overview["full_corpus"]["distribution_pct"],
            "agreement_vs_llm": block_full.get("exact_agreement"),
            "kappa": block_full.get("cohen_kappa"),
            "calibrated": overview["calibration"]["corrected"],
            "observed": overview["calibration"]["observed"],
            "flairs": overview["full_corpus"]["flairs"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
