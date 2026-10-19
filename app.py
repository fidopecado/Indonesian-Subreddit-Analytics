"""
Indonesian Subreddit Analytics Dashboard
========================================

An interactive, reactive Streamlit dashboard over the r/indonesia dataset.

Run:
    streamlit run app.py

Data comes from ./data/cache (built by src/build_cache.py). If the cache is
missing the app offers a one-click build in the sidebar.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "cache"

st.set_page_config(
    page_title="r/indonesia Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NO_FLAIR = "No Flair"
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]

# --------------------------------------------------------------------------- #
# Theme — "e-ink" grayscale palette (github.com/e-ink-colorscheme/e-ink.nvim,
# light mode) with its Everforest accent, styled as brutalist / engineering
# chrome: monospace, hard 2px rules, offset shadows, zero border-radius.
# --------------------------------------------------------------------------- #
EINK = {
    "paper": "#FBFBF9",       # page
    "panel": "#EFEFED",       # raised surfaces
    "grid": "#CCCCCC",        # hairline grid (ramp step 1)
    "ink": "#333333",         # body text / primary marks (ramp step 16)
    "ink_strong": "#1A1A1A",  # rules, borders, headings
    "muted": "#909090",       # secondary text (ramp step 7)
    "accent": "#F85552",      # Everforest red — the single signal colour
    "green": "#8DA101",
    "blue": "#3A94C5",
    "yellow": "#DFA000",
}
# Monochrome ramp taken verbatim from the e-ink palette (light -> dark).
EINK_RAMP = ["#CCCCCC", "#C2C2C2", "#B8B8B8", "#AEAEAE", "#A4A4A4", "#9A9A9A",
             "#909090", "#868686", "#7C7C7C", "#727272", "#686868", "#5E5E5E",
             "#545454", "#4A4A4A", "#474747", "#333333"]
EINK_CATEGORICAL = ["#333333", "#7C7C7C", "#A4A4A4", "#545454", "#909090",
                    "#C2C2C2", "#686868", "#AEAEAE", "#4A4A4A", "#B8B8B8"]
EINK_SEQ = [[0.0, "#FBFBF9"], [0.35, "#CCCCCC"], [0.7, "#727272"], [1.0, "#333333"]]
EINK_SEQ_WARM = [[0.0, "#FBFBF9"], [0.4, "#B8B8B8"], [0.78, "#545454"], [1.0, "#F85552"]]
EINK_MONO = ('"IBM Plex Mono", ui-monospace, "Cascadia Mono", "JetBrains Mono", '
             'Consolas, "Courier New", monospace')

SENTIMENT_COLORS = {
    "positive": EINK["green"],
    "neutral": EINK["muted"],
    "negative": EINK["accent"],
}

# The stylesheet is a Template so the palette lives in exactly one place.
CSS_TEMPLATE = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

* { border-radius: 0 !important; }

html, body, .stApp, [data-testid="stAppViewContainer"], [class*="st-"],
input, textarea, select, button, .stMarkdown, .stText, .stCaption {
  font-family: ${mono} !important;
}

/* paper + faint blueprint grid ------------------------------------------- */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: ${paper} !important;
  background-image:
    linear-gradient(rgba(26,26,26,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26,26,26,.05) 1px, transparent 1px);
  background-size: 26px 26px;
  color: ${ink};
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
#MainMenu, footer { visibility: hidden; }

/* typography -------------------------------------------------------------- */
h1, h2, h3, h4, h5, h6 {
  color: ${ink_strong} !important;
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 700 !important;
}
h1 { border-bottom: 3px solid ${ink_strong}; padding-bottom: .35rem; }
h2 { border-bottom: 2px solid ${ink_strong}; padding-bottom: .3rem; }
h3 { border-left: 8px solid ${accent}; padding-left: .55rem; }
p, li, span, label, blockquote { color: ${ink}; }
a { color: ${ink_strong}; text-decoration: underline; text-underline-offset: 2px; }
hr { border: 0; border-top: 2px solid ${ink_strong}; margin: 1.1rem 0; }

/* sidebar ----------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: ${panel} !important;
  border-right: 3px solid ${ink_strong};
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { border: 0 !important; padding-left: 0; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
  text-transform: uppercase; letter-spacing: .07em; color: ${muted} !important;
}

/* metrics as spec panels -------------------------------------------------- */
[data-testid="stMetric"] {
  background: ${panel};
  border: 2px solid ${ink_strong};
  box-shadow: 4px 4px 0 0 ${ink_strong};
  padding: 12px 14px 10px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  text-transform: uppercase; letter-spacing: .14em;
  font-size: .66rem !important; color: ${muted} !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
  font-weight: 700 !important; color: ${ink_strong} !important;
  font-variant-numeric: tabular-nums;
}

/* tabs -------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 2px solid ${ink_strong}; background: transparent;
}
.stTabs [data-baseweb="tab"] {
  border: 2px solid ${ink_strong}; border-bottom: none; margin-right: -2px;
  background: ${panel}; color: ${ink} !important;
  text-transform: uppercase; letter-spacing: .08em; font-size: .78rem;
  padding: 6px 14px; height: auto;
}
.stTabs [aria-selected="true"] { background: ${ink_strong} !important; color: ${paper} !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* buttons ----------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border: 2px solid ${ink_strong} !important; background: ${paper} !important;
  color: ${ink_strong} !important; text-transform: uppercase; letter-spacing: .1em;
  font-weight: 600 !important; box-shadow: 3px 3px 0 0 ${ink_strong};
  transition: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: ${ink_strong} !important; color: ${paper} !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translate(3px, 3px); box-shadow: none;
}
.stButton > button[kind="primary"] { background: ${accent} !important; color: #FFFFFF !important; }
.stButton > button[kind="primary"]:hover { background: ${ink_strong} !important; }

/* inputs ------------------------------------------------------------------ */
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="base-input"],
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
.stDateInput input, .stTextArea textarea {
  border: 2px solid ${ink_strong} !important; background: ${paper} !important;
  box-shadow: 2px 2px 0 0 ${ink_strong}; color: ${ink_strong} !important;
}
[data-baseweb="tag"] { background: ${ink_strong} !important; color: ${paper} !important; border: none !important; }

/* data surfaces ----------------------------------------------------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 2px solid ${ink_strong}; box-shadow: 4px 4px 0 0 ${ink_strong}; background: ${paper};
}
[data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }

/* notifications & code ---------------------------------------------------- */
[data-testid="stAlert"], [data-testid="stNotification"] {
  border: 2px solid ${ink_strong} !important; box-shadow: 3px 3px 0 0 ${ink_strong};
  background: ${panel} !important; color: ${ink} !important;
}
code, pre, [data-testid="stCodeBlock"] {
  background: ${panel} !important; color: ${ink_strong} !important;
  border: 2px solid ${ink_strong};
}

/* expander ---------------------------------------------------------------- */
[data-testid="stExpander"] { border: 2px solid ${ink_strong} !important; background: ${paper} !important; }
[data-testid="stExpander"] summary { text-transform: uppercase; letter-spacing: .08em; }
</style>
""")

CSS = CSS_TEMPLATE.substitute(
    mono=EINK_MONO, paper=EINK["paper"], panel=EINK["panel"], ink=EINK["ink"],
    ink_strong=EINK["ink_strong"], grid=EINK["grid"], muted=EINK["muted"],
    accent=EINK["accent"],
)
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Cached data loading
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Loading cached dataset…")
def load(name: str) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    path = CACHE / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = CACHE / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cache_ready() -> bool:
    required = ["threads.parquet", "comments.parquet", "dashboard.parquet"]
    return all((CACHE / f).exists() for f in required)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "-"


def fmt_compact(n) -> str:
    try:
        n = float(n)
    except Exception:
        return "-"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:,.0f}"


def fmt_date(s) -> str:
    try:
        return pd.Timestamp(s).strftime("%d %b %Y")
    except Exception:
        return "-"


def empty_fig(msg: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=EINK["muted"], family=EINK_MONO),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor=EINK["paper"], paper_bgcolor=EINK["paper"],
                      font=dict(family=EINK_MONO, color=EINK["ink"]))
    return fig


def style(fig: go.Figure, height: int = 340, **kw) -> go.Figure:
    """Apply the e-ink / engineering look shared by every chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family=EINK_MONO, size=12, color=EINK["ink"]),
        paper_bgcolor=EINK["paper"],
        plot_bgcolor=EINK["paper"],
        colorway=EINK_CATEGORICAL,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", bordercolor=EINK["ink_strong"], borderwidth=1,
            font=dict(family=EINK_MONO, size=11, color=EINK["ink"]),
        ),
        hoverlabel=dict(
            bgcolor=EINK["ink_strong"], bordercolor=EINK["ink_strong"],
            font=dict(family=EINK_MONO, color=EINK["paper"], size=12),
        ),
        **kw,
    )
    fig.update_xaxes(
        showgrid=False, linecolor=EINK["ink_strong"], linewidth=1.4, ticks="outside",
        tickcolor=EINK["ink_strong"], tickfont=dict(family=EINK_MONO, color=EINK["ink"]),
        title_font=dict(family=EINK_MONO, color=EINK["ink"]),
    )
    fig.update_yaxes(
        gridcolor=EINK["grid"], gridwidth=1, griddash="dot", zeroline=False,
        linecolor=EINK["ink_strong"], linewidth=1.4, ticks="outside",
        tickcolor=EINK["ink_strong"], tickfont=dict(family=EINK_MONO, color=EINK["ink"]),
        title_font=dict(family=EINK_MONO, color=EINK["ink"]),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — reactive filters
# --------------------------------------------------------------------------- #


def date_bounds(df: pd.DataFrame, col: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if df.empty or col not in df:
        return None
    lo, hi = df[col].min(), df[col].max()
    if pd.isna(lo) or pd.isna(hi):
        return None
    return pd.Timestamp(lo), pd.Timestamp(hi)


def sidebar_filters(th: pd.DataFrame, cm: pd.DataFrame) -> dict:
    st.sidebar.title("🔎 Filters")
    st.sidebar.caption("Every change re-filters the data and refreshes all charts.")

    # ---- time range (driven by threads, the primary calendar) ----
    st.sidebar.subheader("Time range")
    bounds = date_bounds(th, "created_wib")
    if bounds:
        lo, hi = bounds
        st.sidebar.caption(f"Data spans {fmt_date(lo)} → {fmt_date(hi)}")
        preset = st.sidebar.selectbox(
            "Quick range",
            ["All time", "Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
            index=0,
        )
        spans = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        if preset in spans:
            start = max(lo, hi - pd.Timedelta(days=spans[preset]))
            end = hi
        elif preset == "All time":
            start, end = lo, hi
        else:
            picked = st.sidebar.date_input(
                "Custom range", value=(lo.date(), hi.date()),
                min_value=lo.date(), max_value=hi.date(),
            )
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
            else:
                start, end = lo, hi
        if start > end:
            start, end = end, start
    else:
        start = end = None

    st.sidebar.divider()

    # ---- flair ----
    st.sidebar.subheader("Flair")
    flairs = sorted(th["link_flair_text"].dropna().unique().tolist()) if not th.empty else []
    counts = th["link_flair_text"].value_counts() if not th.empty else pd.Series(dtype=int)
    default_flairs = [f for f in flairs if counts.get(f, 0) >= 20] or flairs
    sel_flairs = st.sidebar.multiselect(
        "Topics", options=flairs, default=default_flairs,
        help="All flairs are selected by default except very rare ones (<20 threads).",
    )
    if not sel_flairs:
        sel_flairs = flairs

    # ---- author / text ----
    st.sidebar.subheader("Search")
    author_q = st.sidebar.text_input("Author UUID contains", "").strip()
    text_q = st.sidebar.text_input("Title contains", "").strip()

    # ---- thresholds ----
    st.sidebar.subheader("Thresholds")
    if not th.empty:
        smin, smax = int(th["score"].min()), int(th["score"].max())
        score_rng = st.sidebar.slider(
            "Thread score", min_value=smin, max_value=smax,
            value=(smin, min(smax, int(th["score"].quantile(0.99)))),
        )
    else:
        score_rng = (0, 0)

    # ---- flags ----
    st.sidebar.subheader("Flags")
    c1, c2 = st.sidebar.columns(2)
    f_nsfw = c1.checkbox("Include NSFW", value=True)
    f_locked = c2.checkbox("Include locked", value=True)
    f_mod = c1.checkbox("Include mod posts", value=True)
    f_stick = c2.checkbox("Include stickied", value=True)
    f_spoil = c1.checkbox("Include spoilers", value=True)
    f_edited = c2.checkbox("Include edited", value=True)

    post_type = st.sidebar.radio("Post type", ["All", "Text", "Link"], horizontal=True)

    min_comments = st.sidebar.number_input(
        "Minimum comments (actual)", min_value=0, value=0, step=5,
        help="Uses total_comments_actual — the true joined count, not Reddit's stale counter.",
    )

    return {
        "start": start, "end": end,
        "flairs": sel_flairs,
        "author_q": author_q,
        "text_q": text_q,
        "score_rng": score_rng,
        "f_nsfw": f_nsfw, "f_locked": f_locked, "f_mod": f_mod,
        "f_stick": f_stick, "f_spoil": f_spoil, "f_edited": f_edited,
        "post_type": post_type,
        "min_comments": int(min_comments),
    }


def apply_filters(th: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = th

    if f["start"] is not None:
        d = d[d["created_wib"].between(f["start"], f["end"])]

    if f["flairs"]:
        d = d[d["link_flair_text"].isin(f["flairs"])]

    if f["author_q"]:
        d = d[d["author"].astype(str).str.contains(f["author_q"], case=False, na=False)]

    if f["text_q"]:
        d = d[d["title"].str.contains(f["text_q"], case=False, na=False, regex=False)]

    lo, hi = f["score_rng"]
    d = d[(d["score"] >= lo) & (d["score"] <= hi)]

    if not f["f_nsfw"]:
        d = d[~d["over_18"]]
    if not f["f_locked"]:
        d = d[~d["locked"]]
    if not f["f_mod"]:
        d = d[~d["is_mod_post"]]
    if not f["f_stick"]:
        d = d[~d["stickied"]]
    if not f["f_spoil"]:
        d = d[~d["spoiler"]]
    if not f["f_edited"]:
        d = d[~d["was_edited"]]

    if f["post_type"] != "All":
        d = d[d["post_type"] == f["post_type"]]

    if f["min_comments"] > 0:
        d = d[d["total_comments_actual"] >= f["min_comments"]]

    return d


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_overview(th: pd.DataFrame, cm: pd.DataFrame, f: dict) -> None:
    st.header("Overview")

    threads = len(th)
    comments = len(cm)
    total_actual = int(th["total_comments_actual"].sum())
    authors = th["author"].nunique()

    c = st.columns(5)
    c[0].metric("Threads", fmt_int(threads))
    c[1].metric("Comments", fmt_int(comments))
    c[2].metric("Contributors", fmt_int(authors))
    c[3].metric("Total score", fmt_compact(th["score"].sum()))
    c[4].metric("Avg score", f"{th['score'].mean():.1f}" if threads else "-")

    st.caption(
        f"Comments shown are those belonging to the {fmt_int(threads)} filtered threads. "
        f"Actual comment total across them: **{fmt_int(total_actual)}**."
    )

    if th.empty:
        st.plotly_chart(empty_fig(), use_container_width=True, key="ov_empty")
        return

    # ---- activity over time ----
    st.subheader("Activity over time (WIB)")
    granularity = st.radio(
        "Bucket", ["Daily", "Weekly", "Monthly"], horizontal=True, key="ov_gran"
    )
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]

    ts = (
        th.set_index("created_wib")
        .resample(freq)
        .agg(threads=("thread_id", "count"), score=("score", "sum"))
        .reset_index()
    )
    ts = ts[ts["threads"] > 0]

    fig = px.bar(ts, x="created_wib", y="threads", labels={"threads": "Threads", "created_wib": ""})
    fig.update_traces(marker_color=EINK["ink"])
    st.plotly_chart(style(fig, 330), use_container_width=True, key="ov_ts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top flairs")
        fl = (
            th.groupby("link_flair_text")
            .agg(threads=("thread_id", "count"), avg_score=("score", "mean"))
            .sort_values("threads", ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            fl.sort_values("threads"), x="threads", y="link_flair_text", orientation="h",
            labels={"threads": "Threads", "link_flair_text": ""},
            hover_data={"avg_score": ":.1f"},
        )
        fig.update_traces(marker_color="#7C7C7C")
        st.plotly_chart(style(fig, 430), use_container_width=True, key="ov_flair")

    with col2:
        st.subheader("Score vs. comments")
        samp = th.sample(min(len(th), 3000), random_state=0) if len(th) > 3000 else th
        fig = px.scatter(
            samp, x="score", y="total_comments_actual", color="link_flair_text",
            opacity=0.6, labels={"score": "Thread score", "total_comments_actual": "Comments (actual)",
                                 "link_flair_text": "Flair"},
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(style(fig, 430), use_container_width=True, key="ov_scatter")

    st.subheader("Distribution of thread scores")
    pos = th[th["score"] >= 0]["score"]
    fig = px.histogram(pos, nbins=60, labels={"value": "Score", "count": "Threads"})
    fig.update_traces(marker_color="#545454")
    fig.update_layout(showlegend=False, yaxis_title="Threads", xaxis_title="Score")
    st.plotly_chart(style(fig, 300), use_container_width=True, key="ov_hist")


def page_threads(th: pd.DataFrame) -> None:
    st.header("Threads")
    if th.empty:
        st.info("No threads match the current filters.")
        return

    c = st.columns(4)
    c[0].metric("Threads", fmt_int(len(th)))
    c[1].metric("Median score", f"{th['score'].median():.0f}")
    c[2].metric("Median comments", f"{th['total_comments_actual'].median():.0f}")
    c[3].metric("Text / Link", f"{int((th['post_type'] == 'Text').sum())} / {int((th['post_type'] == 'Link').sum())}")

    tab1, tab2, tab3 = st.tabs(["Top threads", "Browse / search", "Flair detail"])

    with tab1:
        metric = st.selectbox(
            "Rank by", ["score", "total_comments_actual", "upvote_ratio", "average_comment_score"],
            format_func=lambda x: {
                "score": "Score", "total_comments_actual": "Comments (actual)",
                "upvote_ratio": "Upvote ratio", "average_comment_score": "Avg comment score",
            }[x],
            key="th_rank",
        )
        n = st.slider("Show top N", 10, 100, 25, step=5, key="th_n")
        top = th.nlargest(n, metric)[
            ["title", "link_flair_text", "score", "total_comments_actual",
             "upvote_ratio", "created_wib", "author"]
        ].copy()
        top["created_wib"] = top["created_wib"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            top.rename(columns={
                "title": "Title", "link_flair_text": "Flair", "score": "Score",
                "total_comments_actual": "Comments", "upvote_ratio": "Ratio",
                "created_wib": "Created (WIB)", "author": "Author",
            }),
            use_container_width=True, hide_index=True, height=460,
        )

    with tab2:
        st.caption("Search within the currently filtered thread set.")
        q = st.text_input("Search title or body text", "", key="th_q")
        view = th
        if q:
            mask = (
                th["title"].str.contains(q, case=False, na=False, regex=False)
                | th["selftext"].str.contains(q, case=False, na=False, regex=False)
            )
            view = th[mask]
            st.caption(f"{fmt_int(len(view))} matching threads")
        show = view.sort_values("created_wib", ascending=False)[
            ["created_wib", "title", "link_flair_text", "score", "total_comments_actual",
             "upvote_ratio", "was_edited", "locked"]
        ].copy()
        show["created_wib"] = show["created_wib"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            show.rename(columns={
                "created_wib": "Created (WIB)", "title": "Title", "link_flair_text": "Flair",
                "score": "Score", "total_comments_actual": "Comments",
                "upvote_ratio": "Ratio", "was_edited": "Edited", "locked": "Locked",
            }),
            use_container_width=True, hide_index=True, height=460,
        )
        st.download_button(
            "⬇ Download filtered threads (CSV)",
            view.drop(columns=[c for c in ["selftext"] if c in view]).to_csv(index=False).encode("utf-8"),
            file_name="filtered_threads.csv", mime="text/csv",
        )

    with tab3:
        st.subheader("Flair comparison")
        g = (
            th.groupby("link_flair_text")
            .agg(
                threads=("thread_id", "count"),
                avg_score=("score", "mean"),
                med_score=("score", "median"),
                avg_comments=("total_comments_actual", "mean"),
                avg_ratio=("upvote_ratio", "mean"),
                total_score=("score", "sum"),
            )
            .reset_index()
            .sort_values("threads", ascending=False)
        )
        g = g[g["threads"] >= 3]

        m = st.selectbox(
            "Metric", ["avg_score", "avg_comments", "avg_ratio", "total_score", "threads"],
            format_func=lambda x: {
                "avg_score": "Average score", "avg_comments": "Average comments",
                "avg_ratio": "Average upvote ratio", "total_score": "Total score",
                "threads": "Thread count",
            }[x], key="th_flair_metric",
        )
        fig = px.bar(
            g.sort_values(m), x=m, y="link_flair_text", orientation="h",
            labels={m: m.replace("_", " ").title(), "link_flair_text": ""},
            hover_data={"threads": True, "avg_score": ":.1f", "avg_comments": ":.1f"},
        )
        fig.update_traces(marker_color="#A4A4A4")
        st.plotly_chart(style(fig, max(340, 26 * len(g))), use_container_width=True, key="th_flair_bar")

        disp = g.copy()
        for c_ in ["avg_score", "med_score", "avg_comments", "avg_ratio", "total_score"]:
            disp[c_] = disp[c_].round(2)
        st.dataframe(
            disp.rename(columns={
                "link_flair_text": "Flair", "threads": "Threads", "avg_score": "Avg score",
                "med_score": "Median score", "avg_comments": "Avg comments",
                "avg_ratio": "Avg ratio", "total_score": "Total score",
            }),
            use_container_width=True, hide_index=True,
        )


def page_comments(th: pd.DataFrame, cm: pd.DataFrame, cm_text: pd.DataFrame, f: dict) -> None:
    st.header("Comments")

    if cm.empty:
        st.info("No comments for the current thread filters.")
        return

    c = st.columns(4)
    c[0].metric("Comments", fmt_int(len(cm)))
    c[1].metric("Commenters", fmt_int(cm["author"].nunique()))
    c[2].metric("Median score", f"{cm['score'].median():.0f}")
    c[3].metric("Avg length", f"{cm['body_len'].mean():.0f} chars")

    st.caption("Comment scores reach back to the raw comments file (728K rows), "
               "so these stats recompute live with your filters.")

    tab1, tab2, tab3 = st.tabs(["Activity", "Top & bottom comments", "Text length"])

    with tab1:
        st.subheader("Comment volume over time (WIB)")
        freq = st.radio("Bucket", ["Daily", "Weekly", "Monthly"], horizontal=True, key="cm_gran")
        fr = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
        ts = cm.set_index("created_wib").resample(fr).agg(
            comments=("comment_id", "count"), avg_score=("score", "mean")
        ).reset_index()
        ts = ts[ts["comments"] > 0]
        fig = px.bar(ts, x="created_wib", y="comments", labels={"comments": "Comments", "created_wib": ""})
        fig.update_traces(marker_color=EINK["ink"])
        st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_ts")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Score distribution")
            lo, hi = cm["score"].quantile([0.01, 0.99])
            clipped = cm["score"].clip(lo, hi)
            fig = px.histogram(clipped, nbins=60, labels={"value": "Comment score", "count": "Comments"})
            fig.update_traces(marker_color="#545454")
            fig.update_layout(showlegend=False, yaxis_title="Comments", xaxis_title="Score")
            st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_hist")
        with col2:
            st.subheader("Comments per thread")
            per = th["total_comments_actual"]
            per = per[per > 0]
            fig = px.histogram(per, nbins=50, labels={"value": "Comments per thread", "count": "Threads"})
            fig.update_traces(marker_color="#7C7C7C")
            fig.update_layout(showlegend=False, yaxis_title="Threads", xaxis_title="Comments per thread")
            st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_per")

    with tab2:
        st.subheader("Top comment threads")
        st.caption("Comments are matched to your filtered threads by `thread_id`.")
        tmap = th.set_index("thread_id")["title"].to_dict()
        work = cm.copy()
        if len(work) > 200_000:
            work = work.sample(200_000, random_state=0)
            st.caption("Sampling 200K comments for responsiveness on this view.")

        joinable = cm_text if not cm_text.empty else pd.DataFrame(columns=["comment_id", "body"])
        merged = work.merge(joinable, on="comment_id", how="left")
        merged["thread_title"] = merged["thread_id"].map(tmap)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Highest-scored**")
            up = merged.nlargest(15, "score")[["score", "body", "thread_title"]]
            up["body"] = up["body"].fillna("").astype(str).str.slice(0, 160)
            st.dataframe(
                up.rename(columns={"score": "Score", "body": "Comment", "thread_title": "Thread"}),
                use_container_width=True, hide_index=True, height=380,
            )
        with col2:
            st.markdown("**Lowest-scored**")
            dn = merged.nsmallest(15, "score")[["score", "body", "thread_title"]]
            dn["body"] = dn["body"].fillna("").astype(str).str.slice(0, 160)
            st.dataframe(
                dn.rename(columns={"score": "Score", "body": "Comment", "thread_title": "Thread"}),
                use_container_width=True, hide_index=True, height=380,
            )

    with tab3:
        st.subheader("Comment length distribution")
        lw = pd.DataFrame({
            "chars": cm["body_len"], "words": cm["word_count"],
        })
        fig = px.histogram(lw, x="words", nbins=60, labels={"words": "Words per comment", "count": "Comments"})
        fig.update_traces(marker_color="#A4A4A4")
        fig.update_layout(showlegend=False, yaxis_title="Comments")
        st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_len")

        c1, c2 = st.columns(2)
        c1.metric("Median words", f"{cm['word_count'].median():.0f}")
        c2.metric("Median characters", f"{cm['body_len'].median():.0f}")

        st.markdown("**Longest comments in the filtered set**")
        long_ct = cm.nlargest(15, "word_count")[["word_count", "score", "body_len"]]
        st.dataframe(
            long_ct.rename(columns={"word_count": "Words", "score": "Score", "body_len": "Chars"}),
            use_container_width=True, hide_index=True,
        )


def page_authors(th: pd.DataFrame, cm: pd.DataFrame) -> None:
    st.header("Authors")

    if th.empty and cm.empty:
        st.info("No data for the current filters.")
        return

    st.caption("Leaderboard file is a static snapshot and ignores filters. "
               "The live table below is recomputed from your selections.")

    live = th.groupby("author").agg(
        threads=("thread_id", "count"),
        thread_score=("score", "sum"),
        avg_thread_score=("score", "mean"),
        comments_received=("total_comments_actual", "sum"),
    )
    cm_agg = cm.groupby("author").agg(
        comments=("comment_id", "count"), comment_score=("score", "sum")
    ) if not cm.empty else pd.DataFrame(columns=["comments", "comment_score"])

    combined = live.join(cm_agg, how="outer").fillna(0)
    combined["total_engagement"] = combined["thread_score"] + combined["comment_score"]
    combined = combined.reset_index().rename(columns={"index": "author"})

    tab1, tab2 = st.tabs(["Live ranking", "Static leaderboard file"])

    with tab1:
        metric = st.selectbox(
            "Rank authors by",
            ["total_engagement", "thread_score", "comment_score", "threads", "comments"],
            format_func=lambda x: x.replace("_", " ").title(), key="au_metric",
        )
        n = st.slider("Show top N", 5, 50, 20, key="au_n")
        top = combined.nlargest(n, metric)

        fig = px.bar(
            top.sort_values(metric), x=metric, y="author", orientation="h",
            labels={metric: metric.replace("_", " ").title(), "author": ""},
        )
        fig.update_traces(marker_color=EINK["ink"])
        fig.update_yaxes(tickmode="linear")
        st.plotly_chart(style(fig, max(340, 24 * len(top))), use_container_width=True, key="au_bar")

        # Scatter: are the loudest commenters also the most upvoted?
        fig = px.scatter(
            combined[combined[metric] > 0], x="threads", y="comments",
            size=combined[combined[metric] > 0]["total_engagement"].abs() + 1,
            labels={"threads": "Threads created", "comments": "Comments written"},
            hover_name="author",
        )
        fig.update_traces(marker=dict(color="#545454", opacity=0.75))
        st.plotly_chart(style(fig, 400), use_container_width=True, key="au_scatter")

        disp = top.copy()
        for c_ in ["thread_score", "comment_score", "total_engagement", "avg_thread_score"]:
            disp[c_] = disp[c_].round(1)
        st.dataframe(
            disp.rename(columns={
                "author": "Author", "threads": "Threads", "thread_score": "Thread score",
                "avg_thread_score": "Avg thread score", "comments_received": "Comments received",
                "comments": "Comments written", "comment_score": "Comment score",
                "total_engagement": "Total engagement",
            }),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇ Download this ranking (CSV)",
            combined.sort_values(metric, ascending=False).to_csv(index=False).encode("utf-8"),
            file_name="live_author_ranking.csv", mime="text/csv",
        )

    with tab2:
        lb = load("author_leaderboard")
        if lb.empty:
            st.warning("author_leaderboard.parquet not found in cache.")
            return
        st.info("Static snapshot from `author_leaderboard.csv` — shows the dataset's own "
                "top authors, independent of filters.")
        top = lb.nlargest(20, "total_engagement_score")
        fig = px.bar(
            top.sort_values("total_engagement_score"), x="total_engagement_score", y="author",
            orientation="h", labels={"total_engagement_score": "Total engagement", "author": ""},
        )
        fig.update_traces(marker_color="#545454")
        st.plotly_chart(style(fig, 500), use_container_width=True, key="au_static")

        cm1, cm2 = st.columns(2)
        cm1.metric("Top author threads", fmt_int(top.iloc[0]["threads_created"]))
        cm2.metric("Top author comments", fmt_int(top.iloc[0]["comments_written"]))
        st.dataframe(
            lb.head(50).rename(columns=lambda c: c.replace("_", " ").title()),
            use_container_width=True, hide_index=True,
        )


def page_flairs(th: pd.DataFrame) -> None:
    st.header("Flairs")

    if th.empty:
        st.info("No threads for the current filters.")
        return

    st.caption("Live view recomputed from your filtered threads.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Threads per flair")
        g = th["link_flair_text"].value_counts().reset_index()
        g.columns = ["flair", "threads"]
        fig = px.pie(g, names="flair", values="threads", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(style(fig, 460), use_container_width=True, key="fl_pie")

    with col2:
        st.subheader("Engagement profile by flair")
        profile = (
            th.groupby("link_flair_text")
            .agg(
                threads=("thread_id", "count"),
                avg_score=("score", "mean"),
                avg_comments=("total_comments_actual", "mean"),
                avg_ratio=("upvote_ratio", "mean"),
            )
            .reset_index()
        )
        profile = profile[profile["threads"] >= 3]
        fig = px.scatter(
            profile, x="avg_comments", y="avg_score", size="threads",
            color="avg_ratio", hover_name="link_flair_text",
            labels={"avg_comments": "Avg comments", "avg_score": "Avg score", "avg_ratio": "Avg ratio"},
            color_continuous_scale=EINK_SEQ_WARM,
        )
        st.plotly_chart(style(fig, 460), use_container_width=True, key="fl_scatter")

    st.subheader("Normalized metric heatmap")
    profile = (
        th.groupby("link_flair_text")
        .agg(
            threads=("thread_id", "count"),
            avg_score=("score", "mean"),
            avg_comments=("total_comments_actual", "mean"),
            avg_ratio=("upvote_ratio", "mean"),
            avg_comment_score=("average_comment_score", "mean"),
        )
        .reset_index()
    )
    profile = profile[profile["threads"] >= 3].set_index("link_flair_text")
    if profile.empty:
        st.info("Not enough threads per flair after filtering to build a heatmap.")
    else:
        metrics = ["threads", "avg_score", "avg_comments", "avg_ratio", "avg_comment_score"]
        norm = (profile[metrics] - profile[metrics].min()) / (
            profile[metrics].max() - profile[metrics].min()
        ).replace(0, 1)
        fig = px.imshow(
            norm.T, aspect="auto", color_continuous_scale=EINK_SEQ,
            labels=dict(x="Flair", y="Metric", color="Normalized"),
            text_auto=".2f",
        )
        fig.update_layout(height=max(320, 40 * len(metrics)))
        st.plotly_chart(style(fig, max(320, 40 * len(metrics))), use_container_width=True, key="fl_heat")

    tab1, tab2 = st.tabs(["Top words per flair", "Reference flair performance"])
    with tab1:
        fw = load("flair_top_words")
        if fw.empty:
            st.warning("flair_top_words.parquet not found in cache.")
        else:
            selected = st.multiselect(
                "Compare flairs",
                sorted(fw["flair"].unique()),
                default=sorted(fw["flair"].unique())[:3],
                key="fl_words_sel",
            )
            rows = []
            for fl in selected:
                sub = fw[fw["flair"] == fl].nlargest(10, "count")
                rows.append(sub)
            if rows:
                combo = pd.concat(rows)
                fig = px.bar(
                    combo, x="count", y="word", color="flair", orientation="h",
                    barmode="group", labels={"count": "Occurrences", "word": "", "flair": "Flair"},
                )
                st.plotly_chart(style(fig, max(400, 30 * len(selected) * 2)), use_container_width=True, key="fl_words")
            else:
                st.info("Select at least one flair.")

    with tab2:
        fp = load("flair_performance")
        if fp.empty:
            st.warning("flair_performance.parquet not found in cache.")
        else:
            st.info("Static snapshot from `flair_performance.csv`.")
            fp2 = fp.sort_values("thread_count", ascending=False)
            fig = px.bar(
                fp2.head(20).sort_values("avg_score"), x="avg_score", y="link_flair_text",
                orientation="h", labels={"avg_score": "Avg score", "link_flair_text": ""},
                hover_data={"thread_count": True, "avg_comments_actual": ":.1f"},
            )
            fig.update_traces(marker_color="#A4A4A4")
            st.plotly_chart(style(fig, 520), use_container_width=True, key="fl_ref")
            st.dataframe(
                fp2.rename(columns=lambda c: c.replace("_", " ").title()),
                use_container_width=True, hide_index=True,
            )


def page_timing(th: pd.DataFrame, cm: pd.DataFrame) -> None:
    st.header("Timing")
    st.caption("All times converted to WIB (UTC+7). Live from filtered threads & comments.")

    if th.empty:
        st.info("No threads for the current filters.")
        return

    t = th.copy()
    t["dow"] = t["created_wib"].dt.day_name()
    t["hour"] = t["created_wib"].dt.hour

    c = cm.copy()
    if not c.empty:
        c["dow"] = c["created_wib"].dt.day_name()
        c["hour"] = c["created_wib"].dt.hour

    st.subheader("Thread activity heatmap (day × hour, WIB)")
    pivot = (
        t.pivot_table(index="dow", columns="hour", values="thread_id", aggfunc="count")
        .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
    )
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale=EINK_SEQ,
        labels=dict(x="Hour (WIB)", y="", color="Threads"),
    )
    fig.update_xaxes(dtick=2)
    st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_heat_t")

    if not c.empty:
        st.subheader("Comment activity heatmap (day × hour, WIB)")
        cp = (
            c.pivot_table(index="dow", columns="hour", values="comment_id", aggfunc="count")
            .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
        )
        fig = px.imshow(
            cp, aspect="auto", color_continuous_scale=EINK_SEQ_WARM,
            labels=dict(x="Hour (WIB)", y="", color="Comments"),
        )
        fig.update_xaxes(dtick=2)
        st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_heat_c")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Threads by day of week")
        g = t["dow"].value_counts().reindex(DOW_ORDER).fillna(0).reset_index()
        g.columns = ["dow", "threads"]
        fig = px.bar(g, x="dow", y="threads", labels={"dow": "", "threads": "Threads"})
        fig.update_traces(marker_color=EINK["ink"])
        st.plotly_chart(style(fig, 330), use_container_width=True, key="tm_dow")
    with col2:
        st.subheader("Threads by hour of day")
        g = t["hour"].value_counts().reindex(range(24)).fillna(0).reset_index()
        g.columns = ["hour", "threads"]
        fig = px.bar(g, x="hour", y="threads", labels={"hour": "Hour (WIB)", "threads": "Threads"})
        fig.update_traces(marker_color="#7C7C7C")
        st.plotly_chart(style(fig, 330), use_container_width=True, key="tm_hour")

    st.subheader("Best hours for engagement")
    eng = (
        t.groupby("hour").agg(threads=("thread_id", "count"), avg_score=("score", "mean"),
                              avg_comments=("total_comments_actual", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eng["hour"], y=eng["avg_score"], mode="lines+markers",
                             name="Avg thread score", line=dict(color=EINK["accent"], width=2)))
    fig.add_trace(go.Bar(x=eng["hour"], y=eng["avg_comments"], name="Avg comments",
                         marker_color=EINK["ink"], opacity=0.45, yaxis="y2"))
    fig.update_layout(
        xaxis=dict(title="Hour (WIB)", dtick=2),
        yaxis=dict(title="Avg score"),
        yaxis2=dict(title="Avg comments", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_eng")

    st.divider()
    st.subheader("Reference: WIB hourly activity file")
    wh = load("wib_hourly_activity")
    if wh.empty:
        st.warning("wib_hourly_activity.parquet not found in cache.")
    else:
        st.info("Static snapshot from `wib_hourly_activity.csv` (full 7×24 grid).")
        grid = (
            wh.pivot_table(index="wib_day_of_week", columns="wib_hour",
                           values="threads_count", aggfunc="sum")
            .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
        )
        fig = px.imshow(grid, aspect="auto", color_continuous_scale=EINK_SEQ,
                        labels=dict(x="Hour (WIB)", y="", color="Threads"))
        fig.update_xaxes(dtick=2)
        st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_ref")


def page_sentiment(sent: pd.DataFrame, llm_sent: pd.DataFrame | None = None) -> None:
    """Sentiment analytics: full-corpus model labels, LLM reference, and calibration.

    Data provenance matters here, so the page keeps four artifacts visually and
    verbally distinct:
      1. HF model on the full corpus  -> fast, biased, high coverage.
      2. Assistant (LLM) labels, 10K  -> slow, rubric-based, the reference.
      3. HF model on the same 10K     -> an apples-to-apples agreement check.
      4. 40-comment gold benchmark    -> a small external sanity check.
    """
    st.header("Sentiment analytics")

    overview = load_json("sentiment_overview")
    hf_full = load("sentiment_full_slim") if (CACHE / "sentiment_full_slim.parquet").exists() else pd.DataFrame()

    if not overview or hf_full.empty:
        st.error(
            "Sentiment artifacts are missing. Run `python src/build_sentiment_cache.py` "
            "after the full-corpus model output is in the workspace root."
        )
        return

    full = overview.get("full_corpus", {})
    comp = overview.get("comparison_full_run_vs_llm_10k", {})
    llm10 = overview.get("llm_10k_sample", {})
    hf10 = overview.get("hf_10k_sample", {})
    calib = overview.get("calibration", {})

    # --------------------------------------------------------------- overview
    st.subheader("Coverage and headline distribution")
    c = st.columns(4)
    c[0].metric("Full corpus comments", fmt_int(full.get("rows", 0)))
    c[1].metric("LLM reference sample", fmt_int(llm10.get("rows", 0)))
    c[2].metric("Model mean confidence", f"{full.get('mean_confidence', 0):.2f}")
    c[3].metric("Model labels, low conf <0.60", f"{full.get('low_confidence_lt_0_6', 0):.1%}")

    obs = full.get("distribution_pct", {})
    obs_df = pd.DataFrame(
        {"label": SENTIMENT_ORDER,
         "share": [obs.get(l, 0) for l in SENTIMENT_ORDER]}
    )
    corr = calib.get("corrected", {})
    ref = llm10.get("distribution", {})
    ref_total = sum(ref.values()) or 1
    cmp_df = pd.DataFrame({
        "label": SENTIMENT_ORDER,
        "Model (raw, 678K)": [obs.get(l, 0) for l in SENTIMENT_ORDER],
        "LLM reference (10K)": [ref.get(l, 0) / ref_total for l in SENTIMENT_ORDER],
        "Model (bias-corrected)": [corr.get(l, 0) for l in SENTIMENT_ORDER],
    }).melt(id_vars="label", var_name="series", value_name="share")

    fig = px.bar(
        cmp_df, x="label", y="share", color="series", barmode="group",
        color_discrete_sequence=[EINK["accent"], EINK["ink"], "#9A9A9A"],
        labels={"label": "", "share": "Share of comments", "series": ""},
        category_orders={"label": SENTIMENT_ORDER},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 360), use_container_width=True, key="se_dist_compare")
    st.caption(
        "The raw model calls two-thirds of comments negative and nearly eliminates "
        "neutral. The bias-corrected estimate — derived from the 10K confusion matrix — "
        "lands close to the independent LLM reference, which is strong evidence the "
        "raw full-corpus distribution is an artifact, not a finding."
    )

    # ------------------------------------------------------------- calibration
    st.subheader("Calibration: correcting the model's negative skew")
    k = st.columns(3)
    k[0].metric("Agreement vs. LLM reference", f"{comp.get('exact_agreement', 0):.1%}")
    k[1].metric("Cohen's kappa", f"{comp.get('cohen_kappa', 0):.3f}")
    k[2].metric("Disagreements", fmt_int(comp.get("disagreements", 0)))

    st.markdown(
        f"**Model transition behavior** (rows = LLM reference, columns = model): "
        f"when the rubric says *neutral*, the model keeps neutral only "
        f"**{comp.get('confusion_matrix', {}).get('neutral', {}).get('neutral', 0) / max(1, sum(comp.get('confusion_matrix', {}).get('neutral', {}).values())):.1%}** "
        f"of the time and otherwise relabels it negative. This is the mechanism behind "
        f"the inflated negative share."
    )

    cm = comp.get("confusion_matrix", {})
    if cm:
        cm_df = pd.DataFrame(
            [[cm.get(r, {}).get(col, 0) for col in SENTIMENT_ORDER] for r in SENTIMENT_ORDER],
            index=[f"LLM {r}" for r in SENTIMENT_ORDER],
            columns=[f"Model {col}" for col in SENTIMENT_ORDER],
        )
        fig = px.imshow(
            cm_df, text_auto=True, aspect="auto",
            color_continuous_scale=EINK_SEQ,
            labels={"x": "Model label", "y": "LLM reference label", "color": "Comments"},
        )
        st.plotly_chart(style(fig, 340), use_container_width=True, key="se_cm")

    pl = pd.DataFrame(comp.get("per_label_vs_reference", []))
    if not pl.empty:
        pl = pl.rename(columns={
            "label": "Label", "support": "LLM support",
            "precision": "Precision vs. LLM", "recall": "Recall vs. LLM", "f1": "F1",
        })
        st.dataframe(pl, use_container_width=True, hide_index=True)

    ac = pd.DataFrame(
        [{"Confidence": k2, **v} for k2, v in comp.get("agreement_by_confidence", {}).items()]
    )
    if not ac.empty:
        ac = ac.rename(columns={"n": "Comments", "agreement": "Agreement vs. LLM"})
        ac["Agreement vs. LLM"] = (ac["Agreement vs. LLM"] * 100).round(1).astype(str) + "%"
        st.caption("Agreement barely improves with confidence — a sign the model is confidently wrong.")
        st.dataframe(ac, use_container_width=True, hide_index=True)

    st.info(
        "Interpretation: the LLM reference is a 10K rubric sample, not absolute truth, "
        "so these figures measure agreement between two labelers, not the model's true "
        "accuracy. Use the bias-corrected shares for any claim about the corpus."
    )

    # ------------------------------------------------------- full-corpus slices
    st.divider()
    st.subheader("Full-corpus sentiment by flair and time")
    st.caption(
        "Model labels across all 678,522 comments, broken down by thread flair and month. "
        "Slices inherit the model's negative skew; read shapes and relative differences, "
        "not absolute levels."
    )

    view_mode = st.radio(
        "Distribution mode", ["Raw model labels", "Bias-corrected"], horizontal=True,
        key="se_view_mode",
    )

    bf = load("sentiment_by_flair") if (CACHE / "sentiment_by_flair.parquet").exists() else pd.DataFrame()
    if not bf.empty:
        piv = bf.pivot_table(index="flair", columns="label", values="n", aggfunc="sum").fillna(0)
        piv = piv.reindex(columns=SENTIMENT_ORDER, fill_value=0)
        piv["total"] = piv.sum(axis=1)
        piv = piv.sort_values("total", ascending=False)
        shares = piv[SENTIMENT_ORDER].div(piv["total"], axis=0)

        if view_mode == "Bias-corrected":
            tm = pd.DataFrame(
                calib.get("transition_matrix_p_model_given_reference", {})
            ).reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER).fillna(0)
            tm = tm.div(tm.sum(axis=0).replace(0, 1), axis=1)  # normalize per model label
            # Solve shares_true @ P = shares_model  ->  shares_true = shares_model @ pinv(P)
            pinv = np.linalg.pinv(tm.to_numpy(dtype=float))
            corrected = np.clip(shares.to_numpy(dtype=float) @ pinv, 0, None)
            row = corrected.sum(axis=1, keepdims=True)
            corrected = np.divide(corrected, row, out=np.zeros_like(corrected), where=row > 0)
            shares = pd.DataFrame(corrected, index=shares.index, columns=SENTIMENT_ORDER)
            st.caption("Bias-corrected shares can be noisy for small flairs; treat low-volume flairs cautiously.")

        top = shares.head(15)
        long = top.reset_index().melt(
            id_vars="flair", var_name="label", value_name="share",
            value_vars=SENTIMENT_ORDER,
        )
        fig = px.bar(
            long, x="share", y="flair", color="label", orientation="h", barmode="stack",
            color_discrete_map=SENTIMENT_COLORS,
            labels={"share": "Share of comments", "flair": "", "label": ""},
            category_orders={"label": SENTIMENT_ORDER},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 460), use_container_width=True, key="se_flair")

    bfm = load("sentiment_by_flair_month") if (CACHE / "sentiment_by_flair_month.parquet").exists() else pd.DataFrame()
    if not bfm.empty:
        flairs = sorted(bfm["flair"].unique())
        picked = st.multiselect(
            "Flairs to compare over time", flairs,
            default=flairs[: min(3, len(flairs))], key="se_flair_month_chart",
        )
        sub = bfm[bfm["flair"].isin(picked)] if picked else bfm
        if not sub.empty:
            agg = sub.groupby(["month", "label"], observed=True)["n"].sum().reset_index()
            tot = agg.groupby("month")["n"].transform("sum")
            agg["share"] = agg["n"] / tot
            fig = px.line(
                agg, x="month", y="share", color="label", markers=True,
                color_discrete_map=SENTIMENT_COLORS,
                labels={"month": "", "share": "Share of comments", "label": ""},
                category_orders={"label": SENTIMENT_ORDER},
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 340), use_container_width=True, key="se_flair_month")

    # ------------------------------------------------------ disagreement explorer
    st.divider()
    st.subheader("Where the model and the rubric disagree")
    cmp_path = CACHE / "sentiment_compare_10k.parquet"
    if cmp_path.exists():
        cmp_df = load("sentiment_compare_10k")
        cc = st.columns(3)
        cc[0].metric("Compared comments", fmt_int(len(cmp_df)))
        cc[1].metric("Agree", fmt_int(int(cmp_df["agree"].sum())))
        cc[2].metric("Disagree", fmt_int(int((~cmp_df["agree"]).sum())))

        only_dis = st.checkbox("Show disagreements only", value=True, key="se_only_dis")
        view = cmp_df[~cmp_df["agree"]] if only_dis else cmp_df

        fl = st.multiselect(
            "Filter by flair", sorted(view["flair"].dropna().unique()),
            key="se_dis_flair",
        )
        if fl:
            view = view[view["flair"].isin(fl)]

        q = st.text_input("Search comment text", "", key="se_dis_q")
        if q:
            view = view[view["body"].fillna("").str.contains(q, case=False, regex=False)]

        show_cols = [c2 for c2 in ["body", "llm_label", "hf_label", "hf_confidence", "flair", "route"] if c2 in view]
        view = view.sort_values("hf_confidence", ascending=False)
        st.dataframe(view[show_cols].head(400), use_container_width=True, hide_index=True, height=380)
        st.caption(
            "High-confidence disagreements at the top are the most informative: the model "
            "is sure and the rubric disagrees. These rows are a practical audit list."
        )
    else:
        st.info("Run `python src/build_sentiment_cache.py` to enable the disagreement explorer.")

    # ------------------------------------------------------------- gold 40 check
    st.divider()
    st.subheader("Small external check: 40-comment gold benchmark")
    bench_path = CACHE / "sentiment_router_gold40.benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        b = st.columns(4)
        b[0].metric("Gold rows matched", fmt_int(bench.get("matched_rows", 0)))
        b[1].metric("Agreement", f"{bench.get('accuracy', 0):.1%}")
        b[2].metric("Mean confidence", f"{bench.get('mean_confidence', 0):.2f}")
        hc = bench.get("accuracy_high_confidence_ge_0_8")
        b[3].metric("Agreement, conf ≥0.8", f"{hc:.1%}" if hc is not None else "-")
        st.caption(
            "Only 40 rows and drawn from the same LLM rubric, so it is a sanity check, "
            "not independent validation. It agrees with the 10K finding: the model is "
            "confident and skewed negative."
        )

    # ------------------------------------------------------------------- samples
    st.divider()
    st.subheader("Inspect labeled comments")
    which = st.radio(
        "Dataset", ["LLM reference (10K)", "Model (10K sample)"], horizontal=True,
        key="se_sample_which",
    )
    if which.startswith("LLM") and llm_sent is not None and not llm_sent.empty:
        sdf = llm_sent.copy()
        lcol = "label" if "label" in sdf.columns else "llm_label"
        q = st.text_input("Search LLM-labeled text", "", key="se_llm_q2")
        if q:
            sdf = sdf[sdf["body"].fillna("").str.contains(q, case=False, regex=False)]
        cols = [c2 for c2 in ["body", lcol, "score", "created_wib", "comment_id"] if c2 in sdf]
        st.dataframe(sdf[cols].head(400), use_container_width=True, hide_index=True, height=340)
    else:
        sdf = load("sentiment_hf_10k") if (CACHE / "sentiment_hf_10k.parquet").exists() else pd.DataFrame()
        if sdf.empty:
            st.info("No 10K model sample available.")
        else:
            q = st.text_input("Search model-labeled text", "", key="se_hf_q")
            if q:
                sdf = sdf[sdf["body"].fillna("").str.contains(q, case=False, regex=False)]
            cols = [c2 for c2 in ["body", "label", "confidence", "route", "comment_id"] if c2 in sdf]
            st.dataframe(sdf[cols].head(400), use_container_width=True, hide_index=True, height=340)


def main() -> None:
    if not cache_ready():
        st.title("📊 r/indonesia Analytics")
        st.error("Cache not found. Build it once from the raw CSVs (takes ~1–2 minutes).")
        st.code("python src/build_cache.py", language="bash")
        st.caption(f"Expected location: `{CACHE}`")
        if st.button("🔨 Build cache now", type="primary"):
            with st.spinner("Building cache… this can take a couple of minutes."):
                import subprocess  # noqa: PLC0415
                import sys  # noqa: PLC0415

                res = subprocess.run(
                    [sys.executable, str(BASE / "src" / "build_cache.py"), "--force"],
                    capture_output=True, text=True, cwd=str(BASE),
                )
            if res.returncode == 0:
                st.success("Cache built. Reloading…")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Build failed.")
                st.code(res.stderr[-3000:] or res.stdout[-3000:])
        return

    th_all = load("threads")
    cm_all = load("comments")
    # Comment text is loaded lazily; aggregate views never need the large body column.
    cm_text = load("comments_text") if (CACHE / "comments_text.parquet").exists() else pd.DataFrame()
    sent = load("sentiment")
    llm_sent = load("comment_sentiment_llm_10k") if (CACHE / "comment_sentiment_llm_10k.parquet").exists() else pd.DataFrame()
    meta = load_meta()

    st.sidebar.title("📊 r/indonesia")
    st.sidebar.caption("Indonesian subreddit dataset")

    f = sidebar_filters(th_all, cm_all)

    th = apply_filters(th_all, f)
    # Comments are kept to the threads that survived filtering (join on thread_id).
    keep_ids = set(th["thread_id"])
    cm = cm_all[cm_all["thread_id"].isin(keep_ids)]

    st.sidebar.divider()
    st.sidebar.metric("Threads selected", fmt_int(len(th)))
    st.sidebar.metric("Comments selected", fmt_int(len(cm)))
    st.sidebar.caption(
        f"{len(th) / max(len(th_all), 1):.1%} of threads • "
        f"{len(cm) / max(len(cm_all), 1):.1%} of comments"
    )
    if st.sidebar.button("🔄 Clear cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    pages = {
        "📈 Overview": lambda: page_overview(th, cm, f),
        "🧵 Threads": lambda: page_threads(th),
        "💬 Comments": lambda: page_comments(th, cm, cm_text, f),
        "👥 Authors": lambda: page_authors(th, cm),
        "🏷️ Flairs": lambda: page_flairs(th),
        "🕐 Timing": lambda: page_timing(th, cm),
        "🎭 Sentiment": lambda: page_sentiment(sent, llm_sent),
    }
    tabs = st.tabs(list(pages.keys()))
    for tab, render in zip(tabs, pages.values()):
        with tab:
            render()

    st.divider()
    if meta:
        s = meta.get("sources", {})
        st.caption(
            f"Cache built {meta.get('built_at', '?')} • "
            f"threads: {fmt_int(s.get('threads', {}).get('rows', 0))} • "
            f"comments: {fmt_int(s.get('comments', {}).get('rows', 0))} • "
            f"Built by `src/build_cache.py` — re-run it after the CSVs change."
        )


if __name__ == "__main__":
    main()
