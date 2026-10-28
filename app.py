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

import gc
import html
import json
import os
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "cache"

# The stylesheet targets Streamlit 1.40's DOM (BaseWeb tabs, selects, tags).
# Later releases swapped those widgets for react-aria, which silently drops
# most of the theme — so the pin is asserted here and in requirements.txt.
STREAMLIT_PIN = "1.40.1"

st.set_page_config(
    page_title="r/indonesia Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NO_FLAIR = "No Flair"
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]

# Auto-posted / scheduled threads. Automod publishes these on a fixed clock, so
# they behave nothing like organic posts: ~2% of threads but ~22% of all comments,
# and every one of the top comment threads lands on the same hour (07:00 WIB).
# Any "when should I post?" reading has to be able to exclude them.
AUTO_POST_FLAIRS = {"Daily Chat Thread", "Weekend Chat Thread", "Special Thread"}

# Quick-range presets. Ids stay stable across languages (only the label is
# translated) so switching language never resets the selection, and so a URL
# written in one language restores in the other.
RANGE_PRESETS = {
    "all": "sb.all_time", "d7": "sb.last7", "d30": "sb.last30",
    "d90": "sb.last90", "custom": "sb.custom",
}
# Every widget key the filter sidebar owns. Kept in one place because the URL
# restore and the reset button both have to clear exactly this set.
FILTER_KEYS = (
    "flt_preset", "flt_drange", "flt_flairs", "flt_author", "flt_title",
    "flt_score", "flt_nsfw", "flt_locked", "flt_mod", "flt_stick",
    "flt_spoil", "flt_edited", "flt_auto", "flt_ptype", "flt_minc",
)

# --------------------------------------------------------------------------- #
# Theme — "e-ink" grayscale palette (github.com/e-ink-colorscheme/e-ink.nvim)
# with its Everforest accent, styled as brutalist / engineering chrome:
# monospace, hard 2px rules, offset shadows, zero border-radius.
#
# Two inks share the same *roles*, not the same values: "light" is paper with
# dark ink, "dark" is a dark board with light ink. Hues (accent red, the flair
# palette) stay put so a flair keeps its identity across a theme switch.
# `EINK` is mutated in place by `apply_theme()` because every chart, page and
# CSS block reads it at call time — no reload, no restart.
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict[str, str]] = {
    "light": {
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
        # de-emphasised greys used by charts for "everything else" series
        "dim1": "#545454",
        "dim2": "#7C7C7C",
        "dim3": "#A4A4A4",
        "hover": "#F3F3F0",       # tab hover wash
        "blueprint": "rgba(26,26,26,.05)",   # page grid
        "on_accent": "#FFFFFF",   # text on the accent fill
    },
    "dark": {
        "paper": "#141414",
        "panel": "#1F1F1F",
        "grid": "#3A3A3A",
        "ink": "#D6D6D1",
        "ink_strong": "#F2F2EE",
        "muted": "#8E8E8E",
        "accent": "#F85552",
        "green": "#A8BF3E",
        "blue": "#5FB0DE",
        "yellow": "#E8B54A",
        "dim1": "#A6A6A6",
        "dim2": "#7E7E7E",
        "dim3": "#555555",
        "hover": "#262626",
        "blueprint": "rgba(255,255,255,.045)",
        "on_accent": "#FFFFFF",
    },
}
EINK: dict[str, str] = dict(THEMES["light"])

# Categorical colours for *topics* (flairs). The chrome stays e-ink, but flairs
# need real hue separation to stay readable, so this is deliberately wider than
# the monochrome ramp: muted enough to sit on paper, distinct enough to tell apart.
EINK_TOPICS = [
    "#1F6FB2",  # blue
    "#D1495B",  # red
    "#2E8B57",  # green
    "#E0A02E",  # amber
    "#7B4FBF",  # violet
    "#148F8F",  # teal
    "#E4572E",  # orange
    "#6A8E23",  # olive
    "#C23B72",  # magenta
    "#3C5A99",  # navy
    "#8E6E3A",  # tan
    "#35A7A7",  # aqua
    "#9B5DE5",  # purple
    "#A83232",  # dark red
    "#4C8C3F",  # forest
    "#D9A441",  # gold
    "#4E8CC7",  # cornflower
    "#8FA31E",  # chartreuse
    "#8C4A6B",  # plum
    "#2C7873",  # pine
]
# Sequential ramps. Single-series charts stay ink; only *magnitude* encodings
# (heatmaps, continuous colour) get hue, so colour always carries information.
# Each ramp starts at paper and is monotonic in lightness, so it still reads as
# e-ink and stays legible if printed in greyscale.
SEQ_LIGHT = {
    "neutral": [[0.0, "#FBFBF9"], [0.35, "#CCCCCC"], [0.7, "#5E5E5E"], [1.0, "#0D0D0D"]],
    "blue": [[0.0, "#FBFBF9"], [0.2, "#E4EDF5"], [0.4, "#C2D6E8"],
             [0.6, "#8FB3D4"], [0.8, "#4A7FB0"], [1.0, "#17324F"]],       # threads
    "ember": [[0.0, "#FBFBF9"], [0.2, "#F8E2D6"], [0.4, "#F2C3A6"],
              [0.6, "#E59468"], [0.8, "#D06A3C"], [1.0, "#7A2B1E"]],      # comments
}
# On a dark board "more" has to mean *brighter*, so the ramps run the other way:
# they start at the dark paper and climb to light.
SEQ_DARK = {
    "neutral": [[0.0, "#141414"], [0.35, "#3E3E3E"], [0.7, "#9C9C9C"], [1.0, "#F2F2EE"]],
    "blue": [[0.0, "#141414"], [0.2, "#132433"], [0.4, "#1E4059"],
             [0.6, "#3A6E92"], [0.8, "#7FA9C8"], [1.0, "#DCE9F3"]],
    "ember": [[0.0, "#141414"], [0.2, "#2E1610"], [0.4, "#5A2A1A"],
              [0.6, "#9C4E2E"], [0.8, "#D08A62"], [1.0, "#F5DFD2"]],
}
EINK_SEQ = [list(r) for r in SEQ_LIGHT["neutral"]]        # neutral / fallback
EINK_SEQ_BLUE = [list(r) for r in SEQ_LIGHT["blue"]]      # threads / neutral magnitude
EINK_SEQ_EMBER = [list(r) for r in SEQ_LIGHT["ember"]]    # comments
EINK_MONO = ('"IBM Plex Mono", ui-monospace, "Cascadia Mono", "JetBrains Mono", '
             'Consolas, "Courier New", monospace')

SENTIMENT_COLORS = {
    "positive": EINK["green"],
    "neutral": EINK["muted"],
    "negative": EINK["accent"],
}

# Default theme. Override with DASHBOARD_THEME=light|dark — handy in Docker,
# where it can change without rebuilding the image.
DEFAULT_THEME = os.environ.get("DASHBOARD_THEME", "light")
if DEFAULT_THEME not in THEMES:
    DEFAULT_THEME = "light"


def theme() -> str:
    """Active theme id: sidebar toggle, then URL, then environment default."""
    for key in ("theme_select", "theme"):
        value = st.session_state.get(key)
        if value in THEMES:
            return value
    return DEFAULT_THEME


def apply_theme(name: str) -> None:
    """Swap the live palette in place so every existing reference follows."""
    palette = THEMES.get(name, THEMES["light"])
    EINK.clear()
    EINK.update(palette)
    ramps = SEQ_DARK if name == "dark" else SEQ_LIGHT
    for target, ramp in ((EINK_SEQ, "neutral"), (EINK_SEQ_BLUE, "blue"),
                         (EINK_SEQ_EMBER, "ember")):
        target[:] = [list(step) for step in ramps[ramp]]
    SENTIMENT_COLORS.update(
        positive=EINK["green"], neutral=EINK["muted"], negative=EINK["accent"],
    )

# --------------------------------------------------------------------------- #
# i18n — the interface is bilingual (Indonesian / English).
#
# Only UI *chrome* is translated: headings, captions, metric labels, axis
# titles, column headers, buttons. The data itself (flair names, sentiment
# labels, author UUIDs, post titles) is never touched.
#
# Keys are dotted ids. A missing key renders as the literal id, which makes
# gaps obvious in the UI instead of silently falling back to the wrong language.
# --------------------------------------------------------------------------- #
LANGS = {"id": "Bahasa Indonesia", "en": "English"}

# Default UI language. Override with DASHBOARD_LANG=id|en — useful in Docker,
# where it can be changed without rebuilding the image.
DEFAULT_LANG = os.environ.get("DASHBOARD_LANG", "en")
if DEFAULT_LANG not in LANGS:
    DEFAULT_LANG = "en"

def lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str) -> str:
    """Look up a UI string: (Indonesian, English). Unknown keys pass through."""
    pair = STRINGS.get(key)
    if pair is None:
        return key
    return pair[0] if lang() == "id" else pair[1]


STRINGS: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------- navigation
    "nav.overview": ("Ikhtisar", "Overview"),
    "nav.threads": ("Thread", "Threads"),
    "nav.comments": ("Komentar", "Comments"),
    "nav.authors": ("Penulis", "Authors"),
    "nav.flairs": ("Flair", "Flairs"),
    "nav.timing": ("Waktu", "Timing"),
    "nav.sentiment": ("Sentimen", "Sentiment"),
    "nav.methodology": ("Metodologi", "Methodology"),
    # ------------------------------------------------------------- sidebar
    "sb.subtitle": ("Dataset subreddit Indonesia", "Indonesian subreddit dataset"),
    "sb.filters": ("Filter", "Filters"),
    "sb.filters_help": ("Setiap perubahan menyaring ulang data dan menyegarkan semua grafik.",
                        "Every change re-filters the data and refreshes all charts."),
    "sb.time_range": ("Rentang waktu", "Time range"),
    "sb.spans": ("Data mencakup {span}", "Data spans {span}"),
    "sb.quick_range": ("Rentang cepat", "Quick range"),
    "sb.all_time": ("Sepanjang waktu", "All time"),
    "sb.last7": ("7 hari terakhir", "Last 7 days"),
    "sb.last30": ("30 hari terakhir", "Last 30 days"),
    "sb.last90": ("90 hari terakhir", "Last 90 days"),
    "sb.custom": ("Khusus…", "Custom…"),
    "sb.custom_range": ("Rentang khusus", "Custom range"),
    "sb.flair": ("Flair", "Flair"),
    "sb.topics": ("Topik", "Topics"),
    "sb.topics_help": ("Semua flair dipilih secara bawaan kecuali yang sangat langka (<20 thread).",
                       "All flairs are selected by default except very rare ones (<20 threads)."),
    "sb.search": ("Pencarian", "Search"),
    "sb.search_help": ("Cari berdasarkan UUID penulis atau judul thread.",
                       "Filter by author UUID or thread title."),
    "sb.theme": ("Tampilan", "Theme"),
    "theme.light": ("☀ Terang", "☀ Light"),
    "theme.dark": ("☾ Gelap", "☾ Dark"),
    "sb.reset": ("↺ Atur ulang filter", "↺ Reset filters"),
    "sb.url_hint": ("Tampilan ini tersimpan di URL — salin bilah alamat untuk membagikannya.",
                    "This view lives in the URL — copy the address bar to share it."),
    "sb.author_q": ("UUID penulis mengandung", "Author UUID contains"),
    "sb.title_q": ("Judul mengandung", "Title contains"),
    "sb.thread_score": ("Skor thread", "Thread score"),
    "sb.flags": ("Penanda", "Flags"),
    "sb.flags_help": ("Sertakan thread dengan penanda ini:", "Include threads that are:"),
    "sb.inc_nsfw": ("NSFW", "NSFW"),
    "sb.inc_locked": ("Terkunci", "Locked"),
    "sb.inc_mod": ("Kiriman mod", "Mod posts"),
    "sb.inc_stick": ("Tersemat", "Stickied"),
    "sb.inc_spoil": ("Spoiler", "Spoilers"),
    "sb.inc_edited": ("Diedit", "Edited"),
    "sb.content": ("Konten", "Content"),
    "sb.exclude_auto": ("Kecualikan thread otomatis", "Exclude auto-posted threads"),
    "sb.exclude_auto_help": (
        "Daily Chat, Weekend Chat, dan Special Thread — diposting automod dengan jadwal "
        "tetap. Hanya ~2% thread tetapi ~22% dari seluruh komentar, sehingga "
        "menyimpangkan rata-rata dan analisis waktu.",
        "Daily Chat, Weekend Chat and Special Threads — posted by automod on a fixed "
        "schedule. Only ~2% of threads but ~22% of all comments, so they skew averages "
        "and any timing analysis."),
    "sb.post_type": ("Tipe kiriman", "Post type"),
    "sb.all": ("Semua", "All"),
    "sb.text": ("Teks", "Text"),
    "sb.link": ("Tautan", "Link"),
    "sb.min_comments": ("Minimum komentar (aktual)", "Minimum comments (actual)"),
    "sb.min_comments_help": (
        "Memakai total_comments_actual — hitungan gabungan sebenarnya, bukan penghitung "
        "Reddit yang usang.",
        "Uses total_comments_actual — the true joined count, not Reddit's stale counter."),
    "sb.threads_selected": ("Thread terpilih", "Threads selected"),
    "sb.comments_selected": ("Komentar terpilih", "Comments selected"),
    "sb.pct_of": ("{tp} dari thread • {cp} dari komentar", "{tp} of threads • {cp} of comments"),
    "sb.clear_cache": ("🔄 Bersihkan cache", "🔄 Clear cached data"),
    # -------------------------------------------------------- shared labels
    "lbl.threads": ("Thread", "Threads"),
    "lbl.comments": ("Komentar", "Comments"),
    "lbl.score": ("Skor", "Score"),
    "lbl.thread_score": ("Skor thread", "Thread score"),
    "lbl.comments_actual": ("Komentar (aktual)", "Comments (actual)"),
    "lbl.flair": ("Flair", "Flair"),
    "lbl.share": ("Bagian komentar", "Share of comments"),
    "lbl.hour": ("Jam (WIB)", "Hour (WIB)"),
    "lbl.word": ("Kata", "Word"),
    "lbl.occurrences": ("Kemunculan", "Occurrences"),
    "lbl.metric": ("Metrik", "Metric"),
    "lbl.normalized": ("Ternormalisasi", "Normalized"),
    "lbl.avg_ratio": ("Rata-rata rasio", "Avg ratio"),
    "lbl.avg_score": ("Rata-rata skor", "Avg score"),
    "lbl.avg_comments": ("Rata-rata komentar", "Avg comments"),
    "lbl.created": ("Dibuat (WIB)", "Created (WIB)"),
    "lbl.author": ("Penulis", "Author"),
    "lbl.title": ("Judul", "Title"),
    "lbl.replies_per_thread": ("Balasan / thread", "Replies / thread"),
    "lbl.score_per_thread": ("Skor / thread", "Score / thread"),
    "lbl.ratio": ("Rasio", "Ratio"),
    "lbl.edited": ("Diedit", "Edited"),
    "lbl.locked": ("Terkunci", "Locked"),
    "lbl.comment": ("Komentar", "Comment"),
    "lbl.thread": ("Thread", "Thread"),
    "lbl.words": ("Kata", "Words"),
    "lbl.chars": ("Karakter", "Chars"),
    "lbl.med_score": ("Median skor", "Median score"),
    "lbl.total_score": ("Total skor", "Total score"),
    "lbl.avg_comment_score": ("Rata-rata skor komentar", "Avg comment score"),
    "lbl.comments_received": ("Komentar diterima", "Comments received"),
    "lbl.comments_written": ("Komentar ditulis", "Comments written"),
    "lbl.comment_score": ("Skor komentar", "Comment score"),
    "lbl.total_engagement": ("Total interaksi", "Total engagement"),
    "lbl.avg_thread_score": ("Rata-rata skor thread", "Avg thread score"),
    "lbl.threads_created": ("Thread dibuat", "Threads created"),
    "lbl.label": ("Label", "Label"),
    "lbl.model_label": ("Label model", "Model label"),
    "lbl.ref_label": ("Label rujukan LLM", "LLM reference label"),
    "lbl.llm_label": ("Label LLM", "LLM label"),
    "lbl.hf_label": ("Label model", "Model label"),
    "lbl.confidence": ("Keyakinan", "Confidence"),
    "lbl.route": ("Rute", "Route"),
    "lbl.support": ("Dukungan LLM", "LLM support"),
    "lbl.precision": ("Presisi vs. LLM", "Precision vs. LLM"),
    "lbl.recall": ("Recall vs. LLM", "Recall vs. LLM"),
    "lbl.f1": ("F1", "F1"),
    # ------------------------------------------------------------ sentiment
    "sent.positive": ("positif", "positive"),
    "sent.neutral": ("netral", "neutral"),
    "sent.negative": ("negatif", "negative"),
    # -------------------------------------------------------------- overview
    "ov.title": ("Ikhtisar", "Overview"),
    "ov.comments": ("Komentar", "Comments"),
    "ov.contributors": ("Kontributor", "Contributors"),
    "ov.total_score": ("Total skor", "Total score"),
    "ov.avg_score": ("Rata-rata skor", "Avg score"),
    "ov.caption": (
        "Komentar yang ditampilkan hanya milik {n} thread yang tersaring. Total komentar "
        "aktual pada thread tersebut: **{a}**.",
        "Comments shown are those belonging to the {n} filtered threads. Actual comment "
        "total across them: **{a}**."),
    "ov.activity": ("Aktivitas dari waktu ke waktu (WIB)", "Activity over time (WIB)"),
    "ov.bucket": ("Rentang", "Bucket"),
    "ov.daily": ("Harian", "Daily"),
    "ov.weekly": ("Mingguan", "Weekly"),
    "ov.monthly": ("Bulanan", "Monthly"),
    "ov.top_flairs": ("Flair teratas", "Top flairs"),
    "ov.score_vs_comments": ("Skor vs. komentar", "Score vs. comments"),
    "ov.scatter_key": (
        "Warna titik = flair — kuncinya di grafik Flair teratas (kiri).",
        "Point colour = flair — key in the Top flairs chart (left).",
    ),
    "ov.score_dist": ("Sebaran skor thread", "Distribution of thread scores"),
    # --------------------------------------------------------------- threads
    "th.title": ("Thread", "Threads"),
    "th.median_score": ("Median skor", "Median score"),
    "th.median_comments": ("Median komentar", "Median comments"),
    "th.text_link": ("Teks / Tautan", "Text / Link"),
    "th.tab_top": ("Thread teratas", "Top threads"),
    "th.tab_browse": ("Jelajahi / cari", "Browse / search"),
    "th.tab_detail": ("Detail flair", "Flair detail"),
    "th.rank_by": ("Urutkan berdasarkan", "Rank by"),
    "th.opt_score": ("Skor", "Score"),
    "th.opt_comments": ("Komentar (aktual)", "Comments (actual)"),
    "th.opt_ratio": ("Rasio upvote", "Upvote ratio"),
    "th.opt_avgcs": ("Rata-rata skor komentar", "Avg comment score"),
    "th.opt_avg_score": ("Rata-rata skor", "Average score"),
    "th.opt_avg_comments": ("Rata-rata komentar", "Average comments"),
    "th.opt_avg_ratio": ("Rata-rata rasio upvote", "Average upvote ratio"),
    "th.opt_total_score": ("Total skor", "Total score"),
    "th.opt_threads": ("Jumlah thread", "Thread count"),
    "th.show_top_n": ("Tampilkan N teratas", "Show top N"),
    "th.search_caption": ("Cari di dalam kumpulan thread yang sedang tersaring.",
                          "Search within the currently filtered thread set."),
    "th.search_input": ("Cari di judul atau isi thread", "Search title or body text"),
    "th.matching": ("{n} thread cocok", "{n} matching threads"),
    "th.download": ("⬇ Unduh thread tersaring (CSV)", "⬇ Download filtered threads (CSV)"),
    "th.flair_comparison": ("Perbandingan flair", "Flair comparison"),
    # -------------------------------------------------------------- comments
    "cm.title": ("Komentar", "Comments"),
    "cm.commenters": ("Komentator", "Commenters"),
    "cm.median_score": ("Median skor", "Median score"),
    "cm.avg_length": ("Rata-rata panjang", "Avg length"),
    "cm.caption": ("Skor komentar diambil langsung dari berkas mentah (728K baris), "
                   "jadi statistik ini dihitung ulang secara live mengikuti filter Anda.",
                   "Comment scores reach back to the raw comments file (728K rows), so "
                   "these stats recompute live with your filters."),
    "cm.tab_activity": ("Aktivitas", "Activity"),
    "cm.tab_top": ("Komentar terbaik & terburuk", "Top & bottom comments"),
    "cm.tab_len": ("Panjang teks", "Text length"),
    "cm.volume": ("Volume komentar dari waktu ke waktu (WIB)", "Comment volume over time (WIB)"),
    "cm.bucket": ("Rentang", "Bucket"),
    "cm.score_dist": ("Sebaran skor komentar", "Score distribution"),
    "cm.per_thread": ("Komentar per thread", "Comments per thread"),
    "cm.top_threads": ("Thread dengan komentar terbanyak", "Top comment threads"),
    "cm.top_caption": ("Komentar dicocokkan ke thread tersaring Anda lewat `thread_id`.",
                       "Comments are matched to your filtered threads by `thread_id`."),
    "cm.highest": ("**Skor tertinggi**", "**Highest-scored**"),
    "cm.lowest": ("**Skor terendah**", "**Lowest-scored**"),
    "cm.sampling": ("Mengambil sampel 200K komentar agar tampilan tetap responsif.",
                    "Sampling 200K comments for responsiveness on this view."),
    "cm.len_dist": ("Sebaran panjang komentar", "Comment length distribution"),
    "cm.median_words": ("Median kata", "Median words"),
    "cm.median_chars": ("Median karakter", "Median characters"),
    "cm.longest": ("**Komentar terpanjang pada kumpulan tersaring**",
                   "**Longest comments in the filtered set**"),
    "cm.chars": ("{n} karakter", "{n} chars"),
    "cm.words_per_comment": ("Kata per komentar", "Words per comment"),
    "cm.score_axis": ("Skor komentar", "Comment score"),
    # --------------------------------------------------------------- authors
    "au.title": ("Penulis", "Authors"),
    "au.caption": ("Berkas papan peringkat adalah snapshot statis dan mengabaikan filter. "
                   "Tabel live di bawah dihitung ulang dari pilihan Anda.",
                   "Leaderboard file is a static snapshot and ignores filters. The live "
                   "table below is recomputed from your selections."),
    "au.tab_live": ("Peringkat live", "Live ranking"),
    "au.tab_static": ("Berkas papan peringkat", "Static leaderboard file"),
    "au.rank_by": ("Urutkan penulis berdasarkan", "Rank authors by"),
    "au.opt_total_engagement": ("Total interaksi", "Total engagement"),
    "au.opt_thread_score": ("Skor thread", "Thread score"),
    "au.opt_comment_score": ("Skor komentar", "Comment score"),
    "au.opt_threads": ("Thread", "Threads"),
    "au.opt_comments": ("Komentar", "Comments"),
    "au.show_top_n": ("Tampilkan N teratas", "Show top N"),
    "au.scatter_x": ("Thread dibuat", "Threads created"),
    "au.scatter_y": ("Komentar ditulis", "Comments written"),
    "au.download": ("⬇ Unduh peringkat ini (CSV)", "⬇ Download this ranking (CSV)"),
    "au.static_info": ("Snapshot statis dari `author_leaderboard.csv` — menampilkan penulis "
                       "teratas versi dataset, tidak terpengaruh filter.",
                       "Static snapshot from `author_leaderboard.csv` — shows the dataset's "
                       "own top authors, independent of filters."),
    "au.top_threads": ("Thread penulis teratas", "Top author threads"),
    "au.top_comments": ("Komentar penulis teratas", "Top author comments"),
    "au.missing": ("author_leaderboard.parquet tidak ditemukan di cache.",
                   "author_leaderboard.parquet not found in cache."),
    # ---------------------------------------------------------------- flairs
    "lbl.thread_count": ("Jumlah thread", "Thread count"),
    "lbl.avg_comments_actual": ("Rata-rata komentar (aktual)", "Avg comments (actual)"),
    "fl.title": ("Flair", "Flairs"),
    "fl.caption": ("Tampilan live dihitung ulang dari thread tersaring Anda.",
                   "Live view recomputed from your filtered threads."),
    "fl.threads_per_flair": ("Thread per flair", "Threads per flair"),
    "fl.other": ("Lainnya ({n} flair)", "Other ({n} flairs)"),
    "fl.engagement_profile": ("Profil interaksi per flair", "Engagement profile by flair"),
    "fl.norm_heatmap": ("Heatmap metrik ternormalisasi", "Normalized metric heatmap"),
    "fl.not_enough": ("Thread per flair tidak cukup setelah difilter untuk membuat heatmap.",
                      "Not enough threads per flair after filtering to build a heatmap."),
    "fl.tab_words": ("Kata teratas per flair", "Top words per flair"),
    "fl.tab_reference": ("Referensi performa flair", "Reference flair performance"),
    "fl.compare_flairs": ("Bandingkan flair", "Compare flairs"),
    "fl.select_one": ("Pilih minimal satu flair.", "Select at least one flair."),
    "fl.words_missing": ("flair_top_words.parquet tidak ditemukan di cache.",
                         "flair_top_words.parquet not found in cache."),
    "fl.perf_missing": ("flair_performance.parquet tidak ditemukan di cache.",
                        "flair_performance.parquet not found in cache."),
    "fl.ref_info": ("Snapshot statis dari `flair_performance.csv`.",
                    "Static snapshot from `flair_performance.csv`."),
    # ---------------------------------------------------------------- timing
    "tm.title": ("Waktu", "Timing"),
    "tm.caption": ("Semua waktu dikonversi ke WIB (UTC+7). Live dari thread & komentar tersaring.",
                   "All times converted to WIB (UTC+7). Live from filtered threads & comments."),
    "tm.no_threads": ("Tidak ada thread untuk filter saat ini.", "No threads for the current filters."),
    "tm.thread_heat": ("Heatmap aktivitas thread (hari × jam, WIB)",
                       "Thread activity heatmap (day × hour, WIB)"),
    "tm.comment_heat": ("Heatmap aktivitas komentar (hari × jam, WIB)",
                        "Comment activity heatmap (day × hour, WIB)"),
    "tm.comment_caption": (
        "Komentar mengikuti thread induknya, jadi tidak disaring per tanggal secara "
        "individual — thread di dalam rentang tetap bisa menyumbang komentar yang "
        "diposting setelahnya. Jam 07:00 mendominasi karena automod memposting "
        "Daily/Weekend Chat Thread saat itu.",
        "Comments ride along with their parent thread, so they are *not* date-filtered "
        "individually — a thread inside the window can still contribute comments posted "
        "after it. Hour 07:00 dominates because automod posts the Daily/Weekend Chat "
        "Threads then."),
    "tm.by_dow": ("Thread per hari", "Threads by day of week"),
    "tm.by_hour": ("Thread per jam", "Threads by hour of day"),
    "tm.replies_score": ("Balasan & skor per jam (thread organik)",
                         "Replies & score per hour (organic threads)"),
    "tm.no_organic": ("Tidak ada thread organik untuk filter saat ini.",
                      "No organic threads for the current filters."),
    "tm.trace_median_replies": ("Median balasan", "Median replies"),
    "tm.trace_mean": ("Rata-rata", "Mean replies"),
    "tm.trace_median_score": ("Median skor", "Median score"),
    "tm.eng_caption": (
        "Median dipakai sebagai angka utama karena sebaran balasan dan skor berekor berat; "
        "belah ketupat merah adalah rata-ratanya, sebagai pembanding. Jam tersibuk: "
        "**{hour} WIB** ({med:.0f} median balasan). n = {n} thread organik — thread "
        "otomatis dikecualikan.",
        "Median is the headline because the reply and score distributions are heavy-tailed; "
        "the red diamond is the mean, for contrast. Busiest hour: **{hour} WIB** "
        "({med:.0f} median replies). n = {n} organic threads — auto-posted chat threads "
        "are excluded."),
    "tm.reference": ("Referensi: berkas aktivitas per jam WIB", "Reference: WIB hourly activity file"),
    "tm.ref_missing": ("wib_hourly_activity.parquet tidak ditemukan di cache.",
                       "wib_hourly_activity.parquet not found in cache."),
    "tm.hover_replies": ("Jam %{x}:00 WIB<br>median %{y:.0f} balasan<br>rata-rata "
                         "%{customdata[1]:.0f} · n=%{customdata[0]}<extra></extra>",
                         "Hour %{x}:00 WIB<br>median %{y:.0f} replies<br>mean "
                         "%{customdata[1]:.0f} · n=%{customdata[0]}<extra></extra>"),
    "tm.hover_mean": ("Jam %{x}:00 WIB<br>rata-rata %{y:.0f} balasan<extra></extra>",
                      "Hour %{x}:00 WIB<br>mean %{y:.0f} replies<extra></extra>"),
    "tm.hover_score": ("Jam %{x}:00 WIB<br>median skor %{y:.0f}<extra></extra>",
                       "Hour %{x}:00 WIB<br>median score %{y:.0f}<extra></extra>"),
    "tm.ref_info": (
        "Snapshot statis dari `wib_hourly_activity.csv` (grid 7×24 penuh). **Baseline "
        "sepanjang waktu — sengaja tidak mengikuti filter sidebar.** Kolom `total_comments` "
        "dan `avg_score` juga ada di berkas tetapi tidak digambar di sini.",
        "Static snapshot from `wib_hourly_activity.csv` (full 7×24 grid). **All-time "
        "baseline — it deliberately does not follow the sidebar filters.** Columns "
        "`total_comments` and `avg_score` also exist in the file but are not plotted here."),
    # ------------------------------------------------------------- sentiment
    "se.title": ("Analitik sentimen", "Sentiment analytics"),
    "se.coverage": ("Cakupan dan distribusi utama", "Coverage and headline distribution"),
    "se.full_corpus": ("Komentar korpus penuh", "Full corpus comments"),
    "se.llm_sample": ("Sampel rujukan LLM", "LLM reference sample"),
    "se.mean_conf": ("Rata-rata keyakinan model", "Model mean confidence"),
    "se.low_conf": ("Label model, keyakinan rendah <0,60", "Model labels, low conf <0.60"),
    "se.series_raw": ("Model (mentah, 678K)", "Model (raw, 678K)"),
    "se.series_llm": ("Rujukan LLM (10K)", "LLM reference (10K)"),
    "se.series_corrected": ("Model (koreksi bias)", "Model (bias-corrected)"),
    "se.dist_caption": (
        "Model mentah menyebut dua per tiga komentar negatif dan nyaris menghapus kategori "
        "netral. Estimasi terkoreksi bias — diturunkan dari matriks konfusi 10K — mendekati "
        "rujukan LLM independen, bukti kuat bahwa distribusi korpus penuh yang mentah adalah "
        "artefak, bukan temuan.",
        "The raw model calls two-thirds of comments negative and nearly eliminates neutral. "
        "The bias-corrected estimate — derived from the 10K confusion matrix — lands close "
        "to the independent LLM reference, which is strong evidence the raw full-corpus "
        "distribution is an artifact, not a finding."),
    "se.calibration": ("Kalibrasi: mengoreksi kemiringan negatif model",
                       "Calibration: correcting the model's negative skew"),
    "se.agreement": ("Kesepakatan vs. rujukan LLM", "Agreement vs. LLM reference"),
    "se.kappa": ("Cohen's kappa", "Cohen's kappa"),
    "se.disagreements": ("Ketidaksepakatan", "Disagreements"),
    "se.agreement_conf": ("Kesepakatan vs. LLM", "Agreement vs. LLM"),
    "se.confusion": ("Matriks konfusi", "Confusion matrix"),
    "se.transition": (
        "**Perilaku transisi model** (baris = rujukan LLM, kolom = model): ketika rubrik "
        "bilang *netral*, model mempertahankan netral hanya **{pct}** dan sisanya dilabeli "
        "negatif. Inilah mekanisme di balik melonjaknya porsi negatif.",
        "**Model transition behavior** (rows = LLM reference, columns = model): when the "
        "rubric says *neutral*, the model keeps neutral only **{pct}** of the time and "
        "otherwise relabels it negative. This is the mechanism behind the inflated negative "
        "share."),
    "se.agree_conf_caption": ("Kesepakatan hampir tidak membaik seiring keyakinan — tanda "
                              "model yakin tetapi salah.",
                              "Agreement barely improves with confidence — a sign the model "
                              "is confidently wrong."),
    "se.interpretation": (
        "Interpretasi: rujukan LLM adalah sampel rubrik 10K, bukan kebenaran mutlak, jadi "
        "angka-angka ini mengukur kesepakatan antara dua pelabel, bukan akurasi sejati "
        "model. Gunakan porsi terkoreksi bias untuk klaim tentang korpus.",
        "Interpretation: the LLM reference is a 10K rubric sample, not absolute truth, so "
        "these figures measure agreement between two labelers, not the model's true "
        "accuracy. Use the bias-corrected shares for any claim about the corpus."),
    "se.slices": ("Sentimen korpus penuh per flair dan waktu",
                  "Full-corpus sentiment by flair and time"),
    "se.slices_caption": (
        "Label model di seluruh 678.522 komentar, dipecah per flair thread dan bulan. "
        "Potongan ini mewarisi kemiringan negatif model; baca bentuk dan perbedaan relatif, "
        "bukan tingkat absolutnya.",
        "Model labels across all 678,522 comments, broken down by thread flair and month. "
        "Slices inherit the model's negative skew; read shapes and relative differences, "
        "not absolute levels."),
    "se.view_mode": ("Mode distribusi", "Distribution mode"),
    "se.raw": ("Label model mentah", "Raw model labels"),
    "se.corrected": ("Koreksi bias", "Bias-corrected"),
    "se.corrected_caption": ("Porsi terkoreksi bias bisa berisik untuk flair kecil; "
                             "tafsirkan flair bervolume rendah dengan hati-hati.",
                             "Bias-corrected shares can be noisy for small flairs; treat "
                             "low-volume flairs cautiously."),
    "se.flairs_over_time": ("Flair yang dibandingkan sepanjang waktu",
                            "Flairs to compare over time"),
    "se.disagreement_title": ("Di mana model dan rubrik berbeda",
                              "Where the model and the rubric disagree"),
    "se.compared": ("Komentar yang dibandingkan", "Compared comments"),
    "se.agree": ("Setuju", "Agree"),
    "se.disagree": ("Beda", "Disagree"),
    "se.only_dis": ("Tampilkan hanya yang berbeda", "Show disagreements only"),
    "se.filter_flair": ("Saring per flair", "Filter by flair"),
    "se.search_comment": ("Cari teks komentar", "Search comment text"),
    "se.disagreement_hint": (
        "Ketidaksepakatan berkeyakinan tinggi di bagian atas paling informatif: model yakin "
        "dan rubrik tidak setuju. Baris-baris ini adalah daftar audit yang praktis.",
        "High-confidence disagreements at the top are the most informative: the model is "
        "sure and the rubric disagrees. These rows are a practical audit list."),
    "se.build_hint": ("Jalankan `python src/build_sentiment_cache.py` untuk mengaktifkan "
                      "penjelajah ketidaksepakatan.",
                      "Run `python src/build_sentiment_cache.py` to enable the disagreement "
                      "explorer."),
    "se.gold_title": ("Pengecekan eksternal kecil: benchmark emas 40 komentar",
                      "Small external check: 40-comment gold benchmark"),
    "se.gold_rows": ("Baris emas yang cocok", "Gold rows matched"),
    "se.gold_hc": ("Kesepakatan, keyakinan ≥0.8", "Agreement, conf ≥0.8"),
    "se.gold_agreement": ("Kesepakatan", "Agreement"),
    "se.gold_mean_conf": ("Rata-rata keyakinan", "Mean confidence"),
    "se.gold_caption": (
        "Hanya 40 baris dan diambil dari rubrik LLM yang sama, jadi ini hanya cek "
        "kewajaran, bukan validasi independen. Hasilnya sejalan dengan temuan 10K: "
        "model percaya diri dan condong negatif.",
        "Only 40 rows and drawn from the same LLM rubric, so it is a sanity check, not "
        "independent validation. It agrees with the 10K finding: the model is confident "
        "and skewed negative."),
    "se.sample_title": ("Periksa komentar berlabel", "Inspect labeled comments"),
    "se.dataset": ("Dataset", "Dataset"),
    "se.opt_llm": ("Rujukan LLM (10K)", "LLM reference (10K)"),
    "se.opt_hf": ("Model (sampel 10K)", "Model (10K sample)"),
    "se.search_llm": ("Cari teks berlabel LLM", "Search LLM-labeled text"),
    "se.search_hf": ("Cari teks berlabel model", "Search model-labeled text"),
    "se.no_10k": ("Tidak ada sampel model 10K.", "No 10K model sample available."),
    "lbl.comment_id": ("ID komentar", "Comment ID"),
    "se.missing": ("Artefak sentimen tidak ditemukan. Jalankan "
                   "`python src/build_sentiment_cache.py` setelah keluaran model korpus "
                   "penuh berada di root workspace.",
                   "Sentiment artifacts are missing. Run "
                   "`python src/build_sentiment_cache.py` after the full-corpus model "
                   "output is in the workspace root."),
    # ----------------------------------------------------------- misc messages
    "msg.no_cache_title": ("📊 Analitik r/indonesia", "📊 r/indonesia Analytics"),
    "msg.no_cache": ("Cache tidak ditemukan. Bangun sekali dari CSV mentah (± 1–2 menit).",
                     "Cache not found. Build it once from the raw CSVs (takes ~1–2 minutes)."),
    "msg.no_cache_expected": ("Lokasi yang diharapkan: `{path}`", "Expected location: `{path}`"),
    "msg.build_now": ("🔨 Bangun cache sekarang", "🔨 Build cache now"),
    "msg.building": ("Membangun cache… ini bisa memakan beberapa menit.",
                     "Building cache… this can take a couple of minutes."),
    "msg.built": ("Cache selesai dibangun. Memuat ulang…", "Cache built. Reloading…"),
    "msg.build_failed": ("Pembangunan cache gagal.", "Build failed."),
    "msg.no_threads": ("Tidak ada thread yang cocok dengan filter saat ini.",
                       "No threads match the current filters."),
    "msg.no_comments": ("Tidak ada komentar untuk filter thread saat ini.",
                        "No comments for the current thread filters."),
    "msg.no_data": ("Tidak ada data untuk filter saat ini.", "No data for the current filters."),
    "msg.no_filter_hits": ("Tidak ada data untuk filter saat ini",
                           "No data for the current filters"),
    "msg.pin": ("Streamlit {installed} terpasang, tetapi tema ini ditulis untuk "
                "{pinned}. Setelah 1.40, widget Streamlit berganti implementasi "
                "dan sebagian besar CSS tidak lagi berlaku — pasang ulang dengan "
                "`pip install streamlit=={pinned}`.",
                "Streamlit {installed} is installed, but this theme targets "
                "{pinned}. Streamlit changed its widget internals after 1.40, so "
                "much of the CSS no longer applies — reinstall with "
                "`pip install streamlit=={pinned}`."),
    "ft.built": ("Cache dibangun {at} • thread: {th} • komentar: {cm} • Dibuat oleh "
                 "`src/build_cache.py` — jalankan ulang setelah CSV berubah.",
                 "Cache built {at} • threads: {th} • comments: {cm} • Built by "
                 "`src/build_cache.py` — re-run it after the CSVs change."),
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
    linear-gradient(${blueprint} 1px, transparent 1px),
    linear-gradient(90deg, ${blueprint} 1px, transparent 1px);
  background-size: 26px 26px;
  color: ${ink};
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
#MainMenu, footer { visibility: hidden; }
/* the "Deploy" chrome button belongs to Streamlit, not to this dashboard */
[data-testid="stBaseButton-header"] { display: none !important; }
/* the per-heading "copy link" anchor only adds visual noise here */
[data-testid="stHeaderActionElements"] { display: none !important; }

/* one grid for the whole page -------------------------------------------- */
[data-testid="stMainBlockContainer"], .block-container {
  padding: 3.1rem clamp(1rem, 2.1vw, 2.4rem) 4rem !important;
  max-width: 1580px !important;
  margin: 0 auto !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: .9rem !important; }
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* Streamlit gives every markdown block a -16px bottom margin to cancel the
   paragraph margin it assumes. Our headings carry their own padding + border,
   so that negative margin drags the next widget straight into the rule. */
[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }

/* typography -------------------------------------------------------------- */
h1, h2, h3, h4, h5, h6 {
  color: ${ink_strong} !important;
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 700 !important;
  line-height: 1.25 !important;
}
h1 {
  font-size: 1.5rem !important;
  border-bottom: 3px solid ${ink_strong};
  padding: 0 0 .4rem !important;
  margin: 0 0 .8rem !important;
}
h2 {
  font-size: 1.25rem !important;
  border-left: 6px solid ${ink_strong};
  padding: .3rem 0 .3rem .6rem !important;
  margin: 0 0 .6rem !important;
}
h3 {
  font-size: .94rem !important;
  border-left: 6px solid ${accent};
  padding: .28rem 0 .28rem .6rem !important;
  margin: 1.1rem 0 .55rem !important;
}
/* Two charts side by side must start on the same baseline even when one
   heading wraps and the other does not. */
[data-testid="stMainBlockContainer"] h3 {
  min-height: 2.95rem;
  display: flex;
  align-items: center;
}
p, li, span, label, blockquote { color: ${ink}; }
a { color: ${ink_strong}; text-decoration: underline; text-underline-offset: 2px; }
hr { border: 0; border-top: 2px solid ${ink_strong}; margin: 1.1rem 0; }

/* sidebar ----------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: ${panel} !important;
  border-right: 3px solid ${ink_strong};
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .5rem !important; }
[data-testid="stSidebar"] h1 {
  font-size: 1.2rem !important;
  line-height: 1.3 !important;
  border-bottom: 3px solid ${ink_strong} !important;
  padding: 0 0 .45rem !important;
  margin: 0 0 .15rem !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  border-left: 0 !important;
  padding-left: 0 !important;
  margin: 0 !important;
}
/* One rule per section, sitting in the whitespace above its heading. */
[data-testid="stSidebar"] h3 {
  font-size: .6rem !important;
  letter-spacing: .18em;
  color: ${muted} !important;
  border-top: 2px solid ${ink_strong} !important;
  padding: .55rem 0 0 !important;
  line-height: 1.5 !important;
}
/* Sidebar captions carry data — date spans, help text — so they read as quiet
   sentence-case notes instead of shouting every word. */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  text-transform: none !important;
  letter-spacing: .01em !important;
  font-size: .67rem !important;
  line-height: 1.5 !important;
  color: ${muted} !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom: .1rem !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: .62rem !important;
  letter-spacing: .13em !important;
  line-height: 1.4 !important;
  text-transform: uppercase;
  color: ${muted} !important;
}
/* Ticked boxes are content, not captions: keep them sentence-cased and dark. */
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
  text-transform: none !important;
  letter-spacing: .01em !important;
  font-size: .72rem !important;
  color: ${ink} !important;
}
/* One control height for every widget in the column. */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stDateInput input { min-height: 38px !important; }
[data-testid="stSidebar"] hr { display: none !important; }

/* 20+ selected flairs must not turn the sidebar into a 900px column of tags */
[data-testid="stSidebar"] div:has(> [data-baseweb="tag"]) {
  max-height: 8.5rem;
  overflow-y: auto;
  scrollbar-width: none;
}
[data-testid="stSidebar"] div:has(> [data-baseweb="tag"])::-webkit-scrollbar {
  display: none;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  font-size: .66rem !important;
  letter-spacing: .03em;
  padding: 2px 5px !important;
  min-height: 22px !important;
}

/* the collapse chevron: Streamlit hides it on desktop, we want it back */
[data-testid="stSidebarCollapseButton"] { display: flex !important; align-items: center; }
[data-testid="stSidebarCollapsedControl"] { align-items: center; }
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
  width: 30px; height: 30px; min-width: 30px; min-height: 30px !important;
  display: inline-flex !important; align-items: center; justify-content: center;
  border: 2px solid ${ink_strong} !important; background: ${paper} !important;
  color: ${ink_strong} !important; box-shadow: 2px 2px 0 0 ${ink_strong};
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
  background: ${ink_strong} !important; color: ${paper} !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg { fill: currentColor !important; }

/* metrics as spec panels -------------------------------------------------- */
[data-testid="stMetric"] {
  background: ${panel};
  border: 2px solid ${ink_strong};
  box-shadow: 4px 4px 0 0 ${ink_strong};
  padding: .6rem .7rem .55rem;
  height: 100%;
  min-width: 0;
  display: flex; flex-direction: column; justify-content: space-between; gap: .3rem;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  text-transform: uppercase; letter-spacing: .08em; line-height: 1.3 !important;
  font-size: .58rem !important; color: ${muted} !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
  font-weight: 700 !important; color: ${ink_strong} !important;
  font-size: clamp(1rem, 1.35vw, 1.7rem) !important;
  line-height: 1.1 !important;
  font-variant-numeric: tabular-nums;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
  box-shadow: 2px 2px 0 0 ${ink_strong}; padding: .5rem .65rem;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] * { font-size: .55rem !important; }

/* tabs -------------------------------------------------------------------- */
/* Wrap instead of scroll: at narrow widths the strip becomes a second row of
   tabs, so no tab is ever hidden and no scrollbar is ever drawn. */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 2px solid ${ink_strong}; background: transparent;
  flex-wrap: wrap; align-content: flex-start; overflow: visible;
}
.stTabs [data-baseweb="tab"] {
  border: 2px solid ${ink_strong}; border-bottom: none;
  margin: 0 -2px -2px 0;              /* collapse the shared borders */
  background: ${panel}; color: ${ink} !important;
  text-transform: uppercase; letter-spacing: .04em; font-size: .7rem;
  padding: 6px 9px !important; height: auto !important; min-height: 34px;
  line-height: 1.2 !important; white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover { background: ${hover} !important; }
.stTabs [aria-selected="true"] { background: ${ink_strong} !important; color: ${paper} !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* buttons ----------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border: 2px solid ${ink_strong} !important; background: ${paper} !important;
  color: ${ink_strong} !important; text-transform: uppercase; letter-spacing: .1em;
  font-size: .72rem !important; font-weight: 600 !important;
  padding: .5rem 1.05rem !important; min-height: 38px;
  box-shadow: 3px 3px 0 0 ${ink_strong};
  transition: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: ${ink_strong} !important; color: ${paper} !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translate(3px, 3px); box-shadow: none;
}
.stButton > button[kind="primary"] { background: ${accent} !important; color: ${on_accent} !important; }
.stButton > button[kind="primary"]:hover { background: ${ink_strong} !important; }

/* inputs ------------------------------------------------------------------ */
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="base-input"],
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input,
.stDateInput input, .stTextArea textarea {
  border: 2px solid ${ink_strong} !important; background: ${paper} !important;
  box-shadow: 2px 2px 0 0 ${ink_strong}; color: ${ink_strong} !important;
}
[data-baseweb="select"] svg, [data-baseweb="select"] svg path {
  fill: ${ink_strong} !important; color: ${ink_strong} !important;
}
[data-baseweb="tag"] { background: ${ink_strong} !important; color: ${paper} !important; border: none !important; }
/* BaseWeb paints empty-state text and input placeholders from the *server*
   config theme, so on the dark board they come out grey-on-grey. */
input::placeholder, textarea::placeholder { color: ${muted} !important; opacity: 1 !important; }
[data-testid="stMultiSelect"]:not(:has([data-baseweb="tag"])) [data-baseweb="select"] * {
  color: ${muted} !important;
}

/* chart frames + compact legend ------------------------------------------ */
[data-testid="stPlotlyChart"] {
  background: ${paper};
  outline: 2px solid ${ink_strong};
  outline-offset: -2px;
  box-shadow: 4px 4px 0 0 ${ink_strong};
}
[data-testid="stPlotlyChart"] .modebar { top: 0 !important; right: 0 !important; }
.legend-strip {
  display: flex; flex-wrap: wrap; align-items: center;
  gap: .28rem .9rem; padding: .1rem 0 .15rem;
}
.legend-strip .lg {
  display: inline-flex; align-items: center; gap: .38rem;
  font-size: .63rem; letter-spacing: .06em; text-transform: uppercase;
  color: ${ink}; white-space: nowrap;
}
.legend-strip .lg i {
  width: 10px; height: 10px; display: inline-block;
  border: 1px solid ${ink_strong}; background: ${muted};
}

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

/* contrast fixes — never dark text on an ink (dark) surface ----------------- */
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color: ${paper} !important; }
[data-baseweb="tag"] { background: ${ink_strong} !important; border: none !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] div, [data-baseweb="tag"] p,
[data-baseweb="tag"] label, [data-baseweb="tag"] svg, [data-baseweb="tag"] path {
  color: ${paper} !important; fill: ${paper} !important;
}
.stButton > button:hover, .stButton > button:hover *,
.stDownloadButton > button:hover, .stDownloadButton > button:hover *,
.stButton > button[kind="primary"], .stButton > button[kind="primary"] *,
[data-testid="stFormSubmitButton"] > button, [data-testid="stFormSubmitButton"] > button * {
  color: ${paper} !important;
}
/* popovers, dropdowns and the calendar keep their own surface -------------- */
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul[role="listbox"] {
  background: ${paper} !important;
}
ul[role="listbox"] li, ul[role="listbox"] li * { color: ${ink} !important; }
ul[role="listbox"] li:hover, ul[role="listbox"] li:hover *,
ul[role="listbox"] li[aria-selected="true"], ul[role="listbox"] li[aria-selected="true"] * {
  background-color: ${ink_strong} !important; color: ${paper} !important;
}
[data-baseweb="calendar"], [data-baseweb="calendar"] > div { background: ${paper} !important; }
[data-baseweb="calendar"] div { color: ${ink} !important; }
[data-baseweb="calendar"] [aria-selected="true"],
[data-baseweb="calendar"] [aria-selected="true"] * {
  background: ${ink_strong} !important; color: ${paper} !important;
}

/* checkboxes / radios — BaseWeb paints these from the server config theme, so
   the unchecked box would stay a white square on a dark board. The radio's
   *inner dot* is a separate div that BaseWeb fills white, which reads as a
   white hole on paper and a white square on the dark board. */
[data-baseweb="checkbox"]:has(input:not(:checked)) > span,
[data-baseweb="radio"]:has(input:not(:checked)) > div:first-child {
  background: transparent !important; border: 2px solid ${muted} !important;
}
[data-baseweb="radio"]:has(input:not(:checked)) > div:first-child > div {
  background: transparent !important;
}

/* numeric steppers: Streamlit ships them light-on-light, so they are a white
   block with an invisible glyph on the dark board (and near-invisible on paper) */
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
  background: ${paper} !important; color: ${ink_strong} !important;
  border-left: 2px solid ${ink_strong} !important; box-shadow: none !important;
}
[data-testid="stNumberInputStepDown"] svg, [data-testid="stNumberInputStepUp"] svg {
  fill: currentColor !important; color: ${ink_strong} !important;
}
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover {
  background: ${ink_strong} !important; color: ${paper} !important;
}
[data-testid="stNumberInputStepDown"]:hover svg, [data-testid="stNumberInputStepUp"]:hover svg {
  fill: ${paper} !important;
}

${extra}
</style>
""")

def dark_css() -> str:
    """Extra rules for the dark board.

    Streamlit paints its canvas widgets (the data grid) from the server-side
    config theme, which is fixed at launch, so the grid is inverted to sit on
    the dark board instead of glowing white inside it.
    """
    return f"""
html, .stApp {{ color-scheme: dark; }}
/* Invert the canvas layer only: filtering the root would also invert the 2px
   frame and offset shadow we draw on it, erasing the grid's border. */
[data-testid="stDataFrame"] > div {{ filter: invert(.93); }}
[data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}
[data-testid="stTooltipContent"] {{
  background: {EINK["panel"]} !important; color: {EINK["ink"]} !important;
  border: 2px solid {EINK["ink_strong"]} !important;
}}
[data-testid="stToast"] {{ background: {EINK["panel"]} !important; }}
"""


def inject_css() -> None:
    """Render the stylesheet for the active theme (call once per run)."""
    css = CSS_TEMPLATE.substitute(
        mono=EINK_MONO, paper=EINK["paper"], panel=EINK["panel"], ink=EINK["ink"],
        ink_strong=EINK["ink_strong"], grid=EINK["grid"], muted=EINK["muted"],
        accent=EINK["accent"], hover=EINK["hover"], blueprint=EINK["blueprint"],
        on_accent=EINK["on_accent"],
        extra=dark_css() if theme() == "dark" else "",
    )
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data loading
# --------------------------------------------------------------------------- #


# Columns the dashboard never reads. `comments` carries a second timestamp and
# three ids that no page touches; dropping them saves ~49 MB on the cached
# frame and the same again on the filtered copy built in main().
_UNUSED_COLUMNS = {
    "comments": (
        "created_utc", "distinguished", "is_submitter", "parent_id",
        "version", "edited_dt", "is_mod",
    ),
}


@st.cache_resource(show_spinner="Loading cached dataset…")
def load(name: str) -> pd.DataFrame:
    """Read one cache file, keeping its text Arrow-backed.

    At 728k rows every string column costs ~50 bytes of Python object overhead
    per value, so `comments` alone is 337 MB that way. Reading with the Arrow
    backend keeps the text in contiguous buffers instead - 104 MB for the same
    frame once the columns no page reads are dropped below, and no per-value
    objects to churn through during reloads. Everything that is not text is
    cast back to numpy so the existing pandas and plotly code paths see exactly
    the dtypes they were written against.

    `cache_resource` rather than `cache_data` for the same reason: cache_data
    keeps a pickled copy of the value next to the live one. Nothing mutates
    these frames in place, so sharing them is safe.
    """
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path, dtype_backend="pyarrow")
    for col in df.columns:
        dtype = df[col].dtype
        if not isinstance(dtype, pd.ArrowDtype):
            continue
        arrow_type = dtype.pyarrow_dtype
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            continue
        if pa.types.is_timestamp(arrow_type):
            df[col] = df[col].astype("datetime64[ns]")
        else:
            df[col] = df[col].astype(dtype.numpy_dtype)

    unused = _UNUSED_COLUMNS.get(name)
    if unused:
        df = df.drop(columns=[c for c in unused if c in df.columns])

    _release_memory()
    return df


def _release_memory() -> None:
    """Hand freed Arrow buffers back to the OS between heavy allocations."""
    gc.collect()
    pa.default_memory_pool().release_unused()


def _parquet_rows(name: str) -> int:
    """Row count from the parquet footer, without materialising the frame."""
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        return 0
    try:
        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:  # noqa: BLE001 - fall back to a real read
        return len(load(name))


@st.cache_resource(show_spinner=False)
def _comment_bodies(ids: tuple[str, ...]) -> dict[str, str]:
    """Read the text of only the comments that are about to be displayed.

    `comments_text` is ~146 MB resident and the only thing the dashboard ever
    shows from it is 30 rows (the highest and lowest scoring comments), so it
    never joins the cached working set. The file shipped with the cache is
    sorted by `comment_id` in 2k-row groups, which lets the filter below skip
    almost every group instead of decoding the whole text column.
    """
    path = CACHE / "comments_text.parquet"
    if not ids or not path.exists():
        return {}
    table = ds.dataset(str(path), format="parquet").to_table(
        filter=ds.field("comment_id").isin(list(ids)),
        columns=["comment_id", "body"],
    )
    frame = table.to_pandas()
    return dict(zip(frame["comment_id"], frame["body"]))


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
    """Are the files the dashboard cannot run without present?

    `dashboard.parquet` is deliberately not checked: no page reads it, it is a
    by-product of build_cache.py (threads enriched with the true comment count).
    Requiring it would mean shipping 4 MB that nothing opens.
    """
    required = ["threads.parquet", "comments.parquet", "meta.json"]
    return all((CACHE / f).exists() for f in required)


@st.cache_data(show_spinner=False)
def flair_palette() -> dict:
    """Stable flair -> colour map, ordered by corpus-wide frequency.

    Computed from the *unfiltered* thread cache so a given flair keeps the same
    colour across every view and filter selection.
    """
    th = load("threads")
    if th.empty or "link_flair_text" not in th.columns:
        return {}
    order = th["link_flair_text"].value_counts().index.tolist()
    return {fl: EINK_TOPICS[i % len(EINK_TOPICS)] for i, fl in enumerate(order)}


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
    """Apply the e-ink / engineering look shared by every chart.

    Keyword arguments are forwarded to ``update_layout``; ``margin=`` can be
    passed here and replaces the default frame instead of colliding with it.
    """
    # Only reserve head-room when a legend will actually be drawn, so single
    # series charts do not carry 40px of dead space above the plot.
    legend_on = fig.layout.showlegend is not False and len(fig.data) > 1
    layout = dict(
        height=height,
        margin=dict(l=8, r=14, t=56 if legend_on else 26, b=8),
        font=dict(family=EINK_MONO, size=12, color=EINK["ink"]),
        paper_bgcolor=EINK["paper"],
        plot_bgcolor=EINK["paper"],
        colorway=EINK_TOPICS,
        legend=dict(
            # Container coordinates, not paper ones: a legend anchored to the
            # plot area slides right when automargin widens the y-axis gutter
            # and then overflows the chart frame.
            orientation="h", xref="container", x=0.01, xanchor="left",
            yref="container", y=0.99, yanchor="top",
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(family=EINK_MONO, size=10.5, color=EINK["ink"]),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor=EINK["ink_strong"], bordercolor=EINK["ink_strong"],
            font=dict(family=EINK_MONO, color=EINK["paper"], size=12),
        ),
        # the modebar is drawn by Plotly, not CSS: theme it or it vanishes into
        # the dark board (and stays invisible-ish on paper).
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=EINK["muted"],
                     activecolor=EINK["accent"]),
    )
    layout.update(kw)
    fig.update_layout(**layout)
    # Colour bars: Plotly keeps its own grey (#808495) for their ticks and
    # title, which stays dark-on-dark on the dark board. Theme them explicitly,
    # on the layout axis *and* on each heatmap trace (px binds it per trace).
    bar = dict(
        tickfont=dict(family=EINK_MONO, color=EINK["ink"], size=10.5),
        title=dict(font=dict(family=EINK_MONO, color=EINK["ink"], size=11)),
        outlinewidth=1, outlinecolor=EINK["ink_strong"], thickness=12, len=0.9,
    )
    fig.update_layout(coloraxis_colorbar=bar)
    for trace in fig.data:
        if trace.type == "heatmap":
            trace.update(colorbar=bar)
    # automargin keeps long tick labels and axis titles from being clipped when
    # the chart is squeezed into a narrow column.
    fig.update_xaxes(
        showgrid=False, linecolor=EINK["ink_strong"], linewidth=1.4, ticks="outside",
        tickcolor=EINK["ink_strong"], tickfont=dict(family=EINK_MONO, color=EINK["ink"]),
        title_font=dict(family=EINK_MONO, color=EINK["ink"]), automargin=True,
    )
    fig.update_yaxes(
        gridcolor=EINK["grid"], gridwidth=1, griddash="dot", zeroline=False,
        linecolor=EINK["ink_strong"], linewidth=1.4, ticks="outside",
        tickcolor=EINK["ink_strong"], tickfont=dict(family=EINK_MONO, color=EINK["ink"]),
        title_font=dict(family=EINK_MONO, color=EINK["ink"]), automargin=True,
    )
    return fig


def legend_strip(labels, colors: dict) -> None:
    """Print a compact, wrapping colour key for charts with many categories.

    Plotly's own legend turns into a bordered block that spills over the plot
    once a chart carries ~10+ categories, so those charts hide it and call this
    instead: same colours in a fraction of the vertical space, wrapping with the
    column instead of overflowing it. Rendered *below* the chart so side-by-side
    charts still start on the same line.
    """
    labels = [lab for lab in labels if lab is not None and str(lab) != ""]
    if not labels:
        return
    chips = "".join(
        f'<span class="lg"><i style="background:{colors.get(lab, EINK["muted"])}"></i>'
        f'{html.escape(str(lab))}</span>'
        for lab in labels
    )
    st.markdown(f'<div class="legend-strip">{chips}</div>', unsafe_allow_html=True)


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


def filter_defaults(th: pd.DataFrame) -> dict:
    """State a fresh session starts in.

    Also the baseline the URL is diffed against, so a default view produces a
    short — usually empty — query string, and only deviations are encoded.
    """
    bounds = date_bounds(th, "created_wib")
    lo, hi = bounds if bounds else (None, None)
    if th.empty:
        flairs, counts = [], pd.Series(dtype=int)
        smin = smax = 0
        p99 = 0
    else:
        flairs = sorted(th["link_flair_text"].dropna().unique().tolist())
        counts = th["link_flair_text"].value_counts()
        smin, smax = int(th["score"].min()), int(th["score"].max())
        p99 = min(smax, int(th["score"].quantile(0.99)))
    return {
        "lo": lo, "hi": hi, "flairs": flairs,
        "default_flairs": [f for f in flairs if counts.get(f, 0) >= 20] or flairs,
        "smin": smin, "smax": smax, "score": (smin, p99),
        "preset": "all",
    }


def _qp(name: str) -> str | None:
    """First value of a URL query parameter, or None."""
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _qp_signature() -> str:
    return json.dumps(
        {k: v for k, v in sorted(st.query_params.to_dict().items())}, sort_keys=True,
    )


def seed_filters(defaults: dict) -> None:
    """URL -> widget state. Must run before the filter widgets are created.

    A changed signature means the URL came from somewhere other than this run
    (shared link, bookmark, browser back/forward), so every key is dropped
    first and only the encoded ones are restored — otherwise Back would leave
    the newer values in place.
    """
    signature = _qp_signature()
    if st.session_state.get("_filters_sig") == signature:
        return
    st.session_state["_filters_sig"] = signature
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)

    preset = _qp("preset")
    if preset in RANGE_PRESETS:
        st.session_state["flt_preset"] = preset

    d0, d1 = _qp("d0"), _qp("d1")
    if d0 and d1 and defaults["lo"] is not None:
        try:
            lo, hi = defaults["lo"].date(), defaults["hi"].date()
            start = max(lo, min(hi, pd.Timestamp(d0).date()))
            end = max(lo, min(hi, pd.Timestamp(d1).date()))
            st.session_state["flt_drange"] = (min(start, end), max(start, end))
        except (ValueError, TypeError):
            pass

    raw_flairs = _qp("flairs")
    if raw_flairs:
        try:
            wanted = json.loads(raw_flairs)
        except (json.JSONDecodeError, TypeError):
            wanted = None
        if isinstance(wanted, list):
            allowed = set(defaults["flairs"])
            st.session_state["flt_flairs"] = [f for f in wanted
                                              if isinstance(f, str) and f in allowed]

    for param, key in (("author", "flt_author"), ("title", "flt_title")):
        value = _qp(param)
        if value is not None:
            st.session_state[key] = value

    raw_score = _qp("score")
    if raw_score and "," in raw_score:
        parts = raw_score.split(",")
        if len(parts) == 2 and all(p.strip().isnumeric() for p in parts):
            lo_b, hi_b = int(parts[0]), int(parts[1])
            st.session_state["flt_score"] = (
                max(defaults["smin"], min(defaults["smax"], lo_b)),
                max(defaults["smin"], min(defaults["smax"], hi_b)),
            )

    for param, key in (("nsfw", "flt_nsfw"), ("locked", "flt_locked"),
                       ("mod", "flt_mod"), ("stick", "flt_stick"),
                       ("spoil", "flt_spoil"), ("edited", "flt_edited"),
                       ("auto", "flt_auto")):
        value = _qp(param)
        if value is not None:
            st.session_state[key] = value not in ("0", "false", "no")

    post_type = _qp("ptype")
    if post_type in ("All", "Text", "Link"):
        st.session_state["flt_ptype"] = post_type

    minc = _qp("minc")
    if minc is not None and minc.lstrip("-").isdigit():
        st.session_state["flt_minc"] = max(0, int(minc))


def _url_params(state: dict, defaults: dict) -> dict:
    """Encode only what differs from the defaults, as plain strings."""
    params: dict[str, str] = {}
    if state["preset"] != defaults["preset"]:
        params["preset"] = state["preset"]
    if state["preset"] == "custom" and state["start"] is not None:
        params["d0"] = state["start"].date().isoformat()
        # the filter end is exclusive (start of the next day); the URL is not
        params["d1"] = (state["end"] - pd.Timedelta(days=1)).date().isoformat()
    if state["flairs"] != defaults["default_flairs"]:
        params["flairs"] = json.dumps(state["flairs"], ensure_ascii=False)
    if state["author_q"]:
        params["author"] = state["author_q"]
    if state["text_q"]:
        params["title"] = state["text_q"]
    if tuple(state["score_rng"]) != tuple(defaults["score"]):
        params["score"] = f"{state['score_rng'][0]},{state['score_rng'][1]}"
    for param, key in (("nsfw", "f_nsfw"), ("locked", "f_locked"),
                       ("mod", "f_mod"), ("stick", "f_stick"),
                       ("spoil", "f_spoil"), ("edited", "f_edited")):
        if not state[key]:
            params[param] = "0"
    if state["exclude_auto"]:
        params["auto"] = "1"
    if state["post_type"] != "All":
        params["ptype"] = state["post_type"]
    if state["min_comments"] > 0:
        params["minc"] = str(state["min_comments"])
    if lang() != DEFAULT_LANG:
        params["lang"] = lang()
    if theme() != DEFAULT_THEME:
        params["theme"] = theme()
    return params


def _write_url(params: dict[str, str]) -> None:
    """Mirror the filter state into the address bar so views are shareable."""
    current: dict[str, str] = {}
    for key, value in st.query_params.to_dict().items():
        current[key] = value[0] if isinstance(value, list) and value else value
    if current == params:
        st.session_state["_filters_sig"] = _qp_signature()
        return
    for key in list(current):
        if key not in params:
            del st.query_params[key]
    for key, value in params.items():
        if current.get(key) != value:
            st.query_params[key] = str(value)
    st.session_state["_filters_sig"] = _qp_signature()


def sidebar_filters(th: pd.DataFrame, cm: pd.DataFrame, defaults: dict) -> dict:
    # A reset clears the URL and every filter key *before* the widgets exist in
    # this run — touching widget state after creation is not allowed.
    if st.session_state.pop("_reset_filters", False):
        st.query_params.clear()
        for key in FILTER_KEYS:
            st.session_state.pop(key, None)
        st.session_state["_filters_sig"] = None
    seed_filters(defaults)

    st.sidebar.title(t("sb.filters"))
    st.sidebar.caption(t("sb.filters_help"))

    # ---- time range (driven by threads, the primary calendar) ----
    # Preset ids stay stable across languages so switching never resets the
    # selection; only the displayed label is translated.
    st.sidebar.subheader(t("sb.time_range"))
    lo, hi = defaults["lo"], defaults["hi"]
    if lo is not None:
        st.sidebar.caption(t("sb.spans").format(span=f"{fmt_date(lo)} → {fmt_date(hi)}"))
        preset = st.sidebar.selectbox(
            t("sb.quick_range"),
            options=list(RANGE_PRESETS),
            format_func=lambda k: t(RANGE_PRESETS[k]),
            index=0,
            key="flt_preset",
        )
        spans = {"d7": 7, "d30": 30, "d90": 90}
        if preset in spans:
            start = max(lo, hi - pd.Timedelta(days=spans[preset]))
            end = hi
        elif preset == "all":
            start, end = lo, hi
        else:
            picked = st.sidebar.date_input(
                t("sb.custom_range"), value=(lo.date(), hi.date()),
                min_value=lo.date(), max_value=hi.date(), key="flt_drange",
            )
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
            else:
                start, end = lo, hi
        if start > end:
            start, end = end, start
    else:
        preset, start, end = "all", None, None

    # ---- flair ----
    st.sidebar.subheader(t("sb.flair"))
    st.sidebar.caption(t("sb.topics_help"))
    flairs = defaults["flairs"]
    sel_flairs = st.sidebar.multiselect(
        t("sb.topics"), options=flairs, default=defaults["default_flairs"],
        label_visibility="collapsed", key="flt_flairs",
    )
    if not sel_flairs:
        sel_flairs = flairs

    # ---- author / text ----
    # Folded away by default: they are occasional drills, not part of the
    # everyday filter loop, and two full-width inputs pushed the real filters
    # below the fold. Opened automatically when a shared link carries a query.
    search_open = bool(st.session_state.get("flt_author") or st.session_state.get("flt_title"))
    with st.sidebar.expander(t("sb.search"), expanded=search_open):
        st.caption(t("sb.search_help"))
        author_q = st.text_input(t("sb.author_q"), key="flt_author")
        text_q = st.text_input(t("sb.title_q"), key="flt_title")
    author_q = (author_q or "").strip()
    text_q = (text_q or "").strip()

    # ---- thresholds ----
    # The heading names the control, so the slider's own label would repeat it.
    st.sidebar.subheader(t("sb.thread_score"))
    if not th.empty:
        score_rng = st.sidebar.slider(
            t("sb.thread_score"), min_value=defaults["smin"], max_value=defaults["smax"],
            value=defaults["score"], label_visibility="collapsed", key="flt_score",
        )
    else:
        score_rng = (0, 0)

    # ---- flags ----
    st.sidebar.subheader(t("sb.flags"))
    st.sidebar.caption(t("sb.flags_help"))
    c1, c2 = st.sidebar.columns(2)
    f_nsfw = c1.checkbox(t("sb.inc_nsfw"), value=True, key="flt_nsfw")
    f_locked = c2.checkbox(t("sb.inc_locked"), value=True, key="flt_locked")
    f_mod = c1.checkbox(t("sb.inc_mod"), value=True, key="flt_mod")
    f_stick = c2.checkbox(t("sb.inc_stick"), value=True, key="flt_stick")
    f_spoil = c1.checkbox(t("sb.inc_spoil"), value=True, key="flt_spoil")
    f_edited = c2.checkbox(t("sb.inc_edited"), value=True, key="flt_edited")

    st.sidebar.subheader(t("sb.content"))
    f_exclude_auto = st.sidebar.checkbox(
        t("sb.exclude_auto"), value=False,
        help=t("sb.exclude_auto_help"), key="flt_auto",
    )

    post_type = st.sidebar.radio(
        t("sb.post_type"), options=["All", "Text", "Link"], index=0,
        format_func=lambda v: {"All": t("sb.all"), "Text": t("sb.text"), "Link": t("sb.link")}[v],
        horizontal=True, key="flt_ptype",
    )

    min_comments = st.sidebar.number_input(
        t("sb.min_comments"), min_value=0, value=0, step=5,
        help=t("sb.min_comments_help"), key="flt_minc",
    )

    state = {
        "start": start, "end": end, "preset": preset,
        "flairs": sel_flairs,
        "author_q": author_q,
        "text_q": text_q,
        "score_rng": score_rng,
        "f_nsfw": f_nsfw, "f_locked": f_locked, "f_mod": f_mod,
        "f_stick": f_stick, "f_spoil": f_spoil, "f_edited": f_edited,
        "exclude_auto": f_exclude_auto,
        "post_type": post_type,
        "min_comments": int(min_comments),
    }
    _write_url(_url_params(state, defaults))

    if st.sidebar.button(t("sb.reset"), use_container_width=True):
        st.session_state["_reset_filters"] = True
        st.rerun()
    st.sidebar.caption(t("sb.url_hint"))
    return state


def apply_filters(th: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = th

    if f["start"] is not None:
        d = d[d["created_wib"].between(f["start"], f["end"])]

    if f["flairs"]:
        d = d[d["link_flair_text"].isin(f["flairs"])]

    if f.get("exclude_auto"):
        d = d[~d["link_flair_text"].isin(AUTO_POST_FLAIRS)]

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
    st.header(t("ov.title"))

    threads = len(th)
    comments = len(cm)
    total_actual = int(th["total_comments_actual"].sum())
    authors = th["author"].nunique()

    c = st.columns(5)
    c[0].metric(t("lbl.threads"), fmt_int(threads))
    c[1].metric(t("ov.comments"), fmt_int(comments))
    c[2].metric(t("ov.contributors"), fmt_int(authors))
    c[3].metric(t("ov.total_score"), fmt_compact(th["score"].sum()))
    c[4].metric(t("ov.avg_score"), f"{th['score'].mean():.1f}" if threads else "-")

    st.caption(t("ov.caption").format(n=fmt_int(threads), a=fmt_int(total_actual)))

    if th.empty:
        st.plotly_chart(empty_fig(t("msg.no_filter_hits")), use_container_width=True, key="ov_empty")
        return

    # ---- activity over time ----
    st.subheader(t("ov.activity"))
    granularity = st.radio(
        t("ov.bucket"), ["Daily", "Weekly", "Monthly"], horizontal=True, key="ov_gran",
        format_func=lambda v: {"Daily": t("ov.daily"), "Weekly": t("ov.weekly"),
                               "Monthly": t("ov.monthly")}[v],
    )
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]

    ts = (
        th.set_index("created_wib")
        .resample(freq)
        .agg(threads=("thread_id", "count"), score=("score", "sum"))
        .reset_index()
    )
    ts = ts[ts["threads"] > 0]

    fig = px.bar(ts, x="created_wib", y="threads",
                 labels={"threads": t("lbl.threads"), "created_wib": ""})
    fig.update_traces(marker_color=EINK["ink"])
    st.plotly_chart(style(fig, 330), use_container_width=True, key="ov_ts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t("ov.top_flairs"))
        fl = (
            th.groupby("link_flair_text")
            .agg(threads=("thread_id", "count"), avg_score=("score", "mean"))
            .sort_values("threads", ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            fl.sort_values("threads"), x="threads", y="link_flair_text", orientation="h",
            labels={"threads": t("lbl.threads"), "link_flair_text": ""},
            hover_data={"avg_score": ":.1f"},
            color="link_flair_text", color_discrete_map=flair_palette(),
        )
        fig.update_traces(marker_line_color=EINK["ink_strong"], marker_line_width=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 430), use_container_width=True, key="ov_flair")

    with col2:
        st.subheader(t("ov.score_vs_comments"))
        samp = th.sample(min(len(th), 3000), random_state=0) if len(th) > 3000 else th
        pal = flair_palette()
        fig = px.scatter(
            samp, x="score", y="total_comments_actual", color="link_flair_text",
            opacity=0.6, labels={"score": t("lbl.thread_score"),
                                 "total_comments_actual": t("lbl.comments_actual"),
                                 "link_flair_text": t("lbl.flair")},
            color_discrete_map=pal,
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 430), use_container_width=True, key="ov_scatter")
        # 24 flairs would need a legend taller than the chart itself; the bar
        # chart on the left already prints the same colour key with its labels.
        st.caption(t("ov.scatter_key"))

    st.subheader(t("ov.score_dist"))
    pos = th[th["score"] >= 0]["score"]
    fig = px.histogram(pos, nbins=60,
                       labels={"value": t("lbl.score"), "count": t("lbl.threads")})
    fig.update_traces(marker_color=EINK["dim1"])
    fig.update_layout(showlegend=False, yaxis_title=t("lbl.threads"), xaxis_title=t("lbl.score"))
    st.plotly_chart(style(fig, 300), use_container_width=True, key="ov_hist")


def page_threads(th: pd.DataFrame) -> None:
    st.header(t("th.title"))
    if th.empty:
        st.info(t("msg.no_threads"))
        return

    c = st.columns(4)
    c[0].metric(t("lbl.threads"), fmt_int(len(th)))
    c[1].metric(t("th.median_score"), f"{th['score'].median():.0f}")
    c[2].metric(t("th.median_comments"), f"{th['total_comments_actual'].median():.0f}")
    c[3].metric(
        t("th.text_link"),
        f"{int((th['post_type'] == 'Text').sum())} / {int((th['post_type'] == 'Link').sum())}",
    )

    tab1, tab2, tab3 = st.tabs([t("th.tab_top"), t("th.tab_browse"), t("th.tab_detail")])

    with tab1:
        metric = st.selectbox(
            t("th.rank_by"),
            ["score", "total_comments_actual", "upvote_ratio", "average_comment_score"],
            format_func=lambda x: {
                "score": t("th.opt_score"), "total_comments_actual": t("th.opt_comments"),
                "upvote_ratio": t("th.opt_ratio"), "average_comment_score": t("th.opt_avgcs"),
            }[x],
            key="th_rank",
        )
        n = st.slider(t("th.show_top_n"), 10, 100, 25, step=5, key="th_n")
        top = th.nlargest(n, metric)[
            ["title", "link_flair_text", "score", "total_comments_actual",
             "upvote_ratio", "created_wib", "author"]
        ].copy()
        top["created_wib"] = top["created_wib"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            top.rename(columns={
                "title": t("lbl.title"), "link_flair_text": t("lbl.flair"),
                "score": t("lbl.score"), "total_comments_actual": t("lbl.comments"),
                "upvote_ratio": t("lbl.ratio"), "created_wib": t("lbl.created"),
                "author": t("lbl.author"),
            }),
            use_container_width=True, hide_index=True, height=460,
        )

    with tab2:
        st.caption(t("th.search_caption"))
        q = st.text_input(t("th.search_input"), "", key="th_q")
        view = th
        if q:
            mask = (
                th["title"].str.contains(q, case=False, na=False, regex=False)
                | th["selftext"].str.contains(q, case=False, na=False, regex=False)
            )
            view = th[mask]
            st.caption(t("th.matching").format(n=fmt_int(len(view))))
        show = view.sort_values("created_wib", ascending=False)[
            ["created_wib", "title", "link_flair_text", "score", "total_comments_actual",
             "upvote_ratio", "was_edited", "locked"]
        ].copy()
        show["created_wib"] = show["created_wib"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            show.rename(columns={
                "created_wib": t("lbl.created"), "title": t("lbl.title"),
                "link_flair_text": t("lbl.flair"), "score": t("lbl.score"),
                "total_comments_actual": t("lbl.comments"), "upvote_ratio": t("lbl.ratio"),
                "was_edited": t("lbl.edited"), "locked": t("lbl.locked"),
            }),
            use_container_width=True, hide_index=True, height=460,
        )
        st.download_button(
            t("th.download"),
            view.drop(columns=[c for c in ["selftext"] if c in view]).to_csv(index=False).encode("utf-8"),
            file_name="filtered_threads.csv", mime="text/csv",
        )

    with tab3:
        st.subheader(t("th.flair_comparison"))
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

        metric_labels = {
            "avg_score": t("th.opt_avg_score"), "avg_comments": t("th.opt_avg_comments"),
            "avg_ratio": t("th.opt_avg_ratio"), "total_score": t("th.opt_total_score"),
            "threads": t("th.opt_threads"),
        }
        m = st.selectbox(
            t("lbl.metric"),
            ["avg_score", "avg_comments", "avg_ratio", "total_score", "threads"],
            format_func=lambda x: metric_labels[x], key="th_flair_metric",
        )
        fig = px.bar(
            g.sort_values(m), x=m, y="link_flair_text", orientation="h",
            labels={m: metric_labels[m], "link_flair_text": ""},
            hover_data={"threads": True, "avg_score": ":.1f", "avg_comments": ":.1f"},
        )
        fig.update_traces(marker_color=EINK["dim3"])
        st.plotly_chart(style(fig, max(340, 26 * len(g))), use_container_width=True, key="th_flair_bar")

        disp = g.copy()
        for c_ in ["avg_score", "med_score", "avg_comments", "avg_ratio", "total_score"]:
            disp[c_] = disp[c_].round(2)
        st.dataframe(
            disp.rename(columns={
                "link_flair_text": t("lbl.flair"), "threads": t("lbl.threads"),
                "avg_score": t("lbl.avg_score"), "med_score": t("lbl.med_score"),
                "avg_comments": t("lbl.avg_comments"), "avg_ratio": t("lbl.avg_ratio"),
                "total_score": t("lbl.total_score"),
            }),
            use_container_width=True, hide_index=True,
        )


def page_comments(th: pd.DataFrame, cm: pd.DataFrame, f: dict) -> None:
    st.header(t("cm.title"))

    if cm.empty:
        st.info(t("msg.no_comments"))
        return

    c = st.columns(4)
    c[0].metric(t("lbl.comments"), fmt_int(len(cm)))
    c[1].metric(t("cm.commenters"), fmt_int(cm["author"].nunique()))
    c[2].metric(t("cm.median_score"), f"{cm['score'].median():.0f}")
    c[3].metric(t("cm.avg_length"), t("cm.chars").format(n=f"{cm['body_len'].mean():.0f}"))

    st.caption(t("cm.caption"))

    tab1, tab2, tab3 = st.tabs([t("cm.tab_activity"), t("cm.tab_top"), t("cm.tab_len")])

    with tab1:
        st.subheader(t("cm.volume"))
        freq = st.radio(
            t("cm.bucket"), ["Daily", "Weekly", "Monthly"], horizontal=True, key="cm_gran",
            format_func=lambda v: {"Daily": t("ov.daily"), "Weekly": t("ov.weekly"),
                                   "Monthly": t("ov.monthly")}[v],
        )
        fr = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
        # `cm` carries 16 columns; setting the index on the whole frame would
        # copy every one of them (~150 MB) just to resample two. Resample a
        # two-column stand-in instead - `n` is a literal 1, so its sum is the
        # row count in each bucket.
        ts = (
            pd.DataFrame(
                {"n": 1, "score": cm["score"].to_numpy()},
                index=pd.DatetimeIndex(cm["created_wib"]),
            )
            .resample(fr)
            .agg(comments=("n", "sum"), avg_score=("score", "mean"))
            .reset_index()
        )
        ts = ts[ts["comments"] > 0]
        fig = px.bar(ts, x="created_wib", y="comments",
                     labels={"comments": t("lbl.comments"), "created_wib": ""})
        fig.update_traces(marker_color=EINK["ink"])
        st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_ts")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t("cm.score_dist"))
            lo, hi = cm["score"].quantile([0.01, 0.99])
            # Bin here rather than handing 705k values to the browser: the chart
            # shows 60 bars either way, and the figure payload drops from ~6 MB
            # to a few hundred bytes.
            score_counts, score_edges = np.histogram(
                cm["score"].clip(lo, hi).to_numpy(), bins=60
            )
            hist = pd.DataFrame({
                "score": (score_edges[:-1] + score_edges[1:]) / 2,
                "comments": score_counts,
            })
            fig = px.bar(hist, x="score", y="comments",
                         labels={"score": t("lbl.score"), "comments": t("lbl.comments")})
            fig.update_traces(marker_color=EINK["dim1"])
            fig.update_layout(showlegend=False, yaxis_title=t("lbl.comments"),
                              xaxis_title=t("lbl.score"))
            st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_hist")
        with col2:
            st.subheader(t("cm.per_thread"))
            per = th["total_comments_actual"]
            per = per[per > 0]
            fig = px.histogram(per, nbins=50,
                               labels={"value": t("cm.per_thread"), "count": t("lbl.threads")})
            fig.update_traces(marker_color=EINK["dim2"])
            fig.update_layout(showlegend=False, yaxis_title=t("lbl.threads"),
                              xaxis_title=t("cm.per_thread"))
            st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_per")

    with tab2:
        st.subheader(t("cm.top_threads"))
        st.caption(t("cm.top_caption"))
        tmap = th.set_index("thread_id")["title"].to_dict()
        work = cm
        if len(work) > 200_000:
            work = work.sample(200_000, random_state=0)
            st.caption(t("cm.sampling"))

        # Only the 30 rows below are ever shown, so the text for the whole
        # corpus is not read at all - just the bodies of these comment ids.
        up = work.nlargest(15, "score")[["comment_id", "score", "thread_id"]].copy()
        dn = work.nsmallest(15, "score")[["comment_id", "score", "thread_id"]].copy()
        bodies = _comment_bodies(
            tuple(str(c) for c in pd.concat([up["comment_id"], dn["comment_id"]]))
        )
        for frame in (up, dn):
            frame["body"] = [bodies.get(str(c), "") for c in frame["comment_id"]]
            frame["thread_title"] = frame["thread_id"].map(tmap)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(t("cm.highest"))
            up["body"] = up["body"].astype(str).str.slice(0, 160)
            st.dataframe(
                up[["score", "body", "thread_title"]].rename(
                    columns={"score": t("lbl.score"), "body": t("lbl.comment"),
                             "thread_title": t("lbl.thread")}),
                use_container_width=True, hide_index=True, height=380,
            )
        with col2:
            st.markdown(t("cm.lowest"))
            dn["body"] = dn["body"].astype(str).str.slice(0, 160)
            st.dataframe(
                dn[["score", "body", "thread_title"]].rename(
                    columns={"score": t("lbl.score"), "body": t("lbl.comment"),
                             "thread_title": t("lbl.thread")}),
                use_container_width=True, hide_index=True, height=380,
            )

    with tab3:
        st.subheader(t("cm.len_dist"))
        word_counts, word_edges = np.histogram(cm["word_count"].to_numpy(), bins=60)
        lw = pd.DataFrame({
            "words": (word_edges[:-1] + word_edges[1:]) / 2,
            "comments": word_counts,
        })
        fig = px.bar(lw, x="words", y="comments",
                     labels={"words": t("cm.words_per_comment"), "comments": t("lbl.comments")})
        fig.update_traces(marker_color=EINK["dim3"])
        fig.update_layout(showlegend=False, yaxis_title=t("lbl.comments"))
        st.plotly_chart(style(fig, 320), use_container_width=True, key="cm_len")

        c1, c2 = st.columns(2)
        c1.metric(t("cm.median_words"), f"{cm['word_count'].median():.0f}")
        c2.metric(t("cm.median_chars"), f"{cm['body_len'].median():.0f}")

        st.markdown(t("cm.longest"))
        long_ct = cm.nlargest(15, "word_count")[["word_count", "score", "body_len"]]
        st.dataframe(
            long_ct.rename(columns={"word_count": t("lbl.words"), "score": t("lbl.score"),
                                    "body_len": t("lbl.chars")}),
            use_container_width=True, hide_index=True,
        )


def page_authors(th: pd.DataFrame, cm: pd.DataFrame) -> None:
    st.header(t("au.title"))

    if th.empty and cm.empty:
        st.info(t("msg.no_data"))
        return

    st.caption(t("au.caption"))

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

    metric_labels = {
        "total_engagement": t("au.opt_total_engagement"),
        "thread_score": t("au.opt_thread_score"),
        "comment_score": t("au.opt_comment_score"),
        "threads": t("au.opt_threads"),
        "comments": t("au.opt_comments"),
    }

    tab1, tab2 = st.tabs([t("au.tab_live"), t("au.tab_static")])

    with tab1:
        metric = st.selectbox(
            t("au.rank_by"),
            ["total_engagement", "thread_score", "comment_score", "threads", "comments"],
            format_func=lambda x: metric_labels[x], key="au_metric",
        )
        n = st.slider(t("au.show_top_n"), 5, 50, 20, key="au_n")
        top = combined.nlargest(n, metric)

        fig = px.bar(
            top.sort_values(metric), x=metric, y="author", orientation="h",
            labels={metric: metric_labels[metric], "author": ""},
        )
        fig.update_traces(marker_color=EINK["ink"])
        fig.update_yaxes(tickmode="linear")
        st.plotly_chart(style(fig, max(340, 24 * len(top))), use_container_width=True, key="au_bar")

        # Scatter: are the loudest commenters also the most upvoted?
        fig = px.scatter(
            combined[combined[metric] > 0], x="threads", y="comments",
            size=combined[combined[metric] > 0]["total_engagement"].abs() + 1,
            labels={"threads": t("au.scatter_x"), "comments": t("au.scatter_y")},
            hover_name="author",
        )
        fig.update_traces(marker=dict(color=EINK["dim1"], opacity=0.75))
        st.plotly_chart(style(fig, 400), use_container_width=True, key="au_scatter")

        disp = top.copy()
        for c_ in ["thread_score", "comment_score", "total_engagement", "avg_thread_score"]:
            disp[c_] = disp[c_].round(1)
        st.dataframe(
            disp.rename(columns={
                "author": t("lbl.author"), "threads": t("lbl.threads"),
                "thread_score": t("lbl.thread_score"),
                "avg_thread_score": t("lbl.avg_thread_score"),
                "comments_received": t("lbl.comments_received"),
                "comments": t("lbl.comments_written"),
                "comment_score": t("lbl.comment_score"),
                "total_engagement": t("lbl.total_engagement"),
            }),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            t("au.download"),
            combined.sort_values(metric, ascending=False).to_csv(index=False).encode("utf-8"),
            file_name="live_author_ranking.csv", mime="text/csv",
        )

    with tab2:
        lb = load("author_leaderboard")
        if lb.empty:
            st.warning(t("au.missing"))
            return
        st.info(t("au.static_info"))
        top = lb.nlargest(20, "total_engagement_score")
        fig = px.bar(
            top.sort_values("total_engagement_score"), x="total_engagement_score", y="author",
            orientation="h",
            labels={"total_engagement_score": t("lbl.total_engagement"), "author": ""},
        )
        fig.update_traces(marker_color=EINK["dim1"])
        st.plotly_chart(style(fig, 500), use_container_width=True, key="au_static")

        cm1, cm2 = st.columns(2)
        cm1.metric(t("au.top_threads"), fmt_int(top.iloc[0]["threads_created"]))
        cm2.metric(t("au.top_comments"), fmt_int(top.iloc[0]["comments_written"]))
        lb_cols = {
            "author": t("lbl.author"),
            "threads_created": t("lbl.threads_created"),
            "comments_written": t("lbl.comments_written"),
            "total_engagement_score": t("lbl.total_engagement"),
        }
        st.dataframe(
            lb.head(50).rename(columns=lambda c_: lb_cols.get(c_, c_.replace("_", " ").title())),
            use_container_width=True, hide_index=True,
        )


def page_flairs(th: pd.DataFrame) -> None:
    st.header(t("fl.title"))

    if th.empty:
        st.info(t("msg.no_threads"))
        return

    st.caption(t("fl.caption"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("fl.threads_per_flair"))
        g = th["link_flair_text"].value_counts().reset_index()
        g.columns = ["flair", "threads"]
        pal = flair_palette()
        if len(g) > 10:  # keep the donut legible; bundle the long tail
            head = g.head(10).copy()
            other = pd.DataFrame([{
                "flair": t("fl.other").format(n=len(g) - 10),
                "threads": int(g["threads"].iloc[10:].sum()),
            }])
            g = pd.concat([head, other], ignore_index=True)
        colors = [pal.get(fl, EINK["muted"]) for fl in g["flair"]]
        fig = px.pie(g, names="flair", values="threads", hole=0.45)
        fig.update_traces(
            textposition="outside", textinfo="label+percent",
            textfont=dict(family=EINK_MONO, size=11, color=EINK["ink_strong"]),
            marker=dict(colors=colors, line=dict(color=EINK["ink_strong"], width=1)),
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 460), use_container_width=True, key="fl_pie")

    with col2:
        st.subheader(t("fl.engagement_profile"))
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
            labels={"avg_comments": t("lbl.avg_comments"), "avg_score": t("lbl.avg_score"),
                    "avg_ratio": t("lbl.avg_ratio")},
            color_continuous_scale=EINK_SEQ_BLUE,
        )
        st.plotly_chart(style(fig, 460), use_container_width=True, key="fl_scatter")

    st.subheader(t("fl.norm_heatmap"))
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
        st.info(t("fl.not_enough"))
    else:
        metrics = ["threads", "avg_score", "avg_comments", "avg_ratio", "avg_comment_score"]
        norm = (profile[metrics] - profile[metrics].min()) / (
            profile[metrics].max() - profile[metrics].min()
        ).replace(0, 1)
        fig = px.imshow(
            norm.T, aspect="auto", color_continuous_scale=EINK_SEQ_BLUE,
            labels=dict(x=t("lbl.flair"), y=t("lbl.metric"), color=t("lbl.normalized")),
            text_auto=".2f",
        )
        fig.update_layout(height=max(320, 40 * len(metrics)))
        st.plotly_chart(style(fig, max(320, 40 * len(metrics))), use_container_width=True, key="fl_heat")

    tab1, tab2 = st.tabs([t("fl.tab_words"), t("fl.tab_reference")])
    with tab1:
        fw = load("flair_top_words")
        if fw.empty:
            st.warning(t("fl.words_missing"))
        else:
            selected = st.multiselect(
                t("fl.compare_flairs"),
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
                    barmode="group",
                    labels={"count": t("lbl.occurrences"), "word": "", "flair": t("lbl.flair")},
                    color_discrete_map=flair_palette(),
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style(fig, max(400, 30 * len(selected) * 2)), use_container_width=True, key="fl_words")
                legend_strip(selected, flair_palette())
            else:
                st.info(t("fl.select_one"))

    with tab2:
        fp = load("flair_performance")
        if fp.empty:
            st.warning(t("fl.perf_missing"))
        else:
            st.info(t("fl.ref_info"))
            fp2 = fp.sort_values("thread_count", ascending=False)
            fig = px.bar(
                fp2.head(20).sort_values("avg_score"), x="avg_score", y="link_flair_text",
                orientation="h",
                labels={"avg_score": t("lbl.avg_score"), "link_flair_text": ""},
                hover_data={"thread_count": True, "avg_comments_actual": ":.1f"},
                color="link_flair_text", color_discrete_map=flair_palette(),
            )
            fig.update_traces(marker_line_color=EINK["ink_strong"], marker_line_width=1)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 520), use_container_width=True, key="fl_ref")
            fp_cols = {
                "link_flair_text": t("lbl.flair"),
                "thread_count": t("lbl.thread_count"),
                "avg_score": t("lbl.avg_score"),
                "avg_comments_actual": t("lbl.avg_comments_actual"),
            }
            st.dataframe(
                fp2.rename(columns=lambda c_: fp_cols.get(c_, c_.replace("_", " ").title())),
                use_container_width=True, hide_index=True,
            )


def page_timing(th: pd.DataFrame, cm: pd.DataFrame) -> None:
    st.header(t("tm.title"))
    st.caption(t("tm.caption"))

    if th.empty:
        st.info(t("tm.no_threads"))
        return

    # NB: local name is `tt`, not `t` — `t()` is the translation helper.
    tt = th.copy()
    tt["dow"] = tt["created_wib"].dt.day_name()
    tt["hour"] = tt["created_wib"].dt.hour

    c = None
    if not cm.empty:
        # Same idea as the volume chart: the heatmap needs only day-of-week and
        # hour, so derive those as arrays instead of copying all 16 columns.
        c = pd.DataFrame(
            {
                "dow": pd.Categorical(
                    np.asarray(DOW_ORDER, dtype=object)[
                        cm["created_wib"].dt.dayofweek.to_numpy()
                    ],
                    categories=DOW_ORDER,
                ),
                "hour": cm["created_wib"].dt.hour.to_numpy(),
            }
        )

    st.subheader(t("tm.thread_heat"))
    pivot = (
        tt.pivot_table(index="dow", columns="hour", values="thread_id", aggfunc="count")
        .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
    )
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale=EINK_SEQ_BLUE,
        labels=dict(x=t("lbl.hour"), y="", color=t("lbl.threads")),
    )
    fig.update_xaxes(dtick=2)
    st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_heat_t")

    if c is not None:
        st.subheader(t("tm.comment_heat"))
        cp = (
            c.groupby(["dow", "hour"], observed=False).size().unstack("hour")
            .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
        )
        fig = px.imshow(
            cp, aspect="auto", color_continuous_scale=EINK_SEQ_EMBER,
            labels=dict(x=t("lbl.hour"), y="", color=t("lbl.comments")),
        )
        fig.update_xaxes(dtick=2)
        st.plotly_chart(style(fig, 380), use_container_width=True, key="tm_heat_c")
        st.caption(t("tm.comment_caption"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("tm.by_dow"))
        g = tt["dow"].value_counts().reindex(DOW_ORDER).fillna(0).reset_index()
        g.columns = ["dow", "threads"]
        fig = px.bar(g, x="dow", y="threads", labels={"dow": "", "threads": t("lbl.threads")})
        fig.update_traces(marker_color=EINK["ink"])
        st.plotly_chart(style(fig, 330), use_container_width=True, key="tm_dow")
    with col2:
        st.subheader(t("tm.by_hour"))
        g = tt["hour"].value_counts().reindex(range(24)).fillna(0).reset_index()
        g.columns = ["hour", "threads"]
        fig = px.bar(g, x="hour", y="threads",
                     labels={"hour": t("lbl.hour"), "threads": t("lbl.threads")})
        fig.update_traces(marker_color=EINK["dim2"])
        st.plotly_chart(style(fig, 330), use_container_width=True, key="tm_hour")

    # ---- When do threads get replies? -------------------------------------
    # Auto-posted chat threads are excluded here on purpose. They are published
    # by automod at a fixed hour, so including them makes "best hour" mean "when
    # automod runs" rather than when an ordinary post earns replies. Medians are
    # used because replies/score are heavy-tailed: one viral thread moves an
    # hourly mean by tens of points, while the median barely budges.
    st.subheader(t("tm.replies_score"))
    organic = tt[~tt["link_flair_text"].isin(AUTO_POST_FLAIRS)]
    if organic.empty:
        st.info(t("tm.no_organic"))
    else:
        e = (
            organic.groupby("hour")
            .agg(
                n=("thread_id", "count"),
                med_comments=("total_comments_actual", "median"),
                mean_comments=("total_comments_actual", "mean"),
                med_score=("score", "median"),
            )
            .reindex(range(24)).fillna(0).reset_index()
        )
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
        fig.add_trace(
            go.Bar(
                x=e["hour"], y=e["med_comments"], name=t("tm.trace_median_replies"),
                marker_color=EINK["ink"], marker_line_width=0,
                customdata=e[["n", "mean_comments"]],
                hovertemplate=t("tm.hover_replies"),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=e["hour"], y=e["mean_comments"], mode="markers",
                name=t("tm.trace_mean"),
                marker=dict(color=EINK["accent"], size=8, symbol="diamond"),
                hovertemplate=t("tm.hover_mean"),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=e["hour"], y=e["med_score"], mode="lines+markers",
                name=t("tm.trace_median_score"),
                line=dict(color=EINK["ink"], width=2), marker=dict(color=EINK["ink"], size=5),
                hovertemplate=t("tm.hover_score"),
            ), row=2, col=1,
        )
        fig.update_yaxes(title_text=t("lbl.replies_per_thread"), row=1, col=1)
        fig.update_yaxes(title_text=t("lbl.score_per_thread"), row=2, col=1)
        fig.update_xaxes(dtick=2, row=1, col=1)
        fig.update_xaxes(dtick=2, title_text=t("lbl.hour"), row=2, col=1)
        fig = style(fig, 540)
        fig.update_layout(margin=dict(l=10, r=10, t=64, b=10))
        st.plotly_chart(fig, use_container_width=True, key="tm_eng")

        best = e.loc[e["med_comments"].idxmax()]
        st.caption(t("tm.eng_caption").format(
            hour=f"{int(best['hour']):02d}:00", med=best["med_comments"],
            n=f"{int(e['n'].sum()):,}",
        ))

    st.divider()
    st.subheader(t("tm.reference"))
    wh = load("wib_hourly_activity")
    if wh.empty:
        st.warning(t("tm.ref_missing"))
    else:
        st.info(t("tm.ref_info"))
        grid = (
            wh.pivot_table(index="wib_day_of_week", columns="wib_hour",
                           values="threads_count", aggfunc="sum")
            .reindex(DOW_ORDER).reindex(columns=range(24)).fillna(0)
        )
        fig = px.imshow(grid, aspect="auto", color_continuous_scale=EINK_SEQ_BLUE,
                        labels=dict(x=t("lbl.hour"), y="", color=t("lbl.threads")))
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
    st.header(t("se.title"))

    overview = load_json("sentiment_overview")
    # Only the presence of labels is used below, and the full frame is ~259 MB
    # resident. The footer answers "is there data?" without reading the columns.
    has_hf_full = _parquet_rows("sentiment_full_slim") > 0

    if not overview or not has_hf_full:
        st.error(t("se.missing"))
        return

    full = overview.get("full_corpus", {})
    comp = overview.get("comparison_full_run_vs_llm_10k", {})
    llm10 = overview.get("llm_10k_sample", {})
    hf10 = overview.get("hf_10k_sample", {})
    calib = overview.get("calibration", {})

    # Sentiment labels are data values ("positive"/"neutral"/"negative"); only the
    # displayed form is localised, so the underlying order stays canonical.
    sent_labels = {k: t(f"sent.{k}") for k in SENTIMENT_ORDER}
    order_l10n = [sent_labels[k] for k in SENTIMENT_ORDER]

    st.subheader(t("se.coverage"))
    c = st.columns(4)
    c[0].metric(t("se.full_corpus"), fmt_int(full.get("rows", 0)))
    c[1].metric(t("se.llm_sample"), fmt_int(llm10.get("rows", 0)))
    c[2].metric(t("se.mean_conf"), f"{full.get('mean_confidence', 0):.2f}")
    c[3].metric(t("se.low_conf"), f"{full.get('low_confidence_lt_0_6', 0):.1%}")

    obs = full.get("distribution_pct", {})
    corr = calib.get("corrected", {})
    ref = llm10.get("distribution", {})
    ref_total = sum(ref.values()) or 1
    cmp_df = pd.DataFrame({
        "label": order_l10n,
        t("se.series_raw"): [obs.get(l, 0) for l in SENTIMENT_ORDER],
        t("se.series_llm"): [ref.get(l, 0) / ref_total for l in SENTIMENT_ORDER],
        t("se.series_corrected"): [corr.get(l, 0) for l in SENTIMENT_ORDER],
    }).melt(id_vars="label", var_name="series", value_name="share")

    fig = px.bar(
        cmp_df, x="label", y="share", color="series", barmode="group",
        color_discrete_sequence=[EINK["accent"], EINK["ink"], EINK["dim2"]],
        labels={"label": "", "share": t("lbl.share"), "series": ""},
        category_orders={"label": order_l10n},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 360), use_container_width=True, key="se_dist_compare")
    st.caption(t("se.dist_caption"))

    # ------------------------------------------------------------- calibration
    st.subheader(t("se.calibration"))
    k = st.columns(3)
    k[0].metric(t("se.agreement"), f"{comp.get('exact_agreement', 0):.1%}")
    k[1].metric(t("se.kappa"), f"{comp.get('cohen_kappa', 0):.3f}")
    k[2].metric(t("se.disagreements"), fmt_int(comp.get("disagreements", 0)))

    cm = comp.get("confusion_matrix", {})
    neutral_row = cm.get("neutral", {})
    neutral_kept = neutral_row.get("neutral", 0) / max(1, sum(neutral_row.values()))
    st.markdown(t("se.transition").format(pct=f"{neutral_kept:.1%}"))

    if cm:
        cm_df = pd.DataFrame(
            [[cm.get(r, {}).get(col, 0) for col in SENTIMENT_ORDER] for r in SENTIMENT_ORDER],
            index=order_l10n,
            columns=order_l10n,
        )
        fig = px.imshow(
            cm_df, text_auto=True, aspect="auto",
            color_continuous_scale=EINK_SEQ,
            labels={"x": t("lbl.model_label"), "y": t("lbl.ref_label"),
                    "color": t("lbl.comments")},
        )
        st.plotly_chart(style(fig, 340), use_container_width=True, key="se_cm")

    pl = pd.DataFrame(comp.get("per_label_vs_reference", []))
    if not pl.empty:
        pl["label"] = pl["label"].map(sent_labels).fillna(pl["label"])
        pl = pl.rename(columns={
            "label": t("lbl.label"), "support": t("lbl.support"),
            "precision": t("lbl.precision"), "recall": t("lbl.recall"), "f1": t("lbl.f1"),
        })
        st.dataframe(pl, use_container_width=True, hide_index=True)

    ac = pd.DataFrame(
        [{t("lbl.confidence"): k2, **v}
         for k2, v in comp.get("agreement_by_confidence", {}).items()]
    )
    if not ac.empty:
        ac = ac.rename(columns={"n": t("lbl.comments"), "agreement": t("se.agreement_conf")})
        ac[t("se.agreement_conf")] = (ac[t("se.agreement_conf")] * 100).round(1).astype(str) + "%"
        st.caption(t("se.agree_conf_caption"))
        st.dataframe(ac, use_container_width=True, hide_index=True)

    st.info(t("se.interpretation"))

    # ------------------------------------------------------- full-corpus slices
    st.divider()
    st.subheader(t("se.slices"))
    st.caption(t("se.slices_caption"))

    view_mode = st.radio(
        t("se.view_mode"), ["raw", "corrected"], horizontal=True, key="se_view_mode",
        format_func=lambda v: t("se.raw") if v == "raw" else t("se.corrected"),
    )

    # The legend is driven by the label column, so the colour map has to be keyed
    # by the *localised* label once the frame carries localised values.
    colors_l10n = {sent_labels[k]: v for k, v in SENTIMENT_COLORS.items()}

    bf = load("sentiment_by_flair") if (CACHE / "sentiment_by_flair.parquet").exists() else pd.DataFrame()
    if not bf.empty:
        piv = bf.pivot_table(index="flair", columns="label", values="n", aggfunc="sum").fillna(0)
        piv = piv.reindex(columns=SENTIMENT_ORDER, fill_value=0)
        piv["total"] = piv.sum(axis=1)
        piv = piv.sort_values("total", ascending=False)
        shares = piv[SENTIMENT_ORDER].div(piv["total"], axis=0)

        if view_mode == "corrected":
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
            st.caption(t("se.corrected_caption"))

        top = shares.head(15)
        long = top.reset_index().melt(
            id_vars="flair", var_name="label", value_name="share",
            value_vars=SENTIMENT_ORDER,
        )
        long["label"] = long["label"].map(sent_labels)
        fig = px.bar(
            long, x="share", y="flair", color="label", orientation="h", barmode="stack",
            color_discrete_map=colors_l10n,
            labels={"share": t("lbl.share"), "flair": "", "label": ""},
            category_orders={"label": order_l10n},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 460), use_container_width=True, key="se_flair")

    bfm = load("sentiment_by_flair_month") if (CACHE / "sentiment_by_flair_month.parquet").exists() else pd.DataFrame()
    if not bfm.empty:
        flairs = sorted(bfm["flair"].unique())
        picked = st.multiselect(
            t("se.flairs_over_time"), flairs,
            default=flairs[: min(3, len(flairs))], key="se_flair_month_chart",
        )
        sub = bfm[bfm["flair"].isin(picked)] if picked else bfm
        if not sub.empty:
            agg = sub.groupby(["month", "label"], observed=True)["n"].sum().reset_index()
            tot = agg.groupby("month")["n"].transform("sum")
            agg["share"] = agg["n"] / tot
            agg["label"] = agg["label"].map(sent_labels)
            fig = px.line(
                agg, x="month", y="share", color="label", markers=True,
                color_discrete_map=colors_l10n,
                labels={"month": "", "share": t("lbl.share"), "label": ""},
                category_orders={"label": order_l10n},
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 340), use_container_width=True, key="se_flair_month")

    # ------------------------------------------------------ disagreement explorer
    st.divider()
    st.subheader(t("se.disagreement_title"))
    cmp_path = CACHE / "sentiment_compare_10k.parquet"
    if cmp_path.exists():
        cmp_df = load("sentiment_compare_10k")
        cc = st.columns(3)
        cc[0].metric(t("se.compared"), fmt_int(len(cmp_df)))
        cc[1].metric(t("se.agree"), fmt_int(int(cmp_df["agree"].sum())))
        cc[2].metric(t("se.disagree"), fmt_int(int((~cmp_df["agree"]).sum())))

        only_dis = st.checkbox(t("se.only_dis"), value=True, key="se_only_dis")
        view = cmp_df[~cmp_df["agree"]] if only_dis else cmp_df

        fl = st.multiselect(
            t("se.filter_flair"), sorted(view["flair"].dropna().unique()),
            key="se_dis_flair",
        )
        if fl:
            view = view[view["flair"].isin(fl)]

        q = st.text_input(t("se.search_comment"), "", key="se_dis_q")
        if q:
            view = view[view["body"].fillna("").str.contains(q, case=False, regex=False)]

        show_cols = [c2 for c2 in ["body", "llm_label", "hf_label", "hf_confidence", "flair", "route"]
                     if c2 in view]
        view = view.sort_values("hf_confidence", ascending=False)
        disp = view[show_cols].head(400).copy()
        for col in ("llm_label", "hf_label"):
            if col in disp:
                disp[col] = disp[col].map(sent_labels).fillna(disp[col])
        disp = disp.rename(columns={
            "body": t("lbl.comment"), "llm_label": t("lbl.llm_label"),
            "hf_label": t("lbl.hf_label"), "hf_confidence": t("lbl.confidence"),
            "flair": t("lbl.flair"), "route": t("lbl.route"),
        })
        st.dataframe(disp, use_container_width=True, hide_index=True, height=380)
        st.caption(t("se.disagreement_hint"))
    else:
        st.info(t("se.build_hint"))

    # ------------------------------------------------------------- gold 40 check
    st.divider()
    st.subheader(t("se.gold_title"))
    bench_path = CACHE / "sentiment_router_gold40.benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        b = st.columns(4)
        b[0].metric(t("se.gold_rows"), fmt_int(bench.get("matched_rows", 0)))
        b[1].metric(t("se.gold_agreement"), f"{bench.get('accuracy', 0):.1%}")
        b[2].metric(t("se.gold_mean_conf"), f"{bench.get('mean_confidence', 0):.2f}")
        hc = bench.get("accuracy_high_confidence_ge_0_8")
        b[3].metric(t("se.gold_hc"), f"{hc:.1%}" if hc is not None else "-")
        st.caption(t("se.gold_caption"))

    # ------------------------------------------------------------------- samples
    st.divider()
    st.subheader(t("se.sample_title"))
    which = st.radio(
        t("se.dataset"), ["llm", "hf"], horizontal=True, key="se_sample_which",
        format_func=lambda v: t("se.opt_llm") if v == "llm" else t("se.opt_hf"),
    )
    sample_cols = {
        "body": t("lbl.comment"), "label": t("lbl.label"), "llm_label": t("lbl.llm_label"),
        "confidence": t("lbl.confidence"), "route": t("lbl.route"),
        "comment_id": t("lbl.comment_id"), "score": t("lbl.score"),
        "created_wib": t("lbl.created"),
    }
    if which == "llm" and llm_sent is not None and not llm_sent.empty:
        sdf = llm_sent.copy()
        lcol = "label" if "label" in sdf.columns else "llm_label"
        if lcol in sdf.columns:
            sdf[lcol] = sdf[lcol].map(sent_labels).fillna(sdf[lcol])
        q = st.text_input(t("se.search_llm"), "", key="se_llm_q2")
        if q:
            sdf = sdf[sdf["body"].fillna("").str.contains(q, case=False, regex=False)]
        cols = [c2 for c2 in ["body", lcol, "score", "created_wib", "comment_id"] if c2 in sdf]
        st.dataframe(sdf[cols].head(400).rename(columns=sample_cols),
                     use_container_width=True, hide_index=True, height=340)
    else:
        sdf = load("sentiment_hf_10k") if (CACHE / "sentiment_hf_10k.parquet").exists() else pd.DataFrame()
        if sdf.empty:
            st.info(t("se.no_10k"))
        else:
            if "label" in sdf.columns:
                sdf["label"] = sdf["label"].map(sent_labels).fillna(sdf["label"])
            q = st.text_input(t("se.search_hf"), "", key="se_hf_q")
            if q:
                sdf = sdf[sdf["body"].fillna("").str.contains(q, case=False, regex=False)]
            cols = [c2 for c2 in ["body", "label", "confidence", "route", "comment_id"] if c2 in sdf]
            st.dataframe(sdf[cols].head(400).rename(columns=sample_cols),
                         use_container_width=True, hide_index=True, height=340)


METHOD_DOC: dict[str, str] = {
    "id": """\
Halaman ini menjelaskan cara kerja dashboard: apa isinya, apa arti tiap grafik, dan
jebakan interpretasi yang perlu diketahui sebelum menarik kesimpulan.

## Apa ini

Dashboard menganalisis **r/indonesia** dari snapshot hasil scraping — bukan tarikan
langsung dari Reddit. Seluruh waktu dikonversi ke **WIB (UTC+7)** dan nama pengguna
dianonimkan menjadi UUID, sehingga papan peringkat penulis berbasis UUID, bukan username
Reddit.

| Item | Nilai |
| --- | --- |
| Thread | {threads} |
| Komentar | {comments} |
| Rentang data | {span} |
| Zona waktu | WIB (UTC+7) |
| Rata-rata / median skor thread | {avg_score} / {median_score} |
| Thread otomatis (automod) | {auto_threads} thread, {auto_pct} dari seluruh komentar |

## Cara membaca: live vs snapshot

- **Live** — ikut berubah ketika filter di sidebar diubah. Ini mayoritas grafik.
- **Snapshot statis** — dihitung sekali di luar aplikasi dari berkas CSV dan
  **mengabaikan filter** (papan peringkat penulis, `flair_performance`,
  `flair_top_words`, grafik referensi WIB). Selalu diberi label di antarmuka.

## Isi tiap halaman

**📈 Ikhtisar** — lima metrik utama (thread, komentar, kontributor, total skor,
rata-rata skor), aktivitas dari waktu ke waktu, 15 flair teratas, sebaran skor thread,
dan scatter skor vs. komentar.

**🧵 Thread** — tiga tab: *Thread teratas* (urutkan berdasarkan skor / komentar / rasio
upvote / rata-rata skor komentar), *Jelajahi / cari* (cari di judul & isi, unduh CSV),
dan *Detail flair* (perbandingan metrik antar-flair).

**💬 Komentar** — statistik live komentar, volume dari waktu ke waktu, sebaran skor,
sebaran komentar per thread, 15 komentar berskor tertinggi & terendah, sebaran panjang
teks, dan daftar komentar terpanjang.

**👥 Penulis** — peringkat live yang menggabungkan thread yang dibuat dengan komentar
yang ditulis, plus scatter yang memisahkan "pembuat thread" dari "komentator". Tab kedua
menampilkan berkas papan peringkat statis.

**🏷️ Flair** — donut jumlah thread per flair, scatter profil interaksi (rata-rata
komentar vs. rata-rata skor), heatmap metrik ternormalisasi, kata teratas per flair, dan
referensi performa flair.

**🕐 Waktu** — heatmap hari × jam untuk thread dan komentar, jumlah thread per hari dan
per jam, **median balasan & skor per jam untuk thread organik**, dan heatmap referensi
statis.

**🎭 Sentimen** — cakupan dan distribusi label, kalibrasi (kesepakatan, Cohen's kappa,
matriks konfusi), sentimen per flair dan per bulan, penjelajah ketidaksepakatan, serta
benchmark emas 40 komentar.

## Tiga temuan utama

1. **r/indonesia mayoritas netral, bukan negatif.** Model mentah menyebut ~67% komentar
   negatif, tetapi setelah koreksi bias angkanya turun ke ~30% dan mendekati rujukan LLM
   independen. Distribusi mentah adalah artefak model, bukan sifat subreddit.
2. **Ada dua mode sukses.** *Mesin diskusi* (News, Ask Indonesian — banyak komentar,
   skor rendah) berbeda dari *mesin upvote* (Funny/Memes, Current Affair — skor tinggi,
   rasio upvote 0.92).
3. **Partisipasi sangat terpusat.** 1% komentator teratas menulis ~38% dari seluruh
   komentar, sementara ~34% komentator hanya muncul sekali.

## Jebakan interpretasi

1. **Pakai median, bukan rata-rata.** Skor dan jumlah komentar berekor sangat berat,
   sehingga segelintir thread viral menyeret rata-rata naik (rata-rata skor {avg_score}
   vs. median {median_score}).
2. **Kecualikan thread otomatis** untuk pertanyaan tentang perilaku manusia. Thread
   Daily/Weekend Chat hanya ~2% dari thread tetapi ~22% dari komentar, dan diposting
   otomatis pada jam tetap (07:00 WIB).
3. **Filter tanggal berlaku pada thread, bukan komentar.** Komentar mengikuti thread
   induknya, jadi komentar yang diposting setelah jendela waktu bisa ikut terhitung.
4. **Selisih angka komentar bukan bug.** {comments} adalah baris yang benar-benar ada di
   cache; total "aktual" pada kolom thread sedikit lebih besar karena sebagian komentar
   Reddit tidak ikut terscrape; angka korpus sentimen lebih kecil lagi karena penyaringan
   di skrip pelabelan.
5. **Skor menumpuk seiring waktu.** Bulan-bulan terakhir belum "matang" — thread belum
   sempat mengumpulkan upvote dan komentar — jadi tren waktu cenderung meremehkan
   periode terkini.
6. **Keyakinan model bukan akurasi.** Kesepakatan model vs. rubrik hanya 45.5% dan hampir
   tidak membaik pada keyakinan tinggi.
""",
    "en": """\
This page explains how the dashboard works: what it contains, what each chart means, and
the interpretation traps to know before drawing conclusions.

## What this is

The dashboard analyses **r/indonesia** from a scraping snapshot — not a live Reddit feed.
All times are converted to **WIB (UTC+7)** and usernames are anonymised to UUIDs, so the
author leaderboards are keyed by UUID rather than Reddit username.

| Item | Value |
| --- | --- |
| Threads | {threads} |
| Comments | {comments} |
| Data range | {span} |
| Timezone | WIB (UTC+7) |
| Mean / median thread score | {avg_score} / {median_score} |
| Auto-posted (automod) threads | {auto_threads} threads, {auto_pct} of all comments |

## How to read it: live vs snapshot

- **Live** — recomputes when you change the sidebar filters. This is most charts.
- **Static snapshot** — computed once outside the app from a CSV, and **ignores the
  filters** (author leaderboard, `flair_performance`, `flair_top_words`, the WIB reference
  chart). These are always labelled in the interface.

## What each page contains

**📈 Overview** — five headline metrics (threads, comments, contributors, total score,
mean score), activity over time, the top 15 flairs, the thread-score distribution, and a
score vs. comments scatter.

**🧵 Threads** — three tabs: *Top threads* (rank by score / comments / upvote ratio /
mean comment score), *Browse / search* (search title and body, download CSV), and *Flair
detail* (side-by-side flair metrics).

**💬 Comments** — live comment stats, volume over time, score distribution, comments per
thread, the 15 highest- and lowest-scored comments, text-length distribution, and the
longest comments.

**👥 Authors** — a live ranking combining threads created with comments written, plus a
scatter separating "thread creators" from "commenters". The second tab shows the static
leaderboard file.

**🏷️ Flairs** — a donut of threads per flair, an engagement-profile scatter (mean
comments vs. mean score), a normalised metric heatmap, top words per flair, and the
reference flair performance.

**🕐 Timing** — day × hour heatmaps for threads and comments, thread counts by day and by
hour, **median replies and score per hour for organic threads**, and the static reference
heatmap.

**🎭 Sentiment** — label coverage and distribution, calibration (agreement, Cohen's
kappa, confusion matrix), sentiment by flair and by month, a disagreement explorer, and a
40-comment gold benchmark.

## Three key findings

1. **r/indonesia is mostly neutral, not negative.** The raw model calls ~67% of comments
   negative, but after bias correction that drops to ~30%, close to the independent LLM
   reference. The raw distribution is a model artifact, not a property of the subreddit.
2. **There are two modes of success.** *Discussion engines* (News, Ask Indonesian — many
   comments, low scores) differ from *upvote engines* (Funny/Memes, Current Affair — high
   scores, 0.92 upvote ratio).
3. **Participation is heavily concentrated.** The top 1% of commenters write ~38% of all
   comments, while ~34% of commenters appear only once.

## Interpretation traps

1. **Use medians, not means.** Replies and scores are extremely heavy-tailed, so a
   handful of viral threads drags the mean upward (mean score {avg_score} vs. median
   {median_score}).
2. **Exclude auto-posted threads** for questions about human behaviour. Daily/Weekend
   Chat Threads are ~2% of threads but ~22% of comments, and are posted automatically on a
   fixed hour (07:00 WIB).
3. **The date filter applies to threads, not comments.** Comments follow their parent
   thread, so a comment posted after the window can still be counted.
4. **The comment-count differences are not a bug.** {comments} is the number of rows
   actually in the cache; the "actual" column total on threads is slightly higher because
   some Reddit comments were never scraped; the sentiment corpus is lower again because of
   the labelling script's filters.
5. **Scores accumulate over time.** Recent months are not yet "mature" — those threads
   have had less time to collect upvotes and comments — so time trends tend to understate
   the most recent period.
6. **Model confidence is not accuracy.** Model-vs-rubric agreement is only 45.5% and
   barely improves at high confidence.
""",
}


def page_methodology(th_all: pd.DataFrame, n_comments: int) -> None:
    """Plain-language guide to the dataset, the pages, and the caveats."""
    st.header(f"📖 {t('nav.methodology')}")

    if th_all.empty:
        st.info(t("msg.no_data"))
        return

    auto = th_all[th_all["link_flair_text"].isin(AUTO_POST_FLAIRS)]
    total_actual = th_all["total_comments_actual"].sum()
    vals = {
        "threads": fmt_int(len(th_all)),
        "comments": fmt_int(n_comments),
        "span": f"{fmt_date(th_all['created_wib'].min())} → {fmt_date(th_all['created_wib'].max())}",
        "avg_score": f"{th_all['score'].mean():.1f}",
        "median_score": f"{th_all['score'].median():.0f}",
        "auto_threads": fmt_int(len(auto)),
        "auto_pct": f"{auto['total_comments_actual'].sum() / max(total_actual, 1):.1%}",
    }
    st.markdown(METHOD_DOC.get(lang(), METHOD_DOC["en"]).format(**vals))


def main() -> None:
    # Theme comes first: the stylesheet and every chart below read the live
    # palette, and the toggle below only takes effect on the *next* run (which
    # Streamlit triggers as soon as it changes), so resolve it up front.
    if "theme_select" not in st.session_state:
        shared_theme = _qp("theme")
        if shared_theme in THEMES:
            st.session_state["theme"] = shared_theme
    apply_theme(theme())
    inject_css()

    if st.__version__ != STREAMLIT_PIN:
        st.warning(t("msg.pin").format(installed=st.__version__, pinned=STREAMLIT_PIN))

    if not cache_ready():
        st.title(t("msg.no_cache_title"))
        st.error(t("msg.no_cache"))
        st.code("python src/build_cache.py", language="bash")
        st.caption(t("msg.no_cache_expected").format(path=CACHE))
        if st.button(t("msg.build_now"), type="primary"):
            with st.spinner(t("msg.building")):
                import subprocess  # noqa: PLC0415
                import sys  # noqa: PLC0415

                res = subprocess.run(
                    [sys.executable, str(BASE / "src" / "build_cache.py"), "--force"],
                    capture_output=True, text=True, cwd=str(BASE),
                )
            if res.returncode == 0:
                st.success(t("msg.built"))
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(t("msg.build_failed"))
                st.code(res.stderr[-3000:] or res.stdout[-3000:])
        return

    th_all = load("threads")
    cm_all = load("comments")
    # `comments_text` is deliberately absent: it is 146 MB and only 30 rows of it
    # are ever on screen, so page_comments reads those bodies on demand.
    sent = load("sentiment")
    llm_sent = load("comment_sentiment_llm_10k") if (CACHE / "comment_sentiment_llm_10k.parquet").exists() else pd.DataFrame()
    meta = load_meta()

    # Language is picked first so the rest of the sidebar and every page below
    # render in it on the same run.
    st.sidebar.title("r/indonesia")
    if "lang_select" not in st.session_state:
        shared_lang = _qp("lang")
        if shared_lang in LANGS:
            st.session_state["lang_select"] = shared_lang
    # Seed the resolved language *before* anything translates, so a ?lang= link
    # renders in its own language on the first run rather than the second.
    st.session_state["lang"] = st.session_state.get("lang_select", lang())
    st.sidebar.selectbox(
        "Language / Bahasa",
        options=list(LANGS),
        format_func=lambda k: LANGS[k],
        index=list(LANGS).index(lang()),
        key="lang_select",
    )
    st.session_state["lang"] = st.session_state["lang_select"]

    st.sidebar.radio(
        t("sb.theme"),
        options=list(THEMES),
        format_func=lambda k: {"light": t("theme.light"), "dark": t("theme.dark")}[k],
        index=list(THEMES).index(theme()),
        horizontal=True,
        key="theme_select",
    )
    st.session_state["theme"] = st.session_state["theme_select"]

    st.sidebar.caption(t("sb.subtitle"))

    defaults = filter_defaults(th_all)
    f = sidebar_filters(th_all, cm_all, defaults)

    th = apply_filters(th_all, f)
    # Comments are kept to the threads that survived filtering (join on thread_id).
    keep_ids = set(th["thread_id"])
    cm = cm_all[cm_all["thread_id"].isin(keep_ids)]
    # The filtered frame is what every page uses; only its total row count is
    # needed from the full one, so let the 150 MB original go.
    n_comments_all = len(cm_all)
    del cm_all
    _release_memory()

    st.sidebar.metric(t("sb.threads_selected"), fmt_int(len(th)))
    st.sidebar.metric(t("sb.comments_selected"), fmt_int(len(cm)))
    st.sidebar.caption(
        t("sb.pct_of").format(
            tp=f"{len(th) / max(len(th_all), 1):.1%}",
            cp=f"{len(cm) / max(n_comments_all, 1):.1%}",
        )
    )
    if st.sidebar.button(t("sb.clear_cache"), use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    pages = {
        t("nav.overview"): lambda: page_overview(th, cm, f),
        t("nav.threads"): lambda: page_threads(th),
        t("nav.comments"): lambda: page_comments(th, cm, f),
        t("nav.authors"): lambda: page_authors(th, cm),
        t("nav.flairs"): lambda: page_flairs(th),
        t("nav.timing"): lambda: page_timing(th, cm),
        t("nav.sentiment"): lambda: page_sentiment(sent, llm_sent),
        t("nav.methodology"): lambda: page_methodology(th_all, n_comments_all),
    }
    tabs = st.tabs(list(pages.keys()))
    for tab, render in zip(tabs, pages.values()):
        with tab:
            render()

    st.divider()
    if meta:
        s = meta.get("sources", {})
        st.caption(
            t("ft.built").format(
                at=meta.get("built_at", "?"),
                th=fmt_int(s.get("threads", {}).get("rows", 0)),
                cm=fmt_int(s.get("comments", {}).get("rows", 0)),
            )
        )


if __name__ == "__main__":
    main()
