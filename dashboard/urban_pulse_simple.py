"""Urban Infrastructure Pulse — simple single-page Streamlit GUI (refreshed UI).

Place in: <project root>/dashboard/urban_pulse_simple.py
Theme:    <project root>/.streamlit/config.toml  (dark theme so native widgets match)
Run from project root:
    python -m streamlit run dashboard/urban_pulse_simple.py

Uses the project's existing, validated geographic, historical and three-model
pipeline. It does not write uploaded CSVs or modify historical models.
Only presentation changed: every backend call and session_state key is the same
as before. Requires Streamlit >= 1.36 (already needed for st.navigation).
"""

from __future__ import annotations

import hashlib
import math
import sys
from datetime import date
from html import escape as safe_html
from io import BytesIO
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.upload_preflight import run_upload_preflight
from src.integration.bulk_complaint_contract import REQUIRED_COLUMNS
from src.integration.live_uploaded_batch_pipeline import (
    load_upload_baseline,
    process_uploaded_csv_bytes,
)
from src.integration.investigation_review_queue import build_review_queue
from src.ui.service_factory import get_gui_adapter

st.set_page_config(
    page_title="Urban Infrastructure Pulse",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEMPLATE = ROOT / "outputs/templates/new_complaints_template.csv"
TAXONOMY = ROOT / "outputs/tables/dbscan/dbscan_action_taxonomy_v1.csv"
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 5_000

# Review-order tiers. Colours mark review order, not emergency severity.
TIERS = {
    1: {"short": "Two signals", "long": "Two signals · review first",
        "emoji": "🔴", "hex": "#ff8a7a", "rgb": [255, 138, 122]},
    2: {"short": "Reporting increase", "long": "Historical reporting increase",
        "emoji": "🟠", "hex": "#ffc46b", "rgb": [255, 196, 107]},
    3: {"short": "Spatial concentration", "long": "New spatial concentration",
        "emoji": "🔵", "hex": "#6cb8ff", "rgb": [108, 184, 255]},
}


def _tier(item) -> dict:
    return TIERS.get(int(item["tier"]), TIERS[3])


# --------------------------------------------------------------------------
# Backend wiring (unchanged)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def supported_categories() -> list[str]:
    """Use the existing project taxonomy, with a helpful schema error."""
    taxonomy = pd.read_csv(TAXONOMY, dtype="string")
    if "complaint_category" not in taxonomy.columns:
        raise ValueError(
            "Project taxonomy is missing 'complaint_category'. "
            f"Available taxonomy columns: {taxonomy.columns.tolist()}. "
            "Restore the previously verified taxonomy CSV."
        )
    values = taxonomy["complaint_category"]
    return sorted({str(v).strip().upper() for v in values.dropna()})


def check_uploaded_csv_columns(data: bytes) -> list[str]:
    """Give users the exact missing CSV headers before backend validation."""
    if not data:
        raise ValueError("The uploaded CSV is empty. Select your complete complaints CSV.")
    try:
        header = pd.read_csv(BytesIO(data), nrows=0, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError("This file is not a readable CSV. Use the CSV template.") from exc
    found = header.columns.tolist()
    missing = [name for name in REQUIRED_COLUMNS if name not in found]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required column(s): "
            + ", ".join(missing)
            + ". Columns found: " + (", ".join(found) if found else "none")
            + ". Download the CSV template and check your header names."
        )
    return found


@st.cache_resource(show_spinner=False)
def existing_backend_adapter():
    return get_gui_adapter()


@st.cache_resource(show_spinner=False)
def existing_historical_baseline():
    return load_upload_baseline(ROOT)


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
button, input, textarea, select {font-family:'DM Sans','Segoe UI',sans-serif}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 50% 25% at 90% 0%, #10415355, transparent 80%),
              radial-gradient(ellipse 40% 25% at 0% 30%, #213b6850, transparent 80%), #081420;
  color: #eaf4f8;
}
[data-testid="stHeader"] {background:transparent}
[data-testid="stAppViewContainer"] .block-container {max-width:1170px; padding-top:1.4rem; padding-bottom:4rem}
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {display:none}
#MainMenu,footer {visibility:hidden}
h1,h2,h3 {font-family:'Manrope','Segoe UI',sans-serif !important;letter-spacing:-.035em}
[data-testid="stMarkdownContainer"] p {line-height:1.65}
[data-testid="stFileUploader"] {border:1px dashed #39c6bc77;border-radius:15px;background:#12304166;padding:.5rem;
 transition:border-color .2s, background .2s}
[data-testid="stFileUploader"]:hover {border-color:#39c6bccc;background:#14384a88}
[data-testid="stFileUploaderDropzone"] {background:transparent}
[data-testid="stExpander"] {border-color:#526f7f55 !important;border-radius:12px}
div.stButton>button {min-height:3rem;border-radius:12px;font-weight:800;transition:transform .15s, box-shadow .15s}
div.stButton>button:hover:not(:disabled) {transform:translateY(-1px)}
div.stButton>button[kind="primary"] {background:linear-gradient(110deg,#32dac5,#16a3b6)!important;
 color:#032733!important;border:0;box-shadow:0 10px 25px #09afaa33}
div.stButton>button[kind="primary"]:hover:not(:disabled) {box-shadow:0 14px 32px #09afaa55}
div.stButton>button[kind="secondary"] {background:#143943!important;color:#d7fff5!important;border:1px solid #36c2ba88}
div.stButton>button:disabled {opacity:.45}
div.stDownloadButton>button {border-radius:11px;background:#12313c!important;
 color:#c9fff1!important;border:1px solid #43bfb677}
[data-testid="stDataFrame"] {border:1px solid #527e8655;border-radius:12px;overflow:hidden}
[data-testid="stDeckGlJsonChart"] {border:1px solid #527e8655;border-radius:14px;overflow:hidden}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {gap:.35rem;border-bottom:1px solid #2f556355}
[data-testid="stTabs"] button[role="tab"] {padding:.55rem 1rem;border-radius:10px 10px 0 0;font-weight:700}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {background:#12424877;color:#e1fff9}

/* Hero (full on first visit, slim once a file is selected) */
.hero {display:flex;align-items:center;justify-content:space-between;gap:1.4rem;
 border:1px solid #47cfc366;border-radius:22px; padding:2rem 2.25rem;margin:0 0 1rem;
 background:linear-gradient(110deg,#14354d,#0d293b 68%,#11484b);
 box-shadow:0 20px 55px #050c1755}
.brand {font-size:.72rem;letter-spacing:.18em;font-weight:800;color:#6de9d9;text-transform:uppercase}
.hero h1 {font-size:clamp(2rem,3.6vw,3.15rem);line-height:1.12;margin:.55rem 0;letter-spacing:-.055em;color:white}
.hero p {font-size:.98rem;color:#b1c9d6;line-height:1.65;margin:0;max-width:670px}
.skyline {display:flex;align-items:flex-end;gap:8px;height:125px;opacity:.8}
.skyline i {display:block;width:19px;background:linear-gradient(#36c8bb55,#245f8033);
 border:1px solid #57dac777;border-radius:4px 4px 0 0;border-bottom:2px solid #29b9b8}
.topbar {display:flex;align-items:center;justify-content:space-between;gap:1rem;
 border:1px solid #47cfc355;border-radius:14px;padding:.8rem 1.2rem;margin:0 0 .6rem;
 background:linear-gradient(110deg,#14354d,#0d293b 70%,#11484b)}
.topbar .brand {font-size:.78rem}
.topbar span {color:#9fbccb;font-size:.82rem}

/* Step bar, kept visible while scrolling */
[data-testid="stElementContainer"]:has(.steps), .element-container:has(.steps) {
 position:sticky;top:3.6rem;z-index:50}
.steps {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:.6rem 0 1.1rem;
 padding:6px;border-radius:15px;background:#081420dd;backdrop-filter:blur(8px)}
.step {display:flex;align-items:center;gap:.7rem;padding:.75rem 1rem;border:1px solid #4e718455;
 background:#102a3a;border-radius:12px;color:#93adbe;font-weight:750;font-size:.86rem}
.step b {flex:none;width:30px;height:30px;border-radius:50%;background:#1c4151;color:#93c5d0;display:grid;place-items:center;font-size:.74rem}
.step.done {border-color:#3ce0c255;color:#c9efe8;background:#11363d77}
.step.active {border-color:#3ce0c2aa;color:#e1fff9;background:#12484d;box-shadow:0 0 0 3px #2fd3bd1f}
.step.done b,.step.active b {background:#20c5b4;color:#032d32}

/* Sections */
.sectiontitle {display:flex;align-items:center;gap:.75rem;margin:1.35rem 0 .7rem}
.sectiontitle i {font-style:normal;font-size:.83rem;background:#16454a;color:#6ce5d6;font-weight:800;
 border-radius:10px;padding:.45rem .64rem}
.sectiontitle h2 {font-size:1.3rem;margin:0;color:#f0faff}
.sectiontitle.locked {opacity:.42}
.sectiontitle.locked i {background:#1a3341;color:#8aa8b6}
.sectiontitle .lock {font-size:.75rem;color:#8fb0bf;font-weight:700;margin-left:.2rem}
.muted {color:#99b4c4;font-size:.87rem;line-height:1.6}
.info-panel {background:#11303b;border:1px solid #3a777b66;border-radius:13px;padding:.9rem 1rem;color:#bde3e1;font-size:.88rem;line-height:1.65}

/* Stats */
.stat-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin:1rem 0}
.stat {border:1px solid #4b7c8655;border-radius:15px;padding:1rem;background:linear-gradient(145deg,#173548,#0e2637);
 transition:transform .2s, border-color .2s}
.stat:hover {transform:translateY(-2px);border-color:#4fd2c277}
.stat.highlight {background:linear-gradient(130deg,#145251,#10414a);border-color:#39d2bd77}
.stat small {font-size:.75rem;color:#a6c0cf;display:block;min-height:2.15rem;font-weight:700}
.stat strong {display:block;font-size:clamp(1.55rem,2.8vw,2.5rem);font-family:'Manrope',sans-serif;color:white;letter-spacing:-.055em}
.stat.highlight strong {color:#87ffe7}
.stat span {color:#68d3c5;font-size:.73rem}
.meter {height:5px;border-radius:5px;background:#0b2230;margin:.55rem 0 .35rem;overflow:hidden}
.meter div {height:100%;border-radius:5px;background:linear-gradient(90deg,#32dac5,#16a3b6)}

/* Pills and chips */
.pill {display:inline-block;border:1px solid #26c9b277;color:#91f9d9;background:#124a465e;
 border-radius:100px;padding:.3rem .6rem;font-size:.69rem;font-weight:800;letter-spacing:.04em;margin-right:.35rem}
.pill.amber {color:#ffdc97;border-color:#dba55077;background:#634c2350}
.chips {display:flex;flex-wrap:wrap;gap:8px;margin:.4rem 0 .6rem}
.chip {border-radius:100px;padding:.32rem .75rem;font-size:.78rem;font-weight:700;border:1px solid}
.chip.ok {color:#91f9d9;border-color:#26c9b277;background:#124a465e}
.chip.warn {color:#ffdc97;border-color:#dba55077;background:#634c2350}
.chip.bad {color:#ffb3a8;border-color:#e2766a77;background:#5a2a2650}
.legend {display:flex;flex-wrap:wrap;gap:14px;margin:.2rem 0 .7rem;color:#a9c3d0;font-size:.8rem}
.legend span {display:inline-flex;align-items:center;gap:.4rem}
.legend i {width:11px;height:11px;border-radius:50%;display:inline-block}

/* Location detail */
.detail-head {background:linear-gradient(130deg,#163b50,#0e283b);border:1px solid #46d5c566;
 border-radius:16px;padding:1.3rem 1.5rem;margin:.5rem 0 1rem;border-left-width:5px}
.detail-head small {letter-spacing:.13em;font-size:.72rem;font-weight:850}
.detail-head h3 {color:white;font-size:1.55rem;margin:.4rem 0}
.detail-head p {color:#b4cbd9;margin:0;font-size:.87rem}
.signals {display:flex;flex-wrap:wrap;gap:8px;margin:.85rem 0 0}
.sig {display:inline-flex;align-items:center;gap:.45rem;border-radius:100px;padding:.3rem .75rem;
 font-size:.76rem;font-weight:750;border:1px solid}
.sig.on {color:#c9fff3;border-color:#35d0b388;background:#13504a88}
.sig.on::before {content:"";width:8px;height:8px;border-radius:50%;background:#3ff0d0;box-shadow:0 0 8px #3ff0d0}
.sig.off {color:#7f99a7;border-color:#3f5b6955;background:#0f2533}
.sig.off::before {content:"";width:8px;height:8px;border-radius:50%;background:#3d5664}
.sig.ctx {color:#cfe3ee;border-color:#6d8fa255;background:#142f40}
.sig.ctx::before {content:"";width:8px;height:8px;border-radius:2px;background:#8fb6cc}
.evidence {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin:1rem 0}
.evidence article {background:#102b3c;border:1px solid #3b6a795c;border-radius:13px;padding:1rem;
 transition:transform .2s, border-color .2s}
.evidence article:hover {transform:translateY(-2px);border-color:#4fd2c266}
.evidence b {display:block;font-size:.73rem;letter-spacing:.07em;text-transform:uppercase;color:#68ded1;margin-bottom:.45rem}
.evidence p {color:#d7eaf3;font-size:.86rem;line-height:1.6;margin:0}
.bars {margin:.7rem 0 0}
.bar {display:flex;align-items:center;gap:.55rem;margin:.32rem 0}
.bar span {width:112px;flex:none;font-size:.74rem;color:#a6c0cf}
.bar div {flex:1;background:#0b2230;border-radius:6px;height:10px;overflow:hidden}
.bar div i {display:block;height:100%;border-radius:6px}
.bar em {width:30px;font-style:normal;text-align:right;color:white;font-size:.82rem;font-weight:800}
.reason {color:#e5f1f7;line-height:1.72;font-size:.96rem;margin:.5rem 0 1.15rem}
.kicker {color:#77e1d5;letter-spacing:.08em;font-size:.76rem;text-transform:uppercase;font-weight:850;margin:.9rem 0 .25rem}
.action {background:#12504755;border-left:4px solid #35d0a9;border-radius:9px;padding:1rem 1.15rem;margin:.5rem 0}
.action p {color:#e4fff4;margin:.4rem 0;line-height:1.7}

@media(max-width:850px){.skyline{display:none}.steps{grid-template-columns:repeat(2,1fr)}
 .stat-grid{grid-template-columns:repeat(2,1fr)}.evidence{grid-template-columns:1fr}.hero{padding:1.5rem}}
@media(max-width:420px){.stat-grid{grid-template-columns:1fr}.steps{grid-template-columns:1fr}}
</style>
"""


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------

def _html(text):
    return safe_html(str(text), quote=True)


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _number(value):
    """Return a finite float, or None for missing / non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _estimate_rows(data: bytes) -> int:
    """Rough data-row count (line count minus header) for an early sanity check."""
    lines = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
    return max(lines - 1, 0)


def _show_error(stage: str, exc: Exception):
    st.error(f"{stage} failed. {exc}")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}", language=None)


def _hero(compact: bool):
    if compact:
        _md(
            '<div class="topbar"><div class="brand">◈ URBAN INFRASTRUCTURE PULSE</div>'
            '<span>Evidence-backed investigation recommendations from complaint CSVs</span></div>'
        )
        return
    _md(
        '<div class="hero"><div><div class="brand">◈ URBAN INFRASTRUCTURE PULSE</div>'
        '<h1>Know where to<br>investigate first.</h1>'
        '<p>Turn complaint CSV files into clear, evidence-backed investigation '
        'recommendations. Upload, validate, analyze, and see the suggested next action.</p>'
        '</div><div class="skyline" aria-hidden="true">'
        '<i style="height:53px"></i><i style="height:88px"></i><i style="height:65px"></i>'
        '<i style="height:117px"></i><i style="height:83px"></i><i style="height:103px"></i>'
        '</div></div>'
    )


def _section(number, name, locked_hint: str | None = None):
    if locked_hint:
        _md(
            f'<div class="sectiontitle locked"><i>{_html(number)}</i><h2>{_html(name)}</h2>'
            f'<span class="lock">🔒 {_html(locked_hint)}</span></div>'
        )
        return
    _md(f'<div class="sectiontitle"><i>{_html(number)}</i><h2>{_html(name)}</h2></div>')


def _steps_html(stage) -> str:
    labels = ["Upload complaints", "Validate the CSV", "Analyze", "Review advice"]
    parts = []
    for i, label in enumerate(labels, 1):
        cl = "done" if i < stage else "active" if i == stage else ""
        mark = "✓" if i < stage else f"{i:02d}"
        parts.append(f'<div class="step {cl}"><b>{mark}</b>{_html(label)}</div>')
    return '<div class="steps">' + "".join(parts) + "</div>"


def _stats(data):
    cards = []
    for label, n, sub, highlighted, meter in data:
        bar = (f'<div class="meter"><div style="width:{max(0, min(meter, 1)) * 100:.0f}%"></div></div>'
               if meter is not None else "")
        cards.append(
            f'<div class="stat {"highlight" if highlighted else ""}">'
            f'<small>{_html(label)}</small><strong>{_html(n)}</strong>{bar}<span>{_html(sub)}</span></div>'
        )
    _md('<div class="stat-grid">' + "".join(cards) + "</div>")


def _status_chips(counts: dict):
    chips = []
    for key, value in counts.items():
        k = str(key).lower()
        tone = ("bad" if any(w in k for w in ("reject", "invalid", "error", "fail"))
                else "warn" if any(w in k for w in ("warn", "review", "flag", "duplicate"))
                else "ok")
        chips.append(f'<span class="chip {tone}">{_html(key)} · {_html(value)}</span>')
    _md('<div class="chips">' + "".join(chips) + "</div>")


def _tier_legend():
    _md('<div class="legend">' + "".join(
        f'<span><i style="background:{t["hex"]}"></i>{_html(t["short"])}</span>'
        for t in TIERS.values()
    ) + '<span style="opacity:.75">Colours show review order, not emergency severity.</span></div>')


def _compare_bars(now, seasonal, weekday, weekday_plural) -> str:
    values = [("Now", _number(now), "#87ffe7"),
              ("Seasonal max", _number(seasonal), "#4f8aa3"),
              (f"All {weekday_plural}", _number(weekday), "#4f8aa3")]
    values = [(lbl, v, c) for lbl, v, c in values if v is not None]
    if len(values) < 2:
        return ""
    top = max(v for _, v, _ in values) or 1
    rows = "".join(
        f'<div class="bar"><span>{_html(lbl)}</span><div><i style="width:{v / top * 100:.0f}%;'
        f'background:{c}"></i></div><em>{v:g}</em></div>'
        for lbl, v, c in values
    )
    return f'<div class="bars">{rows}</div>'


def _feature_coords(feed) -> dict:
    coords = {}
    for feature in feed["map_geojson"]["features"]:
        try:
            lon, lat = feature["geometry"]["coordinates"][:2]
            coords[str(feature["id"])] = (float(lon), float(lat))
        except (KeyError, TypeError, ValueError):
            continue
    return coords


def _overview_map(items, feed):
    coords = _feature_coords(feed)
    rows = []
    for item in items:
        tier = _tier(item)
        for sid in map(str, item.get("linked_spatial_ids", [])):
            if sid in coords:
                lon, lat = coords[sid]
                rows.append({
                    "lon": lon, "lat": lat, "color": tier["rgb"],
                    "label": f'{item["review_order"]}. {item["location"]}',
                    "issue": str(item["category"]).title(), "evidence": tier["short"],
                })
    if not rows:
        st.info("No mappable coordinates are linked to these findings.")
        return
    points = pd.DataFrame(rows)
    _tier_legend()
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=float(points["lat"].mean()), longitude=float(points["lon"].mean()),
            zoom=10.3, pitch=0,
        ),
        layers=[pdk.Layer(
            "ScatterplotLayer", points, get_position="[lon, lat]",
            get_fill_color="color", get_radius=110, radius_min_pixels=4,
            stroked=True, get_line_color=[8, 20, 32], line_width_min_pixels=1,
            opacity=0.85, pickable=True,
        )],
        tooltip={"html": "<b>{label}</b><br/>{issue} · {evidence}",
                 "style": {"backgroundColor": "#0e2637", "color": "#eaf4f8"}},
        map_style=None,
    ), use_container_width=True, height=470)
    st.caption("Representative submitted coordinates, not verified fault locations.")



def _history_note_for_review(item, comparison):
    """Describe historical routine for linked investigation areas.

    Prefer comparable same-quarter weekdays.
    Fall back to all matching weekdays when supported.
    Never infer an area's history from another area.
    """

    import re

    if (
        not isinstance(comparison, dict)
        or comparison.get("status") != "compared"
    ):
        return "Historical comparison unavailable."

    summary = comparison.get("summary")

    if (
        not isinstance(summary, pd.DataFrame)
        or summary.empty
        or "area_id" not in summary.columns
    ):
        return "Historical comparison unavailable."

    known = set(summary["area_id"].astype(str))
    linked_areas = set()

    def collect(value):
        if isinstance(value, str):
            for candidate in re.split(r"[,;|]", value):
                candidate = candidate.strip().upper()
                if candidate in known:
                    linked_areas.add(candidate)

        elif isinstance(value, (list, tuple, set)):
            for member in value:
                collect(member)

    collect(item.get("area_id"))
    collect(item.get("area_ids"))

    if not linked_areas:
        return (
            "No reliable historical area match "
            "for this investigation."
        )

    descriptions = []

    for area_id in sorted(linked_areas):

        area_rows = summary.loc[
            summary["area_id"].eq(area_id)
        ]

        chosen = None
        reference_label = None

        for window, label in [
            (
                "same_quarter_weekdays",
                "similar-season weekdays",
            ),
            (
                "all_matching_weekdays",
                "similar weekdays",
            ),
        ]:
            candidates = area_rows.loc[
                area_rows["reference_window"].eq(window)
                &
                area_rows["comparison_status"].eq(
                    "supported"
                )
            ]

            if not candidates.empty:
                chosen = candidates.iloc[0]
                reference_label = label
                break

        if chosen is None:
            descriptions.append(
                f"{area_id}: Not enough historical data."
            )
            continue

        category = str(
            chosen["historical_top_category"]
        ).title()

        reports = int(
            chosen["historical_reports"]
        )

        descriptions.append(
            f"{area_id}: {category} "
            f"({reference_label}, 2023–2025; "
            f"{reports} historical reports)"
        )

    return " | ".join(descriptions)



# PHASE 21.7C — K-MEANS CLUSTER EXPLORER

def _pulse_new_kmeans_cluster_areas(integrated):
    """Read new-batch model assignments, not investigation flags."""

    import pandas as pd

    specifications = [
        (
            "area_id",
            "new_batch_cluster",
            "new_batch_description",
            "incoming_complaints",
        ),
        (
            "new_kmeans_area_id",
            "new_kmeans_cluster",
            "new_kmeans_description",
            "new_kmeans_area_complaints",
        ),
        (
            "kmeans_area_id",
            "kmeans_cluster_id",
            "kmeans_reporting_profile",
            None,
        ),
    ]

    candidates = []

    def visit(value, path="integrated", depth=0):
        if depth > 6:
            return

        if isinstance(value, pd.DataFrame):
            for area_col, id_col, label_col, count_col in specifications:

                required = {
                    area_col, id_col, label_col
                }

                if not required.issubset(value.columns):
                    continue

                cols = [area_col, id_col, label_col]
                has_counts = (
                    count_col is not None
                    and count_col in value.columns
                )

                if has_counts:
                    cols.append(count_col)

                frame = value[cols].copy()

                frame = frame.rename(columns={
                    area_col: "area_id",
                    id_col: "cluster_id",
                    label_col: "reporting_profile",
                })

                if has_counts:
                    frame = frame.rename(columns={
                        count_col: "complaints"
                    })
                else:
                    frame["complaints"] = pd.NA

                frame["area_id"] = (
                    frame["area_id"]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                )

                frame["reporting_profile"] = (
                    frame["reporting_profile"]
                    .astype("string")
                    .str.strip()
                )

                frame["cluster_id"] = pd.to_numeric(
                    frame["cluster_id"],
                    errors="coerce",
                )

                frame["complaints"] = pd.to_numeric(
                    frame["complaints"],
                    errors="coerce",
                )

                frame = frame.dropna(
                    subset=[
                        "area_id",
                        "cluster_id",
                        "reporting_profile",
                    ]
                )

                frame = frame.loc[
                    frame["area_id"].str.len().gt(0)
                    &
                    frame["reporting_profile"].str.len().gt(0)
                ].copy()

                if frame.empty:
                    continue

                # A geographic area must have exactly
                # one cluster and profile in this batch.

                if (
                    frame.groupby("area_id")[
                        "cluster_id"
                    ].nunique().gt(1).any()
                ):
                    raise ValueError(
                        "Conflicting K-Means assignments "
                        "for one geographic area."
                    )

                if (
                    frame.groupby("area_id")[
                        "reporting_profile"
                    ].nunique().gt(1).any()
                ):
                    raise ValueError(
                        "Conflicting K-Means profile labels "
                        "for one geographic area."
                    )

                if (
                    frame.groupby("cluster_id")[
                        "reporting_profile"
                    ].nunique().gt(1).any()
                ):
                    raise ValueError(
                        "One cluster ID has conflicting "
                        "reporting-profile descriptions."
                    )

                frame = (
                    frame.sort_values("area_id")
                    .drop_duplicates("area_id")
                    .reset_index(drop=True)
                )

                priority = (
                    100 if "area_summary" in path else
                    50 if "new_kmeans" in path else
                    10
                )

                candidates.append((
                    len(frame),
                    priority,
                    path,
                    frame,
                ))

        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(
                    child,
                    (dict, list, tuple, pd.DataFrame),
                ):
                    visit(
                        child,
                        f"{path}.{key}",
                        depth + 1,
                    )

        elif isinstance(value, (list, tuple)):
            if value and all(
                isinstance(row, dict) for row in value
            ):
                visit(
                    pd.DataFrame(value),
                    path,
                    depth + 1,
                )

    visit(integrated)

    if not candidates:
        return {
            "status": "unavailable",
            "reason": (
                "No complete new-batch K-Means "
                "area-assignment table was found."
            ),
            "areas": pd.DataFrame(),
        }

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, _, source_name, areas = candidates[0]

    return {
        "status": "clustered",
        "source": source_name,
        "areas": areas,
    }


def _pulse_show_cluster_explorer(explorer):
    """Display every area in the selected K-Means cluster."""

    st.markdown("### Explore New-Batch K-Means Clusters")

    st.caption(
        "Select a one-day reporting pattern to see all "
        "geographic areas assigned to that cluster. "
        "These groups describe complaint types, "
        "not infrastructure damage or repair priorities."
    )

    if (
        not isinstance(explorer, dict)
        or explorer.get("status") != "clustered"
    ):
        reason = (
            explorer.get("reason", "No cluster data.")
            if isinstance(explorer, dict)
            else "No cluster data."
        )

        st.info(
            "New-batch K-Means cluster memberships "
            "are unavailable for this uploaded batch. "
            + str(reason)
        )
        return

    areas = explorer["areas"]

    if areas.empty:
        st.info("No assigned K-Means areas to display.")
        return

    profiles = (
        areas[["cluster_id", "reporting_profile"]]
        .drop_duplicates()
        .sort_values("cluster_id")
    )

    labels = {
        row.cluster_id: str(row.reporting_profile)
        for row in profiles.itertuples(index=False)
    }

    selected_cluster = st.selectbox(
        "Choose a K-Means reporting pattern",
        options=list(labels.keys()),
        format_func=lambda cluster_id: (
            labels[cluster_id]
        ),
        key="pulse_kmeans_cluster_explorer",
    )

    selected_areas = (
        areas.loc[
            areas["cluster_id"].eq(selected_cluster)
        ]
        .sort_values("area_id")
        .copy()
    )

    st.markdown(
        f"**{labels[selected_cluster]}**"
    )

    left, right = st.columns(2)

    with left:
        st.metric(
            "Areas in this cluster",
            len(selected_areas),
        )

    with right:
        if selected_areas["complaints"].notna().all():
            st.metric(
                "Complaints in these areas",
                int(selected_areas["complaints"].sum()),
            )
        else:
            st.metric(
                "Complaints in these areas",
                "Unavailable",
            )

    selected_areas["Borough / ZIP"] = (
        selected_areas["area_id"]
        .str.replace("_", " · ZIP ", n=1, regex=False)
        .str.title()
    )

    display_table = selected_areas[
        ["Borough / ZIP", "complaints"]
    ].rename(columns={
        "complaints": "Incoming complaints"
    })

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "All model-assigned areas in this cluster "
        "are shown, including areas without "
        "DBSCAN or anomaly investigation signals."
    )



# PHASE 22.3 — PDF-STYLE HISTORICAL CONTEXT
@st.cache_data(show_spinner=False)
def _pulse_pdf_candidate_table(path_text, file_modified_ns):
    """Read saved retrospective evidence; no retraining, scoring or rewriting."""
    frame = pd.read_csv(path_text)
    needed = {"area_id", "complaint_type", "screening_result", "category_reports",
              "repeat_rate", "peer_repeat_median", "repeat_peer_count",
              "median_closure_days", "peer_closure_median_days", "closure_peer_count"}
    if not needed.issubset(frame.columns):
        raise ValueError("PDF historical screen has missing columns")
    if frame.duplicated(["area_id", "complaint_type"]).any():
        raise ValueError("PDF historical screen has duplicate area/category pairs")
    frame["area_id"] = frame["area_id"].astype(str).str.strip().str.upper()
    frame["complaint_type"] = frame["complaint_type"].astype(str).str.strip().str.upper()
    allowed = {"repeat_only_candidate", "closure_only_candidate",
               "repeat_and_closure_candidate"}
    return frame.loc[frame["screening_result"].isin(allowed)].copy()


def _pulse_pdf_history_for_review(item, candidate_table):
    """Exact matching only: an item must have both area ID and complaint category."""
    import re
    if not isinstance(candidate_table, pd.DataFrame) or candidate_table.empty:
        return []

    area_ids = set()
    for field in ("area_id", "area_ids"):
        value = item.get(field)
        if isinstance(value, (list, tuple, set)):
            chunks = [str(v) for v in value]
        elif isinstance(value, str):
            chunks = [value]
        else:
            chunks = []
        for chunk in chunks:
            area_ids.update(re.findall(
                r"(?:BRONX|BROOKLYN|MANHATTAN|QUEENS|STATEN ISLAND)_[0-9]{5}",
                chunk.upper()
            ))

    category = str(item.get("category") or "").strip().upper()
    if not area_ids or not category:
        return []

    matches = candidate_table.loc[
        candidate_table["area_id"].isin(area_ids) &
        candidate_table["complaint_type"].eq(category)
    ]

    descriptions = []
    for row in matches.sort_values("area_id").itertuples(index=False):
        kind = row.screening_result
        parts = []
        if kind in ("repeat_only_candidate", "repeat_and_closure_candidate"):
            parts.append(
                f"potential repeat-report rate {100*float(row.repeat_rate):.1f}% "
                f"vs peer median {100*float(row.peer_repeat_median):.1f}% "
                f"({int(row.repeat_peer_count)} peers)"
            )
        if kind in ("closure_only_candidate", "repeat_and_closure_candidate"):
            parts.append(
                f"median recorded closure {float(row.median_closure_days):.1f} days "
                f"vs peer median {float(row.peer_closure_median_days):.1f} days "
                f"({int(row.closure_peer_count)} peers)"
            )
        if parts:
            descriptions.append(
                f"{row.area_id} · {category}: " + "; ".join(parts) + "."
            )
    return descriptions

def _show_location_review(item, feed, weekday_plural, comparison=None):
    tier = _tier(item)
    order = int(item["review_order"])
    issue = str(item["category"]).title()
    spatial_only = item["kind"] == "spatial_only"
    patterns = ", ".join(item.get("kmeans_patterns", [])) or "Insufficient area evidence"
    nearby = int(item.get("spatial_overlap_reports", 0) or 0)

    # Which screening signals are behind this item (derived from fields already present).
    dbscan_on = spatial_only or nearby > 0
    anomaly_on = not spatial_only
    has_patterns = bool(item.get("kmeans_patterns"))
    signals = (
        f'<span class="sig {"on" if dbscan_on else "off"}">Nearby reports · DBSCAN</span>'
        f'<span class="sig {"on" if anomaly_on else "off"}">Unusual activity · Anomaly</span>'
        f'<span class="sig ctx">Area pattern · K-Means'
        f'{"" if has_patterns else " (limited)"}</span>'
    )
    _md(
        f'<div class="detail-head" style="border-left-color:{tier["hex"]}">'
        f'<small style="color:{tier["hex"]}">INVESTIGATION {order:02d} · {_html(tier["long"].upper())}</small>'
        f'<h3>{_html(item["location"])}</h3><p>{_html(issue)} · {_html(item["incoming_reports"])} '
        'complaints reported · Human review recommended</p>'
        f'<div class="signals">{signals}</div></div>'
    )

    if spatial_only:
        geographic = (f'{item["incoming_reports"]} reports form spatial grouping '
                      f'{item["candidate_id"]}. The precise fault location is unconfirmed.')
        historical = "No qualifying historical reporting-count anomaly is linked to this grouping."
        bars = ""
    else:
        geographic = (
            f"{nearby} of {item['incoming_reports']} complaints belong to new nearby-report "
            "concentrations." if nearby else "No linked spatial concentration was detected."
        )
        historical = (
            f"{item['incoming_reports']} reports now vs. historical maxima of "
            f"{item['seasonal_max']} (seasonal) and {item['all_weekday_max']} "
            f"(all {weekday_plural})."
        )
        bars = _compare_bars(item["incoming_reports"], item.get("seasonal_max"),
                             item.get("all_weekday_max"), weekday_plural)

    historical_pattern = _history_note_for_review(
        item, comparison
    )

    kmeans_body = (
        "<strong>Current one-day K-Means pattern:</strong> "
        + _html(patterns)
        + "<br><strong>Normally reported historically:</strong> "
        + _html(historical_pattern)
    )

    evidence = [
        ("Area pattern · K-Means + History", kmeans_body),
        ("Nearby reports · DBSCAN", _html(geographic)),
        ("Unusual activity · Anomaly", _html(historical) + bars),
    ]

    # Optional historical evidence only; review order and screening are unchanged.
    _pdf_historical_path = ROOT / "outputs/tables/pdf_style_anomaly/pdf_style_area_category_screen_v1.csv"
    if _pdf_historical_path.is_file():
        try:
            _pdf_candidates = _pulse_pdf_candidate_table(
                str(_pdf_historical_path), _pdf_historical_path.stat().st_mtime_ns
            )
            _pdf_notes = _pulse_pdf_history_for_review(item, _pdf_candidates)
        except (OSError, ValueError, KeyError, TypeError) as _pdf_error:
            _pdf_notes = []
            st.caption("PDF-style historical evidence unavailable: " + str(_pdf_error))
        if _pdf_notes:
            _pdf_context = " ".join("<p>" + _html(x) + "</p>" for x in _pdf_notes)
            _pdf_context += (
                "<p>Historical 2023–2025 retrospective screening only. "
                "Repeat reports are a proxy; recorded closure is not repair time. "
                "This does not indicate a current-day anomaly.</p>"
            )
            evidence.append(("Historical persistence · PDF evidence", _pdf_context))

    _md('<div class="evidence">' + "".join(
        f'<article><b>{_html(label)}</b><p>{body}</p></article>' for label, body in evidence
    ) + "</div>")
    _md(
        f'<div class="kicker">Why investigate here?</div>'
        f'<div class="reason">{_html(item["why"])}</div>'
        f'<div class="action"><div class="kicker">Suggested action</div>'
        f'<p>{_html(item["suggested_action"])}</p></div>'
    )
    st.caption(item["evidence_note"])

    coords = _feature_coords(feed)
    ids = [str(i) for i in item.get("linked_spatial_ids", [])]
    points = [{"latitude": coords[i][1], "longitude": coords[i][0]} for i in ids if i in coords]
    if points:
        with st.expander("Show this location's reports on the map", icon=":material/location_on:"):
            st.map(pd.DataFrame(points), color=tier["hex"], size=60)
            st.caption("Representative submitted coordinates, not verified fault locations.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def main():
    _md(THEME_CSS)
    _hero(compact=st.session_state.get("pulse_upload") is not None)
    if not TAXONOMY.is_file():
        st.error("Project category taxonomy is missing. Analysis cannot proceed.")
        st.stop()

    # The step bar is drawn into this slot once the page state is known.
    steps_slot = st.empty()

    # ---- 01 Upload ---------------------------------------------------------
    _section("01", "Upload your complaint CSV")
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        uploaded = st.file_uploader(
            "Select a complaints CSV", type=["csv"], key="pulse_upload",
            help="Six columns: complaint_id, created_date, latitude, longitude, "
                 "complaint_category, incident_zip.",
        )
        date_col, test_col = st.columns([1, 1.15])
        day = date_col.date_input("NYC reporting day", value=date.today())
        with test_col:
            st.write("")
            synthetic = st.toggle(
                "Synthetic / test data", value=True,
                help="Turn off when the file contains real submitted complaints.",
            )
    with right:
        _md(
            '<div class="info-panel"><b>First time using the app?</b><br>'
            'Download the six-column template, then upload your filled CSV.<br>'
            f'Maximum 10 MB / {MAX_ROWS:,} rows.</div>'
        )
        if TEMPLATE.is_file():
            st.download_button(
                "Download CSV template", TEMPLATE.read_bytes(),
                "new_complaints_template.csv", "text/csv",
                icon=":material/download:", use_container_width=True,
            )
        st.caption("Your historical model files are kept unchanged.")

    payload = uploaded.getvalue() if uploaded is not None else None
    if uploaded is not None:
        est_rows = _estimate_rows(payload)
        st.caption(f"Selected file: {uploaded.name} · {_fmt_bytes(len(payload))} · ≈ {est_rows:,} rows")
        if est_rows == 0:
            st.warning("This file seems to contain only a header row. "
                       "Please check that you selected the complete file.")
        elif len(payload) > MAX_BYTES or est_rows > MAX_ROWS:
            st.warning(f"This file looks larger than the {MAX_ROWS:,}-row / 10 MB limit "
                       "and may be rejected during validation.")

    file_key = hashlib.sha256(payload).hexdigest() if payload else None
    run_key = (file_key, day.isoformat(), bool(synthetic))
    preflight = st.session_state.get("pulse_preflight")
    validated = (
        bool(file_key)
        and st.session_state.get("pulse_preflight_key") == file_key
        and preflight is not None
    )
    active_result = st.session_state.get("pulse_analysis")
    analyzed = active_result is not None and st.session_state.get("pulse_analysis_key") == run_key
    steps_slot.markdown(
        _steps_html(4 if analyzed else 3 if validated else 2 if payload else 1),
        unsafe_allow_html=True,
    )

    # ---- 02 Validate -------------------------------------------------------
    if payload is None:
        _section("02", "Validate the uploaded data", locked_hint="Upload a CSV first")
    else:
        _section("02", "Validate the uploaded data")
        _md(
            '<div class="muted">Check the CSV first. During analysis, the existing backend '
            'will also verify geography and select the correct reporting day.</div>'
        )
        if st.button("Validate CSV", type="secondary", icon=":material/fact_check:",
                     use_container_width=True):
            try:
                check_uploaded_csv_columns(payload)
                checked = run_upload_preflight(payload, supported_categories())
            except Exception as exc:
                for name in ("pulse_preflight", "pulse_preflight_key",
                             "pulse_analysis", "pulse_analysis_key"):
                    st.session_state.pop(name, None)
                _show_error("Validation", exc)
            else:
                st.session_state["pulse_preflight"] = checked
                st.session_state["pulse_preflight_key"] = file_key
                st.session_state.pop("pulse_analysis", None)
                st.session_state.pop("pulse_analysis_key", None)
                st.rerun()

        if validated:
            st.success(f"CSV checked successfully · {preflight['row_count']} submitted reports",
                       icon=":material/check_circle:")
            if preflight.get("status_counts"):
                _status_chips(preflight["status_counts"])
            with st.expander("See individual validation results"):
                st.dataframe(preflight["validation_results"], hide_index=True,
                             use_container_width=True)

    # ---- 03 Analyze --------------------------------------------------------
    if not validated:
        _section("03", "Find places that deserve attention",
                 locked_hint="Validate the CSV to unlock")
        if payload is None:
            _md(
                '<div class="info-panel">After analysis, this page shows a ranked '
                'investigation list, a map, a plain-language reason and a suggested action.</div>'
            )
        return

    _section("03", "Find places that deserve attention")
    _md(
        '<div class="muted">The system combines area reporting patterns (K-Means), '
        'nearby complaint concentrations (DBSCAN) and unusual historical reporting '
        '(anomaly screening).</div>'
    )
    if st.button("Analyze complaints & get recommendations", type="primary",
                 icon=":material/auto_awesome:", use_container_width=True):
        try:
            with st.status("Analyzing complaints…", expanded=True) as status:
                st.write("Checking geography and selecting the reporting day…")
                with threadpool_limits(limits=1):
                    result = process_uploaded_csv_bytes(
                        payload, day.isoformat(), ROOT,
                        synthetic_data=bool(synthetic),
                        adapter=existing_backend_adapter(),
                        baseline=existing_historical_baseline(),
                    )
                st.write("Building the investigation review queue…")
                feed = result.get("feed")
                queue = build_review_queue(feed) if feed is not None else None

                #  HISTORICAL CATEGORY COMPARISON
                historical_comparison = None

                try:
                    from src.integration.historical_category_comparison import (
                        compare_historical_current_categories
                    )
                    import pandas as _history_pd

                    _historical_features = _history_pd.read_csv(
                        ROOT / "data/processed/selected_modeling_features.csv"
                    )

                    _expected_count = result.get(
                        "selection_stats", {}
                    ).get("selected_for_new_models")

                    historical_comparison = (
                        compare_historical_current_categories(
                            backend_records=result["backend_records"],
                            historical_features=_historical_features,
                            reporting_day=day.isoformat(),
                            expected_analyzed=_expected_count,
                        )
                    )

                except Exception as history_exc:
                    # Historical comparison is optional.
                    # Existing investigations must continue.
                    historical_comparison = {
                        "status": "unavailable",
                        "reason": str(history_exc),
                    }


                # PHASE 21.5: COMPARABLE WEEKDAY COMPARISON
                comparable_day_comparison = None

                try:
                    import json as _comparison_json
                    import pandas as _comparison_pd

                    from src.integration.historical_comparable_day import (
                        compare_historical_comparable_weekdays
                    )

                    _day_history_path = (
                        ROOT / "outputs/tables/anomaly/"
                        "historical_daily_counts_2023_2025_v1.csv.gz"
                    )
                    _day_manifest_path = (
                        ROOT / "outputs/tables/anomaly/"
                        "historical_daily_counts_2023_2025_v1_manifest.json"
                    )

                    @st.cache_data(show_spinner=False)
                    def _load_comparable_history(
                        csv_path, manifest_path,
                        csv_mtime, manifest_mtime
                    ):
                        _frame = _comparison_pd.read_csv(csv_path)
                        with open(
                            manifest_path, "r", encoding="utf-8"
                        ) as _f:
                            _manifest = _comparison_json.load(_f)
                        return _frame, _manifest

                    _daily_counts, _daily_manifest = (
                        _load_comparable_history(
                            str(_day_history_path),
                            str(_day_manifest_path),
                            _day_history_path.stat().st_mtime_ns,
                            _day_manifest_path.stat().st_mtime_ns,
                        )
                    )

                    _expected_analyzed = result.get(
                        "selection_stats", {}
                    ).get("selected_for_new_models")

                    comparable_day_comparison = (
                        compare_historical_comparable_weekdays(
                            backend_records=result[
                                "backend_records"
                            ],
                            historical_daily_counts=_daily_counts,
                            manifest=_daily_manifest,
                            reporting_day=day.isoformat(),
                            expected_analyzed=_expected_analyzed,
                        )
                    )

                except Exception as _day_exc:
                    # Keep existing investigations working even
                    # when optional historical context fails.
                    comparable_day_comparison = {
                        "status": "unavailable",
                        "reason": str(_day_exc),
                    }

                status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("pulse_analysis", None)
            st.session_state.pop("pulse_analysis_key", None)
            _show_error("Analysis", exc)
        else:
            st.session_state["pulse_analysis_key"] = run_key
            st.session_state["pulse_analysis"] = {
                "selection": result["selection_stats"],
                "uploaded": result["uploaded_rows"],
                "audit": result["backend_records"],
                "feed": feed, "review": queue,
                "new_kmeans_explorer": _pulse_new_kmeans_cluster_areas(result.get("integrated")),
                "historical_comparison": historical_comparison,
                "comparable_day_comparison": comparable_day_comparison,
            }
            st.session_state["pulse_toast"] = True
            st.rerun()

    current = st.session_state.get("pulse_analysis")
    if current is None or st.session_state.get("pulse_analysis_key") != run_key:
        _md(
            '<div class="info-panel">After analysis, this page will show a ranked '
            'investigation list, a map, a plain-language reason and a suggested action.</div>'
        )
        return

    if st.session_state.pop("pulse_toast", False):
        st.toast("Analysis complete", icon=":material/check_circle:")

    # ---- Results -----------------------------------------------------------
    selection = current["selection"]
    feed, review = current["feed"], current["review"]
    analyzed_count = int(selection["selected_for_new_models"])
    uploaded_count = int(current["uploaded"])
    synthetic_run = bool(feed["metadata"].get("synthetic_data")) if feed is not None else bool(synthetic)
    _md(
        '<p><span class="pill">✓ ANALYSIS COMPLETE</span>'
        + ('<span class="pill amber">SYNTHETIC / TEST DATA</span>' if synthetic_run else "")
        + "</p>"
    )
    _stats([
        ("Reports uploaded", uploaded_count, "Submitted CSV rows", False, None),
        ("Reports analyzed", analyzed_count, "Verified for selected day", False,
         analyzed_count / uploaded_count if uploaded_count else 0),
        ("Review-first locations",
         int(review["summary"]["review_first_groups"]) if review else 0,
         "Two screening signals", True, None),
        ("Other spatial findings",
         int(review["summary"]["spatial_only_groups"]) if review else 0,
         "May warrant a check", False, None),
    ])
    st.caption(
        f"{uploaded_count - analyzed_count} reports excluded from selected-day analysis. "
        "All original rows remain in the backend audit."
    )



    # New-batch K-Means cluster explorer
    _pulse_show_cluster_explorer(
        current.get("new_kmeans_explorer")
    )

    st.divider()

    with st.expander("View exclusions and download full processing audit"):
        st.write(
            f"Geography / backend-quality exclusions: "
            f"{selection['excluded_by_backend_quality']}; "
            f"other-day reports: {selection['excluded_by_reporting_day']}."
        )
        st.download_button(
            "Download backend audit", current["audit"].to_csv(index=False).encode("utf-8"),
            "urban_pulse_backend_audit.csv", "text/csv", key="dl_audit_summary",
            icon=":material/download:",
        )
    if feed is None:
        st.info("No verified reports matched this reporting day. No model findings were generated.")
        return
    if synthetic_run:
        st.warning("Synthetic demonstration data — not confirmed NYC incidents.",
                   icon=":material/science:")
    else:
        st.info("Uploaded reports are not independently verified infrastructure faults.")
    if not review or not review["queue"]:
        st.success("No investigation review candidates met the screening rules.")
        return

    items = review["queue"]
    weekday_plural = day.strftime("%A") + "s"
    _md(
        '<div class="sectiontitle"><i>✦</i><h2>Recommended places to investigate</h2></div>'
        '<div class="muted">The review order is based on screening evidence, '
        'not emergency severity or automatic repair priority.</div>'
    )

    tab_list, tab_map, tab_dl = st.tabs([
        ":material/checklist: Investigate", ":material/map: Map", ":material/download: Downloads",
    ])

    with tab_list:
        overview = pd.DataFrame([
            {
                "#": item["review_order"], "Place": item["location"],
                "Issue": str(item["category"]).title(),
                "Reports": int(item["incoming_reports"]),
                "Evidence": f'{_tier(item)["emoji"]} {_tier(item)["short"]}',
            }
            for item in items
        ])
        st.caption("Click a row to see why it was flagged and the suggested action.")
        event = st.dataframe(
            overview, hide_index=True, use_container_width=True,
            height=min(390, 40 + 36 * len(items)),
            key="pulse_queue_table", on_select="rerun", selection_mode="single-row",
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Place": st.column_config.TextColumn("Place", width="large"),
                "Reports": st.column_config.ProgressColumn(
                    "Reports", min_value=0, format="%d",
                    max_value=max(int(overview["Reports"].max()), 1),
                ),
            },
        )
        rows = event.selection.rows if event is not None else []
        selected = rows[0] if rows and rows[0] < len(items) else 0
        _show_location_review(
            items[selected],
            feed,
            weekday_plural,
            current.get("comparable_day_comparison"),
        )

    with tab_map:
        _overview_map(items, feed)

    with tab_dl:
        output = pd.DataFrame([
            {
                "Review order": item["review_order"], "Place": item["location"],
                "Issue": item["category"], "Reports": item["incoming_reports"],
                "Why investigate": item["why"], "Suggested action": item["suggested_action"],
                "Evidence limitation": item["evidence_note"],
            }
            for item in items
        ])
        c1, c2 = st.columns(2)
        c1.download_button(
            "All recommendations (CSV)", output.to_csv(index=False).encode("utf-8"),
            "urban_pulse_investigation_recommendations.csv", "text/csv",
            icon=":material/download:", use_container_width=True, key="dl_recs",
        )
        c2.download_button(
            "Full backend audit (CSV)", current["audit"].to_csv(index=False).encode("utf-8"),
            "urban_pulse_backend_audit.csv", "text/csv",
            icon=":material/download:", use_container_width=True, key="dl_audit_tab",
        )

    st.caption(
        "Screening signals are not confirmed faults. Review original complaint details, "
        "local conditions, and maintenance records before operational decisions."
    )


# One focused page; older multipage navigation stays hidden and unchanged.
st.navigation(
    [st.Page(main, title="Urban Infrastructure Pulse", default=True)],
    position="hidden",
).run()
