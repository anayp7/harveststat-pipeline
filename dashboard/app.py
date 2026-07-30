"""
Interactive dashboard for exploring HarvestStat pipeline outputs:
SIF-yield correlations (annual + season-specific) and QA/QC flags,
by country / crop / season / scenario, without digging through
figures/ folders or CSVs by hand.

Usage:
    streamlit run dashboard/app.py
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DASHBOARD = Path(__file__).resolve().parent
BASE = _DASHBOARD.parent
_PIPELINE = BASE / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country  # noqa: E402
from sif_yield_season import (  # noqa: E402
    SEASON_MONTH_OFFSETS, SEASON_DISPLAY,
    _load_season_yield, _build_sif_lookup, _season_sif_mean, _detrend,
    _build_exclusions,
)

PROC_DIR   = BASE / "data" / "processed"
RAW_DIR    = BASE / "data" / "raw"
# Committed, precomputed stand-ins for the gitignored stablebound outputs, so
# the dashboard runs from a clean clone. See pipeline/prepare_dashboard_bundle.py.
BUNDLE_DIR = BASE / "data" / "dashboard"
SACKS_PATH      = RAW_DIR / "All_data_with_climate.csv"
RESEARCHED_PATH = RAW_DIR / "researched_crop_calendars.csv"

MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# FAO/GIEWS crop-calendar convention
PHASE_COLORS = {
    "Sowing":     "#9C6B10",
    "Growing":    "#12A150",
    "Harvesting": "#F5C518",
    "Growing season (phases unknown)": "#7FA8C9",
}

st.set_page_config(page_title="HarvestStat Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Season / month-window labeling helpers
# ---------------------------------------------------------------------------

def _month_range_from_ordered(month_offsets: list[tuple[int, int]]) -> str:
    """Season-specific windows are already stored start->end (with wraparound
    handled via year_offset), so just take the first/last month."""
    if not month_offsets:
        return ""
    start = month_offsets[0][0]
    end = month_offsets[-1][0]
    return f"{MONTH_ABBR[start]}–{MONTH_ABBR[end]}"


def _month_range_from_set(months: list[int]) -> str:
    """The 'Annual' bucket's window comes from a numerically-sorted month
    list (e.g. from a crop-calendar union), which loses wraparound direction.
    Recover it by finding the largest cyclic gap and starting right after it."""
    months = sorted(set(months))
    if len(months) == 12:
        return "Jan–Dec (full year)"
    n = len(months)
    gaps = []
    for i in range(n):
        cur, nxt = months[i], months[(i + 1) % n]
        gaps.append(((nxt - cur) % 12, i))
    gaps.sort(reverse=True)
    start_idx = (gaps[0][1] + 1) % n
    start = months[start_idx]
    end = months[(start_idx - 1) % n]
    return f"{MONTH_ABBR[start]}–{MONTH_ABBR[end]}"


def _annual_window_label(window_text: str) -> str:
    """Turn aggregate_annual.py's sif_season_window text into a short,
    human-readable range + which calendar tier it came from."""
    if "no calendar entry" in window_text:
        return "Jan–Dec · no crop calendar for this crop, averaged over all 12 months"

    researched = "researched" in window_text
    tier = "researched calendar, unverified" if researched else "Sacks et al. crop calendar"

    m = re.search(r"\[(.*?)\]", window_text)
    if not m:
        return window_text
    months = [int(x) for x in m.group(1).split(",")]
    rng = _month_range_from_set(months)

    if len(months) == 12:
        return f"Jan–Dec · crop is in the field year-round · {tier}"
    return f"{rng} · growing season · {tier}"


RESEARCHED_CALENDAR_HELP = (
    "Growing-season windows come from two sources, and the difference matters:\n\n"
    "**Sacks et al.** — a published, peer-reviewed global crop calendar. Covers rice "
    "(all three countries) and wheat (Bangladesh only).\n\n"
    "**Researched, unverified** — compiled via LLM-assisted literature research "
    "(FAO/GIEWS, USDA FAS, CIMMYT, national ministries) for the 11 crops absent from "
    "Sacks. The month windows were checked for internal consistency, but the underlying "
    "citations have **not** been independently verified against source documents. "
    "Treat correlations built on these windows with more caution.\n\n"
    "Crops in neither source fall back to a flat 12-month average."
)


CLEANED_HELP = (
    "**Full**: every reported year, no exclusions.\n\n"
    "**Cleaned**: removes data flagged by the QA/QC checks before correlating — "
    "single-year anomalies with detrended |z| > 3, exact-repeat runs of length ≥ 4, "
    "reported-vs-calculated mismatches > 10%, and whole series with CV < 5%. "
    "Comparing full vs. cleaned shows whether a correlation is being driven by "
    "data-quality artifacts rather than a real signal."
)

QAQC_CHECK_INFO = {
    "Single-year anomaly (detrended)": (
        "Fits a linear trend to each district-crop yield series, then z-scores the "
        "**residuals** (not the raw values) and flags years where |z| > 2.0. "
        "Detrending first means a genuine multi-year trend doesn't hide an isolated "
        "bad year, and doesn't get mistaken for one either. We're interested because "
        "a single implausible year (e.g. a wrong decimal, a merged/duplicated report) "
        "can otherwise slip past a flat-mean check unnoticed."
    ),
    "Low coefficient of variation": (
        "Flags a district-crop series whose yield **CV (std/mean) is below 5%** — "
        "suspiciously flat for real agricultural output, which normally varies with "
        "weather and management year to year. We're interested because a flat series "
        "usually indicates copy-forward or placeholder reporting rather than genuine "
        "stability."
    ),
    "CV outlier (relative to crop)": (
        "Flags a district whose CV is **more than 2 std away from the mean CV** of all "
        "other districts growing the same crop (a relative check, unlike the fixed 5% "
        "floor above). We're interested because a district can look internally "
        "consistent yet still be reporting in a way that's out of step with every "
        "other district growing the same crop — either far too volatile or far too smooth."
    ),
}


# ---------------------------------------------------------------------------
# Crop calendar (FAO/GIEWS-style phase bars)
# ---------------------------------------------------------------------------

def _date_to_frac(datestr) -> float | None:
    """'5/8' (May 8) -> 5.226, on an axis where month m spans [m, m+1)."""
    if datestr is None or (isinstance(datestr, float) and pd.isna(datestr)):
        return None
    try:
        m, d = str(datestr).strip().split("/")
        m, d = int(m), int(d)
    except (ValueError, IndexError):
        return None
    if not 1 <= m <= 12:
        return None
    return m + (d - 1) / DAYS_IN_MONTH[m]


def _wrap_segments(start: float, end: float) -> list[tuple[float, float]]:
    """
    Split a span into drawable (base, width) pieces, cutting at the Dec/Jan
    boundary so a season crossing the year-end renders at both ends of the
    axis -- the same way the FAO calendars show a Rabi crop.

    `end` may legitimately exceed 13 (the caller adds 12 for a season that
    runs into the following year); a span longer than a full year is clamped
    to 12 months, since a crop can occupy at most the whole calendar.
    """
    if start is None or end is None:
        return []
    if end < start:
        end += 12.0
    if end <= start:
        return []
    if end - start > 12.0:
        end = start + 12.0

    if end <= 13.0:
        return [(start, end - start)]
    return [(start, 13.0 - start), (1.0, end - 12.0 - 1.0)]


@st.cache_data
def load_calendar_rows(cc: str) -> list[dict]:
    """
    One dict per crop-season for this country, each with an ordered list of
    (phase_name, start_frac, end_frac).

    Sacks rows carry all four phase dates, so they get the full
    Sowing/Growing/Harvesting breakdown. The researched fallback file only
    records plant-start and harvest-end, so those render as a single
    undifferentiated span rather than inventing phase boundaries.
    """
    country = get_country(cc)["name"]
    rows: list[dict] = []

    if SACKS_PATH.exists():
        sacks = pd.read_csv(SACKS_PATH)
        sub = sacks[sacks["Location"].str.contains(country, case=False, na=False)]
        for _, r in sub.iterrows():
            ps = _date_to_frac(r["Plant.start.date"])
            pe = _date_to_frac(r["Plant.end.date"])
            hs = _date_to_frac(r["Harvest.start.date"])
            he = _date_to_frac(r["Harvest.end.date"])
            if ps is None or he is None:
                continue
            phases = []
            if pe is not None:
                phases.append(("Sowing", ps, pe))
                if hs is not None:
                    phases.append(("Growing", pe, hs))
            if hs is not None:
                phases.append(("Harvesting", hs, he))
            if not phases:
                phases = [("Growing season (phases unknown)", ps, he)]

            # A few entries (e.g. Thailand irrigated sugarcane) report the same
            # window for sowing and harvesting -- staggered fields are planted
            # and cut at the same time of year. The bars then sit on top of one
            # another, so say so rather than let one silently hide the other.
            overlap = False
            if pe is not None and hs is not None:
                _pe = pe + 12.0 if pe < ps else pe
                _hs = hs + 12.0 if hs < ps else hs
                overlap = _hs < _pe

            qual = r.get("Qualifier")
            qual = "" if pd.isna(qual) else str(qual).strip()
            region = r["Location"].replace(country, "").strip(" ()")
            bits = [b for b in (region, qual) if b]
            label = r["Crop"] + (f" ({', '.join(bits)})" if bits else "")

            rows.append({
                "label": label, "crop": r["Crop"], "phases": phases,
                "source": "Sacks et al. (published)"
                          + (" · sowing & harvesting windows overlap" if overlap else ""),
                "span": f"{_frac_label(ps)} – {_frac_label(he)}",
                "overlap": overlap,
            })

    if RESEARCHED_PATH.exists():
        res = pd.read_csv(RESEARCHED_PATH)
        sub = res[res["Country"] == country]
        for _, r in sub.iterrows():
            ps = float(r["Plant_start_month"])
            he = float(r["Harvest_end_month"]) + 1.0   # inclusive of harvest month
            if str(r["Year_boundary_wrap"]).strip().lower() == "yes":
                # Harvest lands in the following calendar year, so push the end
                # past the boundary. Long-cycle crops (cassava, sugarcane) come
                # out as a full 12 months, which _wrap_segments clamps.
                he += 12.0
            rows.append({
                "label": f"{r['Crop']} ({r['Season_name']})",
                "crop": r["Crop"],
                "phases": [("Growing season (phases unknown)", ps, he)],
                "source": f"Researched, unverified · {r['Confidence']} confidence",
                "span": f"{MONTH_ABBR[int(r['Plant_start_month'])]} – "
                        f"{MONTH_ABBR[int(r['Harvest_end_month'])]}",
                "overlap": False,
            })

    return rows


def _frac_label(f: float) -> str:
    m = int(f)
    m = 12 if m > 12 else m
    day = int(round((f - int(f)) * DAYS_IN_MONTH[m])) + 1
    return f"{MONTH_ABBR[m]} {day}"


def plot_crop_calendar(cc: str, rows: list[dict], sif_rows: list[dict] | None = None):
    fig = go.Figure()
    seen_phases: set[str] = set()

    # Plotly stacks categorical y from the bottom up; reverse so the first
    # crop reads at the top like a printed calendar.
    ordered = list(reversed(rows))
    labels = [r["label"] for r in ordered]

    for r in ordered:
        for phase, s, e in r["phases"]:
            for base, width in _wrap_segments(s, e):
                if width <= 0:
                    continue
                fig.add_trace(go.Bar(
                    y=[r["label"]], x=[width], base=[base],
                    orientation="h", name=phase,
                    legendgroup=phase, showlegend=phase not in seen_phases,
                    marker=dict(color=PHASE_COLORS[phase],
                                line=dict(color="white", width=1),
                                # Coincident sowing/harvest windows would hide
                                # each other at full opacity.
                                opacity=0.62 if r.get("overlap") else 1.0),
                    hovertemplate=(f"<b>{r['label']}</b><br>{phase}"
                                   f"<br>{r['span']}<br><i>{r['source']}</i><extra></extra>"),
                    width=0.62,
                ))
                seen_phases.add(phase)

    # Optional overlay: the months the pipeline actually averages CSIF over.
    if sif_rows:
        for r in reversed(sif_rows):
            for base, width in _wrap_segments(r["start"], r["end"]):
                if width <= 0:
                    continue
                fig.add_trace(go.Bar(
                    y=[r["label"]], x=[width], base=[base],
                    orientation="h", name="CSIF window used in analysis",
                    legendgroup="sif", showlegend="sif" not in seen_phases,
                    marker=dict(color="rgba(120,120,120,0.30)",
                                line=dict(color="#444444", width=1.5)),
                    hovertemplate=(f"<b>{r['label']}</b><br>CSIF averaging window"
                                   f"<br>{r['span']}<extra></extra>"),
                    width=0.62,
                ))
                seen_phases.add("sif")
        labels = [r["label"] for r in reversed(sif_rows)] + labels

    fig.update_layout(
        barmode="overlay",
        height=max(260, 34 * len(labels) + 130),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            range=[1, 13],
            tickvals=[m + 0.5 for m in range(1, 13)],
            ticktext=MONTH_ABBR[1:],
            showgrid=True, gridcolor="rgba(140,140,140,0.35)",
            side="bottom", fixedrange=True,
        ),
        yaxis=dict(categoryorder="array", categoryarray=labels,
                   fixedrange=True, ticksuffix="  "),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28,
    )
    # Month separators
    for m in range(2, 13):
        fig.add_vline(x=m, line_width=1, line_color="rgba(140,140,140,0.35)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_boundary(cc: str) -> gpd.GeoDataFrame:
    """Prefer the committed simplified boundaries; fall back to the full
    stablebound geojson when a local checkout is available."""
    bundled = BUNDLE_DIR / f"boundaries_{cc.lower()}.geojson"
    if bundled.exists():
        return gpd.read_file(bundled)
    return gpd.read_file(get_country(cc)["boundary_path"])


def using_bundle(cc: str) -> bool:
    return (BUNDLE_DIR / f"boundaries_{cc.lower()}.geojson").exists()


@st.cache_data
def load_boundary_geojson(cc: str) -> dict:
    gdf = load_boundary(cc)[["stable_id", "geometry"]]
    return json.loads(gdf.to_json())


@st.cache_data
def load_annual_window_labels(cc: str) -> dict:
    """crop -> descriptive month-window string for the 'Annual' baseline bucket."""
    path = PROC_DIR / f"annual_sif_{cc.lower()}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "sif_season_window" not in df.columns:
        return {}
    out = {}
    for crop, g in df.groupby("crop"):
        out[crop] = _annual_window_label(g["sif_season_window"].iloc[0])
    return out


@st.cache_data
def load_corr(cc: str) -> pd.DataFrame:
    """Combine annual (12-month baseline, all crops) and season-specific
    correlation results into one long table with a descriptive `season_label`."""
    parts = []

    annual_path = PROC_DIR / f"sif_yield_corr_{cc.lower()}.csv"
    if annual_path.exists():
        annual = pd.read_csv(annual_path)
        annual["season_raw"] = "Annual"
        annual["bucket"] = "annual"
        parts.append(annual)

    season_path = PROC_DIR / f"sif_yield_season_corr_{cc.lower()}.csv"
    if season_path.exists():
        season = pd.read_csv(season_path)
        season = season.rename(columns={"season": "season_raw"})
        season["bucket"] = "season_specific"
        parts.append(season)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    annual_windows = load_annual_window_labels(cc)

    def build_label(row) -> str:
        if row["bucket"] == "annual":
            window = annual_windows.get(row["crop"], "")
            return f"Annual baseline ({window})" if window else "Annual baseline"
        key = (cc, row["crop"], row["season_raw"])
        display = SEASON_DISPLAY.get(row["season_raw"], row["season_raw"])
        rng = _month_range_from_ordered(SEASON_MONTH_OFFSETS.get(key, []))
        return f"{display} ({rng})" if rng else display

    combined["season_label"] = combined.apply(build_label, axis=1)
    return combined


@st.cache_data
def load_qaqc_table(cc: str, name: str) -> pd.DataFrame:
    path = PROC_DIR / f"{name}_{cc.lower()}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_sif_lookup(cc: str) -> dict:
    path = PROC_DIR / f"sif_by_crop_{cc.lower()}.csv"
    if not path.exists():
        return {}
    return _build_sif_lookup(pd.read_csv(path))


@st.cache_data
def load_exclusions(cc: str) -> tuple[set, set]:
    """QA point/series exclusions behind the 'cleaned' scenario.

    Prefers the precomputed bundle (deriving these live needs name_history.csv
    to resolve admin names to stable_ids, which isn't committed); otherwise
    calls the same function the pipeline scripts use.
    """
    bundled = BUNDLE_DIR / f"exclusions_{cc.lower()}.csv"
    if bundled.exists():
        df = pd.read_csv(bundled)
        points, series = set(), set()
        for r in df.itertuples(index=False):
            if pd.isna(r.year) or str(r.year).strip() == "":
                series.add((r.stable_id, r.crop))
            else:
                points.add((r.stable_id, r.crop, int(float(r.year))))
        return points, series
    try:
        return _build_exclusions(cc)
    except Exception:
        return set(), set()


@st.cache_data
def district_series(cc: str, stable_id: str, crop: str,
                    season_raw: str, bucket: str) -> pd.DataFrame:
    """
    Rebuild the exact per-year (yield, SIF) pair that fed the correlation for
    one district, so the drilldown shows the same numbers the map is coloured by.

    'annual' bucket reads the precomputed annual_joined file; 'season_specific'
    re-derives season-level yield and the season SIF window the same way
    sif_yield_season.py does, reusing that module's own functions so the two
    can't drift apart.
    """
    if bucket == "annual":
        path = PROC_DIR / f"annual_joined_{cc.lower()}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = df[(df["stable_id"] == stable_id) & (df["crop"] == crop)]
        out = df[["year", "yield_mt_ha", "sif_annual_mean"]].rename(
            columns={"sif_annual_mean": "sif"}
        )
    else:
        offsets = SEASON_MONTH_OFFSETS.get((cc, crop, season_raw))
        if not offsets:
            return pd.DataFrame()

        bundled = BUNDLE_DIR / f"season_yield_{cc.lower()}.csv"
        if bundled.exists():
            sy = pd.read_csv(bundled)
            yld = sy[(sy["stable_id"] == stable_id) & (sy["crop"] == crop)
                     & (sy["season"] == season_raw)][["stable_id", "year", "yield_mt_ha"]]
        else:
            yld = _load_season_yield(get_country(cc), crop, season_raw)
            yld = yld[yld["stable_id"] == stable_id]
        if yld.empty:
            return pd.DataFrame()
        lookup = load_sif_lookup(cc)
        yld = yld.copy()
        yld["sif"] = yld["year"].apply(
            lambda y: _season_sif_mean(lookup, stable_id, crop, y, offsets)
        )
        out = yld[["year", "yield_mt_ha", "sif"]]

    out = out.dropna(subset=["yield_mt_ha", "sif"]).sort_values("year")
    out = out.reset_index(drop=True)

    # Mark which years the 'cleaned' scenario drops, so the drilldown can both
    # reproduce the cleaned r exactly and show the reader what was removed.
    excluded_points, _ = load_exclusions(cc)
    out["qa_excluded"] = out["year"].apply(
        lambda y: (stable_id, crop, int(y)) in excluded_points
    )
    return out


def plot_district_drilldown(series: pd.DataFrame, stable_id: str,
                            crop: str, season_label: str, row: pd.Series,
                            scenario: str = "full"):
    """Two panels: standardised detrended anomalies as paired bars per year,
    and the scatter + fitted line the Pearson r actually describes."""
    dropped = series[series["qa_excluded"]] if scenario == "cleaned" else series.iloc[0:0]
    used = series[~series["qa_excluded"]] if scenario == "cleaned" else series

    if len(used) < 2:
        st.info("Not enough years remain after QA exclusions to plot.")
        return

    if not dropped.empty:
        st.caption(
            f"⚠️ {len(dropped)} year(s) removed by QA in this scenario: "
            f"{', '.join(str(int(y)) for y in dropped['year'])}. "
            "Shown faded below; excluded from the fit and from r."
        )

    years = used["year"].values.astype(float)
    y_raw = used["yield_mt_ha"].values.astype(float)
    s_raw = used["sif"].values.astype(float)

    y_anom = _detrend(years, y_raw)
    s_anom = _detrend(years, s_raw)

    # Yield (mt/ha) and CSIF (mW m-2 nm-1 sr-1) differ by ~an order of
    # magnitude, so raw bars would be unreadable side by side. Dividing each
    # detrended series by its own SD puts both in units of "SDs from trend",
    # which is exactly the quantity Pearson r is computed on.
    y_sd = y_anom.std() or 1.0
    s_sd = s_anom.std() or 1.0
    y_z, s_z = y_anom / y_sd, s_anom / s_sd

    c1, c2 = st.columns([3, 2])

    with c1:
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=years, y=y_z, name="Yield anomaly",
            marker_color="#2c7fb8",
            customdata=y_raw,
            hovertemplate="%{x:.0f}<br>yield anom: %{y:+.2f} SD"
                          "<br>raw: %{customdata:.3f} mt/ha<extra></extra>",
        ))
        bar.add_trace(go.Bar(
            x=years, y=s_z, name="CSIF anomaly",
            marker_color="#31a354",
            customdata=s_raw,
            hovertemplate="%{x:.0f}<br>CSIF anom: %{y:+.2f} SD"
                          "<br>raw: %{customdata:.3f}<extra></extra>",
        ))
        if not dropped.empty:
            bar.add_trace(go.Bar(
                x=dropped["year"].values.astype(float),
                y=[0] * len(dropped), name="QA-excluded year",
                marker_color="rgba(200,60,60,0.28)", width=0.9,
                base=-3, hovertemplate="%{x:.0f}<br>excluded by QA<extra></extra>",
            ))
        bar.add_hline(y=0, line_width=1, line_color="#888888")
        bar.update_layout(
            barmode="group", height=380,
            margin=dict(l=0, r=0, t=40, b=0),
            title="Detrended anomalies by year (bars move together → positive r)",
            yaxis_title="SDs from own linear trend",
            xaxis_title="Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        )
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        fit = np.polyfit(s_z, y_z, 1)
        xs = np.linspace(s_z.min(), s_z.max(), 50)
        sc = go.Figure()
        sc.add_trace(go.Scatter(
            x=s_z, y=y_z, mode="markers+text",
            text=[str(int(y)) for y in years],
            textposition="top center", textfont=dict(size=8, color="#888888"),
            marker=dict(size=9, color="#2c7fb8", line=dict(width=1, color="white")),
            name="year", hovertemplate="CSIF %{x:+.2f} SD<br>yield %{y:+.2f} SD<extra></extra>",
        ))
        sc.add_trace(go.Scatter(
            x=xs, y=np.polyval(fit, xs), mode="lines",
            line=dict(color="#d95f02", width=2.5), name="fit",
        ))
        sc.update_layout(
            height=380, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
            title=f"r = {row['r']:+.3f} · p = {_fmt_value('p_value', row['p_value'])} "
                  f"· n = {int(row['n_years'])}",
            xaxis_title="CSIF anomaly (SD)", yaxis_title="Yield anomaly (SD)",
        )
        st.plotly_chart(sc, use_container_width=True)

    with st.expander("Underlying values"):
        tbl = used.copy()
        tbl["yield_anom_sd"] = y_z
        tbl["sif_anom_sd"] = s_z
        if not dropped.empty:
            tbl = pd.concat([tbl, dropped], ignore_index=True).sort_values("year")
        show_table(tbl, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Shared map helper
# ---------------------------------------------------------------------------

def _fmt_value(col: str, v) -> str:
    """Format one hover value. Counts print as integers (a left-join turns them
    into floats, which would otherwise render as '25.000'), p-values collapse to
    '<0.001' rather than a misleading '0.000', everything else gets 3 dp."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if col.startswith("n_"):
        return f"{int(round(float(v)))}"
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    if isinstance(v, (float, np.floating)):
        if col == "p_value" and v < 0.001:
            return "<0.001"
        return f"{v:.3f}"
    return str(v)


def _hover_text(row: pd.Series, cols: list[str]) -> str:
    parts = [f"<b>{row['stable_id']}</b>"]
    parts += [f"{c}: {_fmt_value(c, row[c])}" for c in cols]
    return "<br>".join(parts)


def show_table(df: pd.DataFrame, **kwargs):
    """st.dataframe with float columns formatted to 3 dp. Uses column_config
    rather than df.round() so the full-precision value is still what sorting
    and the CSV export use."""
    cfg = {}
    for col in df.columns:
        if col.startswith("n_"):
            cfg[col] = st.column_config.NumberColumn(col, format="%d")
        elif pd.api.types.is_float_dtype(df[col]):
            fmt = "%.4f" if col == "p_value" else "%.3f"
            cfg[col] = st.column_config.NumberColumn(col, format=fmt)
    st.dataframe(df, column_config=cfg, **kwargs)


def choropleth(cc: str, df: pd.DataFrame, color_col: str, hover_cols: list[str],
               colorscale: str = "RdBu_r", symmetric: bool = True, title: str = "",
               colorbar_title: str = "", sig_ids: list[str] | None = None,
               sig_label: str = "p < 0.05", select_key: str | None = None):
    """
    Three stacked layers:
      1. every district in the country, flat grey -- keeps the map framed on the
         full national extent even when a crop is grown in only a few districts,
         so the reader isn't silently zoomed into a sub-region. Carries no data
         and never enters any statistic.
      2. the districts that actually have a value, on the colour scale.
      3. (optional) significant districts, transparent fill + heavy black
         outline, matching the emphasis used in the matplotlib figures.
    """
    geojson = load_boundary_geojson(cc)
    all_ids = load_boundary(cc)[["stable_id"]]
    gdf = all_ids.merge(df, on="stable_id", how="left")

    data = gdf.dropna(subset=[color_col]).copy()
    if data.empty:
        st.warning("No data available for this selection.")
        return

    vals = data[color_col]
    if symmetric:
        vmax = max(abs(vals.min()), abs(vals.max()), 0.05)
        zmin, zmax = -vmax, vmax
    else:
        zmin, zmax = vals.min(), vals.max()

    fig = go.Figure()

    # Layer 1 -- full-country base (no data)
    fig.add_trace(go.Choropleth(
        geojson=geojson, locations=all_ids["stable_id"],
        featureidkey="properties.stable_id",
        z=[0] * len(all_ids),
        colorscale=[[0, "#e9e9e9"], [1, "#e9e9e9"]],
        showscale=False,
        marker_line_color="white", marker_line_width=0.5,
        hovertemplate="<b>%{location}</b><br>not in this crop/season<extra></extra>",
    ))

    # Layer 2 -- districts with values
    fig.add_trace(go.Choropleth(
        geojson=geojson, locations=data["stable_id"],
        featureidkey="properties.stable_id",
        z=data[color_col], zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        marker_line_color="#555555", marker_line_width=0.4,
        colorbar=dict(title=colorbar_title or color_col, thickness=14, len=0.7),
        text=data.apply(lambda r: _hover_text(r, hover_cols), axis=1),
        hovertemplate="%{text}<extra></extra>",
    ))

    # Layer 3 -- significance emphasis
    if sig_ids:
        sig = data[data["stable_id"].isin(sig_ids)]
        if not sig.empty:
            fig.add_trace(go.Choropleth(
                geojson=geojson, locations=sig["stable_id"],
                featureidkey="properties.stable_id",
                z=[0] * len(sig),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_color="black", marker_line_width=2.2,
                text=sig.apply(lambda r: _hover_text(r, hover_cols), axis=1),
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=60, b=0),
        height=620,
        showlegend=False,
    )
    if select_key:
        event = st.plotly_chart(
            fig, use_container_width=True, key=select_key,
            on_select="rerun", selection_mode="points",
        )
    else:
        event = None
        st.plotly_chart(fig, use_container_width=True)

    caption = "Districts in **light grey** have no data for this selection and " \
              "are excluded from all statistics above."
    if sig_ids:
        caption = (f"Districts outlined in **black** are significant at {sig_label}. "
                   + caption)
    if select_key:
        caption += " **Click a district** to see its year-by-year series."
    st.caption(caption)

    return event


def selected_district(event) -> str | None:
    """Pull the clicked stable_id out of a plotly selection event. Clicks on the
    grey base layer or the outline layer resolve to the same location id, so any
    of the three traces is a valid way to select a district."""
    if not event:
        return None
    points = (event.get("selection") or {}).get("points") or []
    for p in points:
        loc = p.get("location")
        if loc:
            return loc
    return None


# ---------------------------------------------------------------------------
# Sidebar: country selector (shared across tabs)
# ---------------------------------------------------------------------------

st.title("HarvestStat Interactive Dashboard")

codes = all_codes()
cc = st.sidebar.selectbox(
    "Country", codes, format_func=lambda c: f"{get_country(c)['name']} ({c})"
)

tab_corr, tab_qaqc, tab_cal = st.tabs(
    ["SIF ~ Yield Correlation", "QA/QC Flags", "Crop Calendar"]
)


# ---------------------------------------------------------------------------
# Tab 1: SIF-yield correlation explorer
# ---------------------------------------------------------------------------

with tab_corr:
    corr = load_corr(cc)
    if corr.empty:
        st.warning(f"No correlation results found for {cc}.")
    else:
        col_crop, col_season, col_scenario = st.columns(3)

        with col_crop:
            crop = st.selectbox("Crop", sorted(corr["crop"].unique()), key="corr_crop")

        crop_df = corr[corr["crop"] == crop]

        with col_season:
            season_options = sorted(crop_df["season_label"].unique())
            season_label = st.selectbox(
                "Season", season_options, key="corr_season",
                help="Each option shows the season name and the actual months of SIF "
                     "averaged for it. 'Annual baseline' is the pipeline's default "
                     "window for this crop, not necessarily all 12 months.\n\n"
                     + RESEARCHED_CALENDAR_HELP,
            )

        with col_scenario:
            scenario = st.radio(
                "Scenario", ["full", "cleaned"], horizontal=True,
                key="corr_scenario", help=CLEANED_HELP,
            )

        if "researched calendar, unverified" in season_label:
            st.warning(
                "**Unverified calendar source.** This crop's growing-season window comes "
                "from LLM-assisted literature research, not the peer-reviewed Sacks et al. "
                "calendar. Month ranges were checked for internal consistency, but the "
                "underlying citations have not been verified against source documents.",
                icon="⚠️",
            )

        if len(crop_df[crop_df["season_label"] == season_label]["season_raw"].unique()) == 1 \
                and len(season_options) == 1:
            st.caption(
                "Only one window is available for this crop — it has no "
                "multi-season split in the source data and/or no crop-calendar "
                "entry, so there's nothing to break out by season."
            )

        filtered = crop_df[
            (crop_df["season_label"] == season_label) & (crop_df["scenario"] == scenario)
        ].copy()

        if filtered.empty:
            st.warning("No data for this combination.")
        else:
            is_sig  = filtered["p_value"] < 0.05
            n_sig   = int(is_sig.sum())
            n_pos   = int((is_sig & (filtered["r"] > 0)).sum())
            n_neg   = int((is_sig & (filtered["r"] < 0)).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Districts", len(filtered))
            m2.metric("Mean r", f"{filtered['r'].mean():+.3f}")
            m3.metric("Median r", f"{filtered['r'].median():+.3f}")
            m4.metric(
                "Significant (p<0.05)", f"{n_sig}/{len(filtered)}",
                delta=(f"{n_pos} positive · {n_neg} negative" if n_sig else None),
                delta_color="off",
                help="Direction matters: a significant negative correlation is a "
                     "real finding, not a weaker version of a positive one. R² alone "
                     "would collapse the two.",
            )

            event = choropleth(
                cc, filtered, color_col="r",
                hover_cols=["r", "p_value", "n_years"],
                colorscale="RdBu_r", symmetric=True,
                colorbar_title="Pearson r",
                sig_ids=filtered.loc[is_sig, "stable_id"].tolist(),
                title=f"{get_country(cc)['name']} — {crop} — {season_label} ({scenario}): "
                      f"detrended yield ~ CSIF, Pearson r",
                select_key=f"corrmap_{cc}_{crop}_{season_label}_{scenario}",
            )

            clicked = selected_district(event)
            if clicked:
                st.divider()
                st.subheader(f"{clicked} — {crop} — {season_label}")
                match = filtered[filtered["stable_id"] == clicked]
                if match.empty:
                    st.info(
                        f"{clicked} has no {crop} data for this season/scenario "
                        "(it's one of the grey districts), so there's no series to plot."
                    )
                else:
                    row = match.iloc[0]
                    series = district_series(
                        cc, clicked, crop, row["season_raw"], row["bucket"]
                    )
                    if series.empty or len(series) < 2:
                        st.info("Not enough overlapping yield/CSIF years to plot.")
                    else:
                        plot_district_drilldown(
                            series, clicked, crop, season_label, row, scenario
                        )

            st.divider()
            st.subheader("District-level detail")
            id_filter = st.text_input("Filter by stable_id (optional)", key="corr_id_filter")
            table = filtered.sort_values("r", ascending=False)[
                ["stable_id", "r", "r_squared", "p_value", "n_years"]
            ]
            if id_filter:
                table = table[table["stable_id"].str.contains(id_filter, case=False, na=False)]
            show_table(table, use_container_width=True, height=350)


# ---------------------------------------------------------------------------
# Tab 2: QA/QC flags explorer
# ---------------------------------------------------------------------------

with tab_qaqc:
    check = st.selectbox(
        "Check",
        list(QAQC_CHECK_INFO.keys()),
        key="qaqc_check",
        help="Select a QA/QC check to see its map and per-district detail. "
             "A summary of what each check means, why it exists, and its "
             "flag threshold is shown below once selected.",
    )
    st.info(QAQC_CHECK_INFO[check])

    if check == "Single-year anomaly (detrended)":
        df = load_qaqc_table(cc, "qaqc_year_anomaly_district")
        if df.empty:
            st.warning(f"No anomaly data for {cc}.")
        else:
            crops = sorted(df["crop"].unique())
            crop = st.selectbox("Crop", crops, key="qaqc_anomaly_crop")
            sub = df[df["crop"] == crop].copy()

            m1, m2 = st.columns(2)
            m1.metric("Districts", len(sub))
            m2.metric("Mean anomalous years", f"{sub['n_anomalous_years'].mean():.2f}")

            choropleth(
                cc, sub, color_col="n_anomalous_years",
                hover_cols=["n_anomalous_years", "n_years", "series_mean"],
                colorscale="Reds", symmetric=False,
                colorbar_title="anomalous<br>years",
                sig_ids=sub.loc[sub["n_anomalous_years"] > 0, "stable_id"].tolist(),
                sig_label="≥1 anomalous year",
                title=f"{get_country(cc)['name']} — {crop}: single-year anomaly count (detrended, |z|>2)",
            )

            st.subheader("District-level detail")
            show_table(
                sub.sort_values("n_anomalous_years", ascending=False),
                use_container_width=True, height=350,
            )

    elif check == "Low coefficient of variation":
        df = load_qaqc_table(cc, "qaqc_low_cv")
        if df.empty:
            st.warning(f"No CV data for {cc}.")
        else:
            crops = sorted(df["crop"].unique())
            crop = st.selectbox("Crop", crops, key="qaqc_lowcv_crop")
            sub = df[df["crop"] == crop].copy()

            n_flagged = int(sub["flag_low_cv"].sum())
            m1, m2 = st.columns(2)
            m1.metric("Districts", len(sub))
            m2.metric("Flagged (CV < 5%)", f"{n_flagged}/{len(sub)}")

            choropleth(
                cc, sub, color_col="cv",
                hover_cols=["cv", "mean_yield", "n_years", "flag_low_cv"],
                colorscale="Viridis", symmetric=False,
                colorbar_title="CV",
                sig_ids=sub.loc[sub["flag_low_cv"] == True, "stable_id"].tolist(),
                sig_label="CV < 5% (flagged)",
                title=f"{get_country(cc)['name']} — {crop}: yield coefficient of variation",
            )

            st.subheader("District-level detail")
            show_table(
                sub.sort_values("cv")[
                    ["stable_id", "crop", "n_years", "mean_yield", "cv", "flag_low_cv"]
                ],
                use_container_width=True, height=350,
            )

    else:  # CV outlier
        df = load_qaqc_table(cc, "qaqc_low_cv")
        if df.empty:
            st.warning(f"No CV data for {cc}.")
        else:
            crops = sorted(df["crop"].unique())
            crop = st.selectbox("Crop", crops, key="qaqc_cvoutlier_crop")
            sub = df[df["crop"] == crop].copy()
            sub["flagged"] = sub["flag_cv_outlier_high"] | sub["flag_cv_outlier_low"]

            n_flagged = int(sub["flagged"].sum())
            n_high    = int(sub["flag_cv_outlier_high"].sum())
            n_low     = int(sub["flag_cv_outlier_low"].sum())
            m1, m2 = st.columns(2)
            m1.metric("Districts", len(sub))
            m2.metric(
                "Flagged outliers", f"{n_flagged}/{len(sub)}",
                delta=(f"{n_high} too volatile · {n_low} too smooth" if n_flagged else None),
                delta_color="off",
            )

            choropleth(
                cc, sub, color_col="cv_zscore",
                hover_cols=["cv", "cv_zscore", "flagged"],
                colorscale="RdBu_r", symmetric=True,
                colorbar_title="CV z-score",
                sig_ids=sub.loc[sub["flagged"], "stable_id"].tolist(),
                sig_label="|z| > 2 (flagged)",
                title=f"{get_country(cc)['name']} — {crop}: CV z-score relative to same-crop districts",
            )

            st.subheader("District-level detail")
            show_table(
                sub.sort_values("cv_zscore", key=abs, ascending=False)[
                    ["stable_id", "crop", "cv", "cv_zscore", "flagged"]
                ],
                use_container_width=True, height=350,
            )


# ---------------------------------------------------------------------------
# Tab 3: Crop calendar wizard
# ---------------------------------------------------------------------------

with tab_cal:
    country_name = get_country(cc)["name"]
    rows = load_calendar_rows(cc)

    if not rows:
        st.warning(f"No crop calendar entries found for {country_name}.")
    else:
        st.markdown(
            f"### {country_name} — crop calendar\n"
            "Each bar spans the months a crop is in the field. Seasons that cross "
            "the year boundary (Boro, Rabi, Winter/Spring rice) are drawn at both "
            "ends of the axis."
        )

        crops = sorted({r["crop"] for r in rows})
        c1, c2 = st.columns([2, 1])
        with c1:
            picked = st.multiselect(
                "Crops", crops, default=crops, key="cal_crops",
                help="Filter which crops appear in the calendar.",
            )
        with c2:
            show_sif = st.checkbox(
                "Overlay CSIF analysis window", value=False, key="cal_sif",
                help="Shows the months the pipeline actually averages CSIF over for "
                     "each season, as a grey band. Use it to check whether the "
                     "analysis window lines up with the agronomic calendar.",
            )

        shown = [r for r in rows if r["crop"] in picked]
        if not shown:
            st.info("Select at least one crop.")
        else:
            sif_rows = []
            if show_sif:
                for (kcc, fews_crop, season), offsets in SEASON_MONTH_OFFSETS.items():
                    if kcc != cc or not offsets:
                        continue
                    start = float(offsets[0][0])
                    end   = float(offsets[-1][0]) + 1.0
                    if offsets[0][1] == -1:
                        # Window opens in the prior calendar year (Boro, Rabi,
                        # Winter/Spring rice), so it crosses the Dec/Jan boundary.
                        end += 12.0
                    sif_rows.append({
                        "label": f"▸ CSIF: {fews_crop} / {SEASON_DISPLAY.get(season, season)}",
                        "start": start, "end": end,
                        "span": f"{MONTH_ABBR[offsets[0][0]]} – {MONTH_ABBR[offsets[-1][0]]}",
                    })
                if not sif_rows:
                    st.caption("No season-specific CSIF windows are defined for this country.")

            plot_crop_calendar(cc, shown, sif_rows or None)

            n_sacks = sum(1 for r in shown if r["source"].startswith("Sacks"))
            n_res   = len(shown) - n_sacks
            st.caption(
                f"**{n_sacks}** season(s) from Sacks et al. (published, full "
                f"sowing/growing/harvesting phases) · **{n_res}** from the researched "
                f"fallback (unverified; only plant-start and harvest-end are recorded, "
                f"so no phase breakdown is shown rather than inventing one)."
            )

            st.subheader("Calendar detail")
            show_table(
                pd.DataFrame([
                    {"Crop / season": r["label"], "Span": r["span"],
                     "Phases": ", ".join(p for p, _, _ in r["phases"]),
                     "Source": r["source"]}
                    for r in shown
                ]),
                use_container_width=True, hide_index=True,
            )
