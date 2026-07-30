# HarvestStat Dashboard

Interactive explorer for the pipeline's outputs — SIF–yield correlations,
QA/QC flags, and crop calendars — so you don't have to dig through
`figures/` folders or raw CSVs.

## Quick start

No stablebound checkout, no CSIF/CROPGRIDs downloads — just the repo.

### Option A — conda (recommended, and required on macOS / miniforge)

```bash
git clone https://github.com/anayp7/harveststat-pipeline.git
cd harveststat-pipeline
conda env create -f environment.yml
conda activate harveststat
streamlit run dashboard/app.py
```

### Option B — pip, in a fresh virtual environment

Fine on Windows/Linux, or anywhere you're **not** mixing with conda:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Either way it opens at <http://localhost:8501>. If no browser opens, paste
that URL in.

> **Don't mix the two.** Running `pip install -r requirements.txt` on top of a
> conda environment that already has geopandas is the single most common way to
> break this install — see Troubleshooting.

## Troubleshooting

### `ImportError: Error importing numpy: you should not try to import numpy from its source directory`

This message is misleading — it has nothing to do with your working directory.
numpy raises it whenever its compiled core fails to load, and with a
conda + geopandas setup there are two usual causes.

**1. The `streamlit` command is running against the wrong environment.**
If streamlit is installed in `base` but not in your new env, the `streamlit`
binary on your PATH is base's, so it loads base's packages no matter which env
is active. A traceback with paths like
`miniforge3/lib/python3.12/site-packages/...` — rather than
`miniforge3/envs/<yourenv>/lib/python3.12/site-packages/...` — is this problem.

Check it:

```bash
conda activate harveststat
which python && which streamlit
python -c "import numpy, geopandas; print(numpy.__version__, geopandas.__version__, numpy.__file__)"
```

Both paths must be under `envs/harveststat/`. If `streamlit` points at
`miniforge3/bin/streamlit`, it isn't installed in the env — `environment.yml`
above installs it there. As a one-off workaround, `python -m streamlit run
dashboard/app.py` forces the active env's interpreter.

**2. numpy and geopandas were built against different numpy ABIs.**
geopandas **1.0+** is the first series built for numpy 2. If pip pulled numpy 2
on top of a conda geopandas 0.x (or vice versa), the compiled extensions won't
load. The versions printed above should satisfy geopandas ≥ 1.0 with numpy ≥ 2,
*or* geopandas 0.x with numpy 1.x — not a mix.

The reliable fix for both is one clean conda-forge solve, no pip on top:

```bash
conda deactivate
conda env remove -n harveststat        # if you already made one
conda env create -f environment.yml
conda activate harveststat
streamlit run dashboard/app.py
```

Known-good on this project: Python 3.12, numpy 2.2.6, pandas 2.2.2,
geopandas 1.1.3, shapely 2.1.2, scipy 1.14.1, streamlit 1.58, plotly 6.8.

## What you're looking at

**SIF ~ Yield Correlation** — pick a crop, a season, and a scenario. The map
shows the signed Pearson *r* between detrended yield and detrended CSIF per
district. Districts with a **heavy black outline** are significant at p < 0.05;
**light grey** districts have no data for that selection and are excluded from
every statistic shown. **Click any district** for its year-by-year series:
paired anomaly bars plus the scatter and fitted line the correlation describes.

- *Season* options list the actual months averaged, and which calendar they
  came from. Sacks et al. is published and peer-reviewed; anything marked
  **researched, unverified** was compiled from literature research and its
  citations have not been independently checked — treat those with more caution.
- *Scenario* — `full` uses every reported year; `cleaned` drops QA-flagged
  points first. In `cleaned`, removed years are called out under the chart.

**QA/QC Flags** — the yield-quality checks, each with a plain-English note on
what it means, why it exists, and its threshold.

**Crop Calendar** — FAO/GIEWS-style phase bars (sowing / growing / harvesting)
per crop-season, with an optional overlay showing the exact months the pipeline
averages CSIF over.

## Where the data comes from

The dashboard reads committed outputs only:

| Path | Contents |
|---|---|
| `data/processed/` | correlation results, QA/QC flags, annual + monthly SIF |
| `data/dashboard/` | precomputed bundle (see below) |
| `data/raw/*.csv` | crop calendars |

`data/dashboard/` exists because three inputs the dashboard needs live in the
gitignored `stablebound_starter/` checkout (~40 MB, and `stats_aggregated.csv`
holds proprietary FEWS rows we don't publish). `pipeline/prepare_dashboard_bundle.py`
distills them into ~2 MB of derived files — simplified boundaries, season-level
yields, and resolved QA exclusion sets — so a clean clone just works.

If you *do* have a stablebound checkout, the dashboard automatically prefers the
bundle anyway; delete `data/dashboard/` to force it back to the live files.

**Regenerate the bundle** after re-running the pipeline:

```bash
python pipeline/prepare_dashboard_bundle.py
```

## Note on sharing

The bundle contains **derived district-level yields**. The repository is
private; keep it that way unless you've confirmed with the FEWS data owner
that derived yields can be published. Deploying this to a public host
(Streamlit Community Cloud, a tunnel) makes those values publicly reachable.
