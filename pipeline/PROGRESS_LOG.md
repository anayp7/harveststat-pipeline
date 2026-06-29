# Multi-Country QA/QC Pipeline — Progress Log

Running record of what was built, what was found, and why decisions were made.
Written for a reader with no memory of the working sessions — context and
rationale are spelled out, not just the action taken.

**Goal:** generalize the India crop-yield/remote-sensing QA-QC work into a
pipeline other researchers can run on other countries. Thailand (TH),
Bangladesh (BD), and Vietnam (VN) are the first three test cases, using FEWS
yield data + GAUL boundaries (via the `stablebound` package) + GOSIF SIF.

---

## 1. Boundary + yield ingestion (stablebound)

**Source:** team member's `stablebound` package + a starter bundle
(`stablebound_TH_BD_VN_starter.zip`) containing FEWS-format yield CSVs and
GAUL boundary GeoJSONs for all three countries.

**What we did:**
- Created a fresh virtual environment and installed the package wheel
  (`stablebound-0.1.0-py3-none-any.whl`), per the team member's instructions.
- Ran `prepare_stats.py` (name-match yield rows to lineage units) and
  `build_stable_crops.py` (build stable boundaries + aggregate stats) for
  all three countries.

**Bugs found and fixed** (both pandas-3.0 compatibility issues in the
starter scripts, not in `stablebound` itself):
1. `prepare_stats.py` line 264 — `.map(crop_slug)` crashed on rows where
   both "Source crop" and "crop" were null (Vietnam). Fixed with
   `na_action="ignore"` and extended the dropna to also drop rows with a
   null `variable`.
2. `build_stable_crops.py` line 122 — `sorted(...unique())` crashed once
   the NaN-variable rows existed, because Python can't compare `float`
   and `str`. Fixed with `.dropna()` before `.unique()`.

Both fixes are one-line and worth sending upstream to the package author.

**Output:** per country, `<country>/_out/stable/stable_<year>.geojson` and
`stats_aggregated.csv`, keyed by `stable_id`.

**Key discovery — boundary vintages are NOT interchangeable snapshots.**
Initially assumed every `stable_<year>.geojson` for a country was the same
set of polygons with different attributes, and tried merging the union of
all years into one boundary set. This was wrong. Inspection showed:
- Bangladesh's 1983 file has 21 features, 1984 has 22, 1985 has 64.
- The `n_modern` column reveals why: 1983 (the lineage's *base* year) merges
  up to 6 modern administrative units into one polygon — it's the coarsest
  partition, the one that stays geometrically valid for the *entire*
  1983–2024 window. 1985's 64 features are nearly 1:1 with modern units,
  because by 1985 most of the splits that would otherwise require merging
  had already resolved.
- Confirmed against `stats_aggregated.csv`: Bangladesh has exactly 22
  unique `stable_id` values across all years (21 base groups + 1 raw
  `late_reporting` unit_id) — matching `n_stable_groups_at_base`, not the
  64-feature 1985 file.

**Conclusion:** for any analysis that joins against `stats_aggregated.csv`,
use *only* the lineage's base-year boundary file (TH 1978, BD 1983,
VN 1980). Geometry for any given `stable_id` is identical across every
file it appears in — so each district only needs to be rasterized once,
and those weights are reusable across the full time series.

---

## 2. CROPGRIDs crop-area data

The India work only had CROPGRIDs v1.08 layers for rice, maize, and
sorghum — insufficient for the new countries' crop mix (Thai cassava,
Bangladeshi wheat, Vietnamese sugarcane/groundnut, soybean across all
three).

Downloaded the full `CROPGRIDSv1.08_NC_maps.zip` (769 MB, all 26 crops)
from figshare and extracted only the 5 needed files into
`data/raw/cropgrids/`:
`cassava`, `wheat`, `soybean`, `sugarcane`, `groundnut` (singular — the
plural `groundnuts` does not exist as a filename in the archive).

Crops with no CROPGRIDs equivalent (jute, betel, tobacco, rambutan, etc.)
are out of scope for SIF reweighting — expected, not a gap to fix.

---

## 3. GOSIF SIF extraction

**Source:** GOSIF v2 monthly product (Li & Xiao 2019, doi:10.3390/rs11050517),
University of New Hampshire Global Ecology Group. Downloaded as
`Monthly.tar` containing 298 monthly global GeoTIFFs
(`GOSIF_<year>.M<month>.tif.gz`, March 2000 – December 2024).

The India pipeline previously used a pre-clipped India-only NetCDF; these
are *global* 0.05-degree rasters and needed clipping per country.

**Script:** `pipeline/extract_gosif.py`
- Reads each `.tif.gz` directly out of the tar in memory (no full
  extraction to disk — 298 files x ~52MB decompressed would be ~15GB).
- Clips to each country's bounding box (+1 degree buffer).
- Applies GOSIF's documented encoding: raw int16 x scale factor 0.0001,
  fill value -9999.
- Stacks all months into one NetCDF per country: `data/processed/gosif_<cc>.nc`,
  dimensions `(time, lat, lon)`, variable `sif` in W m-2 um-1 sr-1.

**Output sizes:** TH 45.5MB, BD 17.6MB, VN 42.5MB. SIF value ranges checked
sane (roughly -0.07 to 3.28, the upper bound being an int16 quantization
ceiling on rare extreme pixels, not a real ceiling on typical values).

---

## 4. Crop-fraction-weighted SIF aggregation

**Why this step exists:** a GOSIF pixel (0.05 deg, ~5.5km) mixes the
photosynthetic signal of everything green in it — not just the crop being
studied. Naively averaging SIF over a district pollutes the signal with
forest, other crops, fallow land, etc. CROPGRIDs gives, per pixel, the
physical area planted with a specific crop; using that as a weight before
spatial aggregation recovers the SIF signal attributable to that crop:

    SIF_crop(district, month) = sum_px(SIF_px * croparea_px) / sum_px(croparea_px)

restricted to pixels inside the district. This is done pixel-by-pixel
*before* aggregating to the district — never average SIF to a district
mean first and multiply by a district-level fraction afterward, since that
discards the spatial distribution information the weighting exists to use.

**Decisions made along the way (confirmed with the user):**
- Use `croparea` (physical extent), not `harvarea` (harvested area) —
  harvested area can exceed physical area for multi-season fields, which
  would double-count against a once-per-month SIF signal.
- Accept CROPGRIDs' single ~2020 snapshot applied across the full
  2000–2024 SIF series, i.e. assume cropland geography hasn't shifted
  dramatically over the window. Known limitation, not fixed for this pass.
- FEWS crop slug -> CROPGRIDs file name mapping (no 1:1 match between the
  two naming conventions):

  | Country | FEWS slug | CROPGRIDs file |
  |---|---|---|
  | TH | rice | rice |
  | TH | soybeans | soybean |
  | BD | rice_paddy | rice |
  | BD | wheat_grain | wheat |
  | BD | maize_corn | maize |
  | BD | soybean_unspecified | soybean |
  | BD | groundnuts_in_shell | groundnut |
  | VN | rice_paddy | rice |
  | VN | maize_corn | maize |
  | VN | cassava | cassava |
  | VN | soybean / soybean_unspecified | soybean |
  | VN | sugarcane / sugarcane_for_sugar | sugarcane |
  | VN | groundnuts_in_shell | groundnut |

  Note: VN's soybean/soybean_unspecified and sugarcane/sugarcane_for_sugar
  pairs map to the *same* CROPGRIDs file and currently produce duplicate
  rows under two crop labels with identical SIF values. Not yet collapsed
  into one canonical label — flagged for the yield join step.

**Script:** `pipeline/aggregate_sif_by_crop.py`
- Confirmed CROPGRIDs and GOSIF share the exact same global 0.05-degree
  grid (pixel centers match) — no resampling needed, just index alignment.
- Loads the single base-year boundary file per country (see Section 1),
  rasterizes each district once, reuses those weights across all 298 months
  and every crop.

**Output:** `data/processed/sif_by_crop_<cc>.csv`, columns
`stable_id, crop, year, month, sif_weighted, n_pixels, croparea_ha`.
District counts matched `n_stable_groups_at_base` exactly after the
boundary-vintage fix: TH 72, BD 21, VN 38.

Sanity check on Bangladesh rice showed a sensible seasonal curve (rising
through spring, peaking June–October during the monsoon, dipping in
winter) — consistent with expected rice phenology.

---

## 5. Annual aggregation (yield + SIF to one yearly grain)

**Why:** the FEWS yield data reports multiple season rows per crop per
year (e.g. Bangladesh rice's Aman/Aus/Boro), and SIF was still at monthly
granularity. Needed one row per `(stable_id, crop, year)` to do any
variance/correlation analysis. Season-level growing-calendar windows are
not available yet, so this is a deliberate simplification, not the final
form.

**Key discovery — season rows are not always additive.** Checked, per
crop, whether multiple season rows ever co-occur for the same
district-year, and whether any crop also reports an all-encompassing
annual total alongside its sub-seasons (which would double-count if both
were summed):
- Thailand soybeans: "All Year" co-occurs with "Dry" and "Wet" in the same
  district-year 863 times — "All Year" is already the annual total, not a
  third season. Summing all three would triple-count.
- Thailand rice, Bangladesh rice (Aman/Aus/Boro), and Vietnam rice
  (regional/seasonal splits) have no such overlap — their multiple season
  rows are genuinely disjoint cropping cycles within the same calendar
  year, so summing them is correct.
- All other crops checked report a single season row per district-year
  (mostly "Annual" or "Calendar Year") — trivial, no collapse needed.

**Rule applied:** if an all-encompassing label (`All Year` / `Annual` /
`Calendar Year`) is present in a district-year group, use only that row;
otherwise sum all season rows present.

**Yield recomputed as an intensive ratio** (production_mt / area_ha) on
the collapsed annual totals — never averaged from per-season reported
yields, per the `stablebound` README's explicit warning. Area preference:
harvested area if reported, falling back to planted area (Bangladesh and
Vietnam report only planted area for several crops, confirmed empirically
— not just for rice paddy as the README's caveat singled out).

**Script:** `pipeline/aggregate_annual.py`. Guards against a literal
`ZeroDivisionError` when `area_ha == 0` for some Bangladesh rows
(replaces with NaN rather than dividing).

**Output:**
- `data/processed/annual_yield_<cc>.csv` —
  `stable_id, crop, year, production_mt, area_ha, area_type, yield_mt_ha`
- `data/processed/annual_sif_<cc>.csv` —
  `stable_id, crop, year, sif_annual_mean, n_months` (flat 12-month mean;
  not yet a growing-season-matched window)
- `data/processed/annual_joined_<cc>.csv` — the two merged

Row counts: TH 2,343 (joined), BD 1,584 (1,488 with valid yield), VN 3,518
(3,494 with valid yield).

**Already-visible data quality findings** (not yet formally flagged, just
noticed during sanity checks):
- Vietnam has a yield value of 299 mt/ha — physically impossible for any
  crop in this set (real cassava/sugarcane yields top out around 30–80
  mt/ha). Strong candidate for the suspicious-value check below.
- A handful of rows (2 in BD, 2 in VN) have yield exactly 0 with positive
  reported area — genuine crop-failure events (area planted, nothing
  harvested), not a "crop isn't grown here" artifact. Confirmed by
  inspecting the actual rows rather than assuming.
- The annual tables are not a complete panel: a row only exists where the
  source FEWS data had an actual reported value. A district that has
  never grown a crop is correctly and silently absent (not coded as a
  zero) — but a district with a sporadic reporting gap in an otherwise
  continuous series will also just be silently absent for that year,
  which matters for any variability calculation done downstream (a CV
  computed over a sparse, gappy series is not directly comparable to one
  computed over a complete series).

---

## 6. Yield QA/QC checks (in progress)

Before bringing SIF into any correlation/variance analysis, decided to
build out yield-only data-quality checks first — same spirit as the
India `fig_yield_cv.py` low-CV flag, generalized and expanded.

**Checks selected to start with:**
1. **Reported vs. calculated yield mismatch.** The raw FEWS CSVs carry
   both a "Yield: MT/ha (reported)" and "Yield: MT/ha (calculated)"
   column — calculated is presumably production/area computed by the
   source compiler. Checked the disagreement between the two directly in
   the raw data before building anything: median disagreement is
   negligible (<0.1%) in all three countries, as expected, but Thailand
   has 705 rows and Vietnam 144 rows disagreeing by more than 50% (versus
   only 4 in Bangladesh). These are concrete, free QA flags already
   present in the source data, independent of anything computed
   downstream.
2. **Low coefficient-of-variation ("suspiciously flat yield").** Direct
   generalization of the India `fig_yield_cv.py` check — districts whose
   yield barely changes year to year, which tends to indicate
   copy-forward/stale data entry rather than a genuinely stable harvest.
   This is the basis for a yield-variability choropleth map per crop.

**Checks considered and explicitly deferred:**
- **Implausible yield bounds** (flagging values above a crop's physical
  maximum) — decided against building this now, since a defensible
  ceiling needs crop-specific agronomic research to avoid being an
  arbitrary cutoff. Will revisit if/when that research is done. The
  Vietnam 299 mt/ha case will very likely still be caught indirectly by
  the CV and jump-detection checks even without a hardcoded ceiling.

**Plan:** build checks 1 and 2 one country at a time, starting with
Bangladesh (21 districts — small enough to fully inspect by eye, and the
cleanest reported/calculated agreement of the three).

**Script:** `pipeline/qaqc_yield_checks.py`. Check 1 runs directly on the
raw FEWS CSV (row-level, before any stable-boundary aggregation, since
that's the level at which a mismatch is interpretable). Check 2 (low CV)
runs on the already-collapsed `annual_yield_<cc>.csv`, requires a minimum
of 5 reported years per (district, crop) series before a CV is considered
meaningful, and produces a choropleth per crop with low-CV districts
outlined in red — same visual language as the India `fig_yield_cv.py`
check.

### Bangladesh results (first country run)

- **Reported vs. calculated mismatch:** only 8 rows out of 34,497
  comparable rows exceed 10% disagreement (~0.02%) — confirms Bangladesh
  is the cleanest of the three countries on this measure, consistent with
  the very small overlap seen earlier when this check was first scoped.
- **Low-CV check:** 92 of 98 (district, crop) series have enough years
  (>=5) to evaluate; **zero** were flagged below the 5% CV threshold.
  Checked this wasn't a miscalibrated threshold by inspecting the actual
  CV distribution: minimum CV across all eligible series is 0.126, 25th
  percentile 0.257 — every series genuinely varies well above the
  threshold. This is a real finding, not a non-result: Bangladesh's
  reported yields show healthy year-to-year variation across the board,
  no sign of copy-forward/stale entries by this measure.
- The rice_paddy CV map shows a real, visually striking spatial pattern
  worth investigating on its own merits (independent of any QA flag): one
  central/southwestern district has CV > 0.40 while several southeastern
  coastal districts sit around 0.15 — i.e. genuinely more volatile rice
  yields inland than on the coast. Not a data-quality issue, just an
  observation surfaced by building the map.

**Next:** run the same two checks on Thailand and Vietnam, where the
reported-vs-calculated mismatch counts were known to be much higher (705
and 144 rows respectively) before any district-level aggregation — worth
seeing whether those concentrate in particular crops or districts once
mapped.

### Addition — relative (z-score) CV outlier check

The absolute low-CV threshold (CV<5%) found zero flags in Bangladesh, but
that doesn't mean every district behaves the same — it only checks against
a fixed floor. Added a second, complementary check: for each crop,
compute the mean and std of CV *across that crop's eligible districts*,
then z-score each district's CV against its same-crop peers and flag
|z| > 2. This catches districts that stand out from the pack on either
end (anomalously volatile OR anomalously flat relative to other districts
growing the same crop), even when no single district crosses an absolute
floor.

Guarded against small-sample instability: a crop's mean/std (and
therefore z-scores) is only computed if at least 5 districts are eligible
for that crop; below that, the check is skipped and reported as such
rather than producing an unreliable z-score off a handful of districts.

**Bangladesh relative outliers found:**

| Crop | Eligible districts | Mean CV | Outliers (\|z\|>2) |
|---|---|---|---|
| groundnuts_in_shell | 21 | 0.478 | 2 |
| maize_corn | 21 | 0.416 | 1 |
| rice_paddy | 21 | 0.292 | 2 |
| soybean_unspecified | 8 | 0.342 | 0 |
| wheat_grain | 21 | 0.339 | 1 |

Maize's single outlier is a sharp, visually obvious one district in the
southeast (z>3, clearly separated from the rest of the country which
clusters in the blue/neutral range on the diverging map) — confirms a
pattern the user had already suspected by eye before the check was built.
Soybean had no district cross the z>2 threshold, but one southeastern
district trends high (~z=1.3) on a thin sample (only 8 of 21 districts
grow enough soybean to be eligible) — worth more weight once more years
or districts are available, not dismissed outright.

Output now lands in a per-country figure subfolder (`figures/<cc>/`)
rather than the flat `figures/` directory, with two maps per crop: the
absolute-CV choropleth (as before, red outline = below the absolute
threshold) and a new diverging-colormap z-score map (black outline =
relative outlier).

### Thailand results

- **Reported vs. calculated mismatch:** 761 of 21,910 comparable rows
  (3.5%) exceed 10% disagreement — but 691 of those 761 (91%) are a
  single crop, `coconut_dry`. Investigated rather than just reporting the
  count: the reported/calculated ratio for every flagged coconut_dry row
  is almost exactly 0.001, consistently. This is a systematic unit-
  conversion error (likely kg vs. MT, or an equivalent factor-of-1000
  mismatch) isolated to that one crop in the source data, not scattered
  noise across many crops. This is exactly the kind of issue check 1 is
  meant to catch, and it generalizes cleanly: this crop's reported and
  calculated yields should not be mixed in downstream analysis without
  resolving which one (if either) is correct.
- **Low-CV check:** 0 absolute flags (same pattern as Bangladesh — no
  district crosses the 5% floor).
- **Relative CV outlier check:** rice had 3 outliers among 72 eligible
  districts (mean CV 0.193); soybeans had 1 among 42 (mean CV 0.163).
  The rice z-score map shows two of the three outliers as adjacent
  districts in the far north (both low-CV relative to peers) and one
  separate high-CV outlier in the northwest interior — the two adjacent
  low-CV districts being neighbors suggests a shared regional reporting
  pattern rather than two independent coincidences, worth a closer look.

### Vietnam results

- **Reported vs. calculated mismatch:** 526 of 16,875 comparable rows
  (3.1%) exceed 10% disagreement, but unlike Thailand this is spread
  across crops (soybean 139, peanut 95, sugarcane 90, maize 77, cassava
  62) with no single dominant crop. Checked whether a single systematic
  ratio explains it the way it did for Thailand's coconut_dry — it does
  not: the reported/calculated ratio for these rows ranges widely (e.g.
  soybean 0.089–11.667, sugarcane 0.069–13.5) with no consistent factor.
  This looks like genuine row-level noise/data-entry inconsistency rather
  than one systematic unit bug.
- **Low-CV check:** Vietnam is the first of the three countries with
  **absolute** low-CV flags — 4 total (3 sugarcane, 1 soybean_unspecified).
  Inspected the actual flagged rows rather than just the count: all 3
  sugarcane flags span the *exact same decade*, 1980–1990 — the earliest
  years of Vietnam's lineage (base year 1980). This is a meaningful
  temporal pattern, not three independent coincidences: it suggests early-
  period sugarcane figures may have been estimated, interpolated, or
  otherwise less granularly reported than later years, rather than
  reflecting genuinely flat real-world yields. Worth treating pre-1990
  Vietnamese sugarcane yield with extra caution in any downstream
  analysis. The soybean_unspecified flag (1995–2015) does not share this
  pattern and may be a separate, genuinely stable case.
- **Relative CV outlier check:** outliers found in every crop checked
  (cassava 3/38, groundnuts 1/16, maize 3/38, rice_paddy 1/38, soybean
  2/32, soybean_unspecified 2/19, sugarcane 3/35, sugarcane_for_sugar
  1/25). Cassava's two highest outliers are geographically adjacent
  districts on the south-central coast — again, spatial clustering of
  outliers rather than scattered randomness, suggesting either a shared
  regional data/reporting quirk or a genuine local agroclimatic effect
  worth investigating on its own merits.

**Status:** all three countries now have both checks run, with results
in `data/processed/qaqc_reported_vs_calc_<cc>.csv`,
`data/processed/qaqc_low_cv_<cc>.csv`, and maps in `figures/<cc>/`.
Thailand's coconut_dry unit-error finding and Vietnam's pre-1990
sugarcane pattern are the two most concrete, actionable findings so far —
both are specific enough to act on (exclude/relabel coconut_dry yield in
TH; flag pre-1990 sugarcane in VN as lower-confidence) rather than vague
"something might be wrong here" signals.

### Checks 4 & 5 — single-year anomalies and exact-repeat runs

Added two more checks to `pipeline/qaqc_yield_checks.py`, both operating
on the same collapsed `annual_yield_<cc>.csv` series used for the CV
checks:

**Check 4 — single-year anomaly.** Compares each interior year's yield to
the average of its immediate neighbors (year-1, year+1), flagging a
ratio beyond 2.5x or below 0.4x. Critically, only compares when both
neighbors are *truly adjacent* years in the data (no reporting gap) —
otherwise the comparison spans an unknown gap and isn't meaningful. This
is deliberately built to catch an isolated spike/dip that reverts
afterward, not a genuine multi-year trend shift (which it should not, and
does not, flag).

**Check 5 — exact-repeat run detection.** Flags runs of consecutive,
truly-adjacent years reporting an exactly identical yield value. Reports
all runs of length >=3, hard-flags at length >=4. Also checks whether
`production_mt` and `area_ha` are themselves exactly repeated across the
same run — a stronger signal than yield alone coincidentally matching,
since yield is a recomputed ratio that could in principle repeat even if
the underlying production/area pair changes.

**Results:**

- **Bangladesh:** 36 single-year anomalies (concentrated in maize_corn
  and groundnuts_in_shell — the two crops with the smallest cultivated
  area among BD's tracked crops, plausibly more sensitive to reporting
  noise). Zero exact-repeat runs — consistent with Bangladesh's
  consistently clean profile across every check run so far.
- **Thailand:** only 1 single-year anomaly total, in rice — inspected
  directly: a district's rice yield crashed to 0.33 mt/ha in 1988 against
  neighbors of 1.28 (1987) and 1.56 (1989) mt/ha, then fully reverted.
  Could be a real shock (drought/flood) or a transcription error — not
  distinguishable from yield data alone, correctly flagged either way.
  Zero exact-repeat runs.
- **Vietnam:** 26 single-year anomalies, concentrated in cassava (12) and
  soybean (8). More notably, 11 exact-repeat runs of length >=3, with
  **4 hard-flagged at length >=4 — all four in soybean or
  soybean_unspecified.** Inspected these directly and found a striking
  pattern: every hard-flagged run reports a suspiciously round-number
  yield (1.0, 0.5, 2.0, 1.0 mt/ha) repeated for 4-5 consecutive years,
  and in every one of these four cases `production_mt` and `area_ha` are
  themselves NOT identical across the run -- i.e. the underlying
  production and area figures change year to year, but happen to divide
  out to the exact same round number every time. That is a different and
  arguably more concerning pattern than literal row duplication: it looks
  like production and area may be co-estimated from an assumed fixed
  yield for these district-crop-years, rather than being independently
  measured. Three of the four flagged runs fall in the early 1980s
  (1982-1986, 1985-1988, 1982-1985) -- consistent with, and reinforcing,
  the pre-1990 reliability concern already raised by the sugarcane low-CV
  finding above. Soybean yield data in Vietnam's earliest reporting years
  should be treated with real caution in any downstream analysis.

### Revision — district-level anomaly RATE instead of raw count

The first version of Check 4 mapped a raw count of single-year anomalies
per district. That's not a fair comparison across districts with
different amounts of reported history -- a district with 40 years has
more chances to accumulate one-off anomalies than one with 10, even at
the same true underlying probability. Reworked the check (same approach
already used for the CV outlier check) to compute, per (stable_id, crop):
`n_comparisons` (interior years where both neighbors were truly adjacent,
i.e. the denominator of valid comparisons) and `n_anomalies` out of those,
giving an `anomaly_rate`. That rate is then z-scored against same-crop
peer districts, with the same minimum-eligible-districts guard as the CV
check. One deliberate asymmetry from the CV check: only a HIGH rate is
flagged here (z > 2.0, not |z| > 2.0) -- a district with zero anomalies
isn't suspicious, it's just clean, so the low tail is never flagged.

Output split into two files: `qaqc_year_anomaly_<cc>.csv` (the individual
flagged events, unchanged) and `qaqc_year_anomaly_district_<cc>.csv` (the
new district-level rate + z-score summary, one row per district-crop).
The map now plots `anomaly_rate` as the choropleth with the z-score
outlier flag outlined, replacing the old raw-count map.

**Important finding surfaced by debugging this revision — Bangladesh
2021 maize, a likely COVID-era reporting disruption.** While checking why
`maize_corn` was being skipped entirely by the new district-level z-score
(despite having 17 raw anomaly events under the old count-based map),
found that nearly every maize district has only ~3 valid comparison-years
total, and almost all of them have exactly 1 anomaly out of those 3 (rate
~0.33) -- a suspiciously uniform rate across nearly the whole country.
Inspected the actual flagged events directly: **all 17 anomalies, across
16 of Bangladesh's 21 districts, are the exact same year -- 2021** --
with maize yield collapsing to roughly 7-34% of the surrounding years'
levels nationally, then fully recovering in 2022. A near-universal
single-year collapse across nearly every district, that fully reverses
the very next year, is far more plausible as a reporting/data-collection
disruption than a genuine simultaneous agronomic collapse across the
whole country -- the timing (2021) is consistent with COVID-19 disruption
to agricultural surveying. Bangladesh's 2021 maize figures should be
treated with real caution in any downstream analysis.

This also exposed a genuine blind spot in the relative-outlier (z-score-
vs-peers) framing used throughout this pipeline: when *most or all*
districts move the same anomalous way in the same year, none of them
looks unusual relative to "peers" who are equally affected, so a
same-crop z-score check cannot surface it -- it needs a different lens
(e.g. "what fraction of all districts are anomalous in the same year,"
a cross-sectional/temporal check rather than a per-district one). Not
yet built; noted here as a known gap for a future check.

### Correction — within-series z-score, not neighbor-ratio + cross-district rate

After reviewing the rate-based version above, the user clarified the
actual intent: track a SINGLE district's own yield time series, compute
that series' own mean and std, flag years exceeding |z|>2 against its
own statistics, and map the raw COUNT of anomalous years per
district-crop directly. This is simpler than what had been built (which
used a local neighbor-ratio instead of the series' own mean/std, then
added a second normalization step -- z-scoring the rate across other
districts). Replaced `check_year_anomaly` entirely with this plain,
standard definition. Traded away the neighbor-ratio version's one
advantage (distinguishing an isolated reverting spike from a genuine
multi-year trend shift) for the simpler, more standard approach the user
specifically wants -- a year that's part of a real trend shift will now
also get flagged, which is an accepted tradeoff, not an oversight.

**Statistical caveat checked and noted:** with a fixed |z|>2 threshold,
a series with many years has a meaningfully higher chance of containing
at least one exceedance by pure chance alone (under purely random
variation, P(|z|>2) is about 4.5% per year, compounding across years in
a series). Checked this empirically against Bangladesh's actual data:
correlation between series length (n_years) and anomalous-year count is
0.23 -- a real but modest effect. Importantly, within any single
country's data, `n_years` is fairly constant across districts *for a
given crop* (e.g. Bangladesh wheat_grain districts are nearly all 42
years, maize_corn districts nearly all 7) -- so the district-level
comparison the maps are built for (districts within one crop) stays
fair in practice. The caveat mainly applies if counts are ever compared
*across* crops with very different series lengths (e.g. don't read more
into wheat having more total anomalous years than maize when wheat's
series are also ~6x longer).

**Results (re-run, all three countries):**
- Bangladesh: 78 anomalous-year events; 58 of 92 eligible (district,
  crop) series have at least one. wheat_grain's anomaly map shows a
  clear spatial gradient -- one district has 3 anomalous years (out of
  42) while several show zero, a real spread rather than uniform noise.
- Thailand: 145 anomalous-year events (101 rice, 44 soybeans); 86 of 114
  eligible series have at least one.
- Vietnam: 154 anomalous-year events, spread across all 8 tracked crops;
  105 of 241 eligible series have at least one.

Output files unchanged in name (`qaqc_year_anomaly_<cc>.csv` for
individual events, `qaqc_year_anomaly_district_<cc>.csv` for the
district-level count), but the district file's columns changed:
`n_years, series_mean, n_anomalous_years, eligible` -- the old
rate/z-score-vs-peers columns are gone. Maps now plot
`n_anomalous_years` directly as the choropleth (sequential Reds
colormap), no outlier outline needed since the count itself is already
the quantity of interest.

## 7. District drill-down tool

Decided to leave the anomaly check as a plain z-test for now (Grubbs'
test and the Generalized ESD/Rosner's test were discussed as the more
statistically correct small-sample and multiple-outlier-aware
alternatives, but shelved deliberately to avoid overcomplicating the
early-stage pipeline -- noted here in case revisited later).

The choropleth maps are a first-glance triage tool: they tell you WHICH
district to look at. Built `pipeline/district_drilldown.py` as the
second-glance tool -- given a `(country, stable_id)`, it assembles
everything a researcher would want next, without re-running every check
by hand:

1. Every admin name the district has ever carried (via stablebound's
   `name_history.csv` -- the join key needed to pull the reported-vs-
   calculated check's results, since that check operates on raw admin
   names, not `stable_id`).
2. Per crop: CV flag status, every anomalous year with its yield value
   AND its immediate neighboring years shown alongside (so it's visible
   at a glance whether it's an isolated spike or part of a longer move),
   and any exact-repeat runs.
3. **Cross-crop check**: for every anomalous year flagged for any crop in
   the district, shows what every OTHER crop in that same district was
   doing that same year -- generalizes the lens that surfaced the
   Bangladesh 2021 maize finding into a reusable per-district tool, on
   the premise that several crops moving unusually in the same year
   points to a shared cause (a real shock, or a reporting/methodology
   issue that year) rather than a crop-specific data problem.
4. Reported-vs-calculated mismatch rows for any admin name tied to the
   district.

Read-only: assembles already-computed results plus raw-data context, does
not change any flag. No figures -- console/DataFrame output for ad hoc
use or notebooks.

**Demo run:** `BD.ADM2.00005` (Brahmanbaria/Chandpur/Comilla), one of the
two Bangladesh districts with 3 anomalous wheat years. The cross-crop
check immediately surfaced something the formal per-crop flags missed:
in 2021 (the same year groundnuts spiked to z=2.73 in this district),
maize_corn's yield was 1.17 against a mean of 4.42 -- a large drop, but
its z-score (~-1.93, computed from the CV-implied std) falls just short
of the |z|>2 cutoff, so it isn't formally flagged. The cross-crop view
surfaces it anyway. This lines up with the country-wide 2021 Bangladesh
maize collapse already found (Section 6) -- a likely COVID-era reporting
disruption -- visible again at the individual-district level even where
the formal per-crop check didn't trigger. This is exactly the kind of
near-miss/shared-cause signal the tool was built to surface.

## 8. Cross-sectional anomaly check (closing the named blind spot)

Section 6 named a real gap: the per-district anomaly check and the
drilldown tool's cross-crop view can each *show* a shared-cause pattern
once a researcher already suspects one (as happened by hand for
Bangladesh's 2021 maize), but neither *finds* "most districts moved the
same anomalous way this year" automatically. Built
`pipeline/cross_sectional_anomaly.py` to close that gap directly: for
each calendar year, what fraction of districts had an anomalous yield
that year, built on top of the already-computed per-series |z|>2 results
(same threshold, same >=5-year eligibility rule -- this re-aggregates
existing results across the district dimension, it is not a new anomaly
definition).

Two views, since "fraction of districts" is ambiguous once a district
grows multiple crops: **pooled** (every district-crop pair is its own
unit -- exact and unambiguous for a single-crop filter) and **district**
(a district counts once if ANY selected crop was anomalous that year --
matches the plain-language "fraction of all districts" framing for the
all-crops case). Both split into high/low/total fractions. The
denominator each year is restricted to district-crop pairs that are both
eligible overall AND actually reported a value that specific year -- a
missing report is excluded from both numerator and denominator, never
silently counted as "not anomalous."

Filterable to one or more crops via `--crop` (e.g. `--crop rice_paddy`),
defaulting to all crops pooled/unioned. No second-order threshold is
applied to the fraction itself -- deliberately, consistent with declining
to invent an arbitrary yield-ceiling cutoff earlier: the summary just
ranks the top years by fraction and leaves judgment to the researcher.
Produces a time-series plot per (country, crop-selection) -- a line chart
makes sense here, not a map, since the statistic itself has no spatial
dimension (it's already aggregated across districts).

**Results, immediately useful:**

- **Bangladesh, all crops:** automatically reproduces the 2021 finding
  from Section 6 (90.5% of districts anomalous that year) without any
  manual drilldown -- exactly the gap this check was built to close.
  But the full time series reveals something bigger than the single-year
  framing in Section 6 suggested: **the whole 1983-2020 period sits under
  ~10%, then 2021 onward is a sustained, elevated regime that never
  returns to baseline** -- 2021 (90.5%), 2022 (~52% low-dominated, visible
  in the plot though not in the top-5-by-year table since 2022 wasn't
  itself top-ranked), 2023 (42.9%, entirely high), 2024 (71.4%, entirely
  high). This reframes the earlier finding: it is not just a one-year
  COVID-disruption blip that reverted, but a persistent post-2021 shift
  in Bangladesh's reported yields across multiple crops -- a new survey
  methodology, a reporting-process change, or unsettled post-pandemic
  data are all plausible explanations, not distinguishable from yield
  data alone. The entire 2021-2024 window deserves caution, not just 2021.
- **Decomposed the 2024 spike by filtering to a single crop**, demonstrating
  exactly the filter functionality this check was built for: rice_paddy
  alone only accounts for 14.3% of districts in 2024, far short of the
  71.4% all-crops figure. Traced the rest directly to events: 14 of 17
  flagged 2024 events are wheat_grain, every single one a POSITIVE
  z-score (unusually high yield), spread across nearly all of
  Bangladesh's wheat-growing districts simultaneously. A near-universal
  one-year wheat yield surge across the whole country, landing in the
  most recent year of the series, is consistent with either a genuine
  favorable season or a provisional/incomplete final-year reporting
  artifact (recent-year agricultural statistics are often revised after
  initial release) -- cannot be distinguished from yield data alone, but
  worth flagging specifically rather than folding into the vaguer
  "2021-2024 is elevated" framing above.
- **Thailand:** 2024 is the standout year (35.4% of districts), and
  filtering to rice alone reproduces the exact same 35.4% -- Thailand only
  tracks rice and soybeans in this pipeline, so rice fully explains it.
  All high-direction (22 high, 1 low) -- another simultaneous yield-surge
  pattern landing in the most recent year, same caveat as Bangladesh's
  2024 wheat.
- **Vietnam:** two distinct clusters. 1980 (50.0%, lineage base year,
  17 low/5 high) and 1981 (31.6%, 12 low) sit right at the start of the
  series -- plausibly an edge effect of being among the first few points
  in many series rather than a real event, though not confirmable either
  way from this check alone. More interestingly, **2022 (60.5%) and 2023
  (57.9%) form a recent, low-dominated cluster** (21 low/5 high in 2022,
  20 low/8 high in 2023) -- the opposite direction and opposite recency
  pattern from Bangladesh/Thailand's high-direction 2023-2024 clusters,
  worth treating as a separate, distinct signal rather than assuming a
  shared cross-country cause.

Output: `qaqc_cross_sectional_pooled_<cc>_<croptag>.csv`,
`qaqc_cross_sectional_district_<cc>_<croptag>.csv`,
`figures/<cc>/qaqc_cross_sectional_<cc>_<croptag>.png` (croptag = `all`
or the filtered crop slug(s)).

**Note on interpreting the district-level high/low fractions:** in the
DISTRICT view, `fraction_districts_high` + `fraction_districts_low` can
exceed `fraction_districts_any` (and the two can individually sum past
100%) because a district counts in "high" if ANY of its crops was high
that year and separately in "low" if ANY was low -- these are not
mutually exclusive once a district grows multiple crops. Confirmed
directly for Bangladesh 2021 (17 high-districts + 11 low-districts = 28,
but only 19 distinct districts/90.5% any-direction -- a 9-district
overlap): all 9 overlapping districts pair the same nationwide maize
collapse (z about -2.0 to -2.2) with a simultaneous groundnuts spike
(z up to +3.5) in the very same district -- one single 2021 event
appearing as opposite-direction anomalies depending on which crop you
look at, not two independent phenomena. The POOLED view does not have
this property (every district-crop pair is its own unit there, so
high+low always equals total exactly); the ambiguity is specific to the
any-crop-per-district aggregation used in the DISTRICT view.

## 9. SIF-yield variance analysis (Step 3 of the original plan, resumed)

Resumed the SIF side after building out the yield QA layer. Built
`pipeline/sif_yield_variance.py`, implementing the methodology agreed
over several rounds of discussion:

- **Detrend before correlating.** Both yield (technology/input trends)
  and SIF (potential land-use or sensor-product drift over 24 years)
  likely carry secular trends; correlating raw values risks a high R^2
  driven by shared trends rather than real year-to-year co-movement.
  Each series gets a linear detrend against year; everything downstream
  uses the residual anomalies, mirroring the original India pipeline's
  use of detrended yield anomalies against climate drivers.
- **Report signed Pearson r, plus R^2 and p-value**, not R^2 alone --
  R^2 throws away direction, and direction turned out to matter (see
  findings below).
- **Two scenarios side by side**: "full" (every reported year) and
  "cleaned" (QA-flagged points/series removed first), so the effect of
  the QA work is visible rather than asserted.

**QA exclusion rules locked in for "cleaned"** (each decided explicitly,
not a blanket "remove everything flagged" -- see Sections 6-8 for the
reasoning trail):
- Single-year anomalies at **|z| > 3**, not the check's own |z| > 2
  flagging threshold. Checked empirically before deciding: at |z|>4
  almost nothing survives (Vietnam: 0 of 154 events) and it stops
  catching the clearly-impossible Vietnam cassava values (z~3.1-3.5);
  |z|>3 keeps most of those while leaving milder, more ambiguous events
  (e.g. Bangladesh's 2021 maize, z~-2.0 to -2.2) in the "full" scenario
  only, on the reasoning that z=2-3 is exactly the range where genuine
  large-but-real shocks and likely data errors overlap in this data, and
  excluding too aggressively would bias the analysis against finding real
  SIF-yield signal in exactly the years it should be strongest.
- Exact-repeat runs, hard-flagged (length>=4): every year in the run,
  not just one point.
- Reported-vs-calculated mismatch (>10%, same threshold as the flag
  itself), joined via name_history.csv.
- Absolute low-CV flag: entire series dropped.
- Relative CV outlier (z-score vs. peers): deliberately NOT excluded --
  could be genuine, informative volatility rather than bad data.

### Bug found and fixed while building this: admin-name aliasing

The reported-vs-calculated mismatch CSV preserves RAW admin names as
published in the source (e.g. Bangladesh's post-2018 official renaming
"Chattogram"), while `name_history.csv` (built by stablebound) uses the
pre-2018 FEWS-standard spellings the lineage was actually built on (e.g.
"Chittagong") -- the exact alias mapping `prepare_stats.py` applies
during ingestion. Caught empirically: "Chattogram" resolved to zero
stable_ids without correcting for this. This affected not just the new
script but also the existing `district_drilldown.py`, which has the same
unaliased name join (filtering the mismatch CSV by lineage-spelling names
without expanding for raw aliases) -- fixed in both places. Could not
import the alias dicts directly from `prepare_stats.py` (that module
imports `stablebound` at load time, which only exists in the separate
`stablebound_env` venv, not the main environment these pipeline scripts
run in), so the three countries' alias dictionaries were copied directly
into both files instead, with a comment to keep them in sync if the
source file's aliases ever change.

Verified the fix didn't silently do nothing: traced Bangladesh's
exclusion count (23 points) down to its individual components by hand
and confirmed every one reconciles (11 groundnuts + 5 wheat + 2 rice + 4
non-tracked-crop rows that are harmless no-ops since those crops don't
exist in the SIF-joined table). The Chattogram-mapped exclusion itself
turned out to be redundant for Bangladesh specifically -- that exact
point was already excluded via the z>3 anomaly path independently (an
extreme reported-vs-calculated mismatch and an extreme within-series
anomaly often flag the same bad value from two different angles) -- but
the fix is still necessary and correct for Thailand and Vietnam, which
have their own alias tables and far more mismatch rows (761 and 526
respectively) where the redundancy may not hold.

### Results

Output per country: `data/processed/sif_yield_corr_<cc>.csv`
(stable_id, crop, scenario, n_years, r, r_squared, p_value), plus a
choropleth map per (crop, scenario) in `figures/<cc>/`, signed r on a
diverging colormap with p<0.05 districts outlined.

- **Bangladesh**: 23 points / 0 series excluded for "cleaned." Weak
  correlations across all 5 crops (mean R^2 0.10-0.20), full vs. cleaned
  nearly identical for every crop except small movements in groundnuts
  and wheat. Rice shows a negative mean r (-0.22); wheat positive
  (+0.23, +0.21 cleaned) -- same direction split visible directly in a
  single district's drilldown output (see below).
- **Thailand**: 77 points excluded for "cleaned," 0 series. Both crops
  show weak correlations (mean R^2 0.05-0.11); rice mean r negative
  (-0.02 to -0.04, near zero either way), soybeans positive (+0.11, both
  scenarios identical).
- **Vietnam**: 287 points / 4 whole series excluded (matching the 4
  absolute low-CV flags found in Section 6: 3 sugarcane, 1
  soybean_unspecified). Two crops (`soybean`, `sugarcane` -- the bare
  labels, as distinct from `soybean_unspecified`/`sugarcane_for_sugar`)
  drop out of this analysis entirely: checked and confirmed their data
  only covers 1980-1990, entirely before GOSIF's coverage begins (March
  2000) -- zero overlapping years, not a bug. Possible minor finding on
  its own: Vietnam may have used the bare crop labels for older-era
  records and switched to the "_unspecified"/"_for_sugar" labels for more
  recent ones, worth keeping in mind if crop-label consistency over time
  ever matters elsewhere.

**The standout finding: Vietnam's `sugarcane_for_sugar` shows a strong,
often statistically significant NEGATIVE correlation** between detrended
yield and detrended SIF, concentrated in southern Vietnam (mean r=-0.32
full / -0.20 cleaned, 13-14 of 25 districts significant at p<0.05,
visible as a clear blue cluster on the map). This is the kind of result
signed r was specifically kept to surface -- R^2 alone would have shown
"some relationship" without revealing it runs backwards from the naive
expectation that more photosynthetic signal should mean more yield. Not
explained by this analysis alone; a real candidate for follow-up.

**A more general, cross-country observation**: rice shows weak,
near-zero, sometimes slightly negative mean correlations in **all three
countries** (Bangladesh -0.22, Vietnam -0.03, Thailand -0.02 to -0.04).
Given rice is the crop SIF would most plausibly track well, this weak
showing across the board is more likely a symptom of the known
flat-12-month-mean SIF limitation (not yet growing-season-matched, per
the simplification accepted earlier in the project) than evidence SIF
and rice yield are genuinely unrelated -- a strong argument for revisiting
growing-season-matched SIF aggregation once calendar data is available,
rather than concluding the SIF signal itself is uninformative.

### District drilldown tool: SIF-yield stats added

Added a `sif_yield_corr` section to `district_drilldown.py`, printed
alongside each crop's CV status -- shows r, R^2, p-value, n_years, and
significance for both scenarios. Demo run on `BD.ADM2.00005` (the same
district used in Section 7's demo) surfaced an interesting detail
directly: wheat shows r=+0.379 (p=0.061) while rice in the SAME district
shows r=-0.365 (p=0.073) -- opposite-signed relationships for two
different crops in one place, visible side by side without needing to
cross-reference separate files.

## 10. Reproducibility layer — per-country YAML config

**Motivation:** From a reproducibility standpoint, adding a new country
required editing 5–6 separate pipeline scripts. Per-country values
(bounding box, base year, crop map, admin-name aliases, file paths) were
stored as hardcoded dicts inside each script, with `ADMIN_NAME_ALIASES`
duplicated verbatim between `sif_yield_variance.py` and
`district_drilldown.py` — a sync hazard as well as an onboarding burden.

**What was built:**

- `pipeline/countries.yaml`: single source of truth for all per-country
  config. Sections: `name`, `bbox`, `stablebound_dir`, `raw_csv`,
  `base_year`, `crop_map`, `admin_name_aliases`. Includes detailed inline
  comments explaining each field so a new user knows exactly what to fill
  in without reading the code.
- `pipeline/pipeline_config.py`: loader module. `all_codes()` returns the
  list of configured country codes (in YAML order). `get_country(cc)`
  returns a dict with both the raw YAML values and derived Path objects
  (`boundary_path`, `name_history_path`, `stats_aggregated_path`,
  `raw_csv_path`) so callers never have to assemble file paths manually.
  YAML is parsed once and cached; subsequent calls to `get_country` are
  pure dict lookups.

**Scripts updated** (each now imports `all_codes`, `get_country` from
`pipeline_config` and drops its own hardcoded dicts):
- `extract_gosif.py` — removed `COUNTRIES` dict
- `aggregate_sif_by_crop.py` — removed `CROP_MAP`, `STABLE_DIR`,
  `BASE_YEAR`, `SIF_FILE`
- `aggregate_annual.py` — removed `STATS_PATH`
- `qaqc_yield_checks.py` — removed `RAW_CSV`, `BOUNDARY_PATH`
- `sif_yield_variance.py` — removed `ADMIN_NAME_ALIASES`, `BOUNDARY_PATH`,
  `NAME_HISTORY`, `COUNTRIES`
- `district_drilldown.py` — removed `ADMIN_NAME_ALIASES`, `RAW_CSV`,
  `NAME_HISTORY`
- `cross_sectional_anomaly.py` — removed `COUNTRIES` list

**To add a new country:** copy one YAML block, change the two-letter code,
fill in the four required sections (bbox, stablebound_dir, raw_csv,
crop_map). All scripts pick it up automatically with no code changes.
PyYAML (`pyyaml` package) is required; installed in the main environment.

**Verified:** `pipeline_config.py` imports cleanly, all three countries'
derived paths resolve to existing files (`boundary_path`, `raw_csv_path`,
`name_history_path`, `stats_aggregated_path` all return `exists: True`).
No stale references to the old hardcoded dicts remain in any script
(confirmed by regex grep across all pipeline .py files).

*(To be continued as work proceeds.)*
