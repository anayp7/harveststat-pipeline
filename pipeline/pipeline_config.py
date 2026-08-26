"""
Country configuration loader for the HarvestStat pipeline.

All per-country config lives in pipeline/countries.yaml — the single file
a researcher edits when adding a new country.  Every pipeline script imports
from here instead of maintaining its own hardcoded dicts.

Usage:
    from pipeline_config import all_codes, get_country

    cfg = get_country("BD")
    # cfg["name"]                  -> "Bangladesh"
    # cfg["bbox"]                  -> (87.0, 19.0, 94.0, 28.0)
    # cfg["base_year"]             -> 1983
    # cfg["crop_map"]              -> {"rice_paddy": "rice", ...}
    # cfg["admin_name_aliases"]    -> {"Chattogram": "Chittagong", ...}
    # cfg["raw_csv_path"]          -> Path(...bd_standard_apr2926_forjonah.csv)
    # cfg["stablebound_dir"]       -> Path(...bangladesh)
    # cfg["boundary_path"]         -> Path(...stable_1983.geojson)
    # cfg["name_history_path"]     -> Path(...name_history.csv)
    # cfg["stats_aggregated_path"] -> Path(...stats_aggregated.csv)
"""

from pathlib import Path
import sys
import yaml

_HERE = Path(__file__).resolve().parent
BASE  = _HERE.parent          # project root (one level above pipeline/)
_YAML = _HERE / "countries.yaml"

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with open(_YAML, encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def all_codes() -> list[str]:
    """Return the list of configured country codes, in YAML order."""
    return list(_load().keys())


def get_country(cc: str) -> dict:
    """
    Return a config dict for country code *cc*.

    Adds derived Path keys on top of the raw YAML values so callers
    never have to assemble file paths themselves.
    """
    data = _load()
    if cc not in data:
        raise KeyError(f"Unknown country code '{cc}'. Available: {list(data)}")

    raw = dict(data[cc])

    # Resolve paths relative to project root
    sb_dir = BASE / raw["stablebound_dir"]
    raw["stablebound_dir"]        = sb_dir
    raw["raw_csv_path"]           = BASE / raw["raw_csv"]
    raw["boundary_path"]          = sb_dir / "_out" / "stable" / f"stable_{raw['base_year']}.geojson"
    raw["name_history_path"]      = sb_dir / "_out" / "stable" / "name_history.csv"
    raw["stats_aggregated_path"]  = sb_dir / "_out" / "stable" / "stats_aggregated.csv"

    # Normalise types
    raw["bbox"]               = tuple(raw["bbox"])
    raw["admin_name_aliases"] = raw.get("admin_name_aliases") or {}
    raw["crop_map"]           = raw.get("crop_map") or {}

    # Which stats_aggregated variables carry area, in priority order, and what
    # kind of area each is. FEWS-sourced countries (TH/BD/VN) report harvested
    # and planted separately; India's DESAGRI reports a single planted `area_ha`.
    # Default preserves the original harvested-then-planted behaviour.
    raw["area_measures"] = [
        tuple(m) for m in raw.get("area_measures")
        or [["area_harvested_ha", "harvested"], ["area_planted_ha", "planted"]]
    ]

    return raw
