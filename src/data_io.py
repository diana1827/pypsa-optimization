from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


# -----------------------
# File mapping
# -----------------------

TS_FILES = {
    "loads_p_set": "loads-p_set.xlsx",
    "gen_cost": "generators-marginal_cost.xlsx",
}

STATIC_FILES = {
    "loads": "loads.xlsx",
}


# -----------------------
# Config loading
# -----------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    Must contain 'data_path'.
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "data_path" not in cfg:
        raise KeyError("config.yaml must include 'data_path'")

    return cfg


# -----------------------
# File reading helpers
# -----------------------

def read_excel(path: Path, index_col: int | None = 0) -> pd.DataFrame:
    """
    Read Excel file safely.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_excel(path, index_col=index_col, engine="openpyxl")


def ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensures DataFrame index is a valid DatetimeIndex.
    """
    df = df.copy()

    try:
        df.index = pd.to_datetime(df.index, errors="raise")
    except Exception as e:
        raise ValueError(f"{name}: failed datetime conversion → {e}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex")

    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicated timestamps detected")

    # Convert timezone-aware → naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df


def ensure_numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensures all values in DataFrame are numeric.
    """
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name}: NaNs in columns {bad_cols}")

    return df


# -----------------------
# Main loader
# -----------------------

def load_baseline_data(config_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Loads and validates all required input datasets.
    """

    cfg = load_config(config_path)
    data_path = Path(cfg["data_path"])

    # -----------------------
    # Load static metadata
    # -----------------------
    loads = read_excel(data_path / STATIC_FILES["loads"], index_col=None)

    if "name" not in loads.columns:
        raise ValueError("loads.xlsx must contain a 'name' column")

    loads["name"] = loads["name"].astype(str).str.strip()

    if loads["name"].isna().any():
        raise ValueError("Invalid values in 'name' column")

    # -----------------------
    # Load time series
    # -----------------------
    loads_p_set = read_excel(data_path / TS_FILES["loads_p_set"])
    gen_cost = read_excel(data_path / TS_FILES["gen_cost"])

    loads_p_set = ensure_datetime_index(loads_p_set, "loads_p_set")
    gen_cost = ensure_datetime_index(gen_cost, "gen_cost")

    loads_p_set = ensure_numeric(loads_p_set, "loads_p_set")
    gen_cost = ensure_numeric(gen_cost, "gen_cost")

    # -----------------------
    # Consistency checks
    # -----------------------
    if not loads_p_set.index.equals(gen_cost.index):
        raise ValueError("Time indices do not match")

    load_names = loads["name"].tolist()
    missing = sorted(set(load_names) - set(loads_p_set.columns))

    if missing:
        raise ValueError(f"Missing columns in loads_p_set: {missing}")

    # Reorder columns
    loads_p_set = loads_p_set.reindex(columns=load_names)

    return {
        "loads": loads,
        "loads_p_set": loads_p_set,
        "gen_cost": gen_cost,
    }