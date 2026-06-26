from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yaml


# Mapeamento dos ficheiros usados no baseline/dashboard.
TS_FILES = {
    "loads_p_set": "loads-p_set.xlsx",
    "gen_cost": "generators-marginal_cost.xlsx",
    "pv_data": "PV_Data_2022.csv",
}

STATIC_FILES = {
    "loads": "loads.xlsx",
}

# Ficheiros adicionais para a primeira fase da otimização térmica.
OPT_TS_FILES = {
    "zone_temperatures": "zone_temperatures.xlsx",
    "outdoor_temperature": "outdoor_temperature.xlsx",
    "temperature_bounds": "temperature_bounds.xlsx",
}

OPT_STATIC_FILES = {
    "zone_params": "zone_params.xlsx",
}

DEFAULT_HVAC_CATEGORIES = ["ventilation", "pumping", "cooling"]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Carrega o ficheiro de configuração e valida os campos mínimos.
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "data_path" not in cfg:
        raise KeyError("config.yaml must include 'data_path'")

    return cfg


def read_excel(path: Path, index_col: int | None = 0) -> pd.DataFrame:
    """
    Lê um ficheiro Excel e falha logo se o ficheiro não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_excel(path, index_col=index_col, engine="openpyxl")


def ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Garante que o índice é temporal, sem duplicados e ordenado.
    """
    df = df.copy()

    try:
        df.index = pd.to_datetime(df.index, errors="raise")
    except Exception as e:
        raise ValueError(f"{name}: failed datetime conversion -> {e}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex")

    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicated timestamps detected")

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index()


def ensure_numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Converte todas as colunas para numérico e rejeita NaNs resultantes.
    """
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name}: NaNs in columns {bad_cols}")

    return df


def read_pv_csv(path: Path) -> pd.DataFrame:
    """
    Lê o ficheiro CSV de produção fotovoltaica e devolve potência em W.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"datetime", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"PV CSV is missing required columns: {sorted(missing)}")

    try:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="raise")
        df["datetime"] = dt.dt.tz_convert(None)
    except Exception as e:
        raise ValueError(f"pv_data: failed datetime conversion -> {e}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        raise ValueError("pv_data: NaNs detected in column 'value'")

    unit_factor = 1.0
    if "units" in df.columns and not df["units"].dropna().empty:
        units = {str(u).strip().lower() for u in df["units"].dropna().unique()}
        if len(units) != 1:
            raise ValueError(f"pv_data: mixed units detected: {sorted(units)}")

        unit = next(iter(units))
        if unit == "w":
            unit_factor = 1.0
        elif unit == "kw":
            unit_factor = 1000.0
        elif unit == "mw":
            unit_factor = 1_000_000.0
        else:
            raise ValueError(
                f"pv_data: unsupported unit '{unit}'. Supported: w, kw, mw."
            )

    df["value_w"] = df["value"] * unit_factor

    if "variable" not in df.columns:
        df["variable"] = "pv_total"
    else:
        df["variable"] = df["variable"].astype(str).str.strip()
        df.loc[df["variable"] == "", "variable"] = "pv_total"

    pv_df = (
        df.pivot_table(
            index="datetime",
            columns="variable",
            values="value_w",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    )

    pv_df.columns = [str(c).strip() for c in pv_df.columns]
    pv_df = ensure_datetime_index(pv_df, "pv_data")
    pv_df = ensure_numeric(pv_df, "pv_data")

    return pv_df.rename_axis("datetime")


def _normalize_names(values: Iterable[Any]) -> list[str]:
    """
    Normaliza nomes para texto limpo; valores vazios ficam como string vazia.
    """
    out: list[str] = []
    for v in values:
        if pd.isna(v):
            out.append("")
        else:
            out.append(str(v).strip())
    return out


def _read_optional_timeseries(path: Path, label: str) -> pd.DataFrame | None:
    """
    Lê uma série temporal opcional. Se o ficheiro não existir, devolve None.
    """
    if not path.exists():
        return None

    df = read_excel(path)
    df = ensure_datetime_index(df, label)
    df = ensure_numeric(df, label)
    return df


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Procura a primeira coluna cujo nome corresponda a uma das opções dadas.
    """
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_baseline_data(config_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Carrega e valida os dados necessários para o baseline.

    Se existir ficheiro de PV, tenta também carregá-lo.
    """
    cfg = load_config(config_path)
    data_path = Path(cfg["data_path"])

    loads = read_excel(data_path / STATIC_FILES["loads"], index_col=None)

    if "name" not in loads.columns:
        raise ValueError("loads.xlsx must contain a 'name' column")

    loads["name"] = _normalize_names(loads["name"])
    if any(v == "" for v in loads["name"]):
        raise ValueError("Invalid values in 'name' column")

    loads_p_set = read_excel(data_path / TS_FILES["loads_p_set"])
    gen_cost = read_excel(data_path / TS_FILES["gen_cost"])

    loads_p_set = ensure_datetime_index(loads_p_set, "loads_p_set")
    gen_cost = ensure_datetime_index(gen_cost, "gen_cost")

    loads_p_set = ensure_numeric(loads_p_set, "loads_p_set")
    gen_cost = ensure_numeric(gen_cost, "gen_cost")

    if not loads_p_set.index.equals(gen_cost.index):
        raise ValueError("Time indices do not match")

    load_names = loads["name"].tolist()
    missing = sorted(set(load_names) - set(loads_p_set.columns))
    if missing:
        raise ValueError(f"Missing columns in loads_p_set: {missing}")

    loads_p_set = loads_p_set.reindex(columns=load_names)

    result: Dict[str, pd.DataFrame | str] = {
        "loads": loads,
        "loads_p_set": loads_p_set,
        "gen_cost": gen_cost,
    }

    pv_path = data_path / TS_FILES["pv_data"]
    if pv_path.exists():
        try:
            result["pv_power_w"] = read_pv_csv(pv_path)
        except Exception as exc:
            result["pv_error"] = str(exc)

    return result


def load_optimization_data(config_path: str | Path) -> Dict[str, Any]:
    """
    Carrega e valida os dados usados na primeira fase da otimização térmica.

    Usa a mesma base do baseline e acrescenta a informação térmica necessária
    para separar cargas AVAC, temperaturas de zona, limites de conforto e,
    quando existir, temperatura exterior e parâmetros por zona.
    """
    cfg = load_config(config_path)
    data_path = Path(cfg["data_path"])
    opt_cfg = cfg.get("optimization", {})

    hvac_categories = [str(v).strip() for v in opt_cfg.get("hvac_categories", DEFAULT_HVAC_CATEGORIES)]
    if not hvac_categories:
        raise ValueError("optimization.hvac_categories cannot be empty")

    grid_price_col = str(opt_cfg.get("grid_price_col", "Grid")).strip()

    # Reaproveita os dados base já usados no dashboard / baseline.
    base = load_baseline_data(config_path)
    loads_meta = base["loads"].copy()
    loads_p_set = base["loads_p_set"].copy()
    gen_cost = base["gen_cost"].copy()

    if grid_price_col not in gen_cost.columns:
        raise KeyError(
            f"Price column '{grid_price_col}' not found in generators-marginal_cost.xlsx. "
            f"Available columns: {list(gen_cost.columns)}"
        )

    # Para esta fase é obrigatório conseguir distinguir cargas AVAC das restantes.
    if "name" not in loads_meta.columns:
        raise ValueError("loads.xlsx must contain column 'name'")

    if "category" not in loads_meta.columns:
        raise ValueError(
            "For thermal optimization, loads.xlsx must contain a 'category' column "
            "so that HVAC and non-HVAC demand can be separated."
        )

    loads_meta["name"] = _normalize_names(loads_meta["name"])
    loads_meta["category"] = _normalize_names(loads_meta["category"])

    category_map = loads_meta.set_index("name")["category"].to_dict()
    hvac_categories_lower = {x.lower() for x in hvac_categories}

    hvac_load_cols = [
        c for c in loads_p_set.columns
        if str(category_map.get(c, c)).strip().lower() in hvac_categories_lower
    ]

    if not hvac_load_cols:
        raise ValueError(
            "No HVAC load columns were identified. Check loads.xlsx['category'] "
            f"against optimization.hvac_categories = {hvac_categories}."
        )

    non_hvac_cols = [c for c in loads_p_set.columns if c not in hvac_load_cols]
    if not non_hvac_cols:
        raise ValueError("All load columns were classified as HVAC. A fixed non-HVAC base load is required.")

    base_load_kw = loads_p_set[non_hvac_cols].sum(axis=1).rename("base_load_kw")
    hvac_baseline_kw = loads_p_set[hvac_load_cols].sum(axis=1).rename("hvac_baseline_kw")
    price = gen_cost[grid_price_col].rename("price")

    zone_temperatures = read_excel(data_path / OPT_TS_FILES["zone_temperatures"])
    zone_temperatures = ensure_datetime_index(zone_temperatures, "zone_temperatures")
    zone_temperatures = ensure_numeric(zone_temperatures, "zone_temperatures")

    # O horizonte de otimização fica limitado ao intervalo comum entre séries.
    common_index = loads_p_set.index.intersection(gen_cost.index).intersection(zone_temperatures.index)
    if common_index.empty:
        raise ValueError("No common timestamps between loads, prices and zone temperatures.")

    base_load_kw = base_load_kw.reindex(common_index)
    hvac_baseline_kw = hvac_baseline_kw.reindex(common_index)
    price = price.reindex(common_index)
    zone_temperatures = zone_temperatures.reindex(common_index)

    if base_load_kw.isna().any() or hvac_baseline_kw.isna().any() or price.isna().any():
        raise ValueError("Missing values detected after aligning the core optimization series.")

    outdoor = _read_optional_timeseries(data_path / OPT_TS_FILES["outdoor_temperature"], "outdoor_temperature")
    if outdoor is not None:
        out_col = _first_matching_column(
            outdoor,
            ["tout", "t_out", "outdoor_temperature", "temperature", "temp"],
        )
        if out_col is None:
            if outdoor.shape[1] != 1:
                raise ValueError(
                    "outdoor_temperature.xlsx must have a single column or one of the following names: "
                    "Tout, T_out, outdoor_temperature, temperature, temp"
                )
            out_col = outdoor.columns[0]

        outdoor = outdoor[[out_col]].rename(columns={out_col: "Tout"}).reindex(common_index)

    bounds = _read_optional_timeseries(data_path / OPT_TS_FILES["temperature_bounds"], "temperature_bounds")
    if bounds is not None:
        bounds = bounds.reindex(common_index)

    zone_params = None
    zp_path = data_path / OPT_STATIC_FILES["zone_params"]
    if zp_path.exists():
        zone_params = read_excel(zp_path, index_col=None)
        zone_params.columns = _normalize_names(zone_params.columns)

        needed = {"zone", "a", "g"}
        if not needed.issubset(set(zone_params.columns)):
            raise ValueError(
                "zone_params.xlsx must contain at least the columns: zone, a, g. "
                "Column 'b' is optional if no outdoor temperature is used."
            )

        zone_params["zone"] = _normalize_names(zone_params["zone"])

    # Se não existirem pesos definidos, distribui uniformemente pelas zonas.
    zone_names = [str(c).strip() for c in zone_temperatures.columns]
    weights_cfg = opt_cfg.get("zone_weights", {}) or {}

    if weights_cfg:
        missing_weights = [z for z in zone_names if z not in weights_cfg]
        if missing_weights:
            raise ValueError(f"Missing zone weights in config for zones: {missing_weights}")
        zone_weights = pd.Series({z: float(weights_cfg[z]) for z in zone_names})
    else:
        zone_weights = pd.Series(1.0 / len(zone_names), index=zone_names)

    if (zone_weights < 0).any():
        raise ValueError("Zone weights must be non-negative.")

    if zone_weights.sum() <= 0:
        raise ValueError("Zone weights sum must be positive.")

    zone_weights = zone_weights / zone_weights.sum()

    # Se não houver ficheiro com limites de conforto, usa os valores por defeito da config.
    if bounds is None:
        default_tmin = opt_cfg.get("default_tmin", None)
        default_tmax = opt_cfg.get("default_tmax", None)

        if default_tmin is None or default_tmax is None:
            raise ValueError(
                "Provide either temperature_bounds.xlsx or optimization.default_tmin / default_tmax in config.yaml"
            )

        bounds = pd.DataFrame(index=common_index)
        for z in zone_names:
            bounds[f"{z}__Tmin"] = float(default_tmin)
            bounds[f"{z}__Tmax"] = float(default_tmax)

    required_bound_cols = []
    for z in zone_names:
        required_bound_cols.extend([f"{z}__Tmin", f"{z}__Tmax"])

    missing_bound_cols = [c for c in required_bound_cols if c not in bounds.columns]
    if missing_bound_cols:
        raise ValueError(
            "temperature_bounds.xlsx is missing required columns: "
            f"{missing_bound_cols}. Expected format: <zone>__Tmin and <zone>__Tmax."
        )

    bounds = ensure_numeric(bounds[required_bound_cols], "temperature_bounds")

    hvac_max_multiplier = float(opt_cfg.get("hvac_max_multiplier", 1.0))
    if hvac_max_multiplier <= 0:
        raise ValueError("optimization.hvac_max_multiplier must be > 0")

    hvac_max_kw = (hvac_baseline_kw * hvac_max_multiplier).rename("hvac_max_kw")

    result: Dict[str, Any] = {
        "config": cfg,
        "snapshots": common_index,
        "loads": loads_meta,
        "loads_p_set": loads_p_set.reindex(common_index),
        "gen_cost": gen_cost.reindex(common_index),
        "price": price,
        "hvac_categories": hvac_categories,
        "hvac_load_cols": hvac_load_cols,
        "non_hvac_cols": non_hvac_cols,
        "base_load_kw": base_load_kw,
        "hvac_baseline_kw": hvac_baseline_kw,
        "hvac_max_kw": hvac_max_kw,
        "zone_temperatures": zone_temperatures,
        "zone_weights": zone_weights,
        "temperature_bounds": bounds,
        "outdoor_temperature": outdoor,
        "zone_params": zone_params,
    }

    return result