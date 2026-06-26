#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linprog

REGULARIZATION_EUR_PER_KWH = 1e-3
DEFAULT_TIME_LIMIT_SECONDS = 180
CATEGORIES = ["cooling", "ventilation", "pumping"]
REQUIRED_SHEETS = [
    "timeseries_main",
    "zone_weights",
    "equipment_metadata",
    "equipment_hourly",
    "scenario_config",
]

DEFAULT_SCENARIOS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "Tmin_offset_C": 0.0,
        "Tmax_offset_C": 0.0,
        "cooling_min_pu": 1.0,
        "cooling_max_pu": 1.0,
        "ventilation_min_pu": 1.0,
        "ventilation_max_pu": 1.0,
        "pumping_min_pu": 1.0,
        "pumping_max_pu": 1.0,
    },
    "low_flex": {
        "Tmin_offset_C": -0.5,
        "Tmax_offset_C": 0.5,
        "cooling_min_pu": 0.90,
        "cooling_max_pu": 1.10,
        "ventilation_min_pu": 0.95,
        "ventilation_max_pu": 1.05,
        "pumping_min_pu": 0.90,
        "pumping_max_pu": 1.10,
    },
    "medium_flex": {
        "Tmin_offset_C": -1.0,
        "Tmax_offset_C": 1.0,
        "cooling_min_pu": 0.80,
        "cooling_max_pu": 1.20,
        "ventilation_min_pu": 0.90,
        "ventilation_max_pu": 1.10,
        "pumping_min_pu": 0.80,
        "pumping_max_pu": 1.20,
    },
    "price_response": {
        "Tmin_offset_C": -0.25,
        "Tmax_offset_C": 0.25,
        "cooling_min_pu": 0.75,
        "cooling_max_pu": 1.25,
        "ventilation_min_pu": 0.90,
        "ventilation_max_pu": 1.10,
        "pumping_min_pu": 0.80,
        "pumping_max_pu": 1.20,
    },
    "thermal_inertia": {
        "Tmin_offset_C": -1.5,
        "Tmax_offset_C": 1.5,
        "cooling_min_pu": 0.60,
        "cooling_max_pu": 1.35,
        "ventilation_min_pu": 0.90,
        "ventilation_max_pu": 1.05,
        "pumping_min_pu": 0.70,
        "pumping_max_pu": 1.30,
    },
}

SCENARIO_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "low_flex": "low_flex",
    "lowflex": "low_flex",
    "low_flexibility": "low_flex",
    "medium_flex": "medium_flex",
    "mediumflex": "medium_flex",
    "medium_flexibility": "medium_flex",
    "price_response": "price_response",
    "price response": "price_response",
    "priceresponse": "price_response",
    "thermal_inertia": "thermal_inertia",
    "thermal inertia": "thermal_inertia",
    "thermal_inertia_focus": "thermal_inertia",
}


@dataclass
class ThermalCoef:
    zone: str
    a: float
    b: float
    g: float
    c: float
    r2: float
    n_obs: int
    method: str
    cooling_signal_std: float
    cooling_signal_sum: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    config: Dict[str, float]
    preserve_total_categories: Set[str] = field(default_factory=set)
    fixed_categories: Set[str] = field(default_factory=set)


class LPBuilder:
    def __init__(self) -> None:
        self.names: List[str] = []
        self.bounds: List[Tuple[float, Optional[float]]] = []
        self.obj: List[float] = []

    def add_var(self, name: str, lb: float, ub: Optional[float], cost: float) -> int:
        self.names.append(name)
        self.bounds.append((float(lb), None if ub is None else float(ub)))
        self.obj.append(float(cost))
        return len(self.names) - 1


def _native_timestamp_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return out


def read_excel_inputs(excel_path: str) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Ficheiro Excel não encontrado: {excel_path}")

    sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    missing = [s for s in REQUIRED_SHEETS if s not in sheets]
    if missing:
        raise ValueError(f"Faltam folhas obrigatórias no Excel: {missing}")

    return sheets


def prepare_inputs(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    ts = sheets["timeseries_main"].copy()
    zw = sheets["zone_weights"].copy()
    em = sheets["equipment_metadata"].copy()
    eh = sheets["equipment_hourly"].copy()
    sc = sheets["scenario_config"].copy()

    for df in (ts, zw, em, eh, sc):
        df.columns = [str(c).strip() for c in df.columns]

    if "timestamp" not in ts.columns or "timestamp" not in eh.columns:
        raise ValueError("As folhas timeseries_main e equipment_hourly devem conter a coluna timestamp.")

    ts["timestamp"] = _native_timestamp_series(ts["timestamp"])
    eh["timestamp"] = _native_timestamp_series(eh["timestamp"])
    if ts["timestamp"].isna().any() or eh["timestamp"].isna().any():
        raise ValueError("Existem timestamps inválidos nas folhas timeseries_main ou equipment_hourly.")

    required_ts = {"timestamp", "Tout_C", "price_eur_kWh"}
    if not required_ts.issubset(ts.columns):
        raise ValueError(f"timeseries_main deve conter pelo menos: {sorted(required_ts)}")

    zone_cols = [c for c in ts.columns if str(c).lower().startswith("zone")]
    if not zone_cols:
        raise ValueError("Não foram encontradas colunas zone1..zoneN em timeseries_main.")

    ts = ts.sort_values("timestamp").reset_index(drop=True)
    eh = eh.sort_values("timestamp").reset_index(drop=True)

    needed_meta = {"equipment_column", "category", "service_scope"}
    if not needed_meta.issubset(em.columns):
        raise ValueError(f"equipment_metadata deve conter: {sorted(needed_meta)}")
    if "zone" not in em.columns:
        em["zone"] = np.nan

    em["equipment_column"] = em["equipment_column"].astype(str).str.strip()
    em["category"] = em["category"].astype(str).str.strip().str.lower()
    em["service_scope"] = em["service_scope"].astype(str).str.strip().str.lower()
    em["zone"] = em["zone"].astype(str).str.strip()

    bad_cat = sorted(set(em["category"]) - set(CATEGORIES))
    if bad_cat:
        raise ValueError(f"Existem categorias inválidas em equipment_metadata: {bad_cat}")

    bad_scope = sorted(set(em["service_scope"]) - {"zone", "shared"})
    if bad_scope:
        raise ValueError(f"Existem service_scope inválidos em equipment_metadata: {bad_scope}")

    if not {"zone", "weight"}.issubset(zw.columns):
        raise ValueError("zone_weights deve conter as colunas zone e weight.")

    eh_cols = [str(c).strip() for c in eh.columns if str(c).strip() != "timestamp"]
    unknown = sorted(set(eh_cols) - set(em["equipment_column"]))
    if unknown:
        raise ValueError(f"Existem colunas de equipment_hourly sem metadata correspondente: {unknown[:10]}")

    return {
        "timeseries_main": ts,
        "zone_weights": zw,
        "equipment_metadata": em,
        "equipment_hourly": eh,
        "scenario_config": sc,
        "zone_cols": zone_cols,
    }


def build_zone_weights(zw: pd.DataFrame, zone_cols: List[str]) -> pd.Series:
    s = zw.set_index("zone")["weight"].reindex(zone_cols).fillna(0.0).astype(float)
    pos = s[s > 0]
    if pos.empty:
        s[:] = 1.0 / len(zone_cols)
        return s

    s[:] = 0.0
    s.loc[pos.index] = pos / pos.sum()
    return s


def pivot_equipment_long(eh: pd.DataFrame) -> pd.DataFrame:
    long = eh.melt(id_vars=["timestamp"], var_name="equipment_column", value_name="power_W")
    long["equipment_column"] = long["equipment_column"].astype(str).str.strip()
    long["power_W"] = pd.to_numeric(long["power_W"], errors="coerce").fillna(0.0)
    long["power_kW"] = long["power_W"] / 1000.0
    return long


def build_baseline_tables(
    data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.Series]:
    em = data["equipment_metadata"]
    eh = data["equipment_hourly"]
    zone_cols = data["zone_cols"]
    zone_w = build_zone_weights(data["zone_weights"], zone_cols)

    long = pivot_equipment_long(eh)
    merged = long.merge(em, on="equipment_column", how="inner", validate="many_to_one")

    cat_total = merged.groupby(["timestamp", "category"], as_index=False)["power_kW"].sum()
    cat_total = cat_total.pivot(index="timestamp", columns="category", values="power_kW").fillna(0.0)
    for c in CATEGORIES:
        if c not in cat_total.columns:
            cat_total[c] = 0.0
    cat_total = cat_total[CATEGORIES].sort_index()

    zone_by_cat = {c: pd.DataFrame(0.0, index=cat_total.index, columns=zone_cols) for c in CATEGORIES}

    direct = merged[merged["service_scope"] == "zone"].copy()
    if not direct.empty:
        direct = direct[direct["zone"].isin(zone_cols)]
        grouped = direct.groupby(["timestamp", "category", "zone"], as_index=False)["power_kW"].sum()
        for c in CATEGORIES:
            sub = grouped[grouped["category"] == c]
            if not sub.empty:
                pv = sub.pivot(index="timestamp", columns="zone", values="power_kW").reindex(
                    index=cat_total.index,
                    columns=zone_cols,
                    fill_value=0.0,
                )
                zone_by_cat[c] = zone_by_cat[c].add(pv, fill_value=0.0)

    shared = merged[merged["service_scope"] == "shared"].copy()
    if not shared.empty:
        shared_cat_total = shared.groupby(["timestamp", "category"], as_index=False)["power_kW"].sum()
        for c in CATEGORIES:
            sub = shared_cat_total[shared_cat_total["category"] == c].set_index("timestamp")["power_kW"]
            sub = sub.reindex(cat_total.index).fillna(0.0)
            alloc = pd.DataFrame(
                {z: sub.values * float(zone_w.get(z, 0.0)) for z in zone_cols},
                index=cat_total.index,
            )
            zone_by_cat[c] = zone_by_cat[c].add(alloc, fill_value=0.0)

    return cat_total, zone_by_cat, zone_w


def clip_params(a: float, b: float, g: float, c: float) -> Tuple[float, float, float, float]:
    a = float(np.clip(a, 0.0, 0.999))
    b = float(np.clip(b, -2.0, 2.0))
    g = float(np.clip(g, 0.0, 20.0))
    c = float(np.clip(c, -50.0, 50.0))
    return a, b, g, c


def thermal_residuals(
    params: np.ndarray,
    T: np.ndarray,
    Tout: np.ndarray,
    Ucool: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    a, b, g, c = params
    return a * T + b * Tout - g * Ucool + c - y


def estimate_annual_thermal_coefficients(
    ts_all: pd.DataFrame,
    zone_by_cat_all: Dict[str, pd.DataFrame],
    zone_cols: List[str],
    zone_weights: pd.Series,
) -> List[ThermalCoef]:
    ts_core = ts_all[["timestamp", "Tout_C"] + zone_cols].copy()
    cool_wide = zone_by_cat_all["cooling"].reset_index().rename(columns={"index": "timestamp"})
    merged = ts_core.merge(cool_wide, on="timestamp", how="left", suffixes=("", "__cool"))

    out: List[ThermalCoef] = []
    for z in zone_cols:
        T = pd.to_numeric(merged[z], errors="coerce")
        Tout = pd.to_numeric(merged["Tout_C"], errors="coerce")
        U = pd.to_numeric(merged[f"{z}__cool"], errors="coerce").fillna(0.0)

        df = pd.DataFrame({"T": T, "Tout": Tout, "Ucool": U})
        df["T_next"] = df["T"].shift(-1)
        df = df.dropna().reset_index(drop=True)

        n = len(df)
        u_std = float(df["Ucool"].std()) if n else 0.0
        u_sum = float(df["Ucool"].sum()) if n else 0.0

        if n < 30:
            a, b, g, c = clip_params(
                0.96,
                0.01,
                max(1e-4, float(zone_weights.get(z, 0.0)) * 0.1),
                float((1.0 - 0.96) * df["T"].mean()) if n > 0 else 0.0,
            )
            out.append(
                ThermalCoef(
                    z,
                    a,
                    b,
                    g,
                    c,
                    np.nan,
                    int(n),
                    "heuristic_short_series",
                    u_std,
                    u_sum,
                )
            )
            continue

        x0 = np.array([0.95, 0.01, 0.01, 0.0], dtype=float)
        lb = np.array([0.0, -2.0, 0.0, -50.0], dtype=float)
        ub = np.array([0.999, 2.0, 20.0, 50.0], dtype=float)
        method = "robust_bounded_least_squares"

        try:
            res = least_squares(
                thermal_residuals,
                x0=x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=0.5,
                max_nfev=3000,
                args=(
                    df["T"].values.astype(float),
                    df["Tout"].values.astype(float),
                    df["Ucool"].values.astype(float),
                    df["T_next"].values.astype(float),
                ),
            )
            a, b, g, c = [float(x) for x in res.x]
            a, b, g, c = clip_params(a, b, g, c)

            pred = a * df["T"].values + b * df["Tout"].values - g * df["Ucool"].values + c
            ss_res = float(np.sum((df["T_next"].values - pred) ** 2))
            ss_tot = float(np.sum((df["T_next"].values - np.mean(df["T_next"].values)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

            # Se houver sinal de cooling relevante, força um g mínimo positivo para evitar degenerescência.
            if u_std > 1e-6 and g < 1e-6:
                g = max(1e-4, float(zone_weights.get(z, 0.0)) * 0.1)
                method = "robust_bounded_least_squares_with_positive_g_floor"

            out.append(ThermalCoef(z, a, b, g, c, r2, int(n), method, u_std, u_sum))
        except Exception:
            a, b, g, c = clip_params(
                0.96,
                0.01,
                max(1e-4, float(zone_weights.get(z, 0.0)) * 0.1),
                float((1.0 - 0.96) * df["T"].mean()),
            )
            out.append(
                ThermalCoef(
                    z,
                    a,
                    b,
                    g,
                    c,
                    np.nan,
                    int(n),
                    "heuristic_solver_failure",
                    u_std,
                    u_sum,
                )
            )

    return out


def build_comfort_bounds(
    ts_period: pd.DataFrame,
    zone_cols: List[str],
    tmin_offset: float,
    tmax_offset: float,
) -> Tuple[pd.Series, pd.Series]:
    temp_df = ts_period[zone_cols].apply(pd.to_numeric, errors="coerce")
    tmin = temp_df.min(axis=0).astype(float) + float(tmin_offset)
    tmax = temp_df.max(axis=0).astype(float) + float(tmax_offset)
    initial = temp_df.iloc[0].astype(float)

    tmin = np.minimum(tmin, initial)
    tmax = np.maximum(tmax, initial)
    tmax = np.maximum(tmax, tmin + 0.05)

    return pd.Series(tmin, index=zone_cols), pd.Series(tmax, index=zone_cols)


def simulate_temperatures(
    ts_period: pd.DataFrame,
    cooling_series: pd.Series,
    coefs: List[ThermalCoef],
    zone_cols: List[str],
    zone_weights: pd.Series,
) -> pd.DataFrame:
    coef_map = {c.zone: c for c in coefs}
    tout = pd.to_numeric(ts_period["Tout_C"], errors="coerce").ffill().bfill().values.astype(float)
    total_cool = pd.to_numeric(cooling_series, errors="coerce").fillna(0.0).values.astype(float)

    cols = {"timestamp": ts_period["timestamp"].values}
    for z in zone_cols:
        coef = coef_map[z]
        alloc = total_cool * float(zone_weights.get(z, 0.0))
        actual = pd.to_numeric(ts_period[z], errors="coerce").ffill().bfill().values.astype(float)

        pred = np.zeros(len(ts_period), dtype=float)
        pred[0] = float(actual[0])
        for t in range(len(ts_period) - 1):
            pred[t + 1] = coef.a * pred[t] + coef.b * tout[t] - coef.g * alloc[t] + coef.c

        cols[f"{z}__pred"] = pred
        cols[f"{z}__actual"] = actual
        cols[f"{z}__cool_alloc_kW"] = alloc

    return pd.DataFrame(cols)


def normalize_scenario_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return SCENARIO_ALIASES.get(key, key)


def scenario_specs_from_workbook(
    sc: pd.DataFrame,
    requested: Sequence[str],
) -> Dict[str, ScenarioSpec]:
    canonical_rows: Dict[str, Dict[str, float]] = {}
    if "scenario_name" in sc.columns:
        for _, row in sc.iterrows():
            name = normalize_scenario_name(row.get("scenario_name", ""))
            if not name:
                continue
            canonical_rows[name] = row.to_dict()

    specs: Dict[str, ScenarioSpec] = {}
    for raw_name in requested:
        name = normalize_scenario_name(raw_name)
        if name not in DEFAULT_SCENARIOS:
            raise ValueError(f"Cenário não suportado: {raw_name}")

        cfg = DEFAULT_SCENARIOS[name].copy()
        row = canonical_rows.get(name)
        if row is not None:
            for key in cfg.keys():
                if key in row and pd.notna(row[key]):
                    cfg[key] = float(row[key])

        preserve: Set[str] = set()
        if name == "price_response":
            preserve = {"cooling", "ventilation", "pumping"}
        elif name == "thermal_inertia":
            preserve = {"cooling"}

        specs[name] = ScenarioSpec(
            name=name,
            config=cfg,
            preserve_total_categories=preserve,
            fixed_categories=set(),
        )

    return specs


def describe_energy_mode(spec: ScenarioSpec) -> str:
    if spec.preserve_total_categories:
        keep = ", ".join(sorted(spec.preserve_total_categories))
        return f"preserve_total_energy[{keep}]"
    return "bounded_hourly_modulation"


def last_available_iso_week(ts: pd.DataFrame) -> Tuple[int, int]:
    iso = ts["timestamp"].dt.isocalendar()
    counts = (
        ts.assign(iso_year=iso.year.astype(int), iso_week=iso.week.astype(int))
        .groupby(["iso_year", "iso_week"])
        .size()
        .reset_index(name="n")
    )
    counts = counts[counts["n"] > 0].sort_values(["iso_year", "iso_week"])
    row = counts.iloc[-1]
    return int(row["iso_year"]), int(row["iso_week"])


def select_week_period(
    ts: pd.DataFrame,
    year: Optional[int],
    week: Optional[int],
) -> Tuple[pd.DataFrame, int, int]:
    if year is None or week is None:
        year, week = last_available_iso_week(ts)

    iso = ts["timestamp"].dt.isocalendar()
    out = ts[
        (iso.year.astype(int) == int(year)) & (iso.week.astype(int) == int(week))
    ].copy().reset_index(drop=True)

    if out.empty:
        raise ValueError(f"Não existem dados para a semana ISO {week} de {year}.")

    return out, int(year), int(week)


def select_year_period(ts: pd.DataFrame, year: int) -> pd.DataFrame:
    out = ts[ts["timestamp"].dt.year.astype(int) == int(year)].copy().reset_index(drop=True)
    if out.empty:
        years = sorted(ts["timestamp"].dt.year.astype(int).unique().tolist())
        raise ValueError(f"Não existem dados para o ano {year}. Anos disponíveis: {years}")
    return out


def weekly_blocks_for_year(ts_year: pd.DataFrame) -> List[Tuple[int, int, pd.DataFrame]]:
    iso = ts_year["timestamp"].dt.isocalendar()
    aux = ts_year.assign(iso_year=iso.year.astype(int), iso_week=iso.week.astype(int))

    blocks: List[Tuple[int, int, pd.DataFrame]] = []
    for (iy, iw), grp in aux.groupby(["iso_year", "iso_week"], sort=True):
        block = grp.drop(columns=["iso_year", "iso_week"]).copy().reset_index(drop=True)
        blocks.append((int(iy), int(iw), block))

    return blocks


def build_baseline_period(
    ts_period: pd.DataFrame,
    cat_total_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    baseline = cat_total_all.reindex(index=ts_period["timestamp"]).fillna(0.0).copy()
    baseline = (
        baseline.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in baseline.columns
        else baseline
    )

    if "timestamp" not in baseline.columns:
        baseline.insert(0, "timestamp", ts_period["timestamp"].values)

    baseline["price_eur_kWh"] = ts_period["price_eur_kWh"].values.astype(float)
    baseline["hvac_total_kW"] = baseline[CATEGORIES].sum(axis=1)
    baseline["hvac_cost_eur"] = baseline["hvac_total_kW"] * baseline["price_eur_kWh"]
    baseline["hvac_energy_kWh"] = baseline["hvac_total_kW"]

    summary = {
        "price_variation_ok": bool(ts_period["price_eur_kWh"].nunique() > 1),
        "price_min_eur_kWh": float(ts_period["price_eur_kWh"].min()),
        "price_max_eur_kWh": float(ts_period["price_eur_kWh"].max()),
        "price_mean_eur_kWh": float(ts_period["price_eur_kWh"].mean()),
        "objective_eur": float(baseline["hvac_cost_eur"].sum()),
        "cost_eur": float(baseline["hvac_cost_eur"].sum()),
        "energy_kwh_total": float(baseline["hvac_energy_kWh"].sum()),
        "energy_kwh_cooling": float(baseline["cooling"].sum()),
        "energy_kwh_ventilation": float(baseline["ventilation"].sum()),
        "energy_kwh_pumping": float(baseline["pumping"].sum()),
        "n_snapshots": int(len(ts_period)),
    }

    return baseline, summary


def summary_from_solution(ts_period: pd.DataFrame, period_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "price_variation_ok": bool(ts_period["price_eur_kWh"].nunique() > 1),
        "price_min_eur_kWh": float(ts_period["price_eur_kWh"].min()),
        "price_max_eur_kWh": float(ts_period["price_eur_kWh"].max()),
        "price_mean_eur_kWh": float(ts_period["price_eur_kWh"].mean()),
        "objective_eur": float(period_df["opt_hvac_cost_eur"].sum()),
        "cost_eur": float(period_df["opt_hvac_cost_eur"].sum()),
        "energy_kwh_total": float(period_df["opt_hvac_energy_kWh"].sum()),
        "energy_kwh_cooling": float(period_df["opt_cooling_kW"].sum()),
        "energy_kwh_ventilation": float(period_df["opt_ventilation_kW"].sum()),
        "energy_kwh_pumping": float(period_df["opt_pumping_kW"].sum()),
        "n_snapshots": int(len(ts_period)),
    }


def solve_scenario_period(
    ts_period: pd.DataFrame,
    baseline_period: pd.DataFrame,
    spec: ScenarioSpec,
    time_limit_seconds: int,
    block_label: str = "",
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    n = len(ts_period)
    price = baseline_period["price_eur_kWh"].values.astype(float)
    base = {cat: baseline_period[cat].values.astype(float) for cat in CATEGORIES}
    prefixes = {"cooling": "cool", "ventilation": "vent", "pumping": "pump"}

    bld = LPBuilder()
    idx: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}
    slack_p: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}
    slack_n: Dict[str, List[int]] = {cat: [] for cat in CATEGORIES}

    for cat in CATEGORIES:
        min_key = f"{cat}_min_pu"
        max_key = f"{cat}_max_pu"
        pfx = prefixes[cat]

        for t in range(n):
            lb = max(0.0, base[cat][t] * float(spec.config[min_key]))
            ub = max(lb, base[cat][t] * float(spec.config[max_key]))
            idx[cat].append(bld.add_var(f"{pfx}[{t}]", lb, ub, price[t]))
            slack_p[cat].append(
                bld.add_var(f"d{pfx}p[{t}]", 0.0, None, REGULARIZATION_EUR_PER_KWH)
            )
            slack_n[cat].append(
                bld.add_var(f"d{pfx}n[{t}]", 0.0, None, REGULARIZATION_EUR_PER_KWH)
            )

    nv = len(bld.names)
    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []

    def add_eq(terms: Iterable[Tuple[int, float]], rhs: float) -> None:
        row = np.zeros(nv, dtype=float)
        for j, coef in terms:
            row[j] += float(coef)
        A_eq.append(row)
        b_eq.append(float(rhs))

    # Liga cada variável otimizada ao valor base através das slacks.
    for cat in CATEGORIES:
        for t in range(n):
            add_eq(
                [(idx[cat][t], 1.0), (slack_p[cat][t], -1.0), (slack_n[cat][t], 1.0)],
                base[cat][t],
            )

    # Em alguns cenários, certas categorias mantêm a energia total do bloco.
    for cat in sorted(spec.preserve_total_categories):
        add_eq([(j, 1.0) for j in idx[cat]], float(np.sum(base[cat])))

    tic = time.time()
    res = linprog(
        c=np.array(bld.obj, dtype=float),
        A_eq=np.array(A_eq, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bld.bounds,
        method="highs",
        options={"time_limit": float(time_limit_seconds), "presolve": True},
    )
    toc = time.time()

    attempts_df = pd.DataFrame([{
        "block": block_label,
        "scenario": spec.name,
        "success": bool(res.success),
        "status": int(getattr(res, "status", -999)),
        "message": str(getattr(res, "message", "")),
        "objective_solver": float(res.fun) if getattr(res, "fun", None) is not None else np.nan,
        "runtime_s": float(toc - tic),
        "n_variables": len(bld.obj),
        "n_eq": len(A_eq),
        "n_ub": 0,
    }])

    if not res.success or getattr(res, "x", None) is None:
        fallback = baseline_period[
            [
                "timestamp",
                "price_eur_kWh",
                "cooling",
                "ventilation",
                "pumping",
                "hvac_total_kW",
                "hvac_cost_eur",
                "hvac_energy_kWh",
            ]
        ].copy()
        fallback = fallback.rename(columns={
            "cooling": "opt_cooling_kW",
            "ventilation": "opt_ventilation_kW",
            "pumping": "opt_pumping_kW",
            "hvac_total_kW": "opt_hvac_total_kW",
            "hvac_cost_eur": "opt_hvac_cost_eur",
            "hvac_energy_kWh": "opt_hvac_energy_kWh",
        })
        fallback["solver_status"] = "fallback_baseline"

        summary = summary_from_solution(ts_period, fallback)
        summary["solver_status"] = "fallback_baseline"
        summary["energy_mode"] = describe_energy_mode(spec)
        return fallback, summary, attempts_df

    x = np.array(res.x, dtype=float)
    period_df = pd.DataFrame({
        "timestamp": ts_period["timestamp"].values,
        "price_eur_kWh": price,
        "opt_cooling_kW": x[idx["cooling"]],
        "opt_ventilation_kW": x[idx["ventilation"]],
        "opt_pumping_kW": x[idx["pumping"]],
    })
    period_df["opt_hvac_total_kW"] = period_df[
        ["opt_cooling_kW", "opt_ventilation_kW", "opt_pumping_kW"]
    ].sum(axis=1)
    period_df["opt_hvac_cost_eur"] = period_df["opt_hvac_total_kW"] * period_df["price_eur_kWh"]
    period_df["opt_hvac_energy_kWh"] = period_df["opt_hvac_total_kW"]
    period_df["solver_status"] = "optimal"

    summary = summary_from_solution(ts_period, period_df)
    summary["solver_status"] = "optimal"
    summary["energy_mode"] = describe_energy_mode(spec)
    return period_df, summary, attempts_df


def solve_scenario_year_by_weeks(
    ts_year: pd.DataFrame,
    cat_total_all: pd.DataFrame,
    spec: ScenarioSpec,
    time_limit_seconds: int,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    blocks = weekly_blocks_for_year(ts_year)
    all_weeks: List[pd.DataFrame] = []
    all_attempts: List[pd.DataFrame] = []
    weekly_rows: List[Dict[str, object]] = []

    for iso_year, iso_week, block_ts in blocks:
        baseline_block, _ = build_baseline_period(block_ts, cat_total_all)
        label = f"{iso_year}-W{iso_week:02d}"

        week_df, week_summary, attempts_df = solve_scenario_period(
            block_ts,
            baseline_block,
            spec,
            time_limit_seconds,
            label,
        )

        week_df["iso_year"] = iso_year
        week_df["iso_week"] = iso_week

        all_weeks.append(week_df)
        all_attempts.append(attempts_df)

        weekly_rows.append({
            "iso_year": iso_year,
            "iso_week": iso_week,
            "cost_eur": float(week_summary["cost_eur"]),
            "energy_kwh_total": float(week_summary["energy_kwh_total"]),
            "energy_kwh_cooling": float(week_summary["energy_kwh_cooling"]),
            "energy_kwh_ventilation": float(week_summary["energy_kwh_ventilation"]),
            "energy_kwh_pumping": float(week_summary["energy_kwh_pumping"]),
            "solver_status": week_summary["solver_status"],
        })

    year_df = pd.concat(all_weeks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    attempts = pd.concat(all_attempts, ignore_index=True) if all_attempts else pd.DataFrame()
    weekly_df = pd.DataFrame(weekly_rows)

    summary = summary_from_solution(ts_year, year_df)
    summary["solver_status"] = "optimal" if (not attempts.empty and attempts["success"].all()) else "mixed"
    summary["energy_mode"] = describe_energy_mode(spec)
    return year_df, summary, attempts, weekly_df


def run_thermal_analysis(
    ts_period: pd.DataFrame,
    baseline_period: pd.DataFrame,
    scenario_period: pd.DataFrame,
    coefs: List[ThermalCoef],
    zone_cols: List[str],
    zone_weights: pd.Series,
    spec: ScenarioSpec,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tmin, tmax = build_comfort_bounds(
        ts_period,
        zone_cols,
        spec.config["Tmin_offset_C"],
        spec.config["Tmax_offset_C"],
    )

    baseline_pred = simulate_temperatures(
        ts_period,
        baseline_period["cooling"],
        coefs,
        zone_cols,
        zone_weights,
    )
    scenario_pred = simulate_temperatures(
        ts_period,
        scenario_period["opt_cooling_kW"],
        coefs,
        zone_cols,
        zone_weights,
    )

    rows = []
    mae_baseline, mae_scenario, viol_baseline, viol_scenario = [], [], [], []

    for z in zone_cols:
        b_pred = baseline_pred[f"{z}__pred"].values.astype(float)
        s_pred = scenario_pred[f"{z}__pred"].values.astype(float)
        actual = baseline_pred[f"{z}__actual"].values.astype(float)

        lb = float(tmin[z])
        ub = float(tmax[z])

        b_mae = float(np.mean(np.abs(b_pred - actual)))
        s_mae = float(np.mean(np.abs(s_pred - actual)))
        b_viol = float(np.sum(np.maximum(lb - b_pred, 0.0) + np.maximum(b_pred - ub, 0.0)))
        s_viol = float(np.sum(np.maximum(lb - s_pred, 0.0) + np.maximum(s_pred - ub, 0.0)))

        mae_baseline.append(b_mae)
        mae_scenario.append(s_mae)
        viol_baseline.append(b_viol)
        viol_scenario.append(s_viol)

        rows.append({
            "zone": z,
            "comfort_min_C": lb,
            "comfort_max_C": ub,
            "baseline_pred_mae_C": b_mae,
            f"{spec.name}_pred_mae_C": s_mae,
            "baseline_comfort_violation_degC_sum": b_viol,
            f"{spec.name}_comfort_violation_degC_sum": s_viol,
            f"mean_pred_delta_{spec.name}_minus_baseline_C": float(np.mean(s_pred - b_pred)),
        })

    zone_df = pd.DataFrame(rows)
    summary = {
        "thermal_coefficients_count": int(len(coefs)),
        "thermal_baseline_mae_C_mean": float(np.mean(mae_baseline)) if mae_baseline else np.nan,
        f"thermal_{spec.name}_mae_C_mean": float(np.mean(mae_scenario)) if mae_scenario else np.nan,
        "thermal_baseline_violation_degC_sum": float(np.sum(viol_baseline)) if viol_baseline else np.nan,
        f"thermal_{spec.name}_violation_degC_sum": float(np.sum(viol_scenario)) if viol_scenario else np.nan,
    }

    return zone_df, summary


def build_monthly_summary(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["year"] = out["timestamp"].dt.year.astype(int)
    out["month"] = out["timestamp"].dt.month.astype(int)
    out["month_label"] = out["timestamp"].dt.to_period("M").astype(str)

    if mode == "baseline":
        g = out.groupby(["year", "month", "month_label"], as_index=False).agg(
            cost_eur=("hvac_cost_eur", "sum"),
            energy_kwh_total=("hvac_energy_kWh", "sum"),
            energy_kwh_cooling=("cooling", "sum"),
            energy_kwh_ventilation=("ventilation", "sum"),
            energy_kwh_pumping=("pumping", "sum"),
            price_mean_eur_kWh=("price_eur_kWh", "mean"),
            price_min_eur_kWh=("price_eur_kWh", "min"),
            price_max_eur_kWh=("price_eur_kWh", "max"),
            n_snapshots=("timestamp", "count"),
        )
    else:
        g = out.groupby(["year", "month", "month_label"], as_index=False).agg(
            cost_eur=("opt_hvac_cost_eur", "sum"),
            energy_kwh_total=("opt_hvac_energy_kWh", "sum"),
            energy_kwh_cooling=("opt_cooling_kW", "sum"),
            energy_kwh_ventilation=("opt_ventilation_kW", "sum"),
            energy_kwh_pumping=("opt_pumping_kW", "sum"),
            price_mean_eur_kWh=("price_eur_kWh", "mean"),
            price_min_eur_kWh=("price_eur_kWh", "min"),
            price_max_eur_kWh=("price_eur_kWh", "max"),
            n_snapshots=("timestamp", "count"),
        )

    return g.sort_values(["year", "month"]).reset_index(drop=True)


def build_monthly_comparison(
    baseline_monthly: pd.DataFrame,
    scenario_monthly: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    merged = baseline_monthly.merge(
        scenario_monthly,
        on=["year", "month", "month_label"],
        how="outer",
        suffixes=("_baseline", f"_{scenario_name}"),
    )
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    for metric in [
        "cost_eur",
        "energy_kwh_total",
        "energy_kwh_cooling",
        "energy_kwh_ventilation",
        "energy_kwh_pumping",
    ]:
        merged[f"delta_{metric}"] = merged[f"{metric}_{scenario_name}"] - merged[f"{metric}_baseline"]
        denom = merged[f"{metric}_baseline"].replace(0.0, np.nan)
        merged[f"delta_percent_{metric}"] = merged[f"delta_{metric}"] / denom * 100.0

    return merged


def build_comparison_df(
    baseline_summary: Dict[str, float],
    scenario_summary: Dict[str, float],
    thermal_summary: Dict[str, float],
    scenario_name: str,
) -> pd.DataFrame:
    rows = []
    for metric in [
        "cost_eur",
        "energy_kwh_total",
        "energy_kwh_cooling",
        "energy_kwh_ventilation",
        "energy_kwh_pumping",
    ]:
        b = float(baseline_summary.get(metric, np.nan))
        s = float(scenario_summary.get(metric, np.nan))
        d = s - b
        p = d / b * 100.0 if np.isfinite(b) and abs(b) > 1e-12 else np.nan

        rows.append({
            "metric": metric,
            "baseline": b,
            scenario_name: s,
            f"delta_{scenario_name}_minus_baseline": d,
            "delta_percent": p,
        })

    rows.append({
        "metric": "thermal_violation_degC_sum",
        "baseline": float(thermal_summary.get("thermal_baseline_violation_degC_sum", np.nan)),
        scenario_name: float(thermal_summary.get(f"thermal_{scenario_name}_violation_degC_sum", np.nan)),
        f"delta_{scenario_name}_minus_baseline":
            float(thermal_summary.get(f"thermal_{scenario_name}_violation_degC_sum", np.nan))
            - float(thermal_summary.get("thermal_baseline_violation_degC_sum", np.nan)),
        "delta_percent": np.nan,
    })
    rows.append({
        "metric": "thermal_pred_mae_C_mean",
        "baseline": float(thermal_summary.get("thermal_baseline_mae_C_mean", np.nan)),
        scenario_name: float(thermal_summary.get(f"thermal_{scenario_name}_mae_C_mean", np.nan)),
        f"delta_{scenario_name}_minus_baseline":
            float(thermal_summary.get(f"thermal_{scenario_name}_mae_C_mean", np.nan))
            - float(thermal_summary.get("thermal_baseline_mae_C_mean", np.nan)),
        "delta_percent": np.nan,
    })

    return pd.DataFrame(rows)


def build_master_summary(
    baseline_summary: Dict[str, float],
    runs: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    base_cost = float(baseline_summary.get("cost_eur", np.nan))
    base_energy = float(baseline_summary.get("energy_kwh_total", np.nan))

    for name, run in runs.items():
        summary = run["summary"]
        therm = run["thermal_summary"]
        cost = float(summary.get("cost_eur", np.nan))
        energy = float(summary.get("energy_kwh_total", np.nan))

        rows.append({
            "scenario": name,
            "solver_status": summary.get("solver_status", "unknown"),
            "energy_mode": summary.get("energy_mode", "unknown"),
            "cost_eur": cost,
            "delta_cost_eur": cost - base_cost,
            "delta_cost_percent": (cost - base_cost) / base_cost * 100.0 if abs(base_cost) > 1e-12 else np.nan,
            "energy_kwh_total": energy,
            "delta_energy_kwh": energy - base_energy,
            "delta_energy_percent": (energy - base_energy) / base_energy * 100.0 if abs(base_energy) > 1e-12 else np.nan,
            "thermal_mae_C_mean": float(therm.get(f"thermal_{name}_mae_C_mean", np.nan)),
            "thermal_violation_degC_sum": float(therm.get(f"thermal_{name}_violation_degC_sum", np.nan)),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["delta_cost_eur", "delta_energy_kwh"], ascending=[True, True]).reset_index(drop=True)

    return out


def interpretation_text(
    scope: str,
    period_label: str,
    baseline_summary: Dict[str, float],
    scenario_name: str,
    summary: Dict[str, float],
    thermal_summary: Dict[str, float],
    coefs: List[ThermalCoef],
    monthly_df: pd.DataFrame,
) -> str:
    c0 = float(baseline_summary.get("cost_eur", np.nan))
    c1 = float(summary.get("cost_eur", np.nan))
    e0 = float(baseline_summary.get("energy_kwh_total", np.nan))
    e1 = float(summary.get("energy_kwh_total", np.nan))
    dc = c1 - c0
    de = e1 - e0
    pc = dc / c0 * 100.0 if np.isfinite(c0) and abs(c0) > 1e-12 else np.nan
    pe = de / e0 * 100.0 if np.isfinite(e0) and abs(e0) > 1e-12 else np.nan

    coef_df = pd.DataFrame([asdict(c) for c in coefs]) if coefs else pd.DataFrame()
    methods = coef_df["method"].value_counts().to_dict() if not coef_df.empty else {}

    lines = [
        f"Período analisado: {period_label}.",
        f"Cenário analisado: {scenario_name}.",
        f"Modo energético do cenário: {summary.get('energy_mode', 'unknown')}.",
        (
            "A variação horária dos preços foi verificada explicitamente "
            f"(mín={baseline_summary.get('price_min_eur_kWh', np.nan):.5f} EUR/kWh; "
            f"máx={baseline_summary.get('price_max_eur_kWh', np.nan):.5f} EUR/kWh; "
            f"média={baseline_summary.get('price_mean_eur_kWh', np.nan):.5f} EUR/kWh)."
        ),
        f"Baseline: custo HVAC = {c0:.2f} EUR; energia HVAC = {e0:.2f} kWh.",
        f"{scenario_name}: custo HVAC = {c1:.2f} EUR; energia HVAC = {e1:.2f} kWh.",
        f"Variação de custo {scenario_name} vs baseline = {dc:.2f} EUR ({pc:.2f}%).",
        f"Variação de energia {scenario_name} vs baseline = {de:.2f} kWh ({pe:.2f}%).",
        (
            "Análise térmica extra: "
            f"violação prevista baseline = {thermal_summary.get('thermal_baseline_violation_degC_sum', np.nan):.4f} °C·h; "
            f"{scenario_name} = {thermal_summary.get(f'thermal_{scenario_name}_violation_degC_sum', np.nan):.4f} °C·h; "
            f"MAE térmico médio baseline = {thermal_summary.get('thermal_baseline_mae_C_mean', np.nan):.4f} °C; "
            f"{scenario_name} = {thermal_summary.get(f'thermal_{scenario_name}_mae_C_mean', np.nan):.4f} °C."
        ),
        f"Solver: {summary.get('solver_status', 'unknown')}.",
    ]

    if not coef_df.empty:
        lines.append(
            f"Coeficientes térmicos anuais estimados para {len(coef_df)} zona(s): "
            f"ā={coef_df['a'].mean():.4f}, "
            f"b̄={coef_df['b'].mean():.4f}, "
            f"ḡ={coef_df['g'].mean():.4f}, "
            f"c̄={coef_df['c'].mean():.4f}, "
            f"R² médio={coef_df['r2'].replace([np.inf, -np.inf], np.nan).mean():.4f}."
        )
        lines.append(f"Métodos de estimação utilizados: {methods}.")

    if scope == "year" and not monthly_df.empty:
        best = monthly_df.sort_values("cost_eur", ascending=True).iloc[0]
        lines.append(
            "Resumo mensal guardado nos outputs: "
            f"mês de menor custo neste cenário = {best['month_label']} "
            f"({best['cost_eur']:.2f} EUR; {best['energy_kwh_total']:.2f} kWh)."
        )

    if scenario_name == "price_response":
        lines.append(
            "Leitura recomendada: neste cenário, a interpretação principal deve ser feita sobre o custo, "
            "porque a energia total das categorias flexíveis é preservada por bloco semanal e, por agregação, "
            "também no valor anual; o ganho decorre sobretudo da redistribuição horária do consumo face ao sinal de preço."
        )
    elif scenario_name == "thermal_inertia":
        lines.append(
            "Leitura recomendada: neste cenário, a energia total de cooling é preservada em cada bloco semanal, "
            "permitindo ler a inércia térmica como redistribuição temporal do esforço de arrefecimento, "
            "complementada por modulação admissível das cargas auxiliares."
        )
    else:
        lines.append(
            "Leitura recomendada: a redução de custo/energia é interpretada como flexibilidade mecânico-elétrica "
            "dentro das bandas admissíveis do cenário, enquanto a análise térmica funciona como diagnóstico ex-post "
            "da plausibilidade da resposta prevista das zonas."
        )

    return "\n".join(lines)


def safe_sheet_name(name: str) -> str:
    bad = set("[]:*?/\\")
    clean = "".join("_" if c in bad else c for c in name)
    return clean[:31]


def export_outputs(
    scope: str,
    output_dir: str,
    prefix: str,
    baseline_period: pd.DataFrame,
    baseline_summary: Dict[str, float],
    baseline_monthly: pd.DataFrame,
    baseline_weekly: pd.DataFrame,
    runs: Dict[str, Dict[str, object]],
    master_summary: pd.DataFrame,
    coefs: List[ThermalCoef],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{prefix}_report.txt")
    xlsx_path = os.path.join(output_dir, f"{prefix}_results.xlsx")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BASELINE\n")
        for k, v in baseline_summary.items():
            f.write(f"- {k}: {v}\n")

        if scope == "year":
            f.write("\nBASELINE_MENSAL\n")
            f.write(baseline_monthly.to_string(index=False))
            f.write("\n\nBASELINE_SEMANAL\n")
            f.write(baseline_weekly.to_string(index=False))
            f.write("\n")

        f.write("\nRESUMO GLOBAL\n")
        f.write(master_summary.to_string(index=False) if not master_summary.empty else "Sem cenários resolvidos.")
        f.write("\n")

        for name, run in runs.items():
            f.write(f"\n{name.upper()}\n")
            for k, v in run["summary"].items():
                f.write(f"- {k}: {v}\n")

            if scope == "year":
                f.write("\nMENSAL\n")
                f.write(run["monthly_df"].to_string(index=False))
                f.write("\n\nMENSAL_VS_BASELINE\n")
                f.write(run["monthly_cmp_df"].to_string(index=False))
                f.write("\n\nSEMANAL\n")
                f.write(run["weekly_df"].to_string(index=False))

            f.write("\n\nCOMPARAÇÃO\n")
            f.write(run["comparison_df"].to_string(index=False))
            f.write("\n\nINTERPRETAÇÃO\n")
            f.write(run["interpretation_text"])
            f.write("\n\nTENTATIVAS DO SOLVER\n")
            f.write(run["attempts_df"].to_string(index=False))
            f.write("\n")

    coef_df = pd.DataFrame([asdict(c) for c in coefs]) if coefs else pd.DataFrame()
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        baseline_period.to_excel(writer, sheet_name=safe_sheet_name("baseline_period"), index=False)
        pd.DataFrame(list(baseline_summary.items()), columns=["metric", "value"]).to_excel(
            writer,
            sheet_name=safe_sheet_name("baseline_summary"),
            index=False,
        )
        baseline_monthly.to_excel(writer, sheet_name=safe_sheet_name("baseline_monthly"), index=False)
        baseline_weekly.to_excel(writer, sheet_name=safe_sheet_name("baseline_weekly"), index=False)
        master_summary.to_excel(writer, sheet_name=safe_sheet_name("summary_all"), index=False)

        if not coef_df.empty:
            coef_df.to_excel(writer, sheet_name=safe_sheet_name("thermal_coefs"), index=False)

        all_attempts = []
        for name, run in runs.items():
            run["period_df"].to_excel(writer, sheet_name=safe_sheet_name(f"{name}_period"), index=False)
            pd.DataFrame(list(run["summary"].items()), columns=["metric", "value"]).to_excel(
                writer,
                sheet_name=safe_sheet_name(f"{name}_summary"),
                index=False,
            )
            run["comparison_df"].to_excel(writer, sheet_name=safe_sheet_name(f"cmp_{name}"), index=False)
            run["thermal_zone_df"].to_excel(writer, sheet_name=safe_sheet_name(f"therm_{name}"), index=False)
            run["monthly_df"].to_excel(writer, sheet_name=safe_sheet_name(f"monthly_{name}"), index=False)
            run["monthly_cmp_df"].to_excel(writer, sheet_name=safe_sheet_name(f"mcmp_{name}"), index=False)
            run["weekly_df"].to_excel(writer, sheet_name=safe_sheet_name(f"weekly_{name}"), index=False)
            run["attempts_df"].to_excel(writer, sheet_name=safe_sheet_name(f"try_{name}"), index=False)
            all_attempts.append(run["attempts_df"])

        if all_attempts:
            pd.concat(all_attempts, ignore_index=True).to_excel(
                writer,
                sheet_name=safe_sheet_name("solver_attempts"),
                index=False,
            )

    return {"report_txt": txt_path, "results_xlsx": xlsx_path}


def annual_from_weekly(
    ts_year: pd.DataFrame,
    cat_total_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    blocks = weekly_blocks_for_year(ts_year)
    baseline_parts = []
    weekly_rows = []

    for iso_year, iso_week, block_ts in blocks:
        block_baseline, block_summary = build_baseline_period(block_ts, cat_total_all)
        block_baseline["iso_year"] = iso_year
        block_baseline["iso_week"] = iso_week
        baseline_parts.append(block_baseline)

        weekly_rows.append({
            "iso_year": iso_year,
            "iso_week": iso_week,
            "cost_eur": float(block_summary["cost_eur"]),
            "energy_kwh_total": float(block_summary["energy_kwh_total"]),
            "energy_kwh_cooling": float(block_summary["energy_kwh_cooling"]),
            "energy_kwh_ventilation": float(block_summary["energy_kwh_ventilation"]),
            "energy_kwh_pumping": float(block_summary["energy_kwh_pumping"]),
        })

    baseline_df = pd.concat(baseline_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    weekly_df = pd.DataFrame(weekly_rows)

    summary = {
        "price_variation_ok": bool(ts_year["price_eur_kWh"].nunique() > 1),
        "price_min_eur_kWh": float(ts_year["price_eur_kWh"].min()),
        "price_max_eur_kWh": float(ts_year["price_eur_kWh"].max()),
        "price_mean_eur_kWh": float(ts_year["price_eur_kWh"].mean()),
        "objective_eur": float(baseline_df["hvac_cost_eur"].sum()),
        "cost_eur": float(baseline_df["hvac_cost_eur"].sum()),
        "energy_kwh_total": float(baseline_df["hvac_energy_kWh"].sum()),
        "energy_kwh_cooling": float(baseline_df["cooling"].sum()),
        "energy_kwh_ventilation": float(baseline_df["ventilation"].sum()),
        "energy_kwh_pumping": float(baseline_df["pumping"].sum()),
        "n_snapshots": int(len(ts_year)),
        "aggregation_mode": "year_by_weeks",
    }

    return baseline_df, weekly_df, summary


def run_optimization(
    excel_path: str,
    year: Optional[int],
    week: Optional[int],
    scope: str,
    scenarios: Sequence[str],
    output_prefix: str,
    time_limit_seconds: int,
) -> Dict[str, str]:
    raw = read_excel_inputs(excel_path)
    data = prepare_inputs(raw)

    ts_all = data["timeseries_main"]
    zone_cols = data["zone_cols"]
    cat_total_all, zone_by_cat_all, zone_weights = build_baseline_tables(data)
    coefs = estimate_annual_thermal_coefficients(ts_all, zone_by_cat_all, zone_cols, zone_weights)

    scope = str(scope).strip().lower()
    if scope == "week":
        ts_period, iso_year, iso_week = select_week_period(ts_all, year, week)
        baseline_period, baseline_summary = build_baseline_period(ts_period, cat_total_all)
        baseline_weekly = pd.DataFrame([{
            "iso_year": iso_year,
            "iso_week": iso_week,
            "cost_eur": baseline_summary["cost_eur"],
            "energy_kwh_total": baseline_summary["energy_kwh_total"],
            "energy_kwh_cooling": baseline_summary["energy_kwh_cooling"],
            "energy_kwh_ventilation": baseline_summary["energy_kwh_ventilation"],
            "energy_kwh_pumping": baseline_summary["energy_kwh_pumping"],
        }])
        period_label = f"semana ISO {iso_week} de {iso_year}"
        period_year = iso_year
        period_week = iso_week

    elif scope == "year":
        if year is None:
            year = int(ts_all["timestamp"].dt.year.max())

        ts_period = select_year_period(ts_all, int(year))
        baseline_period, baseline_weekly, baseline_summary = annual_from_weekly(ts_period, cat_total_all)
        period_label = f"ano {year}"
        period_year = int(year)
        period_week = None

    else:
        raise ValueError(f"Scope não suportado: {scope}")

    baseline_monthly = build_monthly_summary(baseline_period, "baseline")
    specs = scenario_specs_from_workbook(data["scenario_config"], scenarios)

    runs: Dict[str, Dict[str, object]] = {}
    for raw_name in scenarios:
        name = normalize_scenario_name(raw_name)
        spec = specs[name]

        if scope == "year":
            period_df, summary, attempts_df, weekly_df = solve_scenario_year_by_weeks(
                ts_period,
                cat_total_all,
                spec,
                time_limit_seconds,
            )
        else:
            period_df, summary, attempts_df = solve_scenario_period(
                ts_period,
                baseline_period,
                spec,
                time_limit_seconds,
                period_label,
            )

            weekly_df = baseline_weekly.copy()
            weekly_df["cost_eur"] = float(summary["cost_eur"])
            weekly_df["energy_kwh_total"] = float(summary["energy_kwh_total"])
            weekly_df["energy_kwh_cooling"] = float(summary["energy_kwh_cooling"])
            weekly_df["energy_kwh_ventilation"] = float(summary["energy_kwh_ventilation"])
            weekly_df["energy_kwh_pumping"] = float(summary["energy_kwh_pumping"])
            weekly_df["solver_status"] = summary["solver_status"]

        thermal_zone_df, thermal_summary = run_thermal_analysis(
            ts_period,
            baseline_period,
            period_df,
            coefs,
            zone_cols,
            zone_weights,
            spec,
        )

        summary["thermal_coefficients_count"] = thermal_summary["thermal_coefficients_count"]
        summary[f"thermal_{name}_mae_C_mean"] = thermal_summary[f"thermal_{name}_mae_C_mean"]
        summary[f"thermal_{name}_violation_degC_sum"] = thermal_summary[f"thermal_{name}_violation_degC_sum"]

        comparison_df = build_comparison_df(baseline_summary, summary, thermal_summary, name)
        monthly_df = build_monthly_summary(period_df, "scenario")
        monthly_cmp_df = build_monthly_comparison(baseline_monthly, monthly_df, name)
        interp = interpretation_text(
            scope,
            period_label,
            baseline_summary,
            name,
            summary,
            thermal_summary,
            coefs,
            monthly_df,
        )

        runs[name] = {
            "period_df": period_df,
            "summary": summary,
            "comparison_df": comparison_df,
            "thermal_zone_df": thermal_zone_df,
            "thermal_summary": thermal_summary,
            "interpretation_text": interp,
            "attempts_df": attempts_df,
            "monthly_df": monthly_df,
            "monthly_cmp_df": monthly_cmp_df,
            "weekly_df": weekly_df,
        }

    master_summary = build_master_summary(baseline_summary, runs)
    suffix = f"y{period_year}" if scope == "year" else f"y{period_year}_w{period_week}"
    prefix = f"{output_prefix}_{suffix}"

    outputs = export_outputs(
        scope,
        os.path.join("data", "outputs"),
        prefix,
        baseline_period,
        baseline_summary,
        baseline_monthly,
        baseline_weekly,
        runs,
        master_summary,
        coefs,
    )
    outputs["scope"] = scope
    outputs["analysis_period"] = period_label
    outputs["analysis_year"] = str(period_year)
    outputs["analysis_week"] = "" if period_week is None else str(period_week)
    return outputs


def default_excel_path() -> Optional[str]:
    candidates = [
        Path("data.xlsx"),
        Path("data") / "data.xlsx",
        Path.cwd() / "data.xlsx",
        Path.cwd() / "data" / "data.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimization v11 - análise semanal ou anual por blocos semanais com outputs mensais"
    )
    p.add_argument("--input", default=None, help="Caminho para o Excel (ex.: data/data.xlsx)")
    p.add_argument("--scope", choices=["week", "year"], default="year", help="Horizonte de otimização")
    p.add_argument("--year", type=int, default=None, help="Ano civil ou ano ISO")
    p.add_argument("--week", type=int, default=None, help="Semana ISO (apenas para scope=week)")
    p.add_argument("--output-prefix", default="optimization_v11", help="Prefixo dos ficheiros de output")
    p.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT_SECONDS,
        help="Limite de tempo do solver em segundos",
    )
    p.add_argument(
        "--scenarios",
        default="low_flex,medium_flex,price_response,thermal_inertia",
        help="Lista separada por vírgulas.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    excel_path = args.input or default_excel_path()

    if excel_path is None:
        print(
            "ERRO: não foi possível localizar automaticamente o ficheiro Excel. Usa --input data/data.xlsx",
            file=sys.stderr,
        )
        return 1

    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]

    try:
        res = run_optimization(
            excel_path=excel_path,
            year=args.year,
            week=args.week,
            scope=args.scope,
            scenarios=scenarios,
            output_prefix=args.output_prefix,
            time_limit_seconds=args.time_limit,
        )
        print("Optimization v11 concluída com sucesso.")
        for k, v in res.items():
            print(f"{k}: {v}")
        return 0

    except Exception as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())