import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

try:
    pd.options.future.infer_string = False
except Exception:
    pass

try:
    import pypsa
except Exception:
    pypsa = None


def _require_pypsa():
    if pypsa is None:
        raise ImportError(
            "PyPSA não está instalado neste ambiente. Instala com: pip install pypsa highspy"
        )


if len(Path(__file__).resolve().parents) > 1:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = Path.cwd()

EXCEL_CANDIDATES = [
    PROJECT_ROOT / "data" / "data.xlsx",
    PROJECT_ROOT / "data.xlsx",
    Path.cwd() / "data" / "data.xlsx",
    Path.cwd() / "data.xlsx",
]

RAW_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "Resultados_14abril.xlsx",
    PROJECT_ROOT / "data" / "Resultados_14abril.xlsx",
    Path.cwd() / "Resultados_14abril.xlsx",
    Path.cwd() / "data" / "Resultados_14abril.xlsx",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs_monthly_heatcool_full"
DEFAULT_SCENARIO = "baseline"
DEFAULT_MONTH = 1
DEFAULT_YEAR = None
SOLVER_NAME = "highs"
EXPORT_OUTPUTS = True

# Por defeito usa o histórico completo para estimar parâmetros térmicos.
ESTIMATE_ON_SELECTED_MONTH = False

AUTO_RELAX_IF_INFEASIBLE = True
RELAXATION_STEPS_C = [0.0, 0.25, 0.5, 1.0, 2.0]

MONTH_NAMES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

SCENARIO_DEFAULTS = {
    "baseline": {
        "active": 0,
        "Tmin_offset_C": 0.0,
        "Tmax_offset_C": 0.0,
        "heating_min_pu": 1.00,
        "heating_max_pu": 1.00,
        "cooling_min_pu": 1.00,
        "cooling_max_pu": 1.00,
        "ventilation_min_pu": 1.00,
        "ventilation_max_pu": 1.00,
        "pumping_min_pu": 1.00,
        "pumping_max_pu": 1.00,
    },
    "starter_flex": {
        "active": 1,
        "Tmin_offset_C": -1.0,
        "Tmax_offset_C": 1.0,
        "heating_min_pu": 0.00,
        "heating_max_pu": 1.50,
        "cooling_min_pu": 0.00,
        "cooling_max_pu": 1.50,
        "ventilation_min_pu": 0.00,
        "ventilation_max_pu": 1.20,
        "pumping_min_pu": 0.00,
        "pumping_max_pu": 1.50,
    },
    "low_flex": {
        "active": 1,
        "Tmin_offset_C": -0.5,
        "Tmax_offset_C": 0.5,
        "heating_min_pu": 0.90,
        "heating_max_pu": 1.10,
        "cooling_min_pu": 0.90,
        "cooling_max_pu": 1.10,
        "ventilation_min_pu": 0.95,
        "ventilation_max_pu": 1.05,
        "pumping_min_pu": 0.90,
        "pumping_max_pu": 1.10,
    },
    "medium_flex": {
        "active": 1,
        "Tmin_offset_C": -1.0,
        "Tmax_offset_C": 1.0,
        "heating_min_pu": 0.80,
        "heating_max_pu": 1.20,
        "cooling_min_pu": 0.80,
        "cooling_max_pu": 1.20,
        "ventilation_min_pu": 0.90,
        "ventilation_max_pu": 1.10,
        "pumping_min_pu": 0.80,
        "pumping_max_pu": 1.20,
    },
    "price_response": {
        "active": 1,
        "Tmin_offset_C": -0.25,
        "Tmax_offset_C": 0.25,
        "heating_min_pu": 0.75,
        "heating_max_pu": 1.25,
        "cooling_min_pu": 0.75,
        "cooling_max_pu": 1.25,
        "ventilation_min_pu": 0.90,
        "ventilation_max_pu": 1.10,
        "pumping_min_pu": 0.80,
        "pumping_max_pu": 1.20,
    },
}

ALL_CATEGORIES = ["heating", "cooling", "ventilation", "pumping"]
THERMAL_CATEGORIES = ["heating", "cooling"]
ELECTRIC_CATEGORIES = ["cooling", "ventilation", "pumping"]


def month_name_pt(month_num: int) -> str:
    return MONTH_NAMES_PT.get(int(month_num), f"Mes_{month_num}")


def resolve_excel_path(user_path=None) -> Path:
    if user_path is not None:
        p = Path(user_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Excel file not found: {p}")

    for p in EXCEL_CANDIDATES:
        if p.exists():
            return p.resolve()

    searched = "\n - ".join(str(p) for p in EXCEL_CANDIDATES)
    raise FileNotFoundError(
        "Não foi encontrado o ficheiro Excel preparado (data.xlsx). Locais procurados:\n - " + searched
    )


def resolve_optional_raw_source(user_path=None):
    """
    Resolve o ficheiro bruto de apoio de forma tolerante.

    Aceita:
    - caminho completo;
    - nome sem extensão;
    - parte do nome do ficheiro.

    Se não encontrar nada, devolve None.
    """
    if user_path is not None:
        raw_str = str(user_path).strip()
        if raw_str:
            p = Path(raw_str)
            if p.exists():
                return p.resolve()

            if p.suffix == "":
                p_xlsx = Path(raw_str + ".xlsx")
                if p_xlsx.exists():
                    return p_xlsx.resolve()

            search_dirs = [
                PROJECT_ROOT,
                PROJECT_ROOT / "data",
                Path.cwd(),
                Path.cwd() / "data",
            ]
            candidates = []
            token = p.stem.lower()

            for d in search_dirs:
                if d.exists():
                    for f in d.glob("*.xlsx"):
                        if token in f.stem.lower():
                            candidates.append(f.resolve())

            if len(candidates) == 1:
                return candidates[0]

            if len(candidates) > 1:
                # Se houver mais do que uma hipótese, dá prioridade ao ficheiro esperado.
                for c in candidates:
                    if "resultados_14abril" in c.stem.lower():
                        return c
                return candidates[0]

    for p in RAW_SOURCE_CANDIDATES:
        if p.exists():
            return p.resolve()

    return None


def _native_string(series: pd.Series) -> pd.Series:
    return series.map(lambda x: "" if pd.isna(x) else str(x).strip()).astype(object)


def _to_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    return dt


def _to_native_datetime_index(index_like) -> pd.DatetimeIndex:
    dt = pd.to_datetime(index_like)
    if isinstance(dt, pd.Series):
        values = dt.dt.to_pydatetime()
    elif isinstance(dt, pd.DatetimeIndex):
        values = dt.to_pydatetime()
    else:
        values = pd.to_datetime(dt).to_pydatetime()
    return pd.DatetimeIndex(values, name="snapshot")


def choose_month(ts: pd.DataFrame, month: int, year: int | None = None) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts.index)
    mask = idx.month == int(month)
    if year is not None:
        mask &= idx.year == int(year)

    selected = idx[mask]
    if len(selected) == 0:
        if year is None:
            raise ValueError(f"Não existem dados para o mês {month_name_pt(month)}.")
        raise ValueError(f"Não existem dados para {month_name_pt(month)} de {year}.")

    return _to_native_datetime_index(selected)


def load_excel_inputs(excel_path: Path) -> dict:
    sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

    required = [
        "timeseries_main",
        "zone_weights",
        "equipment_metadata",
        "equipment_hourly",
        "scenario_config",
    ]
    missing = [s for s in required if s not in sheets]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}")

    ts = sheets["timeseries_main"].copy()
    ts.columns = [str(c).strip() for c in ts.columns]
    ts["timestamp"] = _to_datetime(ts["timestamp"])
    if ts["timestamp"].isna().any():
        raise ValueError("timeseries_main.timestamp contains invalid values")

    numeric_ts_cols = [c for c in ts.columns if c != "timestamp"]
    ts[numeric_ts_cols] = ts[numeric_ts_cols].apply(pd.to_numeric, errors="coerce")
    ts = ts.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts.index = _to_native_datetime_index(ts["timestamp"])
    ts = ts.drop(columns=["timestamp"])

    zw = sheets["zone_weights"].copy()
    zw.columns = [str(c).strip() for c in zw.columns]
    zw["zone"] = _native_string(zw["zone"])
    zw["weight"] = pd.to_numeric(zw["weight"], errors="coerce").fillna(0.0)

    em = sheets["equipment_metadata"].copy()
    em.columns = [str(c).strip() for c in em.columns]
    for col in ["equipment_column", "category", "zone", "service_scope", "unit"]:
        if col in em.columns:
            em[col] = _native_string(em[col])
        else:
            em[col] = ""
    em["category"] = em["category"].str.lower()
    em["service_scope"] = em["service_scope"].str.lower()

    eh = sheets["equipment_hourly"].copy()
    eh.columns = [str(c).strip() for c in eh.columns]
    eh["timestamp"] = _to_datetime(eh["timestamp"])
    eh = eh.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    sc = sheets["scenario_config"].copy()
    sc.columns = [str(c).strip() for c in sc.columns]
    sc["scenario_name"] = _native_string(sc["scenario_name"])
    for col in sc.columns:
        if col != "scenario_name":
            sc[col] = pd.to_numeric(sc[col], errors="coerce")

    return {
        "timeseries_main": ts,
        "zone_weights": zw,
        "equipment_metadata": em,
        "equipment_hourly": eh,
        "scenario_config": sc,
    }


def validate_inputs(data: dict) -> None:
    ts = data["timeseries_main"]
    em = data["equipment_metadata"]
    eh = data["equipment_hourly"]
    zw = data["zone_weights"]

    required_ts_cols = ["Tout_C", "base_kW", "hvac_baseline_kW", "hvac_max_kW", "price_eur_kWh"]
    missing_ts = [c for c in required_ts_cols if c not in ts.columns]
    if missing_ts:
        raise ValueError(f"Missing columns in timeseries_main: {missing_ts}")

    zone_cols = [c for c in ts.columns if str(c).startswith("zone")]
    if not zone_cols:
        raise ValueError("No zone columns found in timeseries_main")

    zone_set = set(zone_cols)
    weight_zone_set = set(zw["zone"])
    if not weight_zone_set.issubset(zone_set):
        bad = sorted(weight_zone_set - zone_set)
        raise ValueError(f"zone_weights has unknown zones: {bad}")

    valid_categories = set(ALL_CATEGORIES)
    cat_bad = sorted(set(em["category"]) - valid_categories)
    if cat_bad:
        raise ValueError(f"equipment_metadata has unsupported categories: {cat_bad}")

    valid_scope = {"zone", "shared"}
    scope_bad = sorted(set(em["service_scope"]) - valid_scope)
    if scope_bad:
        raise ValueError(f"equipment_metadata has unsupported service_scope: {scope_bad}")

    equipment_cols = [str(c).strip() for c in eh.columns if c != "timestamp"]
    meta_cols = em["equipment_column"].tolist()
    missing_equipment = sorted(set(meta_cols) - set(equipment_cols))
    if missing_equipment and set(eh.columns) != {"timestamp", "equipment_column", "power_W"}:
        raise ValueError(
            "equipment_metadata references columns not found in equipment_hourly: "
            f"{missing_equipment[:8]}{'...' if len(missing_equipment) > 8 else ''}"
        )

    zone_scope_bad = em.loc[(em["service_scope"] == "zone") & (~em["zone"].isin(zone_set)), "zone"].tolist()
    if zone_scope_bad:
        raise ValueError(f"Zone-scoped equipment has invalid zone labels: {zone_scope_bad[:8]}")


def build_equipment_kW(data: dict) -> pd.DataFrame:
    eh = data["equipment_hourly"].copy()
    em = data["equipment_metadata"].copy()

    if set(eh.columns) == {"timestamp", "equipment_column", "power_W"}:
        long_df = eh.copy()
        long_df["equipment_column"] = _native_string(long_df["equipment_column"])
        long_df["power_W"] = pd.to_numeric(long_df["power_W"], errors="coerce").fillna(0.0)
    else:
        long_df = eh.melt(id_vars="timestamp", var_name="equipment_column", value_name="power_W")
        long_df["equipment_column"] = _native_string(long_df["equipment_column"])
        long_df["power_W"] = pd.to_numeric(long_df["power_W"], errors="coerce").fillna(0.0)

    merged = long_df.merge(em, on="equipment_column", how="left", validate="many_to_one")
    if merged[["category", "zone", "service_scope"]].isna().any().any():
        bad = merged.loc[merged["category"].isna(), "equipment_column"].drop_duplicates().tolist()
        raise ValueError(f"Equipment without metadata mapping: {bad[:8]}")

    merged["timestamp"] = _to_datetime(merged["timestamp"])
    merged = merged.dropna(subset=["timestamp"]).copy()
    merged["power_kW"] = pd.to_numeric(merged["power_W"], errors="coerce").fillna(0.0) / 1000.0
    return merged


def extract_raw_shared_heating(raw_source_path: Path | None):
    if raw_source_path is None or not raw_source_path.exists():
        return pd.DataFrame(columns=["timestamp", "category", "power_kW"])

    try:
        df = pd.read_excel(raw_source_path, sheet_name="Dados_Base", engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "category", "power_kW"])

    timestamp = pd.to_datetime(df.get("snapshot", pd.Series(dtype=object)), errors="coerce")
    if timestamp.isna().all():
        timestamp = pd.to_datetime(df[df.columns[0]], errors="coerce")

    heat_cols = [c for c in df.columns if "Boiler Gas Rate [W](Hourly)" in str(c)]
    if not heat_cols:
        return pd.DataFrame(columns=["timestamp", "category", "power_kW"])

    heating_kW = df[heat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) / 1000.0
    out = pd.DataFrame({
        "timestamp": timestamp,
        "category": "heating",
        "power_kW": heating_kW,
    }).dropna(subset=["timestamp"])
    return out


def build_zone_category_baseline(data: dict, snapshots: pd.DatetimeIndex, raw_source_path: Path | None = None):
    ts = data["timeseries_main"]
    zw = data["zone_weights"].copy()
    equipment = build_equipment_kW(data)
    raw_heating = extract_raw_shared_heating(raw_source_path)
    zone_cols = [c for c in ts.columns if str(c).startswith("zone")]

    zw = zw.set_index("zone").reindex(zone_cols).fillna(0.0)
    weight_sum = zw["weight"].sum()
    if weight_sum <= 0:
        raise ValueError("zone_weights sum must be positive")
    zw["weight"] = zw["weight"] / weight_sum

    equipment = equipment.loc[equipment["timestamp"].isin(pd.DatetimeIndex(snapshots))].copy()

    zone_parts = equipment.loc[
        equipment["service_scope"] == "zone",
        ["timestamp", "category", "zone", "power_kW"]
    ]
    zone_agg = zone_parts.groupby(["timestamp", "zone", "category"], as_index=False)["power_kW"].sum()

    shared_parts = equipment.loc[
        equipment["service_scope"] == "shared",
        ["timestamp", "category", "power_kW"]
    ]
    shared_agg = shared_parts.groupby(["timestamp", "category"], as_index=False)["power_kW"].sum()

    # Se existir fonte bruta de heating partilhado, usa-a em vez do valor preparado.
    # Isto evita dupla contagem e tende a dar um baseline mensal mais coerente.
    if len(raw_heating) > 0:
        raw_shared = raw_heating.loc[
            raw_heating["timestamp"].isin(pd.DatetimeIndex(snapshots)),
            ["timestamp", "category", "power_kW"]
        ].copy()

        if len(raw_shared) > 0:
            shared_agg = shared_agg.loc[shared_agg["category"] != "heating"].copy()
            shared_agg = pd.concat([shared_agg, raw_shared], ignore_index=True)
            shared_agg = shared_agg.groupby(["timestamp", "category"], as_index=False)["power_kW"].sum()

    shared_rows = []
    for _, row in shared_agg.iterrows():
        for zone, weight in zw["weight"].items():
            if weight > 0:
                shared_rows.append({
                    "timestamp": row["timestamp"],
                    "zone": zone,
                    "category": row["category"],
                    "power_kW": row["power_kW"] * float(weight),
                })

    shared_zone = pd.DataFrame(shared_rows)

    baseline_long = pd.concat([zone_agg, shared_zone], axis=0, ignore_index=True)
    if baseline_long.empty:
        raise ValueError("No HVAC baseline could be built from equipment sheets / raw heating source")

    # Ajusta apenas as componentes elétricas para ficarem alinhadas com hvac_baseline_kW.
    electric_mask = baseline_long["category"].isin(ELECTRIC_CATEGORIES)

    hist_total_elec = (
        baseline_long.loc[electric_mask]
        .groupby("timestamp", as_index=False)["power_kW"]
        .sum()
        .rename(columns={"power_kW": "hist_elec_hvac_kW"})
    )

    hvac_target = ts.loc[snapshots, ["hvac_baseline_kW"]].reset_index().rename(columns={"snapshot": "timestamp"})
    hvac_target.columns = ["timestamp", "hvac_baseline_kW"]

    scaler = hvac_target.merge(hist_total_elec, on="timestamp", how="left")
    scaler["hist_elec_hvac_kW"] = scaler["hist_elec_hvac_kW"].fillna(0.0)
    scaler["scale"] = np.where(
        scaler["hist_elec_hvac_kW"] > 1e-9,
        scaler["hvac_baseline_kW"] / scaler["hist_elec_hvac_kW"],
        1.0,
    )

    baseline_long = baseline_long.merge(scaler[["timestamp", "scale"]], on="timestamp", how="left")
    baseline_long.loc[electric_mask, "power_kW"] = (
        baseline_long.loc[electric_mask, "power_kW"] * baseline_long.loc[electric_mask, "scale"].fillna(1.0)
    )
    baseline_long = baseline_long.drop(columns=["scale"])

    zone_cat = (
        baseline_long.pivot_table(
            index=["timestamp", "zone"],
            columns="category",
            values="power_kW",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=ALL_CATEGORIES, fill_value=0.0)
        .sort_index()
    )

    zone_cat["hvac_electric_kW"] = zone_cat[ELECTRIC_CATEGORIES].sum(axis=1)
    zone_cat["hvac_thermal_total_kW"] = zone_cat[THERMAL_CATEGORIES].sum(axis=1)

    cat_total = (
        zone_cat.reset_index()
        .groupby("timestamp")[ALL_CATEGORIES]
        .sum()
        .reindex(snapshots)
        .fillna(0.0)
    )

    zone_by_cat = {
        cat: zone_cat[cat].unstack("zone").reindex(index=snapshots, columns=zone_cols, fill_value=0.0)
        for cat in ALL_CATEGORIES
    }

    return zone_cat, cat_total, zone_by_cat


def build_zone_shares(zone_cat: pd.DataFrame, snapshots: pd.DatetimeIndex, zone_cols, categories=None):
    categories = categories or ALL_CATEGORIES
    shares = {cat: pd.DataFrame(0.0, index=snapshots, columns=zone_cols) for cat in categories}
    df = zone_cat.reset_index()

    for cat in categories:
        pivot = df.pivot(index="timestamp", columns="zone", values=cat).reindex(
            index=snapshots,
            columns=zone_cols,
            fill_value=0.0,
        )
        total = pivot.sum(axis=1).replace(0.0, np.nan)
        share = pivot.div(total, axis=0)

        fallback = share.mean(axis=0).fillna(0.0)
        if float(fallback.sum()) > 0:
            fallback = fallback / fallback.sum()
        else:
            fallback[:] = 1.0 / max(len(zone_cols), 1)

        for z in zone_cols:
            share[z] = share[z].fillna(float(fallback[z]))

        shares[cat] = share.fillna(0.0)

    return shares


def estimate_thermal_parameters(
    ts_hist: pd.DataFrame,
    zone_heat_hist: pd.DataFrame,
    zone_cool_hist: pd.DataFrame,
) -> pd.DataFrame:
    zone_cols = [c for c in ts_hist.columns if str(c).startswith("zone")]
    rows = []
    tout = ts_hist["Tout_C"].to_numpy(dtype=float)

    for zone in zone_cols:
        y_next = ts_hist[zone].shift(-1).iloc[:-1].to_numpy(dtype=float)
        X = np.column_stack([
            ts_hist[zone].iloc[:-1].to_numpy(dtype=float),
            tout[:-1],
            zone_heat_hist[zone].iloc[:-1].to_numpy(dtype=float),
            zone_cool_hist[zone].iloc[:-1].to_numpy(dtype=float),
            np.ones(len(ts_hist) - 1),
        ])

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_next)
        X = X[mask]
        y = y_next[mask]

        if len(y) < 24:
            a, b, gh, gc, c = 0.90, 0.03, 0.01, -0.01, 0.5
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b, gh, gc, c = beta.tolist()

        a = float(np.clip(a, 0.0, 1.00))
        b = float(np.clip(b, -0.5, 0.5))
        if gh <= 1e-5:
            gh = 0.01
        gh = float(np.clip(gh, 1e-4, 1.0))
        if gc >= -1e-5:
            gc = -0.01
        gc = float(np.clip(gc, -1.0, -1e-4))
        c = float(np.clip(c, -5.0, 5.0))

        rows.append({"zone": zone, "a": a, "b": b, "gh": gh, "gc": gc, "c": c})

    return pd.DataFrame(rows).set_index("zone")


def build_comfort_bounds(ts_month: pd.DataFrame, scenario: dict):
    zone_cols = [c for c in ts_month.columns if str(c).startswith("zone")]
    tmin = ts_month[zone_cols].min(axis=0) + float(scenario.get("Tmin_offset_C", 0.0))
    tmax = ts_month[zone_cols].max(axis=0) + float(scenario.get("Tmax_offset_C", 0.0))
    initial_t = ts_month.iloc[0][zone_cols].astype(float)

    tmin = np.minimum(tmin.astype(float), initial_t)
    tmax = np.maximum(tmax.astype(float), initial_t)
    tmax = np.maximum(tmax, tmin + 0.05)

    return pd.Series(tmin, index=zone_cols, dtype=float), pd.Series(tmax, index=zone_cols, dtype=float)


def relax_comfort_bounds(tmin: pd.Series, tmax: pd.Series, delta_c: float):
    if delta_c <= 0:
        return tmin.copy(), tmax.copy()
    return tmin.astype(float) - float(delta_c), tmax.astype(float) + float(delta_c)


def force_pypsa_native_types(network):
    components = ["carriers", "buses", "generators", "loads", "links", "stores"]
    for comp_name in components:
        df = getattr(network, comp_name, None)
        if df is None or len(df) == 0:
            continue

        df.index = pd.Index(
            [None if pd.isna(x) else str(x) for x in df.index],
            dtype=object,
            name=df.index.name,
        )
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)

        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or str(df[col].dtype).startswith("string"):
                df[col] = df[col].map(lambda x: None if pd.isna(x) else str(x)).astype(object)

    if hasattr(network, "snapshots"):
        network.snapshots = _to_native_datetime_index(network.snapshots)

    for attr_name in ["generators_t", "loads_t", "links_t", "stores_t"]:
        container = getattr(network, attr_name, None)
        if container is None:
            continue

        for subname in dir(container):
            if subname.startswith("_"):
                continue
            obj = getattr(container, subname)
            if isinstance(obj, pd.DataFrame):
                obj.index = _to_native_datetime_index(obj.index)
                obj.columns = pd.Index([str(c) for c in obj.columns], dtype=object)
            elif isinstance(obj, pd.Series):
                obj.index = _to_native_datetime_index(obj.index)


def build_network(ts_month, cat_total_month, zone_shares, params, tmin, tmax, scenario):
    _require_pypsa()

    snapshots = _to_native_datetime_index(ts_month.index)
    ts_month = ts_month.copy()
    ts_month.index = snapshots

    cat_total_month = cat_total_month.copy()
    cat_total_month.index = snapshots

    for cat in zone_shares:
        zone_shares[cat] = zone_shares[cat].copy()
        zone_shares[cat].index = snapshots

    zone_cols = [c for c in ts_month.columns if str(c).startswith("zone")]

    n = pypsa.Network()
    n.set_snapshots(snapshots)

    n.add("Carrier", "electricity")
    n.add("Carrier", "heating")
    n.add("Carrier", "cooling")
    n.add("Carrier", "ventilation")
    n.add("Carrier", "pumping")
    n.add("Carrier", "sink")

    n.add("Bus", "electricity", carrier="electricity")
    n.add("Generator", "grid", bus="electricity", carrier="electricity", p_nom=1e6, marginal_cost=0.0)
    n.generators_t.marginal_cost = pd.DataFrame(
        {"grid": ts_month["price_eur_kWh"].astype(float).to_numpy()},
        index=snapshots,
    )

    n.add("Load", "base_load", bus="electricity", carrier="electricity")
    n.loads_t.p_set = pd.DataFrame(
        {"base_load": ts_month["base_kW"].astype(float).to_numpy()},
        index=snapshots,
    )

    cat_is_electric = {
        "heating": False,
        "cooling": True,
        "ventilation": True,
        "pumping": True,
    }

    for cat in ALL_CATEGORIES:
        link_name = f"flex_{cat}"
        sink_bus = f"sink_{cat}"
        store_name = f"dump_{cat}"

        n.add("Bus", sink_bus, carrier="sink")
        e_nom = float(max(1.0, ts_month["hvac_max_kW"].max() * len(snapshots)))
        n.add("Store", store_name, bus=sink_bus, e_nom=e_nom, e_initial=0.0, e_cyclic=False)

        upper = cat_total_month[cat].astype(float) * float(scenario.get(f"{cat}_max_pu", 1.0))
        lower = cat_total_month[cat].astype(float) * float(scenario.get(f"{cat}_min_pu", 1.0))
        upper = np.maximum(upper, lower)
        p_nom = float(max(1.0, upper.max()))

        bus0 = "electricity" if cat_is_electric[cat] else sink_bus
        n.add("Link", link_name, bus0=bus0, bus1=sink_bus, carrier=cat, p_nom=p_nom, efficiency=1.0)

        if not hasattr(n.links_t, "p_min_pu") or n.links_t.p_min_pu.empty:
            n.links_t.p_min_pu = pd.DataFrame(index=snapshots)
        if not hasattr(n.links_t, "p_max_pu") or n.links_t.p_max_pu.empty:
            n.links_t.p_max_pu = pd.DataFrame(index=snapshots)

        n.links_t.p_min_pu[link_name] = (lower / p_nom).astype(float).to_numpy()
        n.links_t.p_max_pu[link_name] = (upper / p_nom).astype(float).to_numpy()

    force_pypsa_native_types(n)
    n.sanitize()

    extra = {
        "zone_cols": zone_cols,
        "params": params.copy(),
        "tmin": tmin.copy(),
        "tmax": tmax.copy(),
        "tout": ts_month["Tout_C"].astype(float).copy(),
        "initial_tin": ts_month.iloc[0][zone_cols].astype(float).copy(),
        "zone_shares": {k: v.copy() for k, v in zone_shares.items()},
        "hvac_cap": ts_month["hvac_max_kW"].astype(float).copy(),
    }

    def extra_functionality(network, sns):
        m = network.model
        link_p = m.variables["Link-p"]
        sns_index = pd.Index(pd.to_datetime(list(sns)).to_pydatetime(), name="snapshot")
        zone_index = pd.Index(zone_cols, name="zone")
        tin = m.add_variables(lower=-50.0, upper=60.0, coords=[sns_index, zone_index], name="Tin")

        for zone in zone_cols:
            m.add_constraints(tin.loc[sns_index[0], zone] == float(extra["initial_tin"][zone]))

        for sn in sns_index:
            for zone in zone_cols:
                m.add_constraints(tin.loc[sn, zone] >= float(extra["tmin"][zone]))
                m.add_constraints(tin.loc[sn, zone] <= float(extra["tmax"][zone]))

        for i in range(len(sns_index) - 1):
            sn = sns_index[i]
            sn_next = sns_index[i + 1]

            for zone in zone_cols:
                a = float(extra["params"].loc[zone, "a"])
                b = float(extra["params"].loc[zone, "b"])
                gh = float(extra["params"].loc[zone, "gh"])
                gc = float(extra["params"].loc[zone, "gc"])
                c = float(extra["params"].loc[zone, "c"])

                heat_share = float(extra["zone_shares"]["heating"].loc[sn, zone])
                cool_share = float(extra["zone_shares"]["cooling"].loc[sn, zone])

                heat_expr = heat_share * link_p.loc[sn, "flex_heating"]
                cool_expr = cool_share * link_p.loc[sn, "flex_cooling"]

                rhs = (
                    a * tin.loc[sn, zone]
                    + b * float(extra["tout"].loc[sn])
                    + gh * heat_expr
                    + gc * cool_expr
                    + c
                )
                m.add_constraints(tin.loc[sn_next, zone] == rhs)

        for sn in sns_index:
            total_elec_hvac = (
                link_p.loc[sn, "flex_cooling"]
                + link_p.loc[sn, "flex_ventilation"]
                + link_p.loc[sn, "flex_pumping"]
            )
            m.add_constraints(total_elec_hvac <= float(extra["hvac_cap"].loc[sn]))

    return n, extra_functionality


def solve_network(network, extra_functionality, solver_name: str = SOLVER_NAME):
    _require_pypsa()
    force_pypsa_native_types(network)
    return network.optimize(solver_name=solver_name, extra_functionality=extra_functionality)


def is_successful_optimization(status, condition) -> bool:
    status_str = str(status).strip().lower()
    cond_str = str(condition).strip().lower()

    success_status = {"ok", "optimal"}
    success_conditions = {"optimal", "feasible", "feasible_or_unbounded"}

    if status_str in success_status:
        return True
    if status_str == "warning" and cond_str in success_conditions:
        return True
    return False


def extract_hvac_opt(network, snapshots):
    link_cols = [f"flex_{c}" for c in ALL_CATEGORIES]
    p0 = getattr(getattr(network, "links_t", None), "p0", None)

    if not isinstance(p0, pd.DataFrame) or p0.empty:
        raise RuntimeError(
            "O solver não devolveu dispatch para os links HVAC. Isto acontece quando a otimização termina infeasible."
        )

    missing = [c for c in link_cols if c not in p0.columns]
    if missing:
        available = list(map(str, p0.columns.tolist()))
        raise RuntimeError(
            "Faltam colunas de dispatch dos links HVAC nos resultados: "
            f"{missing}. Colunas disponíveis: {available[:12]}{'...' if len(available) > 12 else ''}"
        )

    hvac_opt = p0.reindex(index=_to_native_datetime_index(snapshots))[link_cols].copy()
    hvac_opt.columns = ALL_CATEGORIES
    hvac_opt["hvac_electric_kW"] = hvac_opt[ELECTRIC_CATEGORIES].sum(axis=1)
    hvac_opt["hvac_thermal_total_kW"] = hvac_opt[THERMAL_CATEGORIES].sum(axis=1)
    return hvac_opt


def build_baseline_integrity_report(ts_month, cat_total_month, params, has_raw_heating=False):
    def safe_sum(df, cols):
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series(0.0, index=df.index)
        return df[existing].sum(axis=1)

    electric_hist = safe_sum(cat_total_month, ELECTRIC_CATEGORIES)
    target_hvac = ts_month["hvac_baseline_kW"].astype(float)
    scale_ratio = np.where(target_hvac > 1e-9, electric_hist / target_hvac, np.nan)

    elec_total = float(electric_hist.sum())
    cooling_sum = float(cat_total_month["cooling"].sum()) if "cooling" in cat_total_month.columns else 0.0
    ventilation_sum = float(cat_total_month["ventilation"].sum()) if "ventilation" in cat_total_month.columns else 0.0
    pumping_sum = float(cat_total_month["pumping"].sum()) if "pumping" in cat_total_month.columns else 0.0
    heating_sum = float(cat_total_month["heating"].sum()) if "heating" in cat_total_month.columns else 0.0

    rep = {
        "n_hours": int(len(ts_month)),
        "month": int(pd.DatetimeIndex(ts_month.index)[0].month),
        "month_name": month_name_pt(int(pd.DatetimeIndex(ts_month.index)[0].month)),
        "target_hvac_electric_mean_kW": float(target_hvac.mean()),
        "reconstructed_hvac_electric_mean_kW": float(electric_hist.mean()),
        "target_hvac_electric_sum_kWh": float(target_hvac.sum()),
        "reconstructed_hvac_electric_sum_kWh": elec_total,
        "hvac_electric_gap_kWh": float(electric_hist.sum() - target_hvac.sum()),
        "mean_scale_ratio_reconstructed_to_target": float(np.nanmean(scale_ratio)) if np.isfinite(scale_ratio).any() else np.nan,
        "heating_sum_kWh": heating_sum,
        "cooling_sum_kWh": cooling_sum,
        "ventilation_sum_kWh": ventilation_sum,
        "pumping_sum_kWh": pumping_sum,
        "cooling_share_of_electric_hvac_pct": float(100.0 * cooling_sum / elec_total) if elec_total > 1e-9 else 0.0,
        "ventilation_share_of_electric_hvac_pct": float(100.0 * ventilation_sum / elec_total) if elec_total > 1e-9 else 0.0,
        "pumping_share_of_electric_hvac_pct": float(100.0 * pumping_sum / elec_total) if elec_total > 1e-9 else 0.0,
        "base_load_sum_kWh": float(ts_month["base_kW"].sum()),
        "hvac_cap_mean_kW": float(ts_month["hvac_max_kW"].mean()),
        "a_mean": float(params["a"].mean()),
        "b_mean": float(params["b"].mean()),
        "gh_mean": float(params["gh"].mean()),
        "gc_mean": float(params["gc"].mean()),
        "c_mean": float(params["c"].mean()),
        "heating_from_raw_source_detected": int(bool(has_raw_heating)),
    }
    return rep


def build_flex_diagnostics(ts_month, cat_total_month, zone_shares, params, tmin, tmax, scenario):
    zone_cols = [c for c in ts_month.columns if str(c).startswith("zone")]
    snapshots = list(ts_month.index)

    rows = []
    issues = []
    avail_min = {}
    avail_max = {}

    for cat in ALL_CATEGORIES:
        avail_min[cat] = cat_total_month[cat].astype(float) * float(scenario.get(f"{cat}_min_pu", 1.0))
        avail_max[cat] = cat_total_month[cat].astype(float) * float(scenario.get(f"{cat}_max_pu", 1.0))

    hvac_cap = ts_month["hvac_max_kW"].astype(float)
    max_elec_total = np.minimum(
        avail_max["cooling"] + avail_max["ventilation"] + avail_max["pumping"],
        hvac_cap,
    )

    for i in range(len(snapshots) - 1):
        sn = snapshots[i]
        sn_next = snapshots[i + 1]
        tout = float(ts_month.loc[sn, "Tout_C"])

        for zone in zone_cols:
            Tin = float(ts_month.loc[sn, zone])
            Tmin_next = float(tmin[zone])
            Tmax_next = float(tmax[zone])

            a = float(params.loc[zone, "a"])
            b = float(params.loc[zone, "b"])
            gh = float(params.loc[zone, "gh"])
            gc = float(params.loc[zone, "gc"])
            c = float(params.loc[zone, "c"])

            no_hvac_next = a * Tin + b * tout + c

            req_heat_zone = max(0.0, (Tmin_next - no_hvac_next) / max(gh, 1e-6)) if no_hvac_next < Tmin_next else 0.0
            req_cool_zone = max(0.0, (no_hvac_next - Tmax_next) / max(-gc, 1e-6)) if no_hvac_next > Tmax_next else 0.0

            heat_share = float(zone_shares["heating"].loc[sn, zone]) if "heating" in zone_shares else 0.0
            cool_share = float(zone_shares["cooling"].loc[sn, zone]) if "cooling" in zone_shares else 0.0

            req_heat_system = req_heat_zone / heat_share if heat_share > 1e-9 else np.nan
            req_cool_system = req_cool_zone / cool_share if cool_share > 1e-9 else np.nan

            heat_conflict = np.isfinite(req_heat_system) and req_heat_system > float(avail_max["heating"].loc[sn]) + 1e-6
            cool_conflict = np.isfinite(req_cool_system) and req_cool_system > float(max_elec_total.loc[sn]) + 1e-6

            rows.append({
                "timestamp": sn,
                "timestamp_next": sn_next,
                "zone": zone,
                "Tin": Tin,
                "Tout": tout,
                "Tmin_bound": Tmin_next,
                "Tmax_bound": Tmax_next,
                "required_system_heating_kW": req_heat_system,
                "required_system_cooling_kW": req_cool_system,
                "available_heating_max_kW": float(avail_max["heating"].loc[sn]),
                "available_electric_hvac_max_kW": float(max_elec_total.loc[sn]),
                "heat_conflict": bool(heat_conflict),
                "cool_conflict": bool(cool_conflict),
            })

            if heat_conflict:
                issues.append({
                    "timestamp": sn,
                    "zone": zone,
                    "issue_type": "insufficient_heating",
                    "detail": (
                        f"Para manter Tin_next >= Tmin, a zona exigiria ~{req_heat_system:.2f} kW "
                        f"de heating, acima do máximo disponível ~{float(avail_max['heating'].loc[sn]):.2f} kW."
                    ),
                })

            if cool_conflict:
                issues.append({
                    "timestamp": sn,
                    "zone": zone,
                    "issue_type": "insufficient_cooling_or_electric_cap",
                    "detail": (
                        f"Para manter Tin_next <= Tmax, a zona exigiria ~{req_cool_system:.2f} kW "
                        f"de cooling, acima do máximo elétrico disponível ~{float(max_elec_total.loc[sn]):.2f} kW."
                    ),
                })

    diag_df = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)

    summary = {
        "n_hours": int(len(ts_month)),
        "month": int(pd.DatetimeIndex(ts_month.index)[0].month),
        "month_name": month_name_pt(int(pd.DatetimeIndex(ts_month.index)[0].month)),
        "heating_sum_kWh": float(cat_total_month["heating"].sum()),
        "cooling_sum_kWh": float(cat_total_month["cooling"].sum()),
        "ventilation_sum_kWh": float(cat_total_month["ventilation"].sum()),
        "pumping_sum_kWh": float(cat_total_month["pumping"].sum()),
        "hvac_cap_mean_kW": float(ts_month["hvac_max_kW"].mean()),
        "n_heating_conflicts": int(diag_df["heat_conflict"].sum()) if len(diag_df) else 0,
        "n_cooling_conflicts": int(diag_df["cool_conflict"].sum()) if len(diag_df) else 0,
        "max_required_system_heating_kW": float(pd.to_numeric(diag_df["required_system_heating_kW"], errors="coerce").max()) if len(diag_df) else np.nan,
        "max_required_system_cooling_kW": float(pd.to_numeric(diag_df["required_system_cooling_kW"], errors="coerce").max()) if len(diag_df) else np.nan,
    }

    recommendations = []
    if summary["n_heating_conflicts"] > 0:
        recommendations.append("Há horas/zonas com heating insuficiente. Rever heating_max_pu ou conforto inferior.")
    if summary["n_cooling_conflicts"] > 0:
        recommendations.append("Há horas/zonas com cooling insuficiente ou limite elétrico demasiado baixo. Rever cooling_max_pu e hvac_max_kW.")
    if not recommendations:
        recommendations.append("Sem conflitos evidentes nesta formulação mensal.")

    return {
        "diagnostic_rows": diag_df,
        "diagnostic_issues": issues_df,
        "summary": summary,
        "recommendations": recommendations,
    }


def summarise_baseline_only(scenario_name, ts_month, cat_total_month, params, heating_cost_note=True):
    hvac_hist = cat_total_month[ALL_CATEGORIES].copy()
    hvac_hist["hvac_electric_kW"] = hvac_hist[ELECTRIC_CATEGORIES].sum(axis=1)
    hvac_hist["hvac_thermal_total_kW"] = hvac_hist[THERMAL_CATEGORIES].sum(axis=1)
    hvac_final = hvac_hist.copy()

    price = ts_month["price_eur_kWh"].astype(float)
    electric_cost = float(((ts_month["base_kW"] + hvac_hist["hvac_electric_kW"]) * price).sum())

    summary = {
        "scenario": scenario_name,
        "mode": "baseline_reference_only_month",
        "month": int(pd.DatetimeIndex(ts_month.index)[0].month),
        "month_name": month_name_pt(int(pd.DatetimeIndex(ts_month.index)[0].month)),
        "period_start": str(ts_month.index.min()),
        "period_end": str(ts_month.index.max()),
        "n_snapshots": int(len(ts_month)),
        "baseline_energy_cost_eur": electric_cost,
        "optimized_energy_cost_eur": electric_cost,
        "savings_eur": 0.0,
        "savings_pct": 0.0,
        "base_energy_kWh": float(ts_month["base_kW"].sum()),
        "heating_energy_kWh": float(hvac_hist["heating"].sum()),
        "cooling_energy_kWh": float(hvac_hist["cooling"].sum()),
        "ventilation_energy_kWh": float(hvac_hist["ventilation"].sum()),
        "pumping_energy_kWh": float(hvac_hist["pumping"].sum()),
        "hvac_electric_energy_kWh": float(hvac_hist["hvac_electric_kW"].sum()),
        "hvac_thermal_total_energy_kWh": float(hvac_hist["hvac_thermal_total_kW"].sum()),
        "a_mean": float(params["a"].mean()),
        "b_mean": float(params["b"].mean()),
        "gh_mean": float(params["gh"].mean()),
        "gc_mean": float(params["gc"].mean()),
        "c_mean": float(params["c"].mean()),
        "comment": (
            "Custo calculado apenas sobre energia elétrica modelada (base + cooling/ventilation/pumping). Se o heating vier do boiler/gás, esse custo não está incluído."
            if heating_cost_note else
            "Baseline mensal calculado sem otimização."
        ),
    }

    return {"summary": summary, "hvac_hist": hvac_hist, "hvac_final": hvac_final, "params": params}


def summarise_flex_results(scenario_name, ts_month, cat_total_month, params, network):
    hvac_opt = extract_hvac_opt(network, ts_month.index)

    hvac_hist = cat_total_month[ALL_CATEGORIES].copy()
    hvac_hist["hvac_electric_kW"] = hvac_hist[ELECTRIC_CATEGORIES].sum(axis=1)
    hvac_hist["hvac_thermal_total_kW"] = hvac_hist[THERMAL_CATEGORIES].sum(axis=1)

    price = ts_month["price_eur_kWh"].astype(float)
    baseline_cost = float(((ts_month["base_kW"] + hvac_hist["hvac_electric_kW"]) * price).sum())
    optimized_cost = float(((ts_month["base_kW"] + hvac_opt["hvac_electric_kW"]) * price).sum())
    savings = baseline_cost - optimized_cost

    summary = {
        "scenario": scenario_name,
        "mode": "optimized_flex_scenario_month_heatcool",
        "month": int(pd.DatetimeIndex(ts_month.index)[0].month),
        "month_name": month_name_pt(int(pd.DatetimeIndex(ts_month.index)[0].month)),
        "period_start": str(ts_month.index.min()),
        "period_end": str(ts_month.index.max()),
        "n_snapshots": int(len(ts_month)),
        "baseline_energy_cost_eur": baseline_cost,
        "optimized_energy_cost_eur": optimized_cost,
        "savings_eur": savings,
        "savings_pct": float(100.0 * savings / baseline_cost) if baseline_cost else 0.0,
        "base_energy_kWh": float(ts_month["base_kW"].sum()),
        "heating_hist_kWh": float(hvac_hist["heating"].sum()),
        "cooling_hist_kWh": float(hvac_hist["cooling"].sum()),
        "heating_opt_kWh": float(hvac_opt["heating"].sum()),
        "cooling_opt_kWh": float(hvac_opt["cooling"].sum()),
        "hvac_electric_hist_kWh": float(hvac_hist["hvac_electric_kW"].sum()),
        "hvac_electric_opt_kWh": float(hvac_opt["hvac_electric_kW"].sum()),
        "a_mean": float(params["a"].mean()),
        "b_mean": float(params["b"].mean()),
        "gh_mean": float(params["gh"].mean()),
        "gc_mean": float(params["gc"].mean()),
        "c_mean": float(params["c"].mean()),
        "comment": "Custo calculado apenas sobre energia elétrica (base + cooling/ventilation/pumping). Heating tratado como térmico partilhado nesta versão mensal.",
    }

    return {"summary": summary, "hvac_hist": hvac_hist, "hvac_final": hvac_opt, "params": params}


def solve_with_relaxation(ts_month, cat_total_month, zone_shares, params, tmin_base, tmax_base, scenario):
    attempts = []
    deltas = RELAXATION_STEPS_C if AUTO_RELAX_IF_INFEASIBLE else [0.0]

    final_network = None
    final_status = None
    final_condition = None
    final_bounds = (tmin_base, tmax_base)

    for delta in deltas:
        tmin_try, tmax_try = relax_comfort_bounds(tmin_base, tmax_base, delta)
        network, extra_functionality = build_network(
            ts_month,
            cat_total_month,
            zone_shares,
            params,
            tmin_try,
            tmax_try,
            scenario,
        )
        status, condition = solve_network(network, extra_functionality, SOLVER_NAME)

        attempts.append({
            "relaxation_C": float(delta),
            "status": str(status),
            "condition": str(condition),
        })

        final_network = network
        final_status = status
        final_condition = condition
        final_bounds = (tmin_try, tmax_try)

        if is_successful_optimization(status, condition):
            return final_network, final_status, final_condition, final_bounds, pd.DataFrame(attempts)

    return final_network, final_status, final_condition, final_bounds, pd.DataFrame(attempts)


def print_baseline_integrity_report(report: dict):
    print("\n" + "=" * 110)
    print("RELATÓRIO DE INTEGRIDADE DO BASELINE MENSAL (HEATING + COOLING)")
    print("=" * 110)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k:52s}: {v:.4f}")
        else:
            print(f"{k:52s}: {v}")
    print("=" * 110 + "\n")


def print_monthly_crosscheck(crosscheck: pd.DataFrame):
    if crosscheck is None or len(crosscheck) == 0:
        return

    r = crosscheck.iloc[0]

    print("\n" + "=" * 110)
    print("COMPARAÇÃO MENSAL COM OS RESUMOS BRUTOS")
    print("=" * 110)
    print(f"Mês: {r['month_name']}")
    print(f" - HVAC elétrico target / reconstruído [kWh]: {float(r['hvac_baseline_electric_kWh']):.2f} / {float(r['hvac_electric_reconstructed_kWh']):.2f}")
    print(f" - Gap elétrico reconstruído - target [kWh]: {float(r['hvac_electric_gap_vs_target_kWh']):.2f}")

    if "raw_cooling_kWh" in crosscheck.columns and pd.notna(r.get("raw_cooling_kWh", np.nan)):
        print(f" - Cooling preparado / bruto [kWh]:          {float(r['cooling']):.2f} / {float(r['raw_cooling_kWh']):.2f}")
        if "cooling_gap_kWh" in crosscheck.columns:
            print(f"   Gap cooling [kWh]:                        {float(r['cooling_gap_kWh']):.2f}")

    if "raw_ventilation_kWh" in crosscheck.columns and pd.notna(r.get("raw_ventilation_kWh", np.nan)):
        print(f" - Ventilação preparada / bruta [kWh]:       {float(r['ventilation']):.2f} / {float(r['raw_ventilation_kWh']):.2f}")
        if "ventilation_gap_kWh" in crosscheck.columns:
            print(f"   Gap ventilação [kWh]:                     {float(r['ventilation_gap_kWh']):.2f}")

    if "raw_pumping_kWh" in crosscheck.columns and pd.notna(r.get("raw_pumping_kWh", np.nan)):
        print(f" - Bombas preparada / bruta [kWh]:           {float(r['pumping']):.2f} / {float(r['raw_pumping_kWh']):.2f}")
        if "pumping_gap_kWh" in crosscheck.columns:
            print(f"   Gap bombas [kWh]:                         {float(r['pumping_gap_kWh']):.2f}")

    if "raw_heating_gas_kWh" in crosscheck.columns and pd.notna(r.get("raw_heating_gas_kWh", np.nan)):
        print(f" - Heating preparado / bruto [kWh]:          {float(r['heating_total_kWh']):.2f} / {float(r['raw_heating_gas_kWh']):.2f}")
        if "heating_gap_kWh" in crosscheck.columns:
            print(f"   Gap heating [kWh]:                        {float(r['heating_gap_kWh']):.2f}")

    print("=" * 110 + "\n")


def print_diagnostic_report(diag, scenario_name):
    s = diag["summary"]
    issues = diag["diagnostic_issues"]

    print("\n" + "=" * 110)
    print("RELATÓRIO DE DIAGNÓSTICO MENSAL - CENÁRIO FLEX HEAT/COOL")
    print("=" * 110)
    print(f"Cenário:                              {scenario_name}")
    print(f"Mês:                                  {s['month_name']} ({s['month']})")
    print(f"Horas analisadas:                     {s['n_hours']}")
    print(f"Heating [kWh]:                        {s['heating_sum_kWh']:.2f}")
    print(f"Cooling [kWh]:                        {s['cooling_sum_kWh']:.2f}")
    print(f"Ventilation [kWh]:                    {s['ventilation_sum_kWh']:.2f}")
    print(f"Pumping [kWh]:                        {s['pumping_sum_kWh']:.2f}")
    print(f"Conflitos de heating:                 {s['n_heating_conflicts']} (pico req. {s['max_required_system_heating_kW']:.2f} kW)")
    print(f"Conflitos de cooling:                 {s['n_cooling_conflicts']} (pico req. {s['max_required_system_cooling_kW']:.2f} kW)")

    if len(issues) > 0:
        print("-" * 110)
        print("Principais conflitos detetados:")
        for _, row in issues.head(12).iterrows():
            print(f" - {row['timestamp']} | {row['zone']} | {row['issue_type']} | {row['detail']}")

    print("-" * 110)
    print("Recomendações:")
    for rec in diag["recommendations"]:
        print(f" - {rec}")
    print("=" * 110 + "\n")


def print_solution_report(results, attempts_df):
    s = results["summary"]

    print("\n" + "=" * 110)
    print("RELATÓRIO RESUMIDO - BASELINE / FLEX MENSAL (HEATING + COOLING)")
    print("=" * 110)
    print(f"Cenário:                              {s['scenario']}")
    print(f"Modo:                                 {s.get('mode', '')}")
    print(f"Mês:                                  {s['month_name']} ({s['month']})")
    print(f"Período:                              {s['period_start']} -> {s['period_end']}")
    print(f"Snapshots:                            {s['n_snapshots']}")
    print(f"Custo energia baseline [EUR]:         {s['baseline_energy_cost_eur']:.2f}")
    print(f"Custo energia final [EUR]:            {s['optimized_energy_cost_eur']:.2f}")
    print(f"Poupança [EUR]:                       {s['savings_eur']:.2f}")
    print(f"Poupança [%]:                         {s['savings_pct']:.2f}")
    print(f"Energia base [kWh]:                   {s['base_energy_kWh']:.2f}")

    if "heating_energy_kWh" in s:
        print(f"Heating baseline [kWh]:               {s['heating_energy_kWh']:.2f}")
        print(f"Cooling baseline [kWh]:               {s['cooling_energy_kWh']:.2f}")
        print(f"Ventilation baseline [kWh]:           {s['ventilation_energy_kWh']:.2f}")
        print(f"Pumping baseline [kWh]:               {s['pumping_energy_kWh']:.2f}")
        print(f"HVAC elétrico baseline [kWh]:         {s['hvac_electric_energy_kWh']:.2f}")
        print(f"HVAC thermal total [kWh]:             {s['hvac_thermal_total_energy_kWh']:.2f}")
    else:
        print(f"Heating hist./opt [kWh]:              {s['heating_hist_kWh']:.2f} / {s['heating_opt_kWh']:.2f}")
        print(f"Cooling hist./opt [kWh]:              {s['cooling_hist_kWh']:.2f} / {s['cooling_opt_kWh']:.2f}")
        print(f"HVAC elétrico hist./opt [kWh]:        {s['hvac_electric_hist_kWh']:.2f} / {s['hvac_electric_opt_kWh']:.2f}")

    print(f"Parâmetros médios:                    a={s['a_mean']:.4f} | b={s['b_mean']:.4f} | gh={s['gh_mean']:.4f} | gc={s['gc_mean']:.4f} | c={s['c_mean']:.4f}")
    print(f"Comentário:                           {s['comment']}")

    if len(attempts_df) > 0:
        print("-" * 110)
        print("Tentativas do solver:")
        for _, row in attempts_df.iterrows():
            print(f" - relaxação ±{row['relaxation_C']:.2f} °C -> status={row['status']} | condition={row['condition']}")

    print("=" * 110 + "\n")


def build_monthly_crosscheck(ts_month, cat_total_month, raw_source_path: Path | None):
    def safe_cat_sum(df, col):
        if isinstance(df, pd.DataFrame) and col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        return 0.0

    def safe_electric_sum(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        existing = [c for c in ELECTRIC_CATEGORIES if c in df.columns]
        if not existing:
            return 0.0
        return float(df[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).sum())

    try:
        raw_electric = pd.read_excel(raw_source_path, sheet_name="Resumo_Eletricidade", engine="openpyxl") if raw_source_path else None
    except Exception:
        raw_electric = None

    try:
        raw_gas = pd.read_excel(raw_source_path, sheet_name="Resumo_GásNatural", engine="openpyxl") if raw_source_path else None
    except Exception:
        raw_gas = None

    month = int(pd.DatetimeIndex(ts_month.index)[0].month)

    row = {
        "month": month,
        "month_name": month_name_pt(month),
        "hvac_baseline_electric_kWh": float(pd.to_numeric(ts_month["hvac_baseline_kW"], errors="coerce").fillna(0.0).sum()),
        "hvac_electric_reconstructed_kWh": safe_electric_sum(cat_total_month),
        "cooling": safe_cat_sum(cat_total_month, "cooling"),
        "ventilation": safe_cat_sum(cat_total_month, "ventilation"),
        "pumping": safe_cat_sum(cat_total_month, "pumping"),
        "heating_total_kWh": safe_cat_sum(cat_total_month, "heating"),
    }

    if raw_electric is not None and len(raw_electric) > 0:
        cols = [str(c).strip() for c in raw_electric.columns]
        raw_electric.columns = cols
        raw_electric = raw_electric.rename(columns={raw_electric.columns[0]: "month_name"})
        raw_electric["month_name"] = raw_electric["month_name"].astype(str).str.strip()
        raw_electric = raw_electric[raw_electric["month_name"] == month_name_pt(month)]

        if len(raw_electric) > 0:
            rr = raw_electric.iloc[0]
            mapping = {
                "Ventilação [kWh]": "raw_ventilation_kWh",
                "Bombas [kWh]": "raw_pumping_kWh",
                "Arrefecimento [kWh]": "raw_cooling_kWh",
                "Total Mensal [kWh]": "raw_total_electricity_kWh",
            }
            for src_col, dst_col in mapping.items():
                if src_col in raw_electric.columns:
                    row[dst_col] = float(pd.to_numeric(rr[src_col], errors="coerce"))

    if raw_gas is not None and len(raw_gas) > 0:
        cols = [str(c).strip() for c in raw_gas.columns]
        raw_gas.columns = cols
        raw_gas = raw_gas.rename(columns={raw_gas.columns[0]: "month_name"})
        raw_gas["month_name"] = raw_gas["month_name"].astype(str).str.strip()
        raw_gas = raw_gas[raw_gas["month_name"] == month_name_pt(month)]

        if len(raw_gas) > 0:
            rr = raw_gas.iloc[0]
            gas_col = None
            for c in raw_gas.columns:
                if "Consumo total [kWh]" in c:
                    gas_col = c
                    break
            if gas_col is not None:
                row["raw_heating_gas_kWh"] = float(pd.to_numeric(rr[gas_col], errors="coerce"))

    # Diferenças úteis para perceber desvios entre reconstruído e bruto.
    if "raw_cooling_kWh" in row:
        row["cooling_gap_kWh"] = row["cooling"] - row["raw_cooling_kWh"]
    if "raw_ventilation_kWh" in row:
        row["ventilation_gap_kWh"] = row["ventilation"] - row["raw_ventilation_kWh"]
    if "raw_pumping_kWh" in row:
        row["pumping_gap_kWh"] = row["pumping"] - row["raw_pumping_kWh"]
    if "raw_heating_gas_kWh" in row:
        row["heating_gap_kWh"] = row["heating_total_kWh"] - row["raw_heating_gas_kWh"]

    row["hvac_electric_gap_vs_target_kWh"] = (
        row["hvac_electric_reconstructed_kWh"] - row["hvac_baseline_electric_kWh"]
    )

    return pd.DataFrame([row])


def export_package(
    output_dir: Path,
    scenario_name: str,
    baseline_integrity: dict,
    crosscheck_df: pd.DataFrame | None,
    diag: dict | None,
    attempts_df: pd.DataFrame,
    results: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    month_name = results["summary"]["month_name"]
    txt_path = output_dir / f"report_{scenario_name}_{month_name}.txt"
    xlsx_path = output_dir / f"report_{scenario_name}_{month_name}.xlsx"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO MENSAL BASELINE / FLEX (HEATING + COOLING)\n")
        f.write("=" * 100 + "\n\n")
        f.write("INTEGRIDADE DO BASELINE\n")
        for k, v in baseline_integrity.items():
            f.write(f"{k}: {v}\n")

        if crosscheck_df is not None and len(crosscheck_df) > 0:
            f.write("\nCOMPARAÇÃO MENSAL COM O FICHEIRO BRUTO\n")
            for col in crosscheck_df.columns:
                f.write(f"{col}: {crosscheck_df.iloc[0][col]}\n")

        if diag is not None:
            f.write("\nDIAGNÓSTICO DO CENÁRIO FLEX\n")
            for k, v in diag["summary"].items():
                f.write(f"{k}: {v}\n")
            f.write("\nRECOMENDAÇÕES\n")
            for rec in diag["recommendations"]:
                f.write(f" - {rec}\n")

        if len(attempts_df) > 0:
            f.write("\nTENTATIVAS DO SOLVER\n")
            for _, row in attempts_df.iterrows():
                f.write(f" - relaxação ±{row['relaxation_C']:.2f} °C -> status={row['status']} | condition={row['condition']}\n")

        f.write("\nRESUMO DA SOLUÇÃO\n")
        for k, v in results["summary"].items():
            f.write(f"{k}: {v}\n")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([baseline_integrity]).to_excel(writer, sheet_name="baseline_integrity", index=False)

        if crosscheck_df is not None and len(crosscheck_df) > 0:
            crosscheck_df.to_excel(writer, sheet_name="monthly_crosscheck", index=False)

        if diag is not None:
            pd.DataFrame([diag["summary"]]).to_excel(writer, sheet_name="diagnostic_summary", index=False)
            diag["diagnostic_rows"].to_excel(writer, sheet_name="diagnostic_rows", index=False)
            diag["diagnostic_issues"].to_excel(writer, sheet_name="diagnostic_issues", index=False)
            pd.DataFrame({"recommendation": diag["recommendations"]}).to_excel(
                writer,
                sheet_name="recommendations",
                index=False,
            )
            attempts_df.to_excel(writer, sheet_name="solver_attempts", index=False)

        pd.DataFrame([results["summary"]]).to_excel(writer, sheet_name="solution_summary", index=False)
        results["hvac_hist"].to_excel(writer, sheet_name="hvac_hist")
        results["hvac_final"].to_excel(writer, sheet_name="hvac_final")
        results["params"].to_excel(writer, sheet_name="thermal_params")


def resolve_scenario(config_df: pd.DataFrame, scenario_name: str) -> dict:
    rows = config_df.loc[config_df["scenario_name"] == scenario_name]
    base = dict(SCENARIO_DEFAULTS.get(scenario_name, {}))

    if rows.empty:
        if not base:
            raise ValueError(f"Scenario '{scenario_name}' not found and has no default")
        return base

    row = rows.iloc[0].to_dict()
    row.pop("scenario_name", None)

    for k, v in row.items():
        if pd.notna(v):
            base[k] = float(v)

    return base


def run(
    excel_path=None,
    scenario_name: str = DEFAULT_SCENARIO,
    month: int = DEFAULT_MONTH,
    year: int | None = DEFAULT_YEAR,
    raw_source_path=None,
):
    excel_path = resolve_excel_path(excel_path)
    raw_source = resolve_optional_raw_source(raw_source_path)

    data = load_excel_inputs(excel_path)
    validate_inputs(data)

    ts_all = data["timeseries_main"]
    snapshots = choose_month(ts_all, month=month, year=year)
    ts_month = ts_all.loc[snapshots].copy()

    zone_cols = [c for c in ts_month.columns if str(c).startswith("zone")]

    # Reconstrói o baseline HVAC do mês a partir dos dados preparados.
    zone_cat_month, cat_total_month, zone_by_cat = build_zone_category_baseline(data, snapshots, raw_source)
    zone_shares = build_zone_shares(zone_cat_month, snapshots, zone_cols, categories=ALL_CATEGORIES)

    # Os parâmetros térmicos podem ser estimados só no mês ou no histórico completo.
    if ESTIMATE_ON_SELECTED_MONTH:
        params = estimate_thermal_parameters(ts_month, zone_by_cat["heating"], zone_by_cat["cooling"])
    else:
        _, _, zone_by_cat_full = build_zone_category_baseline(data, ts_all.index, raw_source)
        params = estimate_thermal_parameters(ts_all, zone_by_cat_full["heating"], zone_by_cat_full["cooling"])

    has_raw_heating = bool(raw_source is not None and len(extract_raw_shared_heating(raw_source)) > 0)

    baseline_integrity = build_baseline_integrity_report(
        ts_month,
        cat_total_month,
        params,
        has_raw_heating=has_raw_heating,
    )

    month_crosscheck = build_monthly_crosscheck(ts_month, cat_total_month, raw_source)

    print_baseline_integrity_report(baseline_integrity)
    print_monthly_crosscheck(month_crosscheck)

    scenario = resolve_scenario(data["scenario_config"], scenario_name)

    # No baseline só se resume a referência histórica; não há otimização.
    if str(scenario_name).lower() == "baseline" or int(float(scenario.get("active", 1))) == 0:
        results = summarise_baseline_only(
            scenario_name,
            ts_month,
            cat_total_month,
            params,
            heating_cost_note=has_raw_heating,
        )

        attempts_df = pd.DataFrame(columns=["relaxation_C", "status", "condition"])
        diag = None

        print_solution_report(results, attempts_df)

        if EXPORT_OUTPUTS:
            export_package(
                OUTPUT_DIR,
                scenario_name,
                baseline_integrity,
                month_crosscheck,
                diag,
                attempts_df,
                results,
            )

        return {
            "results": results,
            "baseline_integrity": baseline_integrity,
            "monthly_crosscheck": month_crosscheck,
            "scenario": scenario,
            "snapshots": snapshots,
            "cat_total_month": cat_total_month,
            "thermal_params": params,
        }

    # Só avança para o cenário flexível depois do baseline estar montado e validado.
    tmin, tmax = build_comfort_bounds(ts_month, scenario)
    diag = build_flex_diagnostics(ts_month, cat_total_month, zone_shares, params, tmin, tmax, scenario)
    print_diagnostic_report(diag, scenario_name)

    network, status, condition, bounds_used, attempts_df = solve_with_relaxation(
        ts_month,
        cat_total_month,
        zone_shares,
        params,
        tmin,
        tmax,
        scenario,
    )

    if not is_successful_optimization(status, condition):
        raise RuntimeError(
            "Cenário flexível inviável mesmo após relaxação automática. "
            "Consulta o Excel gerado para perceber horas/zonas e parâmetros problemáticos."
        )

    results = summarise_flex_results(scenario_name, ts_month, cat_total_month, params, network)
    results["summary"]["solver_status"] = str(status)
    results["summary"]["termination_condition"] = str(condition)
    results["summary"]["comfort_relaxation_used_C"] = float(max((bounds_used[1] - tmax).abs().max(), 0.0))

    print_solution_report(results, attempts_df)

    if EXPORT_OUTPUTS:
        export_package(
            OUTPUT_DIR,
            scenario_name,
            baseline_integrity,
            month_crosscheck,
            diag,
            attempts_df,
            results,
        )

    return {
        "results": results,
        "baseline_integrity": baseline_integrity,
        "monthly_crosscheck": month_crosscheck,
        "diagnostics": diag,
        "solver_attempts": attempts_df,
        "scenario": scenario,
        "snapshots": snapshots,
        "cat_total_month": cat_total_month,
        "thermal_params": params,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    scenario = sys.argv[1] if len(sys.argv) > 1 and str(sys.argv[1]).strip() else DEFAULT_SCENARIO
    month = int(sys.argv[2]) if len(sys.argv) > 2 and str(sys.argv[2]).strip() else DEFAULT_MONTH
    excel = sys.argv[3] if len(sys.argv) > 3 and str(sys.argv[3]).strip() else None
    raw_source = sys.argv[4] if len(sys.argv) > 4 and str(sys.argv[4]).strip() else None
    year = int(sys.argv[5]) if len(sys.argv) > 5 and str(sys.argv[5]).strip() else DEFAULT_YEAR

    # Exemplos de execução:
    # python src\optimization_debug_fast.py baseline 1 data\data.xlsx Resultados_14abril.xlsx
    # python src\optimization_debug_fast.py starter_flex 1 data\data.xlsx Resultados_14abril.xlsx
    run(excel_path=excel, scenario_name=scenario, month=month, year=year, raw_source_path=raw_source)