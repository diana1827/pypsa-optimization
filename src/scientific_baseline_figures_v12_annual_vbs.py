from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# ======================================================================================
# SCIENTIFIC BASELINE FIGURES V12 (ANNUAL / THESIS-READY / VBS COEFFICIENT DIAGNOSTICS)
# ======================================================================================

ENGLISH_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

PALETTE = {
    "black": "#111111",
    "dark_gray": "#4d4d4d",
    "mid_gray": "#8e8e8e",
    "light_gray": "#d9d9d9",
    "very_light_gray": "#f4f4f4",
    "thermal_conditioning": "#d7b46a",
    "ventilation": "#8fb9d9",
    "pumping": "#d98c8c",
    "temperature": "#6c757d",
    "price": "#5b8aa6",
    "zone_iqr": "#d0d0d0",
    "zone_median": "#1f1f1f",
    "mean_marker": "#b07f2c",
    "a": "#4c4c4c",
    "b": "#8b8b8b",
    "g": "#8fb9d9",
    "c": "#d7b46a",
}

HVAC_BASE_CATEGORIES = {"heating", "cooling", "ventilation", "pumping"}
COMMUNICATION_ORDER = ["thermal_conditioning", "ventilation", "pumping"]
THERMAL_PARAM_PREFERRED_ORDER = ["a", "b", "g", "c", "a_z", "b_z", "g_z", "c_z"]


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 500,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "lines.linewidth": 1.15,
    })


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    png_dir = base_dir / "figures" / "png"
    pdf_dir = base_dir / "figures" / "pdf"
    svg_dir = base_dir / "figures" / "svg"
    csv_dir = base_dir / "data_exports"
    for d in (png_dir, pdf_dir, svg_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"png": png_dir, "pdf": pdf_dir, "svg": svg_dir, "csv": csv_dir}


def save_figure(fig: plt.Figure, dirs: dict[str, Path], stem: str) -> None:
    png_path = dirs["png"] / f"{stem}.png"
    pdf_path = dirs["pdf"] / f"{stem}.pdf"
    svg_path = dirs["svg"] / f"{stem}.svg"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {svg_path}")
    plt.close(fig)


def save_csv(df: pd.DataFrame, dirs: dict[str, Path], name: str) -> None:
    out_path = dirs["csv"] / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


def english_month_name(month_num: int) -> str:
    return ENGLISH_MONTHS.get(int(month_num), f"Month {int(month_num)}")


def format_period_label(year: int | None, month: int | None, ts_df: pd.DataFrame | None = None) -> str:
    if month is not None and year is not None:
        return f"{english_month_name(month)} {year}"
    if month is not None and year is None:
        return english_month_name(month)
    if year is not None:
        return f"{year}"
    if ts_df is not None and not ts_df.empty and "timestamp" in ts_df.columns:
        start = pd.to_datetime(ts_df["timestamp"], errors="coerce").min()
        end = pd.to_datetime(ts_df["timestamp"], errors="coerce").max()
        if pd.notna(start) and pd.notna(end):
            return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    return "Analysed period"


def load_module_from_path(script_path: str | Path):
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Baseline script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("optimization_debug_fast_dynamic", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_datetime_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() < 0.80:
        dt = pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    return dt


def load_excel_inputs(excel_path: str | Path) -> dict[str, pd.DataFrame]:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {excel_path}")
    required_sheets = ["timeseries_main", "equipment_metadata", "equipment_hourly"]
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    missing = [s for s in required_sheets if s not in xls.sheet_names]
    if missing:
        raise KeyError(f"Missing required sheet(s) in workbook: {missing}")
    sheets = {
        "timeseries_main": pd.read_excel(excel_path, sheet_name="timeseries_main", engine="openpyxl"),
        "equipment_metadata": pd.read_excel(excel_path, sheet_name="equipment_metadata", engine="openpyxl"),
        "equipment_hourly": pd.read_excel(excel_path, sheet_name="equipment_hourly", engine="openpyxl"),
    }
    for _, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = parse_datetime_series(df["timestamp"])
    return sheets


def select_period(df: pd.DataFrame, timestamp_col: str, year: int | None = None, month: int | None = None) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = parse_datetime_series(out[timestamp_col])
    out = out.dropna(subset=[timestamp_col]).copy()
    if year is not None:
        out = out[out[timestamp_col].dt.year == int(year)]
    if month is not None:
        out = out[out[timestamp_col].dt.month == int(month)]
    return out.sort_values(timestamp_col).reset_index(drop=True)


def exact_zone_temperature_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^zone\d+$", re.IGNORECASE)
    zone_cols = [c for c in df.columns if bool(pattern.fullmatch(str(c).strip()))]
    return sorted(zone_cols, key=zone_numeric_id)


def standardize_hvac_category(cat: Any) -> str:
    if pd.isna(cat):
        return "unclassified"
    c = str(cat).strip().lower()
    if c in {"heating", "cooling"}:
        return "thermal_conditioning"
    if c in {"ventilation", "pumping"}:
        return c
    return c


def normalize_equipment_name(text: Any) -> str:
    return str(text).strip()


def infer_interval_hours(timestamps: pd.Series) -> pd.DataFrame:
    ts = pd.Series(parse_datetime_series(timestamps)).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if ts.empty:
        return pd.DataFrame(columns=["timestamp", "interval_h"])
    delta_h = ts.diff().dt.total_seconds().div(3600.0)
    typical = float(delta_h.dropna().median()) if delta_h.dropna().size else 1.0
    if not np.isfinite(typical) or typical <= 0:
        typical = 1.0
    next_delta = ts.shift(-1).sub(ts).dt.total_seconds().div(3600.0)
    interval_h = next_delta.fillna(typical)
    interval_h = interval_h.where(interval_h > 0, typical)
    return pd.DataFrame({"timestamp": ts, "interval_h": interval_h.astype(float)})


def build_equipment_long(hourly_df: pd.DataFrame, metadata_df: pd.DataFrame,
                         year: int | None = None, month: int | None = None) -> pd.DataFrame:
    meta = metadata_df.copy()
    meta.columns = [str(c).strip() for c in meta.columns]
    if "equipment_column" not in meta.columns:
        raise KeyError("equipment_metadata must contain 'equipment_column'.")
    meta["equipment_column"] = meta["equipment_column"].map(normalize_equipment_name)
    meta["category"] = meta.get("category", pd.Series(index=meta.index, dtype=object)).astype(str).str.strip().str.lower()
    if "zone" not in meta.columns:
        meta["zone"] = np.nan
    if "service_scope" not in meta.columns:
        meta["service_scope"] = np.nan

    raw = hourly_df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    if "timestamp" not in raw.columns:
        raise KeyError("equipment_hourly must contain a 'timestamp' column.")
    raw["timestamp"] = parse_datetime_series(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw = select_period(raw, "timestamp", year=year, month=month)

    long_cols = {"timestamp", "equipment_column", "power_W"}
    if long_cols.issubset(set(raw.columns)):
        long_df = raw[["timestamp", "equipment_column", "power_W"]].copy()
        long_df["equipment_column"] = long_df["equipment_column"].map(normalize_equipment_name)
    else:
        value_cols = [c for c in raw.columns if c != "timestamp"]
        long_df = raw.melt(id_vars=["timestamp"], value_vars=value_cols, var_name="equipment_column", value_name="power_W")
        long_df["equipment_column"] = long_df["equipment_column"].map(normalize_equipment_name)

    long_df["power_W"] = pd.to_numeric(long_df["power_W"], errors="coerce")
    long_df = long_df.dropna(subset=["power_W"]).copy()
    intervals = infer_interval_hours(long_df["timestamp"])
    long_df = long_df.merge(intervals, on="timestamp", how="left")
    long_df["interval_h"] = pd.to_numeric(long_df["interval_h"], errors="coerce").fillna(1.0)
    long_df["energy_kWh"] = long_df["power_W"] * long_df["interval_h"] / 1000.0
    long_df = long_df.merge(meta, on="equipment_column", how="left", suffixes=("", "_meta"))
    long_df["category_plot"] = long_df["category"].map(standardize_hvac_category)
    long_df["service_scope"] = long_df["service_scope"].fillna("unspecified").astype(str).str.strip().str.lower()
    long_df["zone"] = long_df["zone"].astype(str).replace({"nan": np.nan})
    long_df = long_df[long_df["category"].isin(HVAC_BASE_CATEGORIES)].copy()
    return long_df.sort_values(["timestamp", "equipment_column"]).reset_index(drop=True)


def build_energy_aggregations(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    category_energy = long_df.groupby("category_plot", as_index=False)["energy_kWh"].sum().rename(columns={"energy_kWh": "energy_kWh_period"})
    category_energy["sort_key"] = category_energy["category_plot"].map({k: i for i, k in enumerate(COMMUNICATION_ORDER)}).fillna(999)
    category_energy = category_energy.sort_values(["sort_key", "energy_kWh_period"], ascending=[True, False]).drop(columns="sort_key")
    equipment_energy = long_df.groupby(["equipment_column", "category_plot"], as_index=False)["energy_kWh"].sum().rename(columns={"energy_kWh": "energy_kWh_period"}).sort_values("energy_kWh_period", ascending=False).reset_index(drop=True)
    category_counts = long_df[["equipment_column", "category_plot", "service_scope"]].drop_duplicates().groupby(["category_plot", "service_scope"], as_index=False).size().rename(columns={"size": "equipment_count"}).sort_values(["category_plot", "service_scope"]).reset_index(drop=True)
    return category_energy, equipment_energy, category_counts


def run_baseline(script_path: str | Path, excel_path: str | Path, raw_path: str | Path | None,
                 year: int | None = None, month: int | None = None) -> dict[str, Any]:
    """Try to run the baseline script. If full-year mode is used, try month=None first, then fallback."""
    module = load_module_from_path(script_path)
    if hasattr(module, "EXPORT_OUTPUTS"):
        module.EXPORT_OUTPUTS = False
    if not hasattr(module, "run"):
        raise AttributeError(f"The script {script_path} does not contain run(...)")

    def _call(run_month, run_year):
        with contextlib.redirect_stdout(io.StringIO()):
            return module.run(excel_path=str(excel_path), scenario_name="baseline", month=run_month, year=run_year, raw_source_path=str(raw_path) if raw_path else None)

    try:
        result = _call(month, year)
    except TypeError:
        if month is None and year is not None:
            print("[warning] Baseline run did not accept month=None. Falling back to January of the selected year for coefficient extraction.")
            result = _call(1, year)
        else:
            raise
    return result if isinstance(result, dict) else {"result": result}


def build_zone_temperature_summary(ts_period: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone in exact_zone_temperature_columns(ts_period):
        s = pd.to_numeric(ts_period[zone], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "zone": zone,
            "min_C": float(s.min()),
            "p05_C": float(s.quantile(0.05)),
            "q25_C": float(s.quantile(0.25)),
            "median_C": float(s.quantile(0.50)),
            "q75_C": float(s.quantile(0.75)),
            "p95_C": float(s.quantile(0.95)),
            "max_C": float(s.max()),
            "mean_C": float(s.mean()),
            "std_C": float(s.std(ddof=1)) if s.size > 1 else np.nan,
            "count": int(s.size),
        })
    return pd.DataFrame(rows)


def build_market_price_summary(ts_period: pd.DataFrame) -> pd.DataFrame:
    if not {"timestamp", "price_eur_kWh"}.issubset(ts_period.columns):
        return pd.DataFrame()
    df = ts_period[["timestamp", "price_eur_kWh"]].copy()
    df["price_eur_kWh"] = pd.to_numeric(df["price_eur_kWh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price_eur_kWh"]).copy()
    if df.empty:
        return df
    df["day"] = df["timestamp"].dt.floor("D")
    daily = df.groupby("day", as_index=False)["price_eur_kWh"].agg(mean="mean", min="min", max="max").sort_values("day").reset_index(drop=True)
    daily["day_of_year"] = daily["day"].dt.dayofyear
    return daily


def zone_numeric_id(zone_name: str) -> int:
    m = re.search(r"(\d+)$", str(zone_name))
    return int(m.group(1)) if m else 10**9


def humanize_equipment_label(text: str, max_len: int = 62) -> str:
    t = re.sub(r"\s+", " ", str(text).strip())
    t = t.replace("[W](Hourly)", "").replace("Electric Power", "Power").replace("Electric Energy", "Energy")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def recursive_items(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from recursive_items(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from recursive_items(v)


def normalize_thermal_params_df(df: pd.DataFrame) -> pd.DataFrame:
    cand = df.copy()
    cand.columns = [str(c).strip() for c in cand.columns]
    lower_map = {str(c).strip().lower(): c for c in cand.columns}
    zone_col = None
    for zc in ["zone", "thermal_zone", "zone_name", "zone_id"]:
        if zc in lower_map:
            zone_col = lower_map[zc]
            break
    if zone_col is None:
        if cand.index.name is not None:
            cand = cand.reset_index().rename(columns={cand.index.name: "zone"})
            zone_col = "zone"
        else:
            return pd.DataFrame()
    out = pd.DataFrame()
    out["zone"] = cand[zone_col].astype(str)
    matched = 0
    for target in ["a", "b", "g", "c"]:
        found = None
        for ch in [target, f"{target}_z", f"coef_{target}", f"{target}z"]:
            if ch in lower_map:
                found = lower_map[ch]
                break
        if found is not None:
            out[target] = pd.to_numeric(cand[found], errors="coerce")
            matched += 1
    if matched == 0:
        return pd.DataFrame()
    out = out.dropna(how="all", subset=[c for c in ["a", "b", "g", "c"] if c in out.columns])
    out = out[out["zone"].str.contains(r"zone", case=False, regex=True)].copy()
    return out.sort_values(by="zone", key=lambda s: s.map(zone_numeric_id)).reset_index(drop=True) if not out.empty else pd.DataFrame()


def extract_thermal_parameters(baseline_results: dict[str, Any]) -> pd.DataFrame:
    best = pd.DataFrame()
    best_score = -1
    for item in recursive_items(baseline_results):
        if isinstance(item, pd.DataFrame):
            cand = normalize_thermal_params_df(item)
            score = cand.shape[0] * cand.shape[1] if not cand.empty else -1
            if score > best_score:
                best = cand
                best_score = score
        elif isinstance(item, dict):
            keys = {str(k).lower() for k in item.keys()}
            if {"a", "b", "g", "c"}.issubset(keys):
                try:
                    tmp = pd.DataFrame(item).reset_index().rename(columns={"index": "zone"})
                    cand = normalize_thermal_params_df(tmp)
                    score = cand.shape[0] * cand.shape[1] if not cand.empty else -1
                    if score > best_score:
                        best = cand
                        best_score = score
                except Exception:
                    pass
    return best


def add_bottom_note(fig: plt.Figure, text: str, y: float = 0.012, fontsize: float = 7.1) -> None:
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=fontsize, color=PALETTE["dark_gray"])


def safe_plot(plot_label: str, plot_func, *args, **kwargs) -> None:
    try:
        plot_func(*args, **kwargs)
        print(f"[ok] {plot_label}")
    except Exception as exc:
        print(f"[warning] Plot failed: {plot_label}: {exc}")


def plot_zone_temperature_distribution(summary_df: pd.DataFrame, dirs: dict[str, Path], period_label: str) -> None:
    if summary_df.empty:
        return
    df = summary_df.copy().sort_values(by="zone", key=lambda s: s.map(zone_numeric_id)).reset_index(drop=True)
    fig_h = max(4.8, 0.30 * len(df))
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        p05, q25, med, q75, p95, mean = row["p05_C"], row["q25_C"], row["median_C"], row["q75_C"], row["p95_C"], row["mean_C"]
        ax.hlines(i, p05, p95, color=PALETTE["mid_gray"], linewidth=1.0, zorder=1)
        ax.vlines([p05, p95], i - 0.16, i + 0.16, color=PALETTE["mid_gray"], linewidth=1.0, zorder=1)
        ax.add_patch(Rectangle((q25, i - 0.29), max(q75 - q25, 1e-9), 0.58, facecolor=PALETTE["zone_iqr"], edgecolor=PALETTE["dark_gray"], linewidth=0.8, alpha=0.97, zorder=2))
        ax.vlines(med, i - 0.29, i + 0.29, color=PALETTE["zone_median"], linewidth=1.4, zorder=3)
        ax.scatter(mean, i, marker="D", s=24, color=PALETTE["mean_marker"], edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["zone"])
    ax.invert_yaxis()
    ax.set_xlim(float(df["p05_C"].min()) - 0.5, float(df["p95_C"].max()) + 0.5)
    ax.set_xlabel(r"Indoor temperature [$^\circ$C]")
    ax.set_ylabel("Thermal zone [-]")
    ax.set_title(f"Indoor temperature distribution by thermal zone — {period_label}")
    ax.grid(axis="x", alpha=0.18, linestyle="--", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([0], [0], color=PALETTE["mid_gray"], lw=1.0, label="P05–P95 range"), Rectangle((0, 0), 1, 1, facecolor=PALETTE["zone_iqr"], edgecolor=PALETTE["dark_gray"], linewidth=0.8, label="Q25–Q75 (IQR)"), Line2D([0], [0], color=PALETTE["zone_median"], lw=1.4, label="Median"), Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["mean_marker"], markeredgecolor="white", markersize=5.8, label="Mean")], loc="lower right", ncol=2)
    add_bottom_note(fig, "Box = Q25–Q75 (interquartile range); whiskers = P05–P95 to reduce the visual influence of extreme hourly values; diamond = mean.")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, dirs, "01_indoor_temperature_distribution_by_thermal_zone")


def plot_market_price_variation(daily_df: pd.DataFrame, dirs: dict[str, Path], period_label: str) -> None:
    if daily_df.empty:
        return
    plot_df = daily_df.copy().sort_values("day")
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.fill_between(plot_df["day"], plot_df["min"], plot_df["max"], color=PALETTE["light_gray"], alpha=0.85, linewidth=0.0, label="Daily min–max band")
    ax.plot(plot_df["day"], plot_df["mean"], color=PALETTE["price"], linewidth=1.0, label="Daily mean price")
    ax.set_title(f"Market electricity price across the analysed period — {period_label}")
    ax.set_ylabel(r"Electricity price [EUR/kWh]")
    ax.set_xlabel("Date [-]")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, dirs, "02_market_electricity_price_across_the_period")


def plot_equipment_grouping(category_counts: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if category_counts.empty:
        return
    df = category_counts.copy()
    order = df.groupby("category_plot", as_index=False)["equipment_count"].sum().sort_values("equipment_count", ascending=True)["category_plot"].tolist()
    scope_pivot = df.pivot_table(index="category_plot", columns="service_scope", values="equipment_count", aggfunc="sum", fill_value=0).reindex(order)
    fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.5 * len(scope_pivot))))
    left = np.zeros(len(scope_pivot))
    for i, scope in enumerate(scope_pivot.columns):
        vals = scope_pivot[scope].values.astype(float)
        ax.barh(scope_pivot.index, vals, left=left, height=0.58, color=[PALETTE["light_gray"], "#d8d8d8", "#c7c7c7", "#b5b5b5"][i % 4], edgecolor="white", linewidth=0.5, label=str(scope).replace("_", " "))
        left += vals
    ax.set_xlabel("Equipment count [-]")
    ax.set_ylabel("HVAC category [-]")
    ax.set_title("HVAC equipment grouping by service scope")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, dirs, "03_hvac_equipment_grouping_by_service_scope")


def plot_aggregated_hvac_energy(category_energy: pd.DataFrame, dirs: dict[str, Path], period_label: str) -> None:
    if category_energy.empty:
        return
    df = category_energy.copy().sort_values("energy_kWh_period", ascending=True).reset_index(drop=True)
    fig_h = max(2.8, 0.85 + 0.58 * len(df))
    fig, ax = plt.subplots(figsize=(7.9, fig_h))
    bars = ax.barh(df["category_plot"], df["energy_kWh_period"], height=0.54, color=[PALETTE.get(c, PALETTE["mid_gray"]) for c in df["category_plot"]], edgecolor="white", linewidth=0.65, zorder=3)
    ax.set_xlabel(r"Energy across analysed period [kWh]")
    ax.set_ylabel("HVAC category [-]")
    ax.set_title(f"Aggregated HVAC energy consumption by category — {period_label}")
    ax.grid(axis="x", alpha=0.18, linestyle="--", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    xmax = float(df["energy_kWh_period"].max()) if not df.empty else 1.0
    ax.set_xlim(0, xmax * 1.14)
    for bar, value in zip(bars, df["energy_kWh_period"]):
        ax.text(value + xmax * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center", ha="left", fontsize=7.35, color=PALETTE["dark_gray"])
    fig.tight_layout()
    save_figure(fig, dirs, "04_aggregated_hvac_energy_by_category")


def plot_hvac_equipment_top_consumers(equipment_energy: pd.DataFrame, dirs: dict[str, Path], period_label: str, top_n: int = 15) -> None:
    if equipment_energy.empty:
        return
    df = equipment_energy.head(int(top_n)).copy().sort_values("energy_kWh_period", ascending=True)
    fig, ax = plt.subplots(figsize=(9.8, max(4.8, 0.38 * len(df))))
    ax.barh([humanize_equipment_label(x) for x in df["equipment_column"]], df["energy_kWh_period"], height=0.56, color=[PALETTE.get(c, PALETTE["mid_gray"]) for c in df["category_plot"]], edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"Energy across analysed period [kWh]")
    ax.set_ylabel("HVAC equipment [-]")
    ax.set_title(f"HVAC equipment consumption — top {min(top_n, len(df))} consumers — {period_label}")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    xmax = float(df["energy_kWh_period"].max()) if not df.empty else 1.0
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2.0
        ax.text(width + xmax * 0.015, y, f"{width:,.0f}", va="center", ha="left", fontsize=7.15, color=PALETTE["dark_gray"])
    fig.tight_layout()
    save_figure(fig, dirs, "05_hvac_equipment_consumption_top_consumers")


def plot_hvac_equipment_ranked_all(equipment_energy: pd.DataFrame, dirs: dict[str, Path], period_label: str) -> None:
    if equipment_energy.empty:
        return
    df = equipment_energy.copy().sort_values("energy_kWh_period", ascending=True)
    fig_h = max(8.5, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.barh([humanize_equipment_label(x, max_len=70) for x in df["equipment_column"]], df["energy_kWh_period"], height=0.52, color=[PALETTE.get(c, PALETTE["mid_gray"]) for c in df["category_plot"]], edgecolor="white", linewidth=0.35)
    ax.set_xlabel(r"Energy across analysed period [kWh]")
    ax.set_ylabel("HVAC equipment [-]")
    ax.set_title(f"HVAC equipment consumption — ranked profile — {period_label}")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, dirs, "06_hvac_equipment_consumption_ranked_all")


def pick_thermal_parameter_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if c != "zone" and pd.api.types.is_numeric_dtype(df[c])]
    ordered = [c for c in THERMAL_PARAM_PREFERRED_ORDER if c in numeric_cols]
    return ordered if ordered else numeric_cols


def coefficient_stats_text(s: pd.Series) -> str:
    vals = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if vals.empty:
        return "n = 0"
    return (
        f"n = {len(vals)}\n"
        f"mean = {vals.mean():.4f}\n"
        f"median = {vals.median():.4f}\n"
        f"Q25–Q75 = [{vals.quantile(0.25):.4f}, {vals.quantile(0.75):.4f}]"
    )


def _style_violin_parts(parts, color: str) -> None:
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.35)
        body.set_linewidth(0.6)


def plot_single_coefficient_vbs(ax, values: pd.Series, color: str, x_label: str, title: str) -> None:
    s = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if s.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, color=PALETTE["dark_gray"])
        ax.set_title(title, loc="left")
        ax.set_yticks([])
        return

    p05 = float(s.quantile(0.05))
    q25 = float(s.quantile(0.25))
    med = float(s.quantile(0.50))
    q75 = float(s.quantile(0.75))
    p95 = float(s.quantile(0.95))
    mean = float(s.mean())
    xmin = float(s.min())
    xmax = float(s.max())
    span = max(xmax - xmin, 1e-9)

    # Violin = density shape
    parts = ax.violinplot([s.values], positions=[0], vert=False, widths=0.62,
                          showmeans=False, showmedians=False, showextrema=False)
    _style_violin_parts(parts, color)

    # Box = IQR
    ax.add_patch(Rectangle((q25, -0.10), max(q75 - q25, 1e-9), 0.20,
                           facecolor=PALETTE["zone_iqr"], edgecolor=PALETTE["dark_gray"],
                           linewidth=0.8, zorder=3))

    # Whiskers = P05-P95
    ax.hlines(0, p05, p95, color=PALETTE["mid_gray"], linewidth=1.0, zorder=4)
    ax.vlines([p05, p95], -0.07, 0.07, color=PALETTE["mid_gray"], linewidth=1.0, zorder=4)

    # Median and mean
    ax.vlines(med, -0.10, 0.10, color=PALETTE["zone_median"], linewidth=1.4, zorder=5)
    ax.scatter(mean, 0, marker="D", s=28, color=PALETTE["mean_marker"], edgecolor="white", linewidth=0.45, zorder=6)

    # Individual zone estimates
    rng = np.random.default_rng(42)
    jitter = rng.normal(0.0, 0.028, len(s))
    ax.scatter(s.values, jitter, s=16, color=color, alpha=0.70, edgecolor="white", linewidth=0.30, zorder=7)

    if xmin < 0 < xmax:
        ax.axvline(0, color=PALETTE["dark_gray"], linewidth=0.75, alpha=0.60, zorder=1)

    ax.set_xlim(xmin - 0.08 * span, xmax + 0.14 * span)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel(x_label)
    ax.set_title(title, loc="left")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.18, linestyle="--", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    ax.text(0.985, 0.92, coefficient_stats_text(s), ha="right", va="top", transform=ax.transAxes,
            fontsize=7.1, color=PALETTE["dark_gray"],
            bbox=dict(facecolor="white", edgecolor=PALETTE["light_gray"], boxstyle="round,pad=0.25", alpha=0.95))


def plot_thermal_coefficient_diagnostics(params_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if params_df.empty:
        return

    df = params_df.copy()
    if "zone" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "zone"})
    df = df.sort_values(by="zone", key=lambda s: s.map(zone_numeric_id)).reset_index(drop=True)

    available = pick_thermal_parameter_columns(df)
    if not available:
        return

    if {"a", "b", "g"}.issubset(set(available)):
        cols_to_plot = ["a", "b", "g"]
    elif {"a_z", "b_z", "g_z"}.issubset(set(available)):
        cols_to_plot = ["a_z", "b_z", "g_z"]
    else:
        cols_to_plot = available[:3]

    title_map = {
        "a": r"Cross-zone summary of thermal persistence coefficient $a_z$",
        "a_z": r"Cross-zone summary of thermal persistence coefficient $a_z$",
        "b": r"Cross-zone summary of outdoor-temperature coefficient $b_z$",
        "b_z": r"Cross-zone summary of outdoor-temperature coefficient $b_z$",
        "g": r"Cross-zone summary of HVAC-effect coefficient $g_z$",
        "g_z": r"Cross-zone summary of HVAC-effect coefficient $g_z$",
        "c": r"Cross-zone summary of additive term $c_z$",
        "c_z": r"Cross-zone summary of additive term $c_z$",
    }
    xlabel_map = {
        "a": r"$a_z$ [-]", "a_z": r"$a_z$ [-]",
        "b": r"$b_z$ [-]", "b_z": r"$b_z$ [-]",
        "g": r"$g_z$ [$\Delta ^\circ$C per unit of $u_z$]", "g_z": r"$g_z$ [$\Delta ^\circ$C per unit of $u_z$]",
        "c": r"$c_z$ [$^\circ$C]", "c_z": r"$c_z$ [$^\circ$C]",
    }
    color_map = {
        "a": PALETTE["a"], "a_z": PALETTE["a"],
        "b": PALETTE["b"], "b_z": PALETTE["b"],
        "g": PALETTE["g"], "g_z": PALETTE["g"],
        "c": PALETTE["c"], "c_z": PALETTE["c"],
    }

    fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=1, figsize=(10.7, 2.55 * len(cols_to_plot) + 0.90))
    if len(cols_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_to_plot):
        plot_single_coefficient_vbs(
            ax=ax,
            values=df[col],
            color=color_map.get(col, PALETTE["mid_gray"]),
            x_label=xlabel_map.get(col, f"{col} [-]"),
            title=title_map.get(col, f"Summary of coefficient {col}"),
        )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=PALETTE["light_gray"], edgecolor="white", alpha=0.35, label="Density shape (violin)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["mid_gray"], markeredgecolor="white", markersize=5.0, label="Individual zone estimate"),
        Line2D([0], [0], color=PALETTE["mid_gray"], lw=1.0, label="P05–P95 range"),
        Rectangle((0, 0), 1, 1, facecolor=PALETTE["zone_iqr"], edgecolor=PALETTE["dark_gray"], linewidth=0.8, label="Q25–Q75 (IQR)"),
        Line2D([0], [0], color=PALETTE["zone_median"], lw=1.4, label="Median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["mean_marker"], markeredgecolor="white", markersize=5.3, label="Mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.025), frameon=False)

    add_bottom_note(
        fig,
        "Integrated violin-box-scatter summary across zones. a_z and b_z are dimensionless, c_z is an additive temperature term, and g_z expresses the temperature response per unit of the identified effective HVAC input u_z.",
        y=0.003,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, dirs, "07_thermal_coefficient_diagnostics")


def plot_normalized_thermal_parameter_patterns(params_df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    if params_df.empty:
        return
    df = params_df.copy()
    if "zone" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "zone"})
    df = df.sort_values(by="zone", key=lambda s: s.map(zone_numeric_id)).reset_index(drop=True)
    ordered = pick_thermal_parameter_columns(df)
    if not ordered:
        return
    mat = df[ordered].astype(float).copy()
    mat_norm = mat.apply(lambda s: pd.Series(np.zeros(len(s)), index=s.index) if (pd.isna(s.std(ddof=0)) or s.std(ddof=0) == 0) else (s - s.mean()) / s.std(ddof=0), axis=0).clip(-2.5, 2.5)
    fig_h = max(4.8, 0.28 * len(df))
    fig, ax = plt.subplots(figsize=(7.1, fig_h))
    im = ax.imshow(mat_norm.values, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5))
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels([r"$a_z$" if c in {"a", "a_z"} else r"$b_z$" if c in {"b", "b_z"} else r"$g_z$" if c in {"g", "g_z"} else r"$c_z$" if c in {"c", "c_z"} else str(c) for c in ordered])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["zone"].astype(str).tolist())
    ax.set_xlabel("Thermal parameter [-]")
    ax.set_ylabel("Thermal zone [-]")
    ax.set_title("Normalized thermal-parameter patterns by zone")
    ax.set_xticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Standardized value [z-score]")
    add_bottom_note(fig, "Column-wise z-score normalization: negative = below the cross-zone mean, positive = above the cross-zone mean.")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, dirs, "08_normalized_thermal_parameter_patterns_by_zone")


def export_audit_tables(dirs: dict[str, Path], zone_summary: pd.DataFrame, daily_price: pd.DataFrame, long_df: pd.DataFrame, category_energy: pd.DataFrame, equipment_energy: pd.DataFrame, category_counts: pd.DataFrame, thermal_params: pd.DataFrame) -> None:
    if not zone_summary.empty:
        save_csv(zone_summary, dirs, "audit_zone_temperature_summary.csv")
    if not daily_price.empty:
        save_csv(daily_price, dirs, "audit_market_price_daily_summary.csv")
    if not long_df.empty:
        save_csv(long_df, dirs, "audit_equipment_hourly_long_with_energy.csv")
    if not category_energy.empty:
        save_csv(category_energy, dirs, "audit_hvac_category_energy.csv")
    if not equipment_energy.empty:
        save_csv(equipment_energy, dirs, "audit_hvac_equipment_energy.csv")
    if not category_counts.empty:
        save_csv(category_counts, dirs, "audit_equipment_grouping_counts.csv")
    if not thermal_params.empty:
        save_csv(thermal_params, dirs, "audit_thermal_parameters_by_zone.csv")


def print_console_report(period_label: str, ts_period: pd.DataFrame, zone_summary: pd.DataFrame, category_energy: pd.DataFrame, equipment_energy: pd.DataFrame, thermal_params: pd.DataFrame, out_dir: Path, dirs: dict[str, Path]) -> None:
    print("\n" + "=" * 92)
    print("SCIENTIFIC BASELINE FIGURES — SUMMARY REPORT")
    print("=" * 92)
    print(f"Analysed period      : {period_label}")
    print(f"Snapshots            : {len(ts_period):,}")
    print(f"Thermal zones        : {len(zone_summary):,}")
    if not category_energy.empty:
        print(f"HVAC energy period   : {category_energy['energy_kWh_period'].sum():,.2f} kWh")
    if not equipment_energy.empty:
        top = equipment_energy.iloc[0]
        print(f"Top HVAC equipment   : {top['equipment_column']} ({top['energy_kWh_period']:,.2f} kWh)")
    if not thermal_params.empty:
        vals = []
        for c in ["a", "b", "g", "c", "a_z", "b_z", "g_z", "c_z"]:
            if c in thermal_params.columns:
                vals.append(f"{c}={pd.to_numeric(thermal_params[c], errors='coerce').mean():.4f}")
        if vals:
            print("Average thermal pars : " + ", ".join(vals))
    print(f"Output root          : {out_dir.resolve()}")
    print(f"PNG directory        : {dirs['png'].resolve()}")
    print(f"PDF directory        : {dirs['pdf'].resolve()}")
    print(f"SVG directory        : {dirs['svg'].resolve()}")
    print(f"CSV directory        : {dirs['csv'].resolve()}")
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready scientific baseline figures. By default, all available data in the selected year are used.")
    parser.add_argument("--script", default="optimization_debug_fast.py", help="Path to optimization_debug_fast.py")
    parser.add_argument("--excel", default="data.xlsx", help="Path to data.xlsx")
    parser.add_argument("--raw", default=None, help="Optional path to raw support workbook")
    parser.add_argument("--year", type=int, default=2022, help="Year to analyse. Default: 2022")
    parser.add_argument("--month", type=int, default=None, help="Optional month to analyse (1-12). If omitted, the full available year is analysed.")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory (figures are saved inside figures/png, figures/pdf, figures/svg)")
    parser.add_argument("--top-n-equipment", type=int, default=15, help="Top N HVAC equipment items in the top-consumption chart")
    args = parser.parse_args()

    configure_matplotlib()
    dirs = ensure_dirs(Path(args.output_dir))
    sheets = load_excel_inputs(args.excel)
    ts_period = select_period(sheets["timeseries_main"], "timestamp", year=args.year, month=args.month)
    if ts_period.empty:
        raise ValueError("The selected year/month produced an empty timeseries subset.")
    period_label = format_period_label(args.year, args.month, ts_df=ts_period)

    zone_summary = build_zone_temperature_summary(ts_period)
    daily_price = build_market_price_summary(ts_period)
    equipment_long = build_equipment_long(sheets["equipment_hourly"], sheets["equipment_metadata"], year=args.year, month=args.month)
    category_energy, equipment_energy, category_counts = build_energy_aggregations(equipment_long)

    thermal_params = pd.DataFrame()
    try:
        baseline_results = run_baseline(args.script, args.excel, args.raw, year=args.year, month=args.month)
        thermal_params = extract_thermal_parameters(baseline_results)
        if thermal_params.empty:
            print("[warning] No thermal-parameter table could be extracted from the baseline script output.")
    except Exception as exc:
        print(f"[warning] Baseline script execution failed or thermal parameters could not be extracted: {exc}")

    safe_plot("01_indoor_temperature_distribution_by_thermal_zone", plot_zone_temperature_distribution, zone_summary, dirs, period_label)
    safe_plot("02_market_electricity_price_across_the_period", plot_market_price_variation, daily_price, dirs, period_label)
    safe_plot("03_hvac_equipment_grouping_by_service_scope", plot_equipment_grouping, category_counts, dirs)
    safe_plot("04_aggregated_hvac_energy_by_category", plot_aggregated_hvac_energy, category_energy, dirs, period_label)
    safe_plot("05_hvac_equipment_consumption_top_consumers", plot_hvac_equipment_top_consumers, equipment_energy, dirs, period_label, top_n=args.top_n_equipment)
    safe_plot("06_hvac_equipment_consumption_ranked_all", plot_hvac_equipment_ranked_all, equipment_energy, dirs, period_label)
    safe_plot("07_thermal_coefficient_diagnostics", plot_thermal_coefficient_diagnostics, thermal_params, dirs)
    safe_plot("08_normalized_thermal_parameter_patterns_by_zone", plot_normalized_thermal_parameter_patterns, thermal_params, dirs)

    export_audit_tables(dirs, zone_summary, daily_price, equipment_long, category_energy, equipment_energy, category_counts, thermal_params)
    print_console_report(period_label, ts_period, zone_summary, category_energy, equipment_energy, thermal_params, Path(args.output_dir), dirs)


if __name__ == "__main__":
    main()
