from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class UnitConfig:
    """
    Configuration for the units used in the input files.

    power_unit:
        Unit of power in the input data.
        Supported values: "kW", "MW"

    price_unit:
        Unit of price in the input data.
        Supported values: "EUR/kWh", "EUR/MWh"
    """
    power_unit: str = "kW"
    price_unit: str = "EUR/kWh"


@dataclass(frozen=True)
class Baseline:
    """
    Container for the baseline outputs.

    Attributes
    ----------
    load_by_group:
        Time series of power by group/category [kW]

    price:
        Electricity price time series [EUR/kWh]

    cost_rate_by_group:
        Instantaneous cost rate by group [EUR/h]

    timestep_hours:
        Time resolution of the series [h]
    """
    load_by_group: pd.DataFrame
    price: pd.Series
    cost_rate_by_group: pd.DataFrame
    timestep_hours: float


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    """
    Infer the most common timestep from a DatetimeIndex.
    """
    if len(index) < 2:
        return 1.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 1.0

    dt = deltas.mode().iloc[0]
    return dt.total_seconds() / 3600.0


def convert_units(
    loads: pd.DataFrame,
    price: pd.Series,
    units: UnitConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert input units into the internal standard system:

    - Load  -> kW
    - Price -> EUR/kWh
    """
    loads = loads.copy()
    price = price.copy()

    # Power conversion
    if units.power_unit == "MW":
        loads = loads * 1000.0
    elif units.power_unit != "kW":
        raise ValueError(
            f"Unsupported power unit: {units.power_unit}. Use 'kW' or 'MW'."
        )

    # Price conversion
    if units.price_unit == "EUR/MWh":
        price = price / 1000.0
    elif units.price_unit != "EUR/kWh":
        raise ValueError(
            f"Unsupported price unit: {units.price_unit}. Use 'EUR/kWh' or 'EUR/MWh'."
        )

    avg_price = float(price.mean())
    if avg_price < 0.0001 or avg_price > 5:
        raise ValueError(
            "Price values look unrealistic after conversion. "
            "Please verify the selected price unit."
        )

    return loads, price


def aggregate_loads_by_category(
    loads_meta: pd.DataFrame,
    loads_p_set: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate load time series by category if the metadata includes a 'category' column.
    If no 'category' column exists, return the original load table.
    """
    if "category" not in loads_meta.columns:
        return loads_p_set.copy()

    if "name" not in loads_meta.columns:
        raise KeyError("Column 'name' not found in loads metadata.")

    group_map = loads_meta.set_index("name")["category"].to_dict()

    grouped_columns: Dict[str, list] = {}
    for col in loads_p_set.columns:
        group = group_map.get(col, col)
        grouped_columns.setdefault(str(group), []).append(col)

    aggregated = pd.DataFrame(
        {
            group: loads_p_set[cols].sum(axis=1)
            for group, cols in grouped_columns.items()
        },
        index=loads_p_set.index,
    )

    return aggregated


def compute_baseline(
    data: Dict[str, pd.DataFrame],
    grid_price_col: str = "Grid",
    units: UnitConfig = UnitConfig(),
) -> Baseline:
    """
    Compute the baseline building consumption metrics.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary with:
        - "loads"
        - "loads_p_set"
        - "gen_cost"

    grid_price_col : str
        Name of the column in gen_cost containing the grid electricity price

    units : UnitConfig
        Input unit configuration

    Returns
    -------
    Baseline
        Baseline outputs
    """
    required_keys = {"loads", "loads_p_set", "gen_cost"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"Missing input datasets: {sorted(missing_keys)}")

    loads_meta = data["loads"].copy()
    loads_p_set = data["loads_p_set"].copy()
    gen_cost = data["gen_cost"].copy()

    loads_p_set = loads_p_set.sort_index()
    gen_cost = gen_cost.sort_index()

    if not isinstance(loads_p_set.index, pd.DatetimeIndex):
        raise TypeError("loads_p_set index must be a DatetimeIndex.")

    if not isinstance(gen_cost.index, pd.DatetimeIndex):
        raise TypeError("gen_cost index must be a DatetimeIndex.")

    if grid_price_col not in gen_cost.columns:
        raise KeyError(
            f"Price column '{grid_price_col}' not found in gen_cost. "
            f"Available columns: {list(gen_cost.columns)}"
        )

    price = gen_cost[grid_price_col].copy()

    # Aggregate loads by category if metadata provides categories
    loads = aggregate_loads_by_category(loads_meta, loads_p_set)

    # Convert to internal standard units
    loads, price = convert_units(loads, price, units)

    # Align price with loads index
    price = price.reindex(loads.index)

    # Fill potential small gaps
    if price.isna().any():
        price = price.interpolate(method="time").ffill().bfill()

    if price.isna().any():
        raise ValueError(
            "Price series could not be aligned to the load timestamps."
        )

    timestep_hours = infer_timestep_hours(loads.index)

    # Cost rate [EUR/h] = load [kW] * price [EUR/kWh]
    cost_rate = loads.mul(price, axis=0)

    return Baseline(
        load_by_group=loads,
        price=price,
        cost_rate_by_group=cost_rate,
        timestep_hours=timestep_hours,
    )