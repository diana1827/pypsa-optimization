from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


# -----------------------
# Configuration classes
# -----------------------

@dataclass(frozen=True)
class UnitConfig:
    """
    Configuration for input units.

    power_unit:
        Unit of power in the input data (kW or MW)

    price_unit:
        Unit of price in the input data (EUR/kWh or EUR/MWh)
    """
    power_unit: str = "kW"
    price_unit: str = "EUR/kWh"


@dataclass(frozen=True)
class Baseline:
    """
    Container for all computed baseline outputs.
    """
    load_by_group: pd.DataFrame        # Power per group [kW]
    price: pd.Series                  # Electricity price [EUR/kWh]
    cost_rate_by_group: pd.DataFrame  # Instantaneous cost rate [EUR/h]
    timestep_hours: float             # Time resolution in hours


# -----------------------
# Helper functions
# -----------------------

def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    """
    Infers the time step duration in hours from a DatetimeIndex.

    Logic:
    - Computes differences between consecutive timestamps
    - Extracts the most common difference (mode)
    - Converts it to hours
    """
    if len(index) < 2:
        return 1.0

    dt = index.to_series().diff().mode().iloc[0]
    return dt.total_seconds() / 3600.0


def convert_units(
    loads: pd.DataFrame,
    price: pd.Series,
    units: UnitConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert input units into:
    - Power: kW
    - Price: EUR/kWh
    """

    # Convert MW - kW
    if units.power_unit == "MW":
        loads = loads * 1000.0

    # Convert EUR/MWh - EUR/kWh
    if units.price_unit == "EUR/MWh":
        price = price / 1000.0

    # Basic sanity check to detect wrong unit selection
    avg_price = price.mean()
    if avg_price < 0.005 or avg_price > 1:
        raise ValueError(
            "Price values look unrealistic. Please verify the selected unit."
        )

    return loads, price


# -----------------------
# Main computation
# -----------------------

def compute_baseline(
    data: Dict[str, pd.DataFrame],
    grid_price_col: str = "Grid",
    units: UnitConfig = UnitConfig(),
) -> Baseline:
    """
    Computes the baseline energy and cost metrics.

    Steps:
    1. Extract relevant datasets
    2. Group loads (if categories exist)
    3. Convert units to standard system
    4. Infer timestep
    5. Compute cost rate
    """

    loads_meta = data["loads"]
    loads_p_set = data["loads_p_set"]
    gen_cost = data["gen_cost"]

    # Extract price time series
    price = gen_cost[grid_price_col]

    # -----------------------
    # Group loads by category
    # -----------------------
    if "category" in loads_meta.columns:
        # Map load name → category
        group_map = loads_meta.set_index("name")["category"].to_dict()

        grouped_columns = {}

        for column in loads_p_set.columns:
            group = group_map.get(column, column)
            grouped_columns.setdefault(group, []).append(column)

        # Aggregate loads per group
        loads = pd.DataFrame({
            group: loads_p_set[columns].sum(axis=1)
            for group, columns in grouped_columns.items()
        })

    else:
        # No grouping → use raw loads
        loads = loads_p_set.copy()

    # -----------------------
    # Unit normalization
    # -----------------------
    loads, price = convert_units(loads, price, units)

    # -----------------------
    # Time resolution
    # -----------------------
    timestep_hours = infer_timestep_hours(loads.index)

    # -----------------------
    # Cost calculation (rate)
    # -----------------------
    cost_rate = loads.mul(price, axis=0)

    return Baseline(
        load_by_group=loads,
        price=price,
        cost_rate_by_group=cost_rate,
        timestep_hours=timestep_hours,
    )