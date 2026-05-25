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
        "kW" or "MW"

    price_unit:
        "EUR/kWh" or "EUR/MWh"
    """
    power_unit: str = "kW"
    price_unit: str = "EUR/kWh"


@dataclass(frozen=True)
class Baseline:
    """
    Container for all computed baseline outputs.
    """
    load_by_group: pd.DataFrame        # [kW]
    price: pd.Series                  # [EUR/kWh]
    cost_rate_by_group: pd.DataFrame  # [EUR/h]
    timestep_hours: float


# -----------------------
# Helpers
# -----------------------

def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0

    dt = index.to_series().diff().mode().iloc[0]
    return dt.total_seconds() / 3600.0


def convert_units(
    loads: pd.DataFrame,
    price: pd.Series,
    units: UnitConfig,
) -> tuple[pd.DataFrame, pd.Series]:

    if units.power_unit == "MW":
        loads = loads * 1000.0

    if units.price_unit == "EUR/MWh":
        price = price / 1000.0

    avg_price = price.mean()
    if avg_price < 0.005 or avg_price > 1:
        raise ValueError(
            "Valores de preço parecem irreais — verifica a unidade selecionada."
        )

    return loads, price


# -----------------------
# Main function
# -----------------------

def compute_baseline(
    data: Dict[str, pd.DataFrame],
    grid_price_col: str = "Grid",
    units: UnitConfig = UnitConfig(),
) -> Baseline:

    loads_meta = data["loads"]
    loads_p_set = data["loads_p_set"]
    gen_cost = data["gen_cost"]

    # -----------------------
    # Price validation
    # -----------------------
    if grid_price_col not in gen_cost.columns:
        raise KeyError(
            f"Coluna '{grid_price_col}' não existe. "
            f"Disponíveis: {list(gen_cost.columns)}"
        )

    price = gen_cost[grid_price_col]

    # -----------------------
    # Group loads
    # -----------------------
    if "category" in loads_meta.columns:

        group_map = loads_meta.set_index("name")["category"].to_dict()
        grouped_columns = {}

        for col in loads_p_set.columns:
            group = group_map.get(col, col)
            grouped_columns.setdefault(group, []).append(col)

        loads = pd.DataFrame({
            g: loads_p_set[cols].sum(axis=1)
            for g, cols in grouped_columns.items()
        })

    else:
        loads = loads_p_set.copy()

    # -----------------------
    # Unit conversion
    # -----------------------
    loads, price = convert_units(loads, price, units)

    # -----------------------
    # Time resolution
    # -----------------------
    timestep_hours = infer_timestep_hours(loads.index)

    # -----------------------
    # Cost rate
    # -----------------------
    cost_rate = loads.mul(price, axis=0)

    return Baseline(
        load_by_group=loads,
        price=price,
        cost_rate_by_group=cost_rate,
        timestep_hours=timestep_hours,
    )