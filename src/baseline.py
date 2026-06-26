from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class UnitConfig:
    """
    Define as unidades dos dados de entrada.

    power_unit:
        Unidade da potência (aceita "kW" ou "MW")

    price_unit:
        Unidade do preço (aceita "EUR/kWh" ou "EUR/MWh")
    """
    power_unit: str = "kW"
    price_unit: str = "EUR/kWh"


@dataclass(frozen=True)
class Baseline:
    """
    Estrutura que guarda os resultados do cálculo base (baseline).

    load_by_group:
        Série temporal de potência agregada por grupo [kW]

    price:
        Série temporal do preço de eletricidade [EUR/kWh]

    cost_rate_by_group:
        Custo instantâneo por grupo [EUR/h]

    timestep_hours:
        Intervalo temporal entre amostras [h]
    """
    load_by_group: pd.DataFrame
    price: pd.Series
    cost_rate_by_group: pd.DataFrame
    timestep_hours: float


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    """
    Determina o passo temporal mais comum de uma série temporal.
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
    Converte os dados para o sistema interno:

    - Potência → kW
    - Preço   → EUR/kWh
    """
    loads = loads.copy()
    price = price.copy()

    # Conversão de potência
    if units.power_unit == "MW":
        loads *= 1000.0
    elif units.power_unit != "kW":
        raise ValueError(
            f"Unidade de potência inválida: {units.power_unit}. Use 'kW' ou 'MW'."
        )

    # Conversão de preço
    if units.price_unit == "EUR/MWh":
        price /= 1000.0
    elif units.price_unit != "EUR/kWh":
        raise ValueError(
            f"Unidade de preço inválida: {units.price_unit}. Use 'EUR/kWh' ou 'EUR/MWh'."
        )

    # Validação simples para evitar erros grosseiros de unidade
    avg_price = float(price.mean())
    if avg_price < 0.0001 or avg_price > 5:
        raise ValueError(
            "Os valores de preço parecem incorretos após a conversão. "
            "Verifique as unidades escolhidas."
        )

    return loads, price


def aggregate_loads_by_category(
    loads_meta: pd.DataFrame,
    loads_p_set: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrupa as cargas por categoria (se existir essa informação no metadata).

    Caso não exista coluna 'category', devolve os dados sem alterações.
    """
    if "category" not in loads_meta.columns:
        return loads_p_set.copy()

    if "name" not in loads_meta.columns:
        raise KeyError("A coluna 'name' não existe no metadata das cargas.")

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
    Calcula o baseline de consumo e custos.

    data deve conter:
        - "loads"        (metadata das cargas)
        - "loads_p_set"  (séries temporais de potência)
        - "gen_cost"     (dados de preços)

    grid_price_col:
        Nome da coluna com o preço da eletricidade

    units:
        Configuração das unidades dos dados de entrada
    """
    required_keys = {"loads", "loads_p_set", "gen_cost"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"Ficheiros em falta: {sorted(missing_keys)}")

    loads_meta = data["loads"].copy()
    loads_p_set = data["loads_p_set"].copy()
    gen_cost = data["gen_cost"].copy()

    loads_p_set = loads_p_set.sort_index()
    gen_cost = gen_cost.sort_index()

    if not isinstance(loads_p_set.index, pd.DatetimeIndex):
        raise TypeError("O índice de loads_p_set tem de ser DatetimeIndex.")

    if not isinstance(gen_cost.index, pd.DatetimeIndex):
        raise TypeError("O índice de gen_cost tem de ser DatetimeIndex.")

    if grid_price_col not in gen_cost.columns:
        raise KeyError(
            f"Coluna de preço '{grid_price_col}' não encontrada. "
            f"Disponíveis: {list(gen_cost.columns)}"
        )

    price = gen_cost[grid_price_col].copy()

    # Agrupamento das cargas (se aplicável)
    loads = aggregate_loads_by_category(loads_meta, loads_p_set)

    # Conversão de unidades
    loads, price = convert_units(loads, price, units)

    # Alinhar índices
    price = price.reindex(loads.index)

    # Preencher pequenos intervalos em falta
    if price.isna().any():
        price = price.interpolate(method="time").ffill().bfill()

    if price.isna().any():
        raise ValueError("Não foi possível alinhar o preço com a série temporal de carga.")

    timestep_hours = infer_timestep_hours(loads.index)

    # Cálculo do custo: potência × preço
    cost_rate = loads.mul(price, axis=0)

    return Baseline(
        load_by_group=loads,
        price=price,
        cost_rate_by_group=cost_rate,
        timestep_hours=timestep_hours,
    )