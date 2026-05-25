from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.data_io import load_baseline_data
    from src.baseline import UnitConfig, compute_baseline
except Exception:
    from data_io import load_baseline_data
    from baseline import UnitConfig, compute_baseline


# -----------------------
# Configuração página
# -----------------------

st.set_page_config(layout="wide")
st.title("Dashboard de Consumo Energético")


# -----------------------
# Sidebar
# -----------------------

st.sidebar.header("Configurações")

config_path = st.sidebar.text_input("Config", "config.yaml")
price_column = st.sidebar.text_input("Preço", "Grid")

power_unit = st.sidebar.selectbox("Unidade potência", ["kW", "MW"])
price_unit = st.sidebar.selectbox("Unidade preço", ["EUR/kWh", "EUR/MWh"])

year = st.sidebar.number_input("Ano", value=2022)


# -----------------------
# Load data
# -----------------------

try:
    data = load_baseline_data(config_path)

    baseline = compute_baseline(
        data,
        grid_price_col=price_column,
        units=UnitConfig(power_unit, price_unit),
    )
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()


# -----------------------
# Cálculos
# -----------------------

load = baseline.load_by_group
price = baseline.price
dt = baseline.timestep_hours

total_load = load.sum(axis=1)

# Filtrar ano
mask = total_load.index.year == year
load = load.loc[mask]
total_load = total_load.loc[mask]
price = price.loc[mask]

energy = total_load * dt
daily_energy = energy.resample("D").sum()
monthly_energy = energy.resample("M").sum()

# KPIs
total_energy = energy.sum()
peak_load = total_load.max()

day_max = daily_energy.idxmax()
day_min = daily_energy.idxmin()

avg_price = (price * total_load).sum() / total_load.sum()


# -----------------------
# KPIs
# -----------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Consumo anual", f"{total_energy:,.0f} kWh")
c2.metric("Pico de carga", f"{peak_load:,.1f} kW")
c3.metric("Preço médio", f"{avg_price:.4f}")
c4.metric("Dia máximo", day_max.strftime("%d-%m-%Y"))

st.divider()


# -----------------------
# Tabs
# -----------------------

tab1, tab2, tab3 = st.tabs(["Resumo", "Perfil Diário", "Categorias"])


# ======================
# TAB 1 - Resumo
# ======================
with tab1:

    st.subheader("Diagrama de carga")
    st.line_chart(total_load)

    st.subheader("Curva de duração da carga")

    ldc = total_load.sort_values(ascending=False).reset_index(drop=True)

    st.line_chart(ldc)

    st.subheader("Consumo mensal")
    st.bar_chart(monthly_energy)

    st.info(f"""
    Dia de maior consumo: {day_max.strftime('%d/%m/%Y')}
    
    Dia de menor consumo: {day_min.strftime('%d/%m/%Y')}
    """)


# ======================
# TAB 2 - Perfil diário
# ======================
with tab2:

    st.subheader("Consumo diário")
    st.bar_chart(daily_energy)

    st.subheader("Perfil horário médio")

    df_profile = pd.DataFrame({
        "load": total_load.values,
        "hour": total_load.index.hour,
        "dow": total_load.index.dayofweek
    })

    profile = df_profile.groupby(["hour"])["load"].mean()

    st.line_chart(profile)

    # dias extremos
    st.subheader("Dias extremos")

    max_day = total_load[total_load.index.date == day_max.date()]
    min_day = total_load[total_load.index.date == day_min.date()]

    c1, c2 = st.columns(2)

    c1.line_chart(max_day)
    c2.line_chart(min_day)

    # períodos de maior procura
    st.subheader("Picos de consumo")

    threshold = total_load.quantile(0.9)
    peaks = total_load[total_load > threshold]

    st.write("Horas com maior procura:")
    st.write(peaks.index.hour.value_counts().sort_index())


# ======================
# TAB 3 - Categorias
# ======================
with tab3:

    st.subheader("Consumo por categoria")

    energy_cat = (load * dt).sum(axis=0).sort_values(ascending=False)

    st.bar_chart(energy_cat)

    st.subheader("Distribuição")
    st.write(energy_cat / energy_cat.sum())

    with st.expander("Tabela completa"):
        st.dataframe(energy_cat)
