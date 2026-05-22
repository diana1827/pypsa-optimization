from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# Imports (works for both local and package runs)
try:
    from src.data_io import load_baseline_data
    from src.baseline import UnitConfig, compute_baseline
except Exception:
    from data_io import load_baseline_data
    from baseline import UnitConfig, compute_baseline


# -----------------------
# Page setup
# -----------------------

st.set_page_config(layout="wide")
st.title("Baseline Dashboard")

# -----------------------
# Sidebar inputs
# -----------------------

st.sidebar.header("Settings")

config_path = st.sidebar.text_input("Config path", "config.yaml")
price_column = st.sidebar.text_input("Price column", "Grid")

power_unit = st.sidebar.selectbox("Power unit", ["kW", "MW"])
price_unit = st.sidebar.selectbox("Price unit", ["EUR/kWh", "EUR/MWh"])


# -----------------------
# Data loading
# -----------------------

try:
    data = load_baseline_data(config_path)

    baseline = compute_baseline(
        data=data,
        grid_price_col=price_column,
        units=UnitConfig(power_unit=power_unit, price_unit=price_unit),
    )

except Exception as error:
    st.error(f"Error: {error}")
    st.stop()


# -----------------------
# Derived metrics
# -----------------------

load = baseline.load_by_group          # kW
price = baseline.price                # EUR/kWh
cost_rate = baseline.cost_rate_by_group
dt = baseline.timestep_hours          # hours

# Energy = Power × Time
energy = load * dt
cost = cost_rate * dt

total_load = load.sum(axis=1)

total_energy = float(energy.sum().sum())
total_cost = float(cost.sum().sum())

# Weighted average price
average_price = float((price * total_load).sum() / total_load.sum())

peak_load = float(total_load.max())


# -----------------------
# KPI display
# -----------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Energy", f"{total_energy:,.0f} kWh")
c2.metric("Total Cost", f"{total_cost:,.0f} EUR")
c3.metric("Average Price", f"{average_price:.4f} EUR/kWh")
c4.metric("Peak Load", f"{peak_load:.1f} kW")

st.divider()


# -----------------------
# Time series plots
# -----------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Load")
    st.line_chart(total_load)

with col2:
    st.subheader("Electricity Price")
    st.line_chart(price)

st.subheader("Total Cost Over Time")
st.line_chart(cost.sum(axis=1))

st.divider()


# -----------------------
# Energy by category
# -----------------------

st.subheader("Energy by Category")

energy_totals = energy.sum(axis=0).reset_index()
energy_totals.columns = ["Category", "Energy"]
energy_totals = energy_totals.sort_values("Energy", ascending=False)

fig_energy = px.bar(
    energy_totals,
    x="Category",
    y="Energy",
    color="Category",
    title="Total Energy by Category",
)

fig_energy.update_layout(showlegend=False)
fig_energy.update_yaxes(title="Energy [kWh]")
fig_energy.update_xaxes(title="Category")

st.plotly_chart(fig_energy, use_container_width=True)


# -----------------------
# Cost by category
# -----------------------

st.subheader("Cost by Category")

cost_totals = cost.sum(axis=0).reset_index()
cost_totals.columns = ["Category", "Cost"]
cost_totals = cost_totals.sort_values("Cost", ascending=False)

fig_cost = px.bar(
    cost_totals,
    x="Category",
    y="Cost",
    color="Category",
    title="Total Cost by Category",
)

fig_cost.update_layout(showlegend=False)
fig_cost.update_yaxes(title="Cost [EUR]")
fig_cost.update_xaxes(title="Category")

st.plotly_chart(fig_cost, use_container_width=True)