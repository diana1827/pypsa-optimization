import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# Tenta primeiro os imports a partir da estrutura do projeto.
# Se a app estiver a correr fora desse contexto, tenta os módulos locais.
try:
    from src.data_io import load_baseline_data
    from src.baseline import UnitConfig, compute_baseline
except Exception:
    try:
        from data_io import load_baseline_data
        from baseline import UnitConfig, compute_baseline
    except Exception:
        load_baseline_data = None
        compute_baseline = None
        UnitConfig = None


st.set_page_config(
    page_title="Building Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Building Energy Dashboard")
st.caption(
    "Annual analysis, daily profile, category breakdown, billing variance table, "
    "and PV production analysis."
)


COLORS = {
    "blue": "#1D4ED8",
    "orange": "#F59E0B",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "teal": "#0F766E",
    "gray": "#475569",
    "yellow": "#EAB308",
}


def get_invoice_data() -> pd.DataFrame:
    """
    Devolve os dados de faturação introduzidos manualmente.

    O consumo modelado não fica guardado aqui; é calculado depois
    com base no intervalo exato de faturação.
    """
    data = [
        {
            "Month": "July",
            "Reference Month": pd.Timestamp("2022-07-01"),
            "Supplier": "EDP",
            "Invoice Number": "11220000212761",
            "Issue Date": pd.Timestamp("2022-08-16"),
            "Billing Start": pd.Timestamp("2022-07-14"),
            "Billing End": pd.Timestamp("2022-08-13"),
            "Billing Days": 31,
            "HSV (kWh)": 8642.00,
            "HVN (kWh)": 13011.00,
            "HP (kWh)": 7025.00,
            "HC (kWh)": 29498.00,
            "Invoice Consumption (kWh)": 58176.00,
            "Reactive Energy (kvarh)": 102.00,
            "PHP (kW)": 106.44,
            "PC (kW)": 372.00,
            "Tax (EUR)": 58.18,
            "Regulation Band (EUR)": 46.31,
            "MIBEL (EUR)": 0.00,
            "Total Cost ex VAT (EUR)": 14135.24,
        },
        {
            "Month": "August",
            "Reference Month": pd.Timestamp("2022-08-01"),
            "Supplier": "EDP",
            "Invoice Number": "11220000240377",
            "Issue Date": pd.Timestamp("2022-09-16"),
            "Billing Start": pd.Timestamp("2022-08-14"),
            "Billing End": pd.Timestamp("2022-09-13"),
            "Billing Days": 31,
            "HSV (kWh)": 7923.00,
            "HVN (kWh)": 13570.00,
            "HP (kWh)": 5854.00,
            "HC (kWh)": 25151.00,
            "Invoice Consumption (kWh)": 52498.00,
            "Reactive Energy (kvarh)": 159.00,
            "PHP (kW)": 92.92,
            "PC (kW)": 372.00,
            "Tax (EUR)": 52.50,
            "Regulation Band (EUR)": 45.05,
            "MIBEL (EUR)": 0.00,
            "Total Cost ex VAT (EUR)": 12726.32,
        },
        {
            "Month": "September",
            "Reference Month": pd.Timestamp("2022-09-01"),
            "Supplier": "EDP",
            "Invoice Number": "11220000255527",
            "Issue Date": pd.Timestamp("2022-10-17"),
            "Billing Start": pd.Timestamp("2022-09-14"),
            "Billing End": pd.Timestamp("2022-10-13"),
            "Billing Days": 30,
            "HSV (kWh)": 7434.00,
            "HVN (kWh)": 12868.00,
            "HP (kWh)": 6877.00,
            "HC (kWh)": 26951.00,
            "Invoice Consumption (kWh)": 54130.00,
            "Reactive Energy (kvarh)": 202.00,
            "PHP (kW)": 109.16,
            "PC (kW)": 372.00,
            "Tax (EUR)": 54.13,
            "Regulation Band (EUR)": 53.15,
            "MIBEL (EUR)": 0.00,
            "Total Cost ex VAT (EUR)": 13213.67,
        },
        {
            "Month": "October",
            "Reference Month": pd.Timestamp("2022-10-01"),
            "Supplier": "EDP",
            "Invoice Number": "11220000291384",
            "Issue Date": pd.Timestamp("2022-11-16"),
            "Billing Start": pd.Timestamp("2022-10-14"),
            "Billing End": pd.Timestamp("2022-11-13"),
            "Billing Days": 31,
            "HSV (kWh)": 6375.00,
            "HVN (kWh)": 12754.00,
            "HP (kWh)": 7462.00,
            "HC (kWh)": 23358.00,
            "Invoice Consumption (kWh)": 49949.00,
            "Reactive Energy (kvarh)": 1490.00,
            "PHP (kW)": 95.67,
            "PC (kW)": 372.00,
            "Tax (EUR)": 49.95,
            "Regulation Band (EUR)": 55.77,
            "MIBEL (EUR)": 0.00,
            "Total Cost ex VAT (EUR)": 12206.62,
        },
        {
            "Month": "November",
            "Reference Month": pd.Timestamp("2022-11-01"),
            "Supplier": "EDP",
            "Invoice Number": "11220000300815",
            "Issue Date": pd.Timestamp("2022-12-19"),
            "Billing Start": pd.Timestamp("2022-11-14"),
            "Billing End": pd.Timestamp("2022-12-13"),
            "Billing Days": 30,
            "HSV (kWh)": 5744.00,
            "HVN (kWh)": 10958.00,
            "HP (kWh)": 8335.00,
            "HC (kWh)": 20291.00,
            "Invoice Consumption (kWh)": 45328.00,
            "Reactive Energy (kvarh)": 2403.00,
            "PHP (kW)": 83.35,
            "PC (kW)": 372.00,
            "Tax (EUR)": 45.33,
            "Regulation Band (EUR)": 43.21,
            "MIBEL (EUR)": -148.93,
            "Total Cost ex VAT (EUR)": 10905.30,
        },
        {
            "Month": "December",
            "Reference Month": pd.Timestamp("2022-12-01"),
            "Supplier": "EDP",
            "Invoice Number": "FT 23BSML11/0000012892",
            "Issue Date": pd.Timestamp("2023-01-16"),
            "Billing Start": pd.Timestamp("2022-12-14"),
            "Billing End": pd.Timestamp("2022-12-31"),
            "Billing Days": 18,
            "HSV (kWh)": 3300.00,
            "HVN (kWh)": 5457.00,
            "HP (kWh)": 4691.00,
            "HC (kWh)": 11719.00,
            "Invoice Consumption (kWh)": 25167.00,
            "Reactive Energy (kvarh)": 1347.00,
            "PHP (kW)": 72.17,
            "PC (kW)": 372.00,
            "Tax (EUR)": 25.17,
            "Regulation Band (EUR)": 35.79,
            "MIBEL (EUR)": 1.70,
            "Total Cost ex VAT (EUR)": 6145.42,
        },
    ]

    df = pd.DataFrame(data)
    df["Year"] = df["Reference Month"].dt.year
    return df.sort_values("Reference Month").reset_index(drop=True)


def aggregate_categories(load_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Agrupa colunas com nomes semelhantes na mesma categoria.

    Se estiver ativo o modo 'Grouped HVAC', tudo o que pertence
    a AVAC fica consolidado numa única categoria.
    """
    category_aliases = {
        "equip": "Equipment",
        "light": "Lighting",
        "pump": "Pumps",
        "cool": "Cooling",
        "vent": "Ventilation",
        "hvac": "HVAC",
        "avac": "HVAC",
    }

    hvac_group = {"Cooling", "Pumps", "Ventilation", "HVAC"}

    def normalize_name(name: str) -> str:
        text = str(name).strip().lower()
        for key, label in category_aliases.items():
            if key in text:
                return label
        return str(name).strip().title()

    def map_name(name: str) -> str:
        base = normalize_name(name)
        if mode == "Grouped HVAC" and base in hvac_group:
            return "HVAC"
        return base

    mapping = {col: map_name(col) for col in load_df.columns}
    return load_df.T.groupby(mapping).sum().T


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    """
    Obtém o passo temporal mais frequente do índice.
    """
    if len(index) < 2:
        return 1.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 1.0

    dt = deltas.mode().iloc[0]
    return dt.total_seconds() / 3600.0


def load_project_data_safe(config_path: str, price_column: str):
    """
    Carrega os dados do projeto e calcula o baseline.

    Em caso de erro, devolve a mensagem para a interface tratar
    sem interromper a execução da app.
    """
    if load_baseline_data is None or compute_baseline is None or UnitConfig is None:
        return None, None, "Project modules could not be imported."

    try:
        data = load_baseline_data(config_path)
        baseline = compute_baseline(
            data=data,
            grid_price_col=price_column,
            units=UnitConfig(
                power_unit="kW",
                price_unit="EUR/kWh",
            ),
        )
        return data, baseline, None
    except Exception as exc:
        return None, None, str(exc)


def compute_modeled_billing_consumption(
    total_energy_ts_kwh: pd.Series,
    invoices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula o consumo modelado para cada período de faturação.

    A data final da fatura é considerada inclusive.
    A variância é calculada como:
        consumo modelado - consumo faturado
    """
    rows = []

    for _, row in invoices_df.iterrows():
        start = pd.Timestamp(row["Billing Start"]).normalize()
        end_exclusive = pd.Timestamp(row["Billing End"]).normalize() + pd.Timedelta(days=1)

        mask = (total_energy_ts_kwh.index >= start) & (total_energy_ts_kwh.index < end_exclusive)
        modeled_kwh = float(total_energy_ts_kwh.loc[mask].sum())

        invoice_kwh = float(row["Invoice Consumption (kWh)"])
        billing_variance_kwh = modeled_kwh - invoice_kwh
        billing_variance_pct = (
            100.0 * billing_variance_kwh / invoice_kwh if invoice_kwh != 0 else np.nan
        )

        rows.append(
            {
                **row.to_dict(),
                "Modeled Consumption Billing Period (kWh)": modeled_kwh,
                "Billing Variance (kWh)": billing_variance_kwh,
                "Billing Variance (%)": billing_variance_pct,
            }
        )

    return pd.DataFrame(rows).sort_values("Reference Month").reset_index(drop=True)


def compute_pv_hourly_production_kwh(pv_power_w_df: pd.DataFrame):
    """
    Converte potência fotovoltaica em energia horária.

    Parte do princípio de que os valores de entrada estão em W e que
    cada amostra representa potência instantânea ou por patamar.
    """
    if pv_power_w_df is None or pv_power_w_df.empty:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            1.0,
        )

    pv_power_w_df = pv_power_w_df.copy().sort_index()
    total_pv_power_w = pv_power_w_df.sum(axis=1).rename("PV Power (W)")

    dt_hours = infer_timestep_hours(total_pv_power_w.index)

    pv_energy_step_kwh = (total_pv_power_w * dt_hours / 1000.0).rename("PV Energy Step (kWh)")
    pv_hourly_kwh = pv_energy_step_kwh.resample("h").sum().rename("PV Production (kWh)")

    return pv_hourly_kwh, total_pv_power_w, dt_hours


def format_date_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Formata colunas de data no formato dd/mm/aaaa.
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%d/%m/%Y")
    return out


def style_figure(fig, x_title="", y_title="", horizontal_grid=True):
    """
    Aplica o estilo base aos gráficos para garantir consistência visual.
    """
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="black",
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=18,
            color="black",
            weight="bold",
        ),
        margin=dict(l=100, r=60, t=60, b=100),
        showlegend=False,
        autosize=True,
        height=500,
    )

    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=14, weight="bold", color="black"),
        title_standoff=15,
        showline=True,
        linewidth=1.5,
        linecolor="#BFBFBF",
        mirror=False,
        ticks="outside",
        tickcolor="#BFBFBF",
        ticklen=6,
        tickwidth=1.5,
        tickfont=dict(size=12, color="black"),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=14, weight="bold", color="black"),
        title_standoff=15,
        showline=True,
        linewidth=1.5,
        linecolor="#BFBFBF",
        mirror=False,
        ticks="outside",
        tickcolor="#BFBFBF",
        ticklen=6,
        tickwidth=1.5,
        tickfont=dict(size=12, color="black"),
        showgrid=horizontal_grid,
        gridcolor="#E0E0E0",
        gridwidth=0.5,
        zeroline=False,
    )

    fig.update_traces(cliponaxis=False)

    return fig


st.sidebar.header("Settings")

config_path = st.sidebar.text_input("Config file", "config.yaml")
price_column = st.sidebar.text_input("Price column", "Grid")
category_mode = st.sidebar.radio(
    "Category view",
    ["Detailed categories", "Grouped HVAC"],
    index=0,
)


invoice_df_all = get_invoice_data()

project_data, baseline, baseline_error = load_project_data_safe(
    config_path=config_path,
    price_column=price_column,
)

baseline_loaded = baseline is not None
pv_loaded = False
pv_error = None

if project_data is not None and "pv_error" in project_data:
    pv_error = project_data["pv_error"]

monthly_energy_kwh = pd.Series(dtype=float)
daily_energy_kwh = pd.Series(dtype=float)
total_energy_ts_kwh = pd.Series(dtype=float)
total_load_kw = pd.Series(dtype=float)
category_df = pd.DataFrame()

pv_hourly_kwh = pd.Series(dtype=float)
pv_daily_kwh = pd.Series(dtype=float)
pv_monthly_kwh = pd.Series(dtype=float)
pv_total_power_w = pd.Series(dtype=float)
pv_source_df = pd.DataFrame()

selected_year = 2022
available_years = sorted(invoice_df_all["Year"].unique().tolist())

if baseline_loaded:
    try:
        load_by_group_raw = baseline.load_by_group.copy().sort_index()
        baseline_years = sorted(load_by_group_raw.index.year.unique().tolist())
        available_years = sorted(set(available_years).union(set(baseline_years)))
    except Exception:
        baseline_loaded = False
        baseline_error = "Baseline object does not provide a valid load_by_group time series."

if project_data is not None and "pv_power_w" in project_data:
    try:
        pv_raw = project_data["pv_power_w"].copy().sort_index()
        pv_years = sorted(pv_raw.index.year.unique().tolist())
        available_years = sorted(set(available_years).union(set(pv_years)))
    except Exception as exc:
        pv_error = str(exc)

selected_year = st.sidebar.selectbox(
    "Analysis year",
    options=available_years,
    index=available_years.index(2022) if 2022 in available_years else len(available_years) - 1,
)

invoice_df = invoice_df_all[invoice_df_all["Year"] == selected_year].copy()

# Cálculo dos indicadores do baseline do edifício.
if baseline_loaded:
    try:
        load_by_group_year = load_by_group_raw.loc[load_by_group_raw.index.year == selected_year].copy()

        if load_by_group_year.empty:
            baseline_loaded = False
            baseline_error = f"No baseline time series available for year {selected_year}."
        else:
            load_by_group_year = aggregate_categories(load_by_group_year, category_mode)

            dt_hours = float(baseline.timestep_hours)
            total_load_kw = load_by_group_year.sum(axis=1).rename("Total Load (kW)")
            total_energy_ts_kwh = (total_load_kw * dt_hours).rename("Total Energy (kWh)")
            daily_energy_kwh = total_energy_ts_kwh.resample("D").sum()
            monthly_energy_kwh = total_energy_ts_kwh.resample("ME").sum()

            category_energy_kwh = load_by_group_year.mul(dt_hours).sum().sort_values(ascending=False)
            category_df = category_energy_kwh.rename_axis("Category").reset_index(name="Energy (kWh)")
            category_df["Share (%)"] = 100 * category_df["Energy (kWh)"] / category_df["Energy (kWh)"].sum()

    except Exception as exc:
        baseline_loaded = False
        baseline_error = str(exc)

# Cálculo dos indicadores da produção fotovoltaica.
if project_data is not None and "pv_power_w" in project_data:
    try:
        pv_power_w_year = project_data["pv_power_w"].loc[
            project_data["pv_power_w"].index.year == selected_year
        ].copy()

        if pv_power_w_year.empty:
            pv_loaded = False
            if pv_error is None:
                pv_error = f"No PV time series available for year {selected_year}."
        else:
            pv_hourly_kwh, pv_total_power_w, pv_dt_hours = compute_pv_hourly_production_kwh(pv_power_w_year)
            pv_daily_kwh = pv_hourly_kwh.resample("D").sum()
            pv_monthly_kwh = pv_hourly_kwh.resample("ME").sum()

            pv_source_energy_kwh = (
                pv_power_w_year.mul(pv_dt_hours / 1000.0)
                .sum()
                .sort_values(ascending=False)
            )

            pv_source_df = pv_source_energy_kwh.rename_axis("PV Source").reset_index(name="Energy (kWh)")
            if not pv_source_df.empty and pv_source_df["Energy (kWh)"].sum() > 0:
                pv_source_df["Share (%)"] = 100 * pv_source_df["Energy (kWh)"] / pv_source_df["Energy (kWh)"].sum()
            else:
                pv_source_df["Share (%)"] = 0.0

            pv_loaded = True

    except Exception as exc:
        pv_loaded = False
        pv_error = str(exc)
else:
    pv_loaded = False
    if pv_error is None:
        pv_error = "Optional PV file 'PV_Data_2022.csv' was not found in the configured data folder."


comparison_df = pd.DataFrame()

if baseline_loaded and not invoice_df.empty:
    comparison_df = compute_modeled_billing_consumption(
        total_energy_ts_kwh=total_energy_ts_kwh,
        invoices_df=invoice_df,
    )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Daily Profile", "Categories", "Billing Comparison", "PV Production"]
)


with tab1:
    st.subheader(f"Overview - {selected_year}")

    if invoice_df.empty:
        st.warning(f"No invoice data configured for year {selected_year}.")
    else:
        total_invoice_kwh = float(invoice_df["Invoice Consumption (kWh)"].sum())
        st.metric("Total Invoice Consumption", f"{total_invoice_kwh:,.2f} kWh")

        if baseline_loaded and not comparison_df.empty:
            st.info(
                "The invoice-to-model comparison was moved to the 'Billing Comparison' tab, "
                "where it is presented only as a billing variance table."
            )
        elif not baseline_loaded:
            st.warning(
                "Invoice data is available, but the baseline time series could not be loaded. "
                "Billing variance analysis requires the baseline load time series in kW."
            )
            st.code(f"Baseline load error: {baseline_error}")

    if baseline_loaded:
        st.markdown("---")
        st.subheader(f"Modeled building summary - {selected_year}")

        annual_consumption_kwh = float(total_energy_ts_kwh.sum())
        average_daily_consumption_kwh = float(daily_energy_kwh.mean())
        peak_load_kw = float(total_load_kw.max())
        peak_timestamp = total_load_kw.idxmax()

        s1, s2, s3 = st.columns(3)
        s1.metric("Annual Consumption", f"{annual_consumption_kwh:,.2f} kWh")
        s2.metric("Average Daily Consumption", f"{average_daily_consumption_kwh:,.2f} kWh")
        s3.metric("Peak Load", f"{peak_load_kw:,.2f} kW")

        st.caption(f"Peak load timestamp: {peak_timestamp.strftime('%d/%m/%Y %H:%M')}")

        monthly_plot_df = monthly_energy_kwh.rename_axis("Date").reset_index(name="Energy (kWh)")
        monthly_plot_df["Month"] = monthly_plot_df["Date"].dt.strftime("%b")

        # Força a ordenação correta dos meses no gráfico.
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_plot_df["Month"] = pd.Categorical(
            monthly_plot_df["Month"],
            categories=month_order,
            ordered=True,
        )
        monthly_plot_df = monthly_plot_df.sort_values("Month")

        fig_monthly = px.bar(
            monthly_plot_df,
            x="Month",
            y="Energy (kWh)",
            text="Energy (kWh)",
            color_discrete_sequence=[COLORS["teal"]],
        )

        fig_monthly.update_traces(
            texttemplate="%{text:,.0f} kWh",
            textposition="outside",
            textfont=dict(size=13, color="black", weight="bold"),
            marker_line_color="#A6A6A6",
            marker_line_width=1.5,
            hovertemplate="<b>%{x}</b><br>Consumption: %{y:,.0f} kWh<br><extra></extra>",
        )

        fig_monthly = style_figure(fig_monthly, x_title="Month", y_title="Energy Consumption (kWh)")
        fig_monthly.update_layout(height=500, margin=dict(l=80, r=60, t=60, b=80))

        st.plotly_chart(
            fig_monthly,
            use_container_width=True,
            key=f"overview_monthly_modeled_{selected_year}",
        )

    if pv_loaded:
        st.markdown("---")
        st.subheader(f"PV summary - {selected_year}")

        annual_pv_kwh = float(pv_hourly_kwh.sum())
        avg_daily_pv_kwh = float(pv_daily_kwh.mean())
        peak_hour_pv_kwh = float(pv_hourly_kwh.max())
        peak_hour_pv_ts = pv_hourly_kwh.idxmax()

        p1, p2, p3 = st.columns(3)
        p1.metric("Annual PV Production", f"{annual_pv_kwh:,.2f} kWh")
        p2.metric("Average Daily PV Production", f"{avg_daily_pv_kwh:,.2f} kWh")
        p3.metric("Peak Hourly PV Production", f"{peak_hour_pv_kwh:,.2f} kWh")

        st.caption(f"Peak PV hour: {peak_hour_pv_ts.strftime('%d/%m/%Y %H:%M')}")


with tab2:
    st.subheader(f"Daily Profile - {selected_year}")

    if not baseline_loaded:
        st.warning(
            "Daily profile is not available because the baseline time series could not be loaded."
        )
        st.code(f"Baseline load error: {baseline_error}")
    else:
        daily_df = pd.DataFrame({
            "Date": daily_energy_kwh.index,
            "Consumption (kWh)": daily_energy_kwh.values,
        })

        fig_daily = px.bar(
            daily_df,
            x="Date",
            y="Consumption (kWh)",
            text="Consumption (kWh)",
            color_discrete_sequence=[COLORS["blue"]],
        )

        fig_daily.update_traces(
            texttemplate="%{text:,.0f} kWh",
            textposition="outside",
            textfont=dict(size=10, color="black"),
            marker_color=COLORS["blue"],
            marker_line_color="#A6A6A6",
            marker_line_width=1,
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Daily consumption: %{y:,.0f} kWh<extra></extra>",
        )

        fig_daily = style_figure(fig_daily, x_title="Date", y_title="Daily Consumption (kWh)")
        fig_daily.update_xaxes(tickangle=-45, tickformat="%d/%m", dtick="M1")
        fig_daily.update_layout(height=500, margin=dict(l=80, r=60, t=60, b=120))

        st.plotly_chart(
            fig_daily,
            use_container_width=True,
            key=f"daily_profile_bar_{selected_year}",
        )

        st.markdown("### Selected day analysis")

        available_days = [d.date() for d in daily_energy_kwh.index]
        default_day = available_days[0] if available_days else None

        if default_day is not None:
            selected_day = st.selectbox(
                "Select day",
                options=available_days,
                index=0,
                format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
            )

            selected_day_ts = pd.Timestamp(selected_day)
            next_day_ts = selected_day_ts + pd.Timedelta(days=1)

            day_load_kw = total_load_kw[
                (total_load_kw.index >= selected_day_ts) & (total_load_kw.index < next_day_ts)
            ]
            day_energy_kwh = total_energy_ts_kwh[
                (total_energy_ts_kwh.index >= selected_day_ts) & (total_energy_ts_kwh.index < next_day_ts)
            ]

            h1, h2, h3 = st.columns(3)
            h1.metric("Selected Day", selected_day_ts.strftime("%d-%m-%Y"))
            h2.metric("Daily Consumption", f"{day_energy_kwh.sum():,.2f} kWh")
            h3.metric("Daily Peak Load", f"{day_load_kw.max():,.2f} kW")

            mode = st.radio(
                "Chart type",
                ["Hourly Load (kW)", "Hourly Consumption (kWh)"],
                horizontal=True,
            )

            if mode == "Hourly Load (kW)":
                hourly = day_load_kw.resample("h").mean()
                plot_df = hourly.rename_axis("Timestamp").reset_index(name="Value")
                plot_df["Hour"] = plot_df["Timestamp"].dt.strftime("%H:%M")

                fig_hour = px.line(
                    plot_df,
                    x="Hour",
                    y="Value",
                    markers=True,
                )

                fig_hour.update_traces(
                    line=dict(color=COLORS["red"], width=2.5),
                    marker=dict(size=7, color=COLORS["red"], line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x}</b><br>Load: %{y:,.2f} kW<extra></extra>",
                )

                fig_hour = style_figure(fig_hour, x_title="Hour", y_title="Load (kW)")
                fig_hour.update_layout(height=450, margin=dict(l=80, r=60, t=40, b=80))

                st.plotly_chart(
                    fig_hour,
                    use_container_width=True,
                    key=f"selected_day_load_{selected_year}_{selected_day_ts.strftime('%Y%m%d')}",
                )

            else:
                hourly = day_energy_kwh.resample("h").sum()
                plot_df = hourly.rename_axis("Timestamp").reset_index(name="Value")
                plot_df["Hour"] = plot_df["Timestamp"].dt.strftime("%H:%M")

                fig_hour = px.bar(
                    plot_df,
                    x="Hour",
                    y="Value",
                    text="Value",
                    color_discrete_sequence=[COLORS["orange"]],
                )

                fig_hour.update_traces(
                    texttemplate="%{text:,.2f} kWh",
                    textposition="outside",
                    textfont=dict(size=11, color="black"),
                    marker_line_color="#A6A6A6",
                    marker_line_width=1,
                    hovertemplate="<b>%{x}</b><br>Consumption: %{y:,.2f} kWh<extra></extra>",
                )

                fig_hour = style_figure(fig_hour, x_title="Hour", y_title="Consumption (kWh)")
                fig_hour.update_layout(height=450, margin=dict(l=80, r=60, t=40, b=80))

                st.plotly_chart(
                    fig_hour,
                    use_container_width=True,
                    key=f"selected_day_energy_{selected_year}_{selected_day_ts.strftime('%Y%m%d')}",
                )


with tab3:
    st.subheader(f"Categories - {selected_year}")

    if not baseline_loaded:
        st.warning(
            "Category analysis is not available because the baseline time series could not be loaded."
        )
        st.code(f"Baseline load error: {baseline_error}")
    else:
        left, right = st.columns([1.1, 1])

        with left:
            fig_pie = px.pie(
                category_df,
                names="Category",
                values="Energy (kWh)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig_pie.update_traces(
                textinfo="percent+label",
                textposition="inside",
                textfont=dict(size=13, color="black", weight="bold"),
                insidetextorientation="auto",
                marker=dict(line=dict(color="white", width=2)),
                hovertemplate="<b>%{label}</b><br>Energy: %{value:,.2f} kWh<br>Share: %{percent}<extra></extra>",
            )

            fig_pie.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial, sans-serif", size=13, color="black"),
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=True,
                legend=dict(
                    bgcolor="white",
                    bordercolor="#D9D9D9",
                    borderwidth=1,
                    font=dict(size=12, color="black"),
                ),
                height=500,
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True,
                key=f"categories_pie_{selected_year}_{category_mode.replace(' ', '_')}",
            )

        with right:
            ranking_df = category_df.sort_values("Energy (kWh)", ascending=True)
            fig_rank = px.bar(
                ranking_df,
                x="Energy (kWh)",
                y="Category",
                orientation="h",
                text="Energy (kWh)",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig_rank.update_traces(
                texttemplate="%{text:,.2f} kWh",
                textposition="outside",
                textfont=dict(size=11, color="black"),
                marker_line_color="#A6A6A6",
                marker_line_width=1,
                hovertemplate="<b>%{y}</b><br>Energy: %{x:,.2f} kWh<extra></extra>",
            )

            fig_rank = style_figure(fig_rank, x_title="Energy (kWh)", y_title="Category")
            fig_rank.update_layout(height=500, margin=dict(l=120, r=60, t=40, b=80), showlegend=False)

            st.plotly_chart(
                fig_rank,
                use_container_width=True,
                key=f"categories_rank_{selected_year}_{category_mode.replace(' ', '_')}",
            )

        st.markdown("### Full category table")
        st.dataframe(
            category_df.style.format(
                {
                    "Energy (kWh)": "{:,.2f}",
                    "Share (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with tab4:
    st.subheader(f"Billing Comparison - {selected_year}")

    if invoice_df.empty:
        st.warning(f"No invoice data configured for year {selected_year}.")
    else:
        with st.expander("Legend for invoice acronyms and electricity market terms", expanded=True):
            st.markdown(
                """
                - **HSV**: *Horas de Super Vazio* — energy consumed in super off-peak tariff period.
                - **HVN**: *Horas de Vazio Normal* — energy consumed in normal off-peak tariff period.
                - **HP**: *Horas de Ponta* — energy consumed during peak tariff period.
                - **HC**: *Horas de Cheias* — energy consumed during full/shoulder tariff period.
                - **Reactive Energy (kvarh)**: reactive energy billed by the supplier.
                - **PHP (kW)**: power registered during peak hours (*Potência em Horas de Ponta*).
                - **PC (kW)**: contracted power (*Potência Contratada*).
                - **MIBEL (EUR)**: adjustment or market component related to the Iberian Electricity Market (*Mercado Ibérico de Eletricidade*).
                - **VAT**: Value Added Tax. "Total Cost ex VAT" means total cost excluding VAT.

                **Note:** exact acronym definitions may vary slightly depending on supplier wording and tariff contract, but these are the standard meanings used in Portuguese electricity billing.
                """
            )

        date_table = format_date_columns(
            invoice_df[
                [
                    "Month",
                    "Invoice Number",
                    "Issue Date",
                    "Billing Start",
                    "Billing End",
                    "Billing Days",
                ]
            ],
            ["Issue Date", "Billing Start", "Billing End"],
        )

        st.markdown("### Invoice dates and billing periods")
        st.dataframe(
            date_table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Invoice energy components")
        invoice_energy_table = invoice_df[
            [
                "Month",
                "HSV (kWh)",
                "HVN (kWh)",
                "HP (kWh)",
                "HC (kWh)",
                "Invoice Consumption (kWh)",
            ]
        ].copy()

        st.dataframe(
            invoice_energy_table.style.format(
                {
                    "HSV (kWh)": "{:,.2f}",
                    "HVN (kWh)": "{:,.2f}",
                    "HP (kWh)": "{:,.2f}",
                    "HC (kWh)": "{:,.2f}",
                    "Invoice Consumption (kWh)": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Electrical and financial details")
        financial_table = invoice_df[
            [
                "Month",
                "Reactive Energy (kvarh)",
                "PHP (kW)",
                "PC (kW)",
                "Tax (EUR)",
                "Regulation Band (EUR)",
                "MIBEL (EUR)",
                "Total Cost ex VAT (EUR)",
            ]
        ].copy()

        st.dataframe(
            financial_table.style.format(
                {
                    "Reactive Energy (kvarh)": "{:,.2f}",
                    "PHP (kW)": "{:,.2f}",
                    "PC (kW)": "{:,.2f}",
                    "Tax (EUR)": "{:,.2f}",
                    "Regulation Band (EUR)": "{:,.2f}",
                    "MIBEL (EUR)": "{:,.2f}",
                    "Total Cost ex VAT (EUR)": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if baseline_loaded and not comparison_df.empty:
            st.markdown("### Billing variance table")

            comparison_display = format_date_columns(
                comparison_df[
                    [
                        "Month",
                        "Invoice Number",
                        "Issue Date",
                        "Billing Start",
                        "Billing End",
                        "Billing Days",
                        "Invoice Consumption (kWh)",
                        "Modeled Consumption Billing Period (kWh)",
                        "Billing Variance (kWh)",
                        "Billing Variance (%)",
                    ]
                ],
                ["Issue Date", "Billing Start", "Billing End"],
            )

            st.dataframe(
                comparison_display.style.format(
                    {
                        "Invoice Consumption (kWh)": "{:,.2f}",
                        "Modeled Consumption Billing Period (kWh)": "{:,.2f}",
                        "Billing Variance (kWh)": "{:,.2f}",
                        "Billing Variance (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "Billing variance is calculated over the exact invoice billing dates. "
                "Formula used: modeled consumption minus invoiced consumption. "
                "Negative values indicate the model is below invoiced consumption; "
                "positive values indicate the model is above invoiced consumption."
            )

        else:
            st.warning(
                "Invoice data is available, but billing variance is not available "
                "because the baseline time series could not be loaded."
            )
            st.code(f"Baseline load error: {baseline_error}")


with tab5:
    st.subheader(f"PV Production - {selected_year}")

    if not pv_loaded:
        st.warning(
            "PV production is not available because the PV file could not be loaded or contains no data for the selected year."
        )
        st.code(f"PV load error: {pv_error}")
    else:
        annual_pv_kwh = float(pv_hourly_kwh.sum())
        average_daily_pv_kwh = float(pv_daily_kwh.mean()) if not pv_daily_kwh.empty else 0.0
        peak_hourly_pv_kwh = float(pv_hourly_kwh.max()) if not pv_hourly_kwh.empty else 0.0
        peak_hourly_pv_ts = pv_hourly_kwh.idxmax() if not pv_hourly_kwh.empty else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual PV Production", f"{annual_pv_kwh:,.2f} kWh")
        c2.metric("Average Daily PV Production", f"{average_daily_pv_kwh:,.2f} kWh")
        c3.metric("Peak Hourly PV Production", f"{peak_hourly_pv_kwh:,.2f} kWh")

        if peak_hourly_pv_ts is not None:
            st.caption(f"Peak PV hour: {peak_hourly_pv_ts.strftime('%d/%m/%Y %H:%M')}")

        st.info(
            "PV hourly production is computed from the CSV values by summing the PV power samples "
            "within each hour after converting them to energy. "
            "Assumption used: values are in W, so kWh = Σ(W × Δt[h]) / 1000."
        )

        monthly_pv_df = pv_monthly_kwh.rename_axis("Date").reset_index(name="Energy (kWh)")
        monthly_pv_df["Month"] = monthly_pv_df["Date"].dt.strftime("%b")

        # Força a ordenação correta dos meses no gráfico.
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_pv_df["Month"] = pd.Categorical(
            monthly_pv_df["Month"],
            categories=month_order,
            ordered=True,
        )
        monthly_pv_df = monthly_pv_df.sort_values("Month")

        fig_monthly_pv = px.bar(
            monthly_pv_df,
            x="Month",
            y="Energy (kWh)",
            text="Energy (kWh)",
            color_discrete_sequence=[COLORS["yellow"]],
        )

        fig_monthly_pv.update_traces(
            texttemplate="%{text:,.0f} kWh",
            textposition="outside",
            textfont=dict(size=13, color="black", weight="bold"),
            marker_line_color="#A6A6A6",
            marker_line_width=1.5,
            hovertemplate="<b>%{x}</b><br>PV production: %{y:,.0f} kWh<extra></extra>",
        )

        fig_monthly_pv = style_figure(fig_monthly_pv, x_title="Month", y_title="PV Production (kWh)")
        fig_monthly_pv.update_layout(height=500, margin=dict(l=80, r=60, t=60, b=80))

        st.plotly_chart(
            fig_monthly_pv,
            use_container_width=True,
            key=f"pv_monthly_{selected_year}",
        )

        daily_pv_df = pd.DataFrame({
            "Date": pv_daily_kwh.index,
            "PV Production (kWh)": pv_daily_kwh.values,
        })

        fig_daily_pv = px.bar(
            daily_pv_df,
            x="Date",
            y="PV Production (kWh)",
            text="PV Production (kWh)",
            color_discrete_sequence=[COLORS["green"]],
        )

        fig_daily_pv.update_traces(
            texttemplate="%{text:,.0f} kWh",
            textposition="outside",
            textfont=dict(size=10, color="black"),
            marker_line_color="#A6A6A6",
            marker_line_width=1,
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>PV production: %{y:,.0f} kWh<extra></extra>",
        )

        fig_daily_pv = style_figure(fig_daily_pv, x_title="Date", y_title="PV Production (kWh)")
        fig_daily_pv.update_xaxes(tickangle=-45, tickformat="%d/%m", dtick="M1")
        fig_daily_pv.update_layout(height=500, margin=dict(l=80, r=60, t=60, b=120))

        st.plotly_chart(
            fig_daily_pv,
            use_container_width=True,
            key=f"pv_daily_{selected_year}",
        )

        st.markdown("### Selected day PV analysis")

        available_pv_days = [d.date() for d in pv_daily_kwh.index]
        default_pv_day = available_pv_days[0] if available_pv_days else None

        if default_pv_day is not None:
            selected_pv_day = st.selectbox(
                "Select PV day",
                options=available_pv_days,
                index=0,
                format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
                key="selected_pv_day",
            )

            selected_pv_day_ts = pd.Timestamp(selected_pv_day)
            next_pv_day_ts = selected_pv_day_ts + pd.Timedelta(days=1)

            day_hourly_pv_kwh = pv_hourly_kwh[
                (pv_hourly_kwh.index >= selected_pv_day_ts) & (pv_hourly_kwh.index < next_pv_day_ts)
            ]

            day_pv_power_w = pv_total_power_w[
                (pv_total_power_w.index >= selected_pv_day_ts) & (pv_total_power_w.index < next_pv_day_ts)
            ]

            if not day_hourly_pv_kwh.empty:
                day_pv_df = pd.DataFrame({
                    "Hour": day_hourly_pv_kwh.index,
                    "Production (kWh)": day_hourly_pv_kwh.values,
                })

                fig_day_pv = px.bar(
                    day_pv_df,
                    x="Hour",
                    y="Production (kWh)",
                    text="Production (kWh)",
                    color_discrete_sequence=[COLORS["yellow"]],
                )

                fig_day_pv.update_traces(
                    texttemplate="%{text:,.3f} kWh",
                    textposition="outside",
                    textfont=dict(size=11, color="black"),
                    marker_line_color="#A6A6A6",
                    marker_line_width=1,
                    hovertemplate="<b>%{x|%d/%m/%Y %H:%M}</b><br>PV production: %{y:,.3f} kWh<extra></extra>",
                )

                fig_day_pv = style_figure(fig_day_pv, x_title="Hour", y_title="PV Production (kWh)")
                fig_day_pv.update_xaxes(tickformat="%H:%M")
                fig_day_pv.update_layout(height=450, margin=dict(l=80, r=60, t=40, b=80))

                st.plotly_chart(fig_day_pv, use_container_width=True)

            if not day_pv_power_w.empty:
                day_power_df = pd.DataFrame({
                    "Hour": day_pv_power_w.index,
                    "Power (kW)": day_pv_power_w.values / 1000.0,
                })

                fig_day_power = px.line(
                    day_power_df,
                    x="Hour",
                    y="Power (kW)",
                    markers=True,
                )

                fig_day_power.update_traces(
                    line=dict(color=COLORS["green"], width=2.5),
                    marker=dict(size=7, color=COLORS["green"], line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%d/%m/%Y %H:%M}</b><br>PV power: %{y:,.2f} kW<extra></extra>",
                )

                fig_day_power = style_figure(fig_day_power, x_title="Hour", y_title="PV Power (kW)")
                fig_day_power.update_xaxes(tickformat="%H:%M")
                fig_day_power.update_layout(height=450, margin=dict(l=80, r=60, t=40, b=80))

                st.plotly_chart(fig_day_power, use_container_width=True)