import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# OPTIONAL PROJECT IMPORTS
# =========================================================

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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Building Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Building Energy Dashboard")
st.caption(
    "Annual analysis, daily profile, category breakdown, and exact billing-period comparison "
    "between invoice consumption and modeled consumption."
)


# =========================================================
# COLORS
# =========================================================

COLORS = {
    "blue": "#1D4ED8",
    "orange": "#F59E0B",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "teal": "#0F766E",
    "gray": "#475569",
}


# =========================================================
# INVOICE DATA (EXACT VALUES FROM THE USER)
# =========================================================

def get_invoice_data() -> pd.DataFrame:
    """
    Exact invoice data. Only invoice-side values are stored here.
    Modeled consumption is computed later from the exact billing period.
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


# =========================================================
# HELPERS
# =========================================================

def aggregate_categories(load_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Simplified category aggregation.
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


def load_baseline_safe(config_path: str, price_column: str):
    """
    Load baseline with fixed internal units for billing comparison:
    power in kW and price in EUR/kWh.
    """
    if load_baseline_data is None or compute_baseline is None or UnitConfig is None:
        return None, "Project modules could not be imported."

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
        return baseline, None
    except Exception as exc:
        return None, str(exc)


def compute_modeled_billing_consumption(
    total_energy_ts_kwh: pd.Series,
    invoices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute modeled consumption in kWh for the exact invoice billing period.
    End date is treated as inclusive.
    """
    rows = []

    for _, row in invoices_df.iterrows():
        start = pd.Timestamp(row["Billing Start"]).normalize()
        end_exclusive = pd.Timestamp(row["Billing End"]).normalize() + pd.Timedelta(days=1)

        mask = (total_energy_ts_kwh.index >= start) & (total_energy_ts_kwh.index < end_exclusive)
        modeled_kwh = float(total_energy_ts_kwh.loc[mask].sum())

        invoice_kwh = float(row["Invoice Consumption (kWh)"])
        difference_kwh = invoice_kwh - modeled_kwh
        absolute_difference_kwh = abs(difference_kwh)
        coverage_pct = 100.0 * modeled_kwh / invoice_kwh if invoice_kwh != 0 else np.nan

        rows.append(
            {
                **row.to_dict(),
                "Modeled Consumption Billing Period (kWh)": modeled_kwh,
                "Difference (kWh)": difference_kwh,
                "Absolute Difference (kWh)": absolute_difference_kwh,
                "Coverage (%)": coverage_pct,
            }
        )

    return pd.DataFrame(rows).sort_values("Reference Month").reset_index(drop=True)


def format_date_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%d/%m/%Y")
    return out


def get_lowest_error_row(comparison_df: pd.DataFrame):
    if comparison_df.empty:
        return None
    idx = comparison_df["Absolute Difference (kWh)"].idxmin()
    return comparison_df.loc[idx]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Settings")

config_path = st.sidebar.text_input("Config file", "config.yaml")
price_column = st.sidebar.text_input("Price column", "Grid")
category_mode = st.sidebar.radio(
    "Category view",
    ["Detailed categories", "Grouped HVAC"],
    index=0,
)


# =========================================================
# DATA LOADING
# =========================================================

invoice_df_all = get_invoice_data()

baseline, baseline_error = load_baseline_safe(
    config_path=config_path,
    price_column=price_column,
)

baseline_loaded = baseline is not None

monthly_energy_kwh = pd.Series(dtype=float)
daily_energy_kwh = pd.Series(dtype=float)
total_energy_ts_kwh = pd.Series(dtype=float)
total_load_kw = pd.Series(dtype=float)
category_df = pd.DataFrame()
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

selected_year = st.sidebar.selectbox(
    "Analysis year",
    options=available_years,
    index=available_years.index(2022) if 2022 in available_years else len(available_years) - 1,
)

invoice_df = invoice_df_all[invoice_df_all["Year"] == selected_year].copy()

if baseline_loaded:
    try:
        load_by_group_year = load_by_group_raw.loc[load_by_group_raw.index.year == selected_year].copy()

        if load_by_group_year.empty:
            baseline_loaded = False
            baseline_error = f"No baseline time series available for year {selected_year}."
        else:
            # Aggregate categories
            load_by_group_year = aggregate_categories(load_by_group_year, category_mode)

            # Internal units:
            # - load in kW
            # - timestep in hours
            # - energy = kW * h = kWh
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


comparison_df = pd.DataFrame()

if baseline_loaded and not invoice_df.empty:
    comparison_df = compute_modeled_billing_consumption(
        total_energy_ts_kwh=total_energy_ts_kwh,
        invoices_df=invoice_df,
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Daily Profile", "Categories", "Billing Comparison"]
)


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:
    st.subheader(f"Overview - {selected_year}")

    if invoice_df.empty:
        st.warning(f"No invoice data configured for year {selected_year}.")
    else:
        total_invoice_kwh = float(invoice_df["Invoice Consumption (kWh)"].sum())

        if baseline_loaded and not comparison_df.empty:
            total_modeled_billing_kwh = float(comparison_df["Modeled Consumption Billing Period (kWh)"].sum())
            total_difference_kwh = float(comparison_df["Difference (kWh)"].sum())
            avg_coverage_pct = float(comparison_df["Coverage (%)"].mean())

            lowest_error_row = get_lowest_error_row(comparison_df)
            lowest_error_kwh = float(lowest_error_row["Absolute Difference (kWh)"])
            lowest_error_month = lowest_error_row["Month"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Invoice Consumption", f"{total_invoice_kwh:,.2f} kWh")
            m2.metric("Total Modeled Consumption", f"{total_modeled_billing_kwh:,.2f} kWh")
            m3.metric("Total Difference", f"{total_difference_kwh:,.2f} kWh")
            m4.metric("Average Coverage", f"{avg_coverage_pct:.2f} %")
            m5.metric("Lowest Absolute Error", f"{lowest_error_kwh:,.2f} kWh")

            st.caption(f"Lowest absolute error observed in: {lowest_error_month}")

            plot_df = comparison_df.melt(
                id_vars=["Month"],
                value_vars=[
                    "Invoice Consumption (kWh)",
                    "Modeled Consumption Billing Period (kWh)",
                ],
                var_name="Series",
                value_name="Energy (kWh)",
            )

            label_map = {
                "Invoice Consumption (kWh)": "Invoice Consumption",
                "Modeled Consumption Billing Period (kWh)": "Modeled Consumption (Exact Billing Period)",
            }
            plot_df["Series"] = plot_df["Series"].map(label_map)

            fig = px.bar(
                plot_df,
                x="Month",
                y="Energy (kWh)",
                color="Series",
                barmode="group",
                text_auto=".2f",
                color_discrete_map={
                    "Invoice Consumption": COLORS["blue"],
                    "Modeled Consumption (Exact Billing Period)": COLORS["orange"],
                },
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Energy (kWh)",
                legend_title="Series",
            )
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.2f} kWh<extra></extra>"
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"overview_compare_{selected_year}",
            )

            fig_cov = px.line(
                comparison_df,
                x="Month",
                y="Coverage (%)",
                markers=True,
            )
            fig_cov.update_traces(
                line=dict(color=COLORS["purple"], width=3),
                marker=dict(size=9, color=COLORS["purple"]),
                hovertemplate="<b>%{x}</b><br>Coverage: %{y:.2f}%<extra></extra>",
            )
            fig_cov.update_layout(
                xaxis_title="Month",
                yaxis_title="Coverage (%)",
                showlegend=False,
            )
            st.plotly_chart(
                fig_cov,
                use_container_width=True,
                key=f"overview_coverage_{selected_year}",
            )

            display_cols = [
                "Month",
                "Invoice Consumption (kWh)",
                "Modeled Consumption Billing Period (kWh)",
                "Difference (kWh)",
                "Coverage (%)",
            ]

            st.markdown("### Exact comparison table")
            st.dataframe(
                comparison_df[display_cols].style.format(
                    {
                        "Invoice Consumption (kWh)": "{:,.2f}",
                        "Modeled Consumption Billing Period (kWh)": "{:,.2f}",
                        "Difference (kWh)": "{:,.2f}",
                        "Coverage (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "Comparison is based on the exact invoice billing dates. "
                "Invoice consumption is read exactly from the invoice data. "
                "Modeled consumption is computed by summing the modeled kWh over the exact billing interval. "
                "The displayed error metric is the lowest absolute difference across invoice months."
            )

        else:
            st.metric("Total Invoice Consumption", f"{total_invoice_kwh:,.2f} kWh")
            st.warning(
                "Invoice data is available, but the baseline time series could not be loaded. "
                "Exact invoice-to-model comparison requires the baseline load time series in kW."
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

        fig_monthly = px.bar(
            monthly_plot_df,
            x="Month",
            y="Energy (kWh)",
            text_auto=".2f",
            color_discrete_sequence=[COLORS["teal"]],
        )
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            showlegend=False,
        )
        fig_monthly.update_traces(
            hovertemplate="<b>%{x}</b><br>Modeled consumption: %{y:,.2f} kWh<extra></extra>"
        )
        st.plotly_chart(
            fig_monthly,
            use_container_width=True,
            key=f"overview_monthly_modeled_{selected_year}",
        )


# =========================================================
# TAB 2 - DAILY PROFILE
# =========================================================

with tab2:
    st.subheader(f"Daily Profile - {selected_year}")

    if not baseline_loaded:
        st.warning(
            "Daily profile is not available because the baseline time series could not be loaded."
        )
        st.code(f"Baseline load error: {baseline_error}")
    else:
        fig_daily = px.bar(
            x=daily_energy_kwh.index,
            y=daily_energy_kwh.values,
            labels={"x": "Day", "y": "Daily Consumption (kWh)"},
        )
        fig_daily.update_traces(
            marker_color=COLORS["blue"],
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Daily consumption: %{y:,.2f} kWh<extra></extra>",
        )
        fig_daily.update_layout(
            xaxis_title="Day",
            yaxis_title="Daily Consumption (kWh)",
            showlegend=False,
        )
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
                    line=dict(color=COLORS["red"], width=3),
                    marker=dict(size=8, color=COLORS["red"]),
                    hovertemplate="<b>%{x}</b><br>Load: %{y:,.2f} kW<extra></extra>",
                )
                fig_hour.update_layout(
                    xaxis_title="Hour",
                    yaxis_title="Load (kW)",
                    showlegend=False,
                )
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
                    text_auto=".2f",
                )
                fig_hour.update_traces(
                    marker_color=COLORS["orange"],
                    hovertemplate="<b>%{x}</b><br>Consumption: %{y:,.2f} kWh<extra></extra>",
                )
                fig_hour.update_layout(
                    xaxis_title="Hour",
                    yaxis_title="Consumption (kWh)",
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_hour,
                    use_container_width=True,
                    key=f"selected_day_energy_{selected_year}_{selected_day_ts.strftime('%Y%m%d')}",
                )


# =========================================================
# TAB 3 - CATEGORIES
# =========================================================

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
                hovertemplate="<b>%{label}</b><br>Energy: %{value:,.2f} kWh<br>Share: %{percent}<extra></extra>",
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
                text_auto=".2f",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_rank.update_layout(
                xaxis_title="Energy (kWh)",
                yaxis_title="Category",
                showlegend=False,
            )
            fig_rank.update_traces(
                hovertemplate="<b>%{y}</b><br>Energy: %{x:,.2f} kWh<extra></extra>"
            )
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


# =========================================================
# TAB 4 - BILLING COMPARISON
# =========================================================

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
                - **VAT**: Value Added Tax. “Total Cost ex VAT” means total cost excluding VAT.

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
            st.markdown("### Exact invoice vs modeled comparison")

            lowest_error_row = get_lowest_error_row(comparison_df)
            st.metric(
                "Lowest Absolute Error",
                f"{lowest_error_row['Absolute Difference (kWh)']:,.2f} kWh",
                help=f"Observed in {lowest_error_row['Month']}. Calculated as the absolute difference between invoice consumption and modeled billing-period consumption.",
            )
            st.caption(f"Month with lowest absolute error: {lowest_error_row['Month']}")

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
                        "Difference (kWh)",
                        "Coverage (%)",
                    ]
                ],
                ["Issue Date", "Billing Start", "Billing End"],
            )

            st.dataframe(
                comparison_display.style.format(
                    {
                        "Invoice Consumption (kWh)": "{:,.2f}",
                        "Modeled Consumption Billing Period (kWh)": "{:,.2f}",
                        "Difference (kWh)": "{:,.2f}",
                        "Coverage (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            plot_df = comparison_df.melt(
                id_vars=["Month"],
                value_vars=[
                    "Invoice Consumption (kWh)",
                    "Modeled Consumption Billing Period (kWh)",
                ],
                var_name="Series",
                value_name="Energy (kWh)",
            )

            label_map = {
                "Invoice Consumption (kWh)": "Invoice Consumption",
                "Modeled Consumption Billing Period (kWh)": "Modeled Consumption (Exact Billing Period)",
            }
            plot_df["Series"] = plot_df["Series"].map(label_map)

            fig_compare = px.bar(
                plot_df,
                x="Month",
                y="Energy (kWh)",
                color="Series",
                barmode="group",
                text_auto=".2f",
                color_discrete_map={
                    "Invoice Consumption": COLORS["blue"],
                    "Modeled Consumption (Exact Billing Period)": COLORS["orange"],
                },
            )
            fig_compare.update_layout(
                xaxis_title="Month",
                yaxis_title="Energy (kWh)",
                legend_title="Series",
            )
            fig_compare.update_traces(
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.2f} kWh<extra></extra>"
            )
            st.plotly_chart(
                fig_compare,
                use_container_width=True,
                key=f"billing_compare_chart_{selected_year}",
            )

            st.markdown("### Selected invoice month detail")

            selected_month = st.selectbox(
                "Select invoice month",
                options=comparison_df["Month"].tolist(),
                index=0,
                key="selected_billing_month_detail",
            )

            selected_row = comparison_df[comparison_df["Month"] == selected_month].iloc[0]

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Invoice Number", selected_row["Invoice Number"])
            d2.metric(
                "Billing Period",
                f"{selected_row['Billing Start'].strftime('%d/%m/%Y')} - {selected_row['Billing End'].strftime('%d/%m/%Y')}",
            )
            d3.metric("Invoice Consumption", f"{selected_row['Invoice Consumption (kWh)']:,.2f} kWh")
            d4.metric(
                "Modeled Consumption",
                f"{selected_row['Modeled Consumption Billing Period (kWh)']:,.2f} kWh",
            )

            selected_components = pd.DataFrame(
                {
                    "Tariff Period": ["HSV", "HVN", "HP", "HC"],
                    "Energy (kWh)": [
                        selected_row["HSV (kWh)"],
                        selected_row["HVN (kWh)"],
                        selected_row["HP (kWh)"],
                        selected_row["HC (kWh)"],
                    ],
                }
            )

            fig_components = px.bar(
                selected_components,
                x="Tariff Period",
                y="Energy (kWh)",
                color="Tariff Period",
                text_auto=".2f",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_components.update_layout(
                xaxis_title="Tariff Period",
                yaxis_title="Energy (kWh)",
                showlegend=False,
            )
            fig_components.update_traces(
                hovertemplate="<b>%{x}</b><br>Energy: %{y:,.2f} kWh<extra></extra>"
            )
            st.plotly_chart(
                fig_components,
                use_container_width=True,
                key=f"billing_components_{selected_year}_{selected_month}",
            )

            summary_table = pd.DataFrame(
                {
                    "Field": [
                        "Supplier",
                        "Invoice Number",
                        "Issue Date",
                        "Billing Start",
                        "Billing End",
                        "Billing Days",
                        "Invoice Consumption (kWh)",
                        "Modeled Consumption Billing Period (kWh)",
                        "Difference (kWh)",
                        "Coverage (%)",
                        "Reactive Energy (kvarh)",
                        "PHP (kW)",
                        "PC (kW)",
                        "Tax (EUR)",
                        "Regulation Band (EUR)",
                        "MIBEL (EUR)",
                        "Total Cost ex VAT (EUR)",
                    ],
                    "Value": [
                        selected_row["Supplier"],
                        selected_row["Invoice Number"],
                        selected_row["Issue Date"].strftime("%d/%m/%Y"),
                        selected_row["Billing Start"].strftime("%d/%m/%Y"),
                        selected_row["Billing End"].strftime("%d/%m/%Y"),
                        f"{selected_row['Billing Days']}",
                        f"{selected_row['Invoice Consumption (kWh)']:,.2f}",
                        f"{selected_row['Modeled Consumption Billing Period (kWh)']:,.2f}",
                        f"{selected_row['Difference (kWh)']:,.2f}",
                        f"{selected_row['Coverage (%)']:.2f}",
                        f"{selected_row['Reactive Energy (kvarh)']:,.2f}",
                        f"{selected_row['PHP (kW)']:,.2f}",
                        f"{selected_row['PC (kW)']:,.2f}",
                        f"{selected_row['Tax (EUR)']:,.2f}",
                        f"{selected_row['Regulation Band (EUR)']:,.2f}",
                        f"{selected_row['MIBEL (EUR)']:,.2f}",
                        f"{selected_row['Total Cost ex VAT (EUR)']:,.2f}",
                    ],
                }
            )

            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True,
            )

            st.success(
                "This comparison uses exact invoice billing dates and computes modeled consumption "
                "from the modeled time series over those exact dates. "
                "Units are consistent: power in kW and energy in kWh."
            )

        else:
            st.warning(
                "Invoice data is available, but modeled billing-period comparison is not available "
                "because the baseline time series could not be loaded."
            )
            st.code(f"Baseline load error: {baseline_error}")