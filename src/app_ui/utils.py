from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

LINE_MAP: dict[str, str] = {
    "KJ": "KJ LRT",
    "AG": "AG LRT",
    "SP": "Kajang MRT",
    "PYL": "Putrajaya MRT",
    "MR": "Monorail",
}

LINE_COLORS: dict[str, str] = {
    "KJ LRT": "#E63946",
    "AG LRT": "#F4A261",
    "Kajang MRT": "#2A9D8F",
    "Putrajaya MRT": "#6A4C93",
    "Monorail": "#E9C46A",
}


def get_station_line(station_code: str) -> str:
    """Extract line from station code prefix, e.g. 'KJ14: Pasar Seni' → 'KJ LRT'."""
    code = station_code.split(":")[0].strip()
    for prefix, line in sorted(LINE_MAP.items(), key=lambda x: -len(x[0])):
        if code.upper().startswith(prefix):
            return line
    return "Other"


def build_station_options(station_daily: pd.DataFrame) -> dict[str, list[dict]]:
    """Return {line: [{label, value}, ...]} grouped by line."""
    stations = sorted(station_daily["station"].unique())
    grouped: dict[str, list[dict]] = {}
    for s in stations:
        line = get_station_line(s)
        grouped.setdefault(line, []).append({"label": s.split(":", 1)[-1].strip(), "value": s})
    return grouped


def load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_mape(actuals: pd.DataFrame, predictions: pd.DataFrame, station: str, days: int = 30) -> float | None:
    """30-day rolling MAPE for selected station."""
    if actuals is None or predictions is None:
        return None
    a = actuals[actuals["station"] == station].copy()
    p = predictions[predictions["station"] == station].copy()
    merged = pd.merge(a, p, on=["station", "date"], how="inner")
    if merged.empty:
        return None
    merged = merged.sort_values("date").tail(days)
    mape = (abs(merged["ridership"] - merged["prediction"]) / (merged["ridership"] + 1)).mean() * 100
    return round(float(mape), 1)


def get_holiday_flag(predictions: pd.DataFrame, station: str) -> tuple[str, int] | None:
    """Return (holiday_name, impact_pct) for the next predicted date if it's a holiday."""
    from mlops.pipelines.training.nodes import HOLIDAY_IMPACT, MY_HOLIDAYS
    if predictions is None or predictions.empty:
        return None
    p = predictions[predictions["station"] == station]
    if p.empty:
        return None
    next_date = p["date"].max()
    date_str = str(next_date.date())
    if date_str in MY_HOLIDAYS:
        name = MY_HOLIDAYS[date_str]
        impact = HOLIDAY_IMPACT.get(name, -15)
        return name, impact
    return None


def create_main_chart(
    actuals: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    station: str,
    lookback_days: int,
) -> go.Figure:
    fig = go.Figure()

    if actuals is not None:
        a = actuals[actuals["station"] == station].sort_values("date")
        now = a["date"].max() if not a.empty else None
        min_date = now - pd.Timedelta(days=lookback_days) if now else None
        a_f = a[a["date"] >= min_date] if min_date is not None else a

        fig.add_trace(go.Scatter(
            x=a_f["date"], y=a_f["ridership"],
            name="Actual", mode="lines+markers",
            line=dict(color="#E63946", width=2),
            marker=dict(symbol="circle", size=5),
        ))

        if now:
            fig.add_vline(
                x=str(now), line_width=1.5, line_dash="dash", line_color="#888",
                annotation_text="now", annotation_position="top right",
                annotation_font=dict(color="#888", size=11),
            )
    else:
        now = None

    if predictions is not None:
        p = predictions[predictions["station"] == station].sort_values("date")
        min_date_p = (now - pd.Timedelta(days=lookback_days)) if now else None
        p_f = p[p["date"] >= min_date_p] if min_date_p else p

        p_hist = p_f[p_f["date"] <= now] if now else p_f
        p_future = p_f[p_f["date"] > now] if now else pd.DataFrame()

        if not p_hist.empty:
            fig.add_trace(go.Scatter(
                x=p_hist["date"], y=p_hist["prediction"],
                name="Predicted", mode="lines+markers",
                line=dict(color="#2A9D8F", width=2),
                marker=dict(size=5),
            ))

        if not p_future.empty:
            fig.add_trace(go.Scatter(
                x=p_future["date"], y=p_future["prediction"],
                name="Forecast", mode="lines+markers",
                line=dict(color="#2A9D8F", width=2, dash="dash"),
                marker=dict(size=5),
                showlegend=False,
            ))

    station_label = station.split(":", 1)[-1].strip() if ":" in station else station
    fig.update_layout(
        template="plotly_white",
        height=320,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Daily Departures",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#ccc"),
        title=dict(text=f"Predicted vs actual departures — {station_label}", font=dict(size=13)),
    )
    return fig


def create_top_destinations(od_daily: pd.DataFrame | None, station: str, lookback_days: int = 30) -> go.Figure:
    fig = go.Figure()
    if od_daily is None or od_daily.empty:
        fig.update_layout(template="plotly_white", height=220, title="Top destination stations")
        return fig

    cutoff = od_daily["date"].max() - pd.Timedelta(days=lookback_days)
    filtered = od_daily[
        (od_daily["origin"] == station) & (od_daily["date"] >= cutoff)
    ]
    top = (
        filtered.groupby("destination")["ridership"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top["dest_label"] = top["destination"].apply(lambda x: x.split(":", 1)[-1].strip() if ":" in x else x)

    fig.add_trace(go.Bar(
        x=top["ridership"],
        y=top["dest_label"],
        orientation="h",
        marker_color="#2A9D8F",
        text=top["ridership"].apply(lambda x: f"{x:,}"),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=10, r=60, t=35, b=10),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
        title=dict(text="Top destination stations", font=dict(size=13)),
        bargap=0.3,
    )
    return fig


def create_dow_pattern(station_daily: pd.DataFrame | None, station: str, lookback_days: int = 30) -> go.Figure:
    fig = go.Figure()
    if station_daily is None or station_daily.empty:
        fig.update_layout(template="plotly_white", height=220, title="Day-of-week pattern (30d avg)")
        return fig

    df = station_daily[station_daily["station"] == station].copy()
    if df.empty:
        fig.update_layout(template="plotly_white", height=220, title="Day-of-week pattern (30d avg)")
        return fig

    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]
    df["dow"] = df["date"].dt.dayofweek

    dow_avg = df.groupby("dow")["ridership"].mean().reindex(range(7), fill_value=0)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#2A9D8F" if i < 5 else "#B0B8C1" for i in range(7)]

    fig.add_trace(go.Bar(
        x=labels,
        y=dow_avg.values,
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_white",
        height=220,
        margin=dict(l=40, r=10, t=35, b=20),
        yaxis=dict(showticklabels=False, showgrid=False),
        title=dict(text="Day-of-week pattern (30d avg)", font=dict(size=13)),
        bargap=0.25,
    )
    return fig
