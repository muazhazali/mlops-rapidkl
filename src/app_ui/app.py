from __future__ import annotations

import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yaml
from dash import Input, Output, callback, dcc, html

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

from app_ui.utils import (
    LINE_COLORS,
    LINE_MAP,
    build_station_options,
    compute_mape,
    create_dow_pattern,
    create_main_chart,
    create_top_destinations,
    get_holiday_flag,
    get_station_line,
    load_parquet,
)

with open(project_root / "conf" / "base" / "parameters.yml") as f:
    config = yaml.safe_load(f)["ui"]

ACTUAL_PATH = project_root / config["actual_data_path"]
PRED_PATH = project_root / config["predictions_path"]
OD_PATH = project_root / config["od_data_path"]
STATION_DAILY_PATH = project_root / config["station_daily_path"]

# Build initial station options from station_daily if available
_sd = load_parquet(STATION_DAILY_PATH)
STATION_OPTIONS = build_station_options(_sd) if _sd is not None else {}
ALL_LINES = list(LINE_MAP.values())
DEFAULT_LINE = ALL_LINES[0] if ALL_LINES else None
DEFAULT_STATION = (STATION_OPTIONS.get(DEFAULT_LINE) or [{}])[0].get("value", "")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="KL Rapid Rail — Ridership Forecast",
)

LINE_BADGES = [
    html.Span(
        line,
        style={
            "background": color,
            "color": "#fff",
            "borderRadius": "4px",
            "padding": "2px 7px",
            "fontSize": "11px",
            "fontWeight": "600",
            "marginRight": "4px",
            "marginBottom": "4px",
            "display": "inline-block",
        },
    )
    for line, color in LINE_COLORS.items()
]

app.layout = dbc.Container(
    [
        dcc.Interval(id="interval", interval=config["update_interval_ms"], n_intervals=0),
        dbc.Row(
            [
                # ── Sidebar ──────────────────────────────────────────────
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5("Control panel", className="sidebar-title"),
                                html.Label("Line", className="ctrl-label"),
                                dcc.Dropdown(
                                    id="line-select",
                                    options=[{"label": l, "value": l} for l in ALL_LINES],
                                    value=DEFAULT_LINE,
                                    clearable=False,
                                    className="mb-2",
                                ),
                                html.Label("Station", className="ctrl-label"),
                                dcc.Dropdown(
                                    id="station-select",
                                    options=STATION_OPTIONS.get(DEFAULT_LINE, []),
                                    value=DEFAULT_STATION,
                                    clearable=False,
                                    className="mb-3",
                                ),
                                html.Label("Display window (last N days)", className="ctrl-label"),
                                dbc.InputGroup(
                                    [
                                        dbc.Button("−", id="lb-dec", n_clicks=0, color="light", size="sm"),
                                        dbc.Input(
                                            id="lookback-days",
                                            type="number",
                                            min=7,
                                            step=1,
                                            value=config["default_lookback_days"],
                                            style={"textAlign": "center"},
                                        ),
                                        dbc.Button("+", id="lb-inc", n_clicks=0, color="light", size="sm"),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                            className="sidebar-card",
                        ),
                        html.Div(
                            [
                                html.P("App overview", className="sidebar-section-title"),
                                html.Ul(
                                    [
                                        html.Li("Forecasts next-day station departures"),
                                        html.Li("CatBoost model trained on 3 years of OD data"),
                                        html.Li("Features: lags, holidays, weather, day-of-week"),
                                        html.Li("Inference replays 1 day per second"),
                                        html.Li("Docker: train + inference + UI containers"),
                                    ],
                                    className="overview-list",
                                ),
                                html.P("Lines", className="sidebar-section-title mt-2"),
                                html.Div(LINE_BADGES, style={"flexWrap": "wrap", "display": "flex"}),
                            ],
                            className="sidebar-card mt-2",
                        ),
                    ],
                    width=3,
                    style={"paddingTop": "10px"},
                ),
                # ── Main panel ───────────────────────────────────────────
                dbc.Col(
                    [
                        # Header row
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H4(
                                        "KL Rapid Rail — daily ridership forecast",
                                        style={"margin": 0, "fontWeight": "700", "fontSize": "18px"},
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Span(id="last-updated", style={"color": "#888", "fontSize": "13px"}),
                                    width="auto",
                                    className="d-flex align-items-center ms-3",
                                ),
                                dbc.Col(
                                    html.Span(
                                        "● Replaying",
                                        style={"color": "#2A9D8F", "fontSize": "13px", "fontWeight": "600"},
                                    ),
                                    width="auto",
                                    className="d-flex align-items-center ms-auto",
                                ),
                            ],
                            className="mb-3 align-items-center",
                        ),
                        # Metric cards
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="card-actual"), width=3),
                                dbc.Col(html.Div(id="card-predicted"), width=3),
                                dbc.Col(html.Div(id="card-mape"), width=3),
                                dbc.Col(html.Div(id="card-holiday"), width=3),
                            ],
                            className="mb-3 g-2",
                        ),
                        # Main chart
                        dbc.Card(
                            dcc.Graph(id="main-chart", config={"displayModeBar": False}),
                            className="chart-card mb-3",
                        ),
                        # Bottom panels
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dcc.Graph(id="top-dest-chart", config={"displayModeBar": False}),
                                        className="chart-card",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dcc.Graph(id="dow-chart", config={"displayModeBar": False}),
                                        className="chart-card",
                                    ),
                                    width=6,
                                ),
                            ],
                            className="g-2",
                        ),
                    ],
                    width=9,
                    style={"paddingTop": "10px"},
                ),
            ],
            align="start",
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#F5F5F7", "minHeight": "100vh", "padding": "16px"},
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("station-select", "options"),
    Output("station-select", "value"),
    Input("line-select", "value"),
)
def update_station_options(line: str):
    sd = load_parquet(STATION_DAILY_PATH)
    opts = build_station_options(sd).get(line, []) if sd is not None else []
    value = opts[0]["value"] if opts else None
    return opts, value


@callback(
    Output("lookback-days", "value"),
    Input("lb-dec", "n_clicks"),
    Input("lb-inc", "n_clicks"),
    Input("lookback-days", "value"),
    prevent_initial_call=True,
)
def adjust_lookback(dec, inc, current):
    from dash import ctx
    val = current or config["default_lookback_days"]
    if ctx.triggered_id == "lb-dec":
        return max(7, val - 1)
    if ctx.triggered_id == "lb-inc":
        return val + 1
    return val


@callback(
    Output("main-chart", "figure"),
    Output("top-dest-chart", "figure"),
    Output("dow-chart", "figure"),
    Output("card-actual", "children"),
    Output("card-predicted", "children"),
    Output("card-mape", "children"),
    Output("card-holiday", "children"),
    Output("last-updated", "children"),
    Input("station-select", "value"),
    Input("lookback-days", "value"),
    Input("interval", "n_intervals"),
)
def update_dashboard(station, lookback_days, _):
    import datetime

    if not station:
        empty = go.Figure()
        return empty, empty, empty, "", "", "", "", ""


    actuals = load_parquet(ACTUAL_PATH)
    predictions = load_parquet(PRED_PATH)
    od_daily = load_parquet(OD_PATH)
    station_daily = load_parquet(STATION_DAILY_PATH)

    lb = lookback_days or config["default_lookback_days"]

    # Main chart
    main_fig = create_main_chart(actuals, predictions, station, lb)

    # Bottom charts
    top_dest_fig = create_top_destinations(od_daily, station, lb)
    dow_fig = create_dow_pattern(station_daily, station, lb)

    # Metric: today's actual
    actual_val = "—"
    actual_sub = ""
    if actuals is not None:
        a = actuals[actuals["station"] == station]
        if not a.empty:
            latest = a.sort_values("date").iloc[-1]
            actual_val = f"{int(latest['ridership']):,}"
            station_label = station.split(":", 1)[-1].strip()
            actual_sub = station_label

    # Metric: predicted
    pred_val = "—"
    pred_sub = ""
    if predictions is not None:
        p = predictions[predictions["station"] == station]
        if not p.empty and actuals is not None:
            a = actuals[actuals["station"] == station]
            latest_pred = p.sort_values("date").iloc[-1]
            pred_val = f"{int(latest_pred['prediction']):,}"
            if not a.empty:
                actual_latest = a.sort_values("date").iloc[-1]["ridership"]
                diff_pct = (latest_pred["prediction"] - actual_latest) / (actual_latest + 1) * 100
                sign = "+" if diff_pct >= 0 else ""
                color = "#E63946" if diff_pct < 0 else "#2A9D8F"
                pred_sub = html.Span(f"{sign}{diff_pct:.1f}% vs actual", style={"color": color, "fontSize": "12px"})

    # Metric: MAPE
    mape_val = "—"
    mape_sub = ""
    mape = compute_mape(actuals, predictions, station)
    if mape is not None:
        mape_val = f"{mape:.1f}%"
        mape_sub = html.Span(
            "Within target" if mape < 10 else "Above target",
            style={"color": "#2A9D8F" if mape < 10 else "#E63946", "fontSize": "12px"},
        )

    # Metric: holiday flag
    holiday_name = "—"
    holiday_sub = ""
    flag = get_holiday_flag(predictions, station)
    if flag:
        h_name, h_impact = flag
        holiday_name = h_name
        holiday_sub = html.Span(
            f"Demand {h_impact}% exp.",
            style={"color": "#E63946", "fontWeight": "600", "fontSize": "12px"},
        )

    def metric_card(title: str, value, subtitle="", value_style=None):
        return html.Div(
            [
                html.Div(title, style={"fontSize": "12px", "color": "#888", "marginBottom": "2px"}),
                html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "lineHeight": "1.2", **(value_style or {})}),
                html.Div(subtitle, style={"fontSize": "12px", "marginTop": "2px", "color": "#555"}),
            ],
            className="metric-card",
        )

    card_actual = metric_card("Today's actual", actual_val, actual_sub)
    card_predicted = metric_card("Predicted", pred_val, pred_sub)
    card_mape = metric_card("30-day MAPE", mape_val, mape_sub)
    card_holiday = metric_card("Holiday flag", holiday_name, holiday_sub)

    now_str = datetime.datetime.now().strftime("Last updated: %d %b %Y")

    return main_fig, top_dest_fig, dow_fig, card_actual, card_predicted, card_mape, card_holiday, now_str


server = app.server

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8050)
