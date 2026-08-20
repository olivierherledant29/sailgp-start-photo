from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from influx_io import ALL_BOATS, get_cfg, load_channels_timeseries


REF_BOAT = "FRA"

CH_BSP = "BOAT_SPEED_km_h_1"
CH_TWA = "TWA_MHU_SGP_deg"
CH_VMG = "VMG_km_h_1"
CH_TARGET_VMG = "TARG_VMG_km_h_1"
CH_JIB_SHEET_LOAD = "LOAD_JIB_SHEET_kgf"
CH_JIB_SHEET_PCT = "PER_JIB_SHEET_pct"

CHANNELS = [
    CH_BSP,
    CH_TWA,
    CH_VMG,
    CH_TARGET_VMG,
    CH_JIB_SHEET_LOAD,
    CH_JIB_SHEET_PCT,
]

RACE_DAYS = [
    ("perth_2026_01_17", "Australia/Perth", "2026-01-17"),
    ("perth_2026_01_18", "Australia/Perth", "2026-01-18"),
    ("auckland_2026_02_14", "Pacific/Auckland", "2026-02-14"),
    ("rio_2026_04_11", "America/Sao_Paulo", "2026-04-11"),
    ("rio_2026_04_12", "America/Sao_Paulo", "2026-04-12"),
    ("bermuda_2026_05_09", "Atlantic/Bermuda", "2026-05-09"),
    ("bermuda_2026_05_10", "Atlantic/Bermuda", "2026-05-10"),
    ("new_york_2026_05_31", "America/New_York", "2026-05-31"),
    ("halifax_2026_06_20", "America/Halifax", "2026-06-20"),
    ("halifax_2026_06_21", "America/Halifax", "2026-06-21"),
]

FULL_DAY_EVENTS = {
    "perth_2026_01_17",
    "perth_2026_01_18",
    "bermuda_2026_05_09",
    "bermuda_2026_05_10",
}

TEAM_COLORS = {
    "FRA": "#0064FF",
    "ESP": "#D62728",
    "AUS": "#7CFC00",
    "GBR": "#7A3DB8",
    "NZL": "#333333",
    "SUI": "#BDBDBD",
    "USA": "#00B8D9",
    "SWE": "#C49A6C",
    "GER": "#FFD400",
    "DEN": "#E6A8A8",
    "CAN": "#D8BFD8",
    "ITA": "#A8DADC",
    "BRA": "#B7E4C7",
}

FALLBACK_COLOR = "#D0C7B8"

TEAM_ORDER = [
    "AUS",
    "BRA",
    "CAN",
    "DEN",
    "ESP",
    "FRA",
    "GBR",
    "GER",
    "ITA",
    "NZL",
    "SUI",
    "SWE",
    "USA",
]


def local_window_to_utc(event_name: str, day: str, tz_name: str):
    tz = ZoneInfo(tz_name)

    if event_name in FULL_DAY_EVENTS:
        start_local = datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=tz)
        stop_local = datetime.fromisoformat(f"{day}T23:59:59").replace(tzinfo=tz)
    else:
        start_local = datetime.fromisoformat(f"{day}T12:00:00").replace(tzinfo=tz)
        stop_local = datetime.fromisoformat(f"{day}T18:00:00").replace(tzinfo=tz)

    return (
        start_local.astimezone(ZoneInfo("UTC")),
        stop_local.astimezone(ZoneInfo("UTC")),
        start_local,
        stop_local,
    )


def prepare_df(df: pd.DataFrame, event_name: str) -> pd.DataFrame:
    out = df.copy()

    for col in CHANNELS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["target_vmg_pct"] = np.where(
            np.abs(out[CH_TARGET_VMG]) > 1e-9,
            100.0 * out[CH_VMG] / out[CH_TARGET_VMG],
            np.nan,
        )

    out = out[
        (out["target_vmg_pct"] > 60.0)
        & (out[CH_JIB_SHEET_LOAD] > 1200.0)
    ].copy()

    out["event_day"] = event_name

    return out.rename(
        columns={
            CH_BSP: "BSP",
            CH_TWA: "TWA",
            CH_JIB_SHEET_LOAD: "jib_sheet_load",
            CH_JIB_SHEET_PCT: "jib_sheet_per",
        }
    )


def make_legend_annotation(boats: list[str]) -> str:
    parts = []
    for boat in boats:
        color = TEAM_COLORS.get(boat, FALLBACK_COLOR)
        parts.append(
            f'<span style="color:{color};font-weight:700;">●</span> {boat}'
        )
    return " &nbsp; ".join(parts)


def add_day_traces(fig, df: pd.DataFrame, event_name: str, row: int):
    if df.empty:
        fig.add_annotation(
            text=f"{event_name}: aucune donnée filtrée",
            x=50,
            y=1950,
            showarrow=False,
            row=row,
            col=1,
        )
        return

    boats = sorted(df["boat"].dropna().astype(str).unique())

    fig.add_annotation(
        text=make_legend_annotation(boats),
        xref=f"x{row} domain" if row > 1 else "x domain",
        yref=f"y{row} domain" if row > 1 else "y domain",
        x=0.01,
        y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(180,180,180,0.6)",
        borderwidth=1,
        font=dict(size=11),
        row=row,
        col=1,
    )

    for boat in boats:
        d = df[df["boat"].astype(str) == boat].copy()
        if d.empty:
            continue

        fig.add_trace(
            go.Scattergl(
                x=d["jib_sheet_per"],
                y=d["jib_sheet_load"],
                mode="markers",
                name=boat,
                legendgroup=boat,
                showlegend=(row == 1),
                marker=dict(
                    size=7,
                    color=TEAM_COLORS.get(boat, FALLBACK_COLOR),
                    opacity=0.85 if boat == REF_BOAT else 0.58,
                    line=dict(
                        width=0.7 if boat == REF_BOAT else 0,
                        color="black" if boat == REF_BOAT else TEAM_COLORS.get(boat, FALLBACK_COLOR),
                    ),
                ),
                hovertemplate=(
                    f"event={event_name}<br>"
                    "boat=%{customdata[0]}<br>"
                    "time=%{customdata[1]}<br>"
                    "BSP=%{customdata[2]:.1f}<br>"
                    "TWA=%{customdata[3]:.1f}<br>"
                    "target VMG=%{customdata[4]:.1f}%<br>"
                    "jib sheet %=%{x:.1f}<br>"
                    "jib sheet load=%{y:.0f} kgf<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        d["boat"].astype(str),
                        d["time_utc"].astype(str),
                        d["BSP"],
                        d["TWA"],
                        d["target_vmg_pct"],
                    ],
                    axis=1,
                ),
            ),
            row=row,
            col=1,
        )


def add_all_days_plot(fig, df_all: pd.DataFrame, row: int):
    if df_all.empty:
        fig.add_annotation(
            text="all_race_days: aucune donnée filtrée",
            x=50,
            y=1950,
            showarrow=False,
            row=row,
            col=1,
        )
        return

    boats = sorted(df_all["boat"].dropna().astype(str).unique())

    fig.add_annotation(
        text=make_legend_annotation(boats),
        xref=f"x{row} domain",
        yref=f"y{row} domain",
        x=0.01,
        y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(180,180,180,0.6)",
        borderwidth=1,
        font=dict(size=11),
        row=row,
        col=1,
    )

    for boat in boats:
        d = df_all[df_all["boat"].astype(str) == boat].copy()
        if d.empty:
            continue

        fig.add_trace(
            go.Scattergl(
                x=d["jib_sheet_per"],
                y=d["jib_sheet_load"],
                mode="markers",
                name=f"{boat} all days",
                legendgroup=boat,
                showlegend=False,
                marker=dict(
                    size=7,
                    color=TEAM_COLORS.get(boat, FALLBACK_COLOR),
                    opacity=0.60 if boat == REF_BOAT else 0.34,
                    line=dict(
                        width=0.7 if boat == REF_BOAT else 0,
                        color="black" if boat == REF_BOAT else TEAM_COLORS.get(boat, FALLBACK_COLOR),
                    ),
                ),
                hovertemplate=(
                    "event=%{customdata[0]}<br>"
                    "boat=%{customdata[1]}<br>"
                    "time=%{customdata[2]}<br>"
                    "BSP=%{customdata[3]:.1f}<br>"
                    "TWA=%{customdata[4]:.1f}<br>"
                    "target VMG=%{customdata[5]:.1f}%<br>"
                    "jib sheet %=%{x:.1f}<br>"
                    "jib sheet load=%{y:.0f} kgf<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        d["event_day"].astype(str),
                        d["boat"].astype(str),
                        d["time_utc"].astype(str),
                        d["BSP"],
                        d["TWA"],
                        d["target_vmg_pct"],
                    ],
                    axis=1,
                ),
            ),
            row=row,
            col=1,
        )


def make_combined_html(day_dfs: dict[str, pd.DataFrame], df_all: pd.DataFrame, out_html: Path):
    event_names = list(day_dfs.keys())
    subplot_titles = [
        f"{name} — jib sheet load vs jib sheet %"
        for name in event_names
    ] + ["ALL RACE DAYS — jib sheet load vs jib sheet %"]

    n = len(subplot_titles)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.025,
        subplot_titles=subplot_titles,
    )

    for i, event_name in enumerate(event_names, start=1):
        add_day_traces(fig, day_dfs[event_name], event_name, i)

        fig.update_yaxes(
            range=[1200, 2400],
            title_text="jib_sheet_load (kgf)",
            row=i,
            col=1,
        )
        fig.update_xaxes(
            title_text="jib_sheet_per (%)",
            row=i,
            col=1,
        )

    all_row = n
    add_all_days_plot(fig, df_all, all_row)
    fig.update_yaxes(
        range=[1200, 2400],
        title_text="jib_sheet_load (kgf)",
        row=all_row,
        col=1,
    )
    fig.update_xaxes(
        title_text="jib_sheet_per (%)",
        row=all_row,
        col=1,
    )

    fig.update_layout(
        title=(
            "Jib sheet load vs jib sheet percentage — race days<br>"
            "<sup>Filters: target VMG > 80%, jib_sheet_load > 1200 kgf. "
            "Perth and Bermuda queried over full local day 00:00–23:59.</sup>"
        ),
        height=max(800, 520 * n),
        template="plotly_white",
        legend_title_text="Team",
        hovermode="closest",
        margin=dict(l=80, r=40, t=120, b=60),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")



def _add_violin_trace(
    fig,
    *,
    row: int,
    col: int,
    team: str,
    values: pd.Series,
    metric_label: str,
    showlegend: bool,
):
    s = pd.to_numeric(values, errors="coerce").dropna()

    if s.empty:
        return

    color = TEAM_COLORS.get(team, FALLBACK_COLOR)

    fig.add_trace(
        go.Violin(
            x=[team] * len(s),
            y=s,
            name=team,
            legendgroup=team,
            showlegend=showlegend,
            fillcolor=color,
            line=dict(
                color=color,
                width=2.6 if team == REF_BOAT else 1.4,
            ),
            opacity=0.92 if team == REF_BOAT else 0.70,
            points=False,
            box_visible=True,
            meanline_visible=True,
            spanmode="hard",
            bandwidth=None,
            scalemode="width",
            width=0.85,
            hoveron="violins+kde",
            hovertemplate=(
                f"team={team}<br>"
                f"metric={metric_label}<br>"
                "value=%{y:.2f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def add_race_day_violins(
    fig,
    *,
    df: pd.DataFrame,
    event_name: str,
    row: int,
):
    """
    Add two violin panels for one race day:
    - left: jib_sheet_load by team
    - right: jib_sheet_per by team

    TEAM_ORDER is imposed on both axes, so teams without data retain an empty slot.
    """
    if df.empty:
        fig.add_annotation(
            text=f"{event_name}: aucune donnée filtrée",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=row,
            col=1,
        )
        fig.add_annotation(
            text=f"{event_name}: aucune donnée filtrée",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=row,
            col=2,
        )
        return

    available_teams = set(df["boat"].dropna().astype(str).unique())

    for team in TEAM_ORDER:
        if team not in available_teams:
            continue

        d_team = df[df["boat"].astype(str) == team]

        _add_violin_trace(
            fig,
            row=row,
            col=1,
            team=team,
            values=d_team["jib_sheet_load"],
            metric_label="jib_sheet_load",
            showlegend=(row == 1),
        )

        _add_violin_trace(
            fig,
            row=row,
            col=2,
            team=team,
            values=d_team["jib_sheet_per"],
            metric_label="jib_sheet_per",
            showlegend=False,
        )


def make_violins_html(
    day_dfs: dict[str, pd.DataFrame],
    out_html: Path,
):
    """
    Create a second HTML page with one row per race day.

    Left: violin plots of jib_sheet_load by team.
    Right: violin plots of jib_sheet_per by team.
    Width represents local data density.
    """
    event_names = list(day_dfs.keys())
    n_rows = len(event_names)

    subplot_titles = []
    for event_name in event_names:
        subplot_titles.extend(
            [
                f"{event_name} — jib sheet load density",
                f"{event_name} — jib sheet % density",
            ]
        )

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.065,
        vertical_spacing=max(0.015, min(0.04, 0.28 / max(n_rows, 1))),
        subplot_titles=subplot_titles,
    )

    for row, event_name in enumerate(event_names, start=1):
        add_race_day_violins(
            fig,
            df=day_dfs[event_name],
            event_name=event_name,
            row=row,
        )

        fig.update_xaxes(
            categoryorder="array",
            categoryarray=TEAM_ORDER,
            tickangle=-45,
            title_text="Team",
            row=row,
            col=1,
        )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=TEAM_ORDER,
            tickangle=-45,
            title_text="Team",
            row=row,
            col=2,
        )

        fig.update_yaxes(
            title_text="Jib sheet load (kgf)",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Jib sheet (%)",
            row=row,
            col=2,
        )

    fig.update_layout(
        title=(
            "Jib sheet density distributions by race day and team<br>"
            "<sup>"
            "Each row is one race day. "
            "Violin width represents local data density; internal boxes show quartiles and median; "
            "the mean line is also displayed. "
            "Empty team positions mean no filtered data. "
            "Filters: target VMG &gt; 60%, jib sheet load &gt; 1200 kgf."
            "</sup>"
        ),
        height=max(900, 500 * n_rows),
        template="plotly_white",
        violinmode="group",
        violingap=0.05,
        violingroupgap=0.02,
        legend_title_text="Team",
        margin=dict(l=75, r=35, t=120, b=70),
        hovermode="closest",
    )

    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
    )

def main():
    cfg = get_cfg()

    out_dir = ROOT / "outputs" / "jib_sheet_race_days"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_filtered = []
    day_dfs = {}

    boats = list(ALL_BOATS)
    if REF_BOAT not in boats:
        boats = [REF_BOAT] + boats

    for event_name, tz_name, day in RACE_DAYS:
        start_utc, stop_utc, start_local, stop_local = local_window_to_utc(
            event_name,
            day,
            tz_name,
        )

        print(f"\n=== {event_name} ===")
        print(f"Local: {start_local.strftime('%Y-%m-%d %H:%M:%S')} -> {stop_local.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}")
        print(f"UTC:   {start_utc.isoformat()} -> {stop_utc.isoformat()}")

        df_raw = load_channels_timeseries(
            cfg=cfg,
            boats=boats,
            channels=CHANNELS,
            start_utc=start_utc,
            stop_utc=stop_utc,
            every="1s",
            level_expr="strm|mdss|mdss_fast|raw",
            agg_fn="mean",
        )

        if df_raw.empty:
            print("Aucune donnée brute.")
            day_dfs[event_name] = pd.DataFrame()
            continue

        df_filtered = prepare_df(df_raw, event_name)

        if df_filtered.empty:
            print("Aucune donnée après filtres.")
            day_dfs[event_name] = pd.DataFrame()
            continue

        keep_cols = [
            "event_day",
            "boat",
            "time_utc",
            "BSP",
            "TWA",
            "target_vmg_pct",
            "jib_sheet_load",
            "jib_sheet_per",
        ]

        df_filtered = df_filtered[keep_cols].sort_values(["event_day", "boat", "time_utc"])

        csv_path = out_dir / f"{event_name}_filtered.csv"
        df_filtered.to_csv(csv_path, index=False)

        print(f"Points filtrés : {len(df_filtered):,}".replace(",", " "))
        print(f"CSV : {csv_path}")

        day_dfs[event_name] = df_filtered
        all_filtered.append(df_filtered)

    if all_filtered:
        df_all = pd.concat(all_filtered, ignore_index=True)

        all_csv = out_dir / "all_race_days_jib_sheet_filtered.csv"
        combined_html = out_dir / "all_race_days_jib_sheet_plots.html"
        violins_html = out_dir / "all_race_days_jib_sheet_violins.html"

        df_all.to_csv(all_csv, index=False)

        make_combined_html(
            day_dfs,
            df_all,
            combined_html,
        )

        make_violins_html(
            day_dfs,
            violins_html,
        )

        print(f"\nCSV global : {all_csv}")
        print(f"HTML scatter global : {combined_html}")
        print(f"HTML violins : {violins_html}")
        print(f"Total points : {len(df_all):,}".replace(",", " "))
    else:
        print("\nAucune donnée filtrée sur l'ensemble des journées.")


if __name__ == "__main__":
    main()