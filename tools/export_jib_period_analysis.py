from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, time, timedelta, timezone

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
CH_JIB_CUNNO_LOAD = "LOAD_JIB_CUNNO_kgf"

CHANNELS = [
    CH_BSP,
    CH_TWA,
    CH_VMG,
    CH_TARGET_VMG,
    CH_JIB_SHEET_LOAD,
    CH_JIB_SHEET_PCT,
    CH_JIB_CUNNO_LOAD,
]

TARGET_VMG_PCT_MIN = 60.0
JIB_SHEET_LOAD_MIN = 1200.0
JIB_CUNNO_LOAD_MIN = 1200.0

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
    "AUS", "BRA", "CAN", "DEN", "ESP", "FRA", "GBR",
    "GER", "ITA", "NZL", "SUI", "SWE", "USA",
]


def prompt_last_24_hours() -> bool:
    while True:
        raw = input("Data des 24 dernières heures ? [oui/non] : ").strip().lower()
        if raw in {"oui", "o", "yes", "y"}:
            return True
        if raw in {"non", "n", "no"}:
            return False
        print("Réponse attendue : oui ou non.")


def prompt_date() -> str:
    while True:
        raw = input("Date locale à analyser [DD-MM-YYYY] : ").strip()
        try:
            datetime.strptime(raw, "%d-%m-%Y")
            return raw
        except ValueError:
            print("Format invalide. Exemple : 21-06-2026")


def prompt_utc_offset() -> int:
    prompt = (
        "Décalage horaire par rapport à UTC, entier de -12 à +12 "
        "[défaut 0]. Repères : New York -5 en hiver / -4 en été ; "
        "Sydney +10 en hiver austral / +11 en été austral ; Dubai +4 : "
    )
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return 0
        try:
            offset = int(raw)
        except ValueError:
            print("Entrer un entier compris entre -12 et +12.")
            continue
        if -12 <= offset <= 12:
            return offset
        print("Le décalage doit être compris entre -12 et +12.")


def offset_label(offset_hours: int) -> str:
    return f"UTC{offset_hours:+d}" if offset_hours else "UTC"


def make_query_window(*, use_last_24_hours: bool, date_local: str | None, offset_hours: int):
    local_tz = timezone(timedelta(hours=offset_hours))
    utc_tz = timezone.utc
    if use_last_24_hours:
        stop_utc = datetime.now(utc_tz)
        start_utc = stop_utc - timedelta(hours=24)
        start_local = start_utc.astimezone(local_tz)
        stop_local = stop_utc.astimezone(local_tz)
        period_key = f"last_24h_{stop_utc.strftime('%Y-%m-%d_%H-%M-%S')}_UTC"
        period_title = "24 dernières heures"
        return start_utc, stop_utc, start_local, stop_local, period_key, period_title
    if date_local is None:
        raise ValueError("La date locale est requise hors mode 24 heures.")
    local_date = datetime.strptime(date_local, "%d-%m-%Y").date()
    start_local = datetime.combine(local_date, time(0, 0, 0), tzinfo=local_tz)
    stop_local = datetime.combine(local_date, time(23, 59, 59), tzinfo=local_tz)
    start_utc = start_local.astimezone(utc_tz)
    stop_utc = stop_local.astimezone(utc_tz)
    period_key = local_date.strftime("%Y-%m-%d")
    period_title = local_date.strftime("%d-%m-%Y")
    return start_utc, stop_utc, start_local, stop_local, period_key, period_title


def prepare_data(df_raw: pd.DataFrame):
    out = df_raw.copy()

    for col in CHANNELS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    missing = [col for col in CHANNELS if col not in out.columns]
    if missing:
        raise RuntimeError(
            "Channels manquants dans les données retournées : "
            + ", ".join(missing)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        out["target_vmg_pct"] = np.where(
            np.abs(out[CH_TARGET_VMG]) > 1e-9,
            100.0 * out[CH_VMG] / out[CH_TARGET_VMG],
            np.nan,
        )

    out = out.rename(
        columns={
            CH_BSP: "BSP",
            CH_TWA: "TWA",
            CH_JIB_SHEET_LOAD: "jib_sheet_load",
            CH_JIB_SHEET_PCT: "jib_sheet_per",
            CH_JIB_CUNNO_LOAD: "jib_cunno_load",
        }
    )

    common = out["target_vmg_pct"] > TARGET_VMG_PCT_MIN

    df_sheet = out[
        common & (out["jib_sheet_load"] > JIB_SHEET_LOAD_MIN)
    ].copy()

    df_cunno = out[
        common & (out["jib_cunno_load"] > JIB_CUNNO_LOAD_MIN)
    ].copy()

    return df_sheet, df_cunno


def add_sheet_scatter(fig, df_sheet: pd.DataFrame):
    if df_sheet.empty:
        fig.add_annotation(
            text="Aucune donnée jib sheet après filtres",
            x=0.5, y=0.5, showarrow=False, row=1, col=1,
        )
        return

    boats = sorted(df_sheet["boat"].dropna().astype(str).unique())

    for boat in boats:
        d = df_sheet[df_sheet["boat"].astype(str) == boat].copy()
        if d.empty:
            continue

        color = TEAM_COLORS.get(boat, FALLBACK_COLOR)

        fig.add_trace(
            go.Scatter(
                x=d["jib_sheet_per"],
                y=d["jib_sheet_load"],
                mode="markers",
                name=boat,
                legendgroup=boat,
                showlegend=True,
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.88 if boat == REF_BOAT else 0.58,
                    line=dict(
                        width=0.8 if boat == REF_BOAT else 0,
                        color="black" if boat == REF_BOAT else color,
                    ),
                ),
                customdata=np.stack(
                    [
                        d["time_utc"].astype(str),
                        d["BSP"],
                        d["TWA"],
                        d["target_vmg_pct"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    f"boat={boat}<br>"
                    "time=%{customdata[0]}<br>"
                    "BSP=%{customdata[1]:.1f}<br>"
                    "TWA=%{customdata[2]:.1f}<br>"
                    "target VMG=%{customdata[3]:.1f}%<br>"
                    "jib sheet %=%{x:.1f}<br>"
                    "jib sheet load=%{y:.0f} kgf"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def add_violin(fig, *, df, value_col, row, col, metric_label):
    if df.empty:
        fig.add_annotation(
            text=f"Aucune donnée {metric_label} après filtres",
            x=0.5, y=0.5, showarrow=False, row=row, col=col,
        )
        return

    available_teams = set(df["boat"].dropna().astype(str).unique())

    for team in TEAM_ORDER:
        if team not in available_teams:
            continue

        values = pd.to_numeric(
            df.loc[df["boat"].astype(str) == team, value_col],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        color = TEAM_COLORS.get(team, FALLBACK_COLOR)

        fig.add_trace(
            go.Violin(
                x=[team] * len(values),
                y=values,
                name=team,
                legendgroup=team,
                showlegend=False,
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
                scalemode="width",
                width=0.85,
                hoveron="violins+kde",
                hovertemplate=(
                    f"team={team}<br>"
                    f"{metric_label}=%{{y:.2f}}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def make_html(*, period_title, utc_offset, df_sheet, df_cunno, out_html):
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "Jib sheet load vs jib sheet %",
            "Jib sheet load density by team",
            "Jib sheet % density by team",
            "Jib cunno load density by team",
        ],
        vertical_spacing=0.085,
        horizontal_spacing=0.08,
        row_heights=[0.40, 0.30, 0.30],
    )

    add_sheet_scatter(fig, df_sheet)

    add_violin(
        fig,
        df=df_sheet,
        value_col="jib_sheet_load",
        row=2,
        col=1,
        metric_label="jib sheet load",
    )
    add_violin(
        fig,
        df=df_sheet,
        value_col="jib_sheet_per",
        row=2,
        col=2,
        metric_label="jib sheet %",
    )
    add_violin(
        fig,
        df=df_cunno,
        value_col="jib_cunno_load",
        row=3,
        col=1,
        metric_label="jib cunno load",
    )

    fig.update_xaxes(title_text="Jib sheet (%)", row=1, col=1)
    fig.update_yaxes(title_text="Jib sheet load (kgf)", row=1, col=1)

    for r, c in [(2, 1), (2, 2), (3, 1)]:
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=TEAM_ORDER,
            tickangle=-45,
            title_text="Team",
            row=r,
            col=c,
        )

    fig.update_yaxes(title_text="Jib sheet load (kgf)", row=2, col=1)
    fig.update_yaxes(title_text="Jib sheet (%)", row=2, col=2)
    fig.update_yaxes(title_text="Jib cunno load (kgf)", row=3, col=1)

    fig.update_layout(
        title=(
            f"Jib analysis — {period_title} — {offset_label(utc_offset)}<br>"
            "<sup>"
            f"Target VMG &gt; {TARGET_VMG_PCT_MIN:.0f}% · "
            f"Jib sheet load &gt; {JIB_SHEET_LOAD_MIN:.0f} kgf · "
            f"Jib cunno load &gt; {JIB_CUNNO_LOAD_MIN:.0f} kgf · "
            "time window shown in the console before loading."
            "</sup>"
        ),
        height=2100,
        template="plotly_white",
        violinmode="group",
        violingap=0.05,
        violingroupgap=0.02,
        legend_title_text="Team",
        hovermode="closest",
        margin=dict(l=80, r=40, t=130, b=70),
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def main():
    use_last_24_hours = prompt_last_24_hours()
    date_local = None
    if not use_last_24_hours:
        date_local = prompt_date()
    utc_offset = prompt_utc_offset()
    (
        start_utc,
        stop_utc,
        start_local,
        stop_local,
        period_key,
        period_title,
    ) = make_query_window(
        use_last_24_hours=use_last_24_hours,
        date_local=date_local,
        offset_hours=utc_offset,
    )
    print(
        f"\nPériode locale ({offset_label(utc_offset)}) : "
        f"{start_local.isoformat()} -> {stop_local.isoformat()}"
    )
    print(f"Période UTC : {start_utc.isoformat()} -> {stop_utc.isoformat()}")
    cfg = get_cfg()
    boats = list(ALL_BOATS)
    if REF_BOAT not in boats:
        boats = [REF_BOAT] + boats
    print("\nChargement des données Influx...")
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
        print("Aucune donnée brute pour cette période.")
        return
    df_sheet, df_cunno = prepare_data(df_raw)
    out_dir = ROOT / "outputs" / "jib_single_day_analysis" / period_key
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_csv = out_dir / f"{period_key}_jib_sheet_filtered.csv"
    cunno_csv = out_dir / f"{period_key}_jib_cunno_filtered.csv"
    html_path = out_dir / f"{period_key}_jib_sheet_cunno_analysis.html"
    sheet_cols = [
        "boat", "time_utc", "BSP", "TWA", "target_vmg_pct",
        "jib_sheet_load", "jib_sheet_per",
    ]
    cunno_cols = [
        "boat", "time_utc", "BSP", "TWA", "target_vmg_pct",
        "jib_cunno_load",
    ]
    df_sheet[sheet_cols].sort_values(["boat", "time_utc"]).to_csv(sheet_csv, index=False)
    df_cunno[cunno_cols].sort_values(["boat", "time_utc"]).to_csv(cunno_csv, index=False)
    make_html(
        period_title=period_title,
        utc_offset=utc_offset,
        df_sheet=df_sheet,
        df_cunno=df_cunno,
        out_html=html_path,
    )
    print(f"\nPoints jib sheet filtrés : {len(df_sheet):,}".replace(",", " "))
    print(f"Points jib cunno filtrés : {len(df_cunno):,}".replace(",", " "))
    print(f"CSV jib sheet : {sheet_csv}")
    print(f"CSV jib cunno : {cunno_csv}")
    print(f"HTML : {html_path}")


if __name__ == "__main__":
    main()
