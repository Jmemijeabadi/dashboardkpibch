from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet


# =========================
# Streamlit page setup
# =========================
st.set_page_config(page_title="CRMBI Comercial (Por EJE)", layout="wide")


# =========================
# CONFIG (your business logic)
# =========================
KEY_SPECIALTIES = [
    {"id": "ORTOPEDIA",    "match": ["ORTOPEDIA"]},
    {"id": "CARDIOLOGIA",  "match": ["CARDIO"]},
    {"id": "NEUROCIRUGIA", "match": ["NEURO"]},
    {"id": "CX TORAX",     "match": ["TORAX", "TÓRAX"]},
    {"id": "CX GENERAL",   "match": ["CIRUGIA GENERAL", "CIRUGÍA GENERAL", "CX GENERAL"]},
    {"id": "ONCOLOGIA",    "match": ["ONCO"]},
]

KPI_CONFIG: Dict[str, List[Dict[str, Any]]] = {
    "eje1": [
        {
            "id": "eje1_key_specialties_total",
            "indicador": "Atracción médicos nuevos",
            "estrategia": "Actividad por especialidades clave (suma de médicos en: Ortopedia, Cardio, Neuro, Tórax, CX General, Onco).",
            "unit": "médicos",
            "defaultTarget": 25,
            "actualSource": "key_specialties_sum",
        },
        {
            "id": "eje1_eventos_medicos",
            "indicador": "Atracción médicos nuevos",
            "estrategia": "Eventos con contacto 1:1 con médicos (interno/colegios).",
            "unit": "eventos",
            "defaultTarget": 2,
            "actualSource": "not_in_excel",
        },
        {
            "id": "eje1_atenciones_medicos",
            "indicador": "Atención de médicos",
            "estrategia": "Atenciones 1:1 por mes con médicos (seguimiento).",
            "unit": "atenciones",
            "defaultTarget": 80,
            "actualSource": "not_in_excel",
        },
        {
            "id": "eje1_insured_rate",
            "indicador": "Rentabilidad (proxy)",
            "estrategia": "% Seguro / Pacientes (Indicadores).",
            "unit": "%",
            "defaultTarget": 0.35,  # decimal
            "actualSource": "indicadores_insured_rate",
        },
    ],
    "eje2": [
        {
            "id": "eje2_prospectos_empresas",
            "indicador": "Nuevos convenios",
            "estrategia": "Prospectos de empresas para convenio.",
            "unit": "prospectos",
            "defaultTarget": 6,
            "actualSource": "not_in_excel",
        },
        {
            "id": "eje2_cierres_empresas",
            "indicador": "Nuevos convenios",
            "estrategia": "Cierres de convenio con empresas.",
            "unit": "cierres",
            "defaultTarget": 1,
            "actualSource": "not_in_excel",
        },
        {
            "id": "eje2_eventos_empresas",
            "indicador": "Eventos",
            "estrategia": "Eventos empresariales (ferias / patrocinios / networking).",
            "unit": "eventos",
            "defaultTarget": 1,
            "actualSource": "not_in_excel",
        },
    ],
    "eje3": [
        {
            "id": "eje3_pacientes_seguro",
            "indicador": "Pacientes de seguros",
            "estrategia": "Pacientes con seguro (Indicadores).",
            "unit": "pacientes",
            "defaultTarget": 33,
            "actualSource": "indicadores_insured",
        },
        {
            "id": "eje3_visitas_brokers",
            "indicador": "Pacientes de seguros",
            "estrategia": "Visitas mensuales de agentes y brokers.",
            "unit": "visitas",
            "defaultTarget": 2,
            "actualSource": "not_in_excel",
        },
    ],
}


# =========================
# Session state (targets, status)
# =========================
if "targets" not in st.session_state:
    # { "YYYY-MM": {kpi_id: meta_value} }
    st.session_state.targets = {}

if "status" not in st.session_state:
    st.session_state.status = "Selecciona Mes/Año y sube el Excel."


# =========================
# UI helpers
# =========================
def badge(text: str, tone: str = "gray"):
    colors = {
        "green": ("#e8f7ee", "#0f5132", "#b7e4c7"),
        "yellow": ("#fff7e6", "#664d03", "#ffe0a3"),
        "red": ("#fdecec", "#842029", "#f5c2c7"),
        "gray": ("#eef2f7", "#334155", "#dbe3ef"),
        "blue": ("#e8f0ff", "#1e3a8a", "#c7d2fe"),
    }
    bg, fg, br = colors.get(tone, colors["gray"])
    st.markdown(
        f"""
        <span style="
          display:inline-block; padding:4px 10px; border-radius:999px;
          background:{bg}; color:{fg}; border:1px solid {br};
          font-size:12px; font-weight:600;">
          {text}
        </span>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div style="
          border-radius:16px; padding:14px 16px; background:#ffffff;
          box-shadow:0 1px 10px rgba(15,23,42,.06); border:1px solid rgba(15,23,42,.06);">
          <div style="font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:#64748b; font-weight:700;">
            {title}
          </div>
          <div style="font-size:34px; font-weight:900; color:#0f172a; line-height:1.1; margin-top:6px;">
            {value}
          </div>
          <div style="font-size:12px; color:#64748b; margin-top:4px;">
            {sub}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_json_button(label: str, data: Dict[str, Any], filename: str):
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(label=label, data=raw, file_name=filename, mime="application/json")


def read_uploaded_json(uploaded_file) -> Dict[str, Any]:
    if uploaded_file is None:
        return {}
    try:
        return json.loads(uploaded_file.read().decode("utf-8"))
    except Exception:
        return {}


def fmt_int(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def traffic_light(ratio: Optional[float]) -> Tuple[str, str]:
    if ratio is None:
        return ("N/D", "gray")
    if ratio >= 1:
        return ("VERDE", "green")
    if ratio >= 0.8:
        return ("AMARILLO", "yellow")
    return ("ROJO", "red")


# =========================
# Excel parsing helpers
# =========================
def _norm(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _parse_month_to_iso(v: Any) -> Optional[str]:
    if v is None or _norm(v) == "":
        return None

    # datetime/date-like
    if hasattr(v, "year") and hasattr(v, "month"):
        try:
            return f"{int(v.year)}-{int(v.month):02d}-01"
        except Exception:
            pass

    # string parse
    s = _norm(v)
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return f"{dt.year}-{dt.month:02d}-01"


def _build_grid_with_merges(ws: Worksheet) -> List[List[Any]]:
    R, C = ws.max_row or 0, ws.max_column or 0
    grid = [[ws.cell(row=r, column=c).value for c in range(1, C + 1)] for r in range(1, R + 1)]

    for merged in ws.merged_cells.ranges:
        v = ws.cell(row=merged.min_row, column=merged.min_col).value
        for r in range(merged.min_row, merged.max_row + 1):
            for c in range(merged.min_col, merged.max_col + 1):
                if grid[r - 1][c - 1] in (None, ""):
                    grid[r - 1][c - 1] = v
    return grid


def _find_cell_exact(grid: List[List[Any]], target: str) -> Optional[Tuple[int, int]]:
    t = target.strip().lower()
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if _norm(v).lower() == t:
                return r, c
    return None


@dataclass
class IndicadoresOut:
    patients: pd.DataFrame
    insured: pd.DataFrame
    debug: Dict[str, Any]


def extract_indicadores(ws: Worksheet) -> IndicadoresOut:
    grid = _build_grid_with_merges(ws)
    if not grid:
        raise ValueError("Hoja 'Indicadores' vacía o ilegible.")

    header = _find_cell_exact(grid, "Estadistica")
    if not header:
        raise ValueError('No encontré "Estadistica" en Indicadores.')

    hr, hc = header
    months: List[Tuple[int, str]] = []
    for c in range(hc + 1, len(grid[hr])):
        iso = _parse_month_to_iso(grid[hr][c])
        if iso:
            months.append((c, iso))
    if not months:
        raise ValueError("No pude parsear meses en Indicadores (fila de 'Estadistica').")

    ppos = _find_cell_exact(grid, "No. de Pacientes")
    spos = _find_cell_exact(grid, "Seguro")
    if not ppos:
        raise ValueError('No encontré "No. de Pacientes" en Indicadores.')
    if not spos:
        raise ValueError('No encontré "Seguro" en Indicadores.')

    def row_series(row_idx: int) -> pd.DataFrame:
        rows = []
        for c, iso in months:
            v = grid[row_idx][c]
            try:
                val = float(v) if v is not None and _norm(v) != "" else None
            except Exception:
                val = None
            if val is not None:
                rows.append({"month": iso, "value": val})
        return pd.DataFrame(rows).sort_values("month")

    patients = row_series(ppos[0])
    insured = row_series(spos[0])

    return IndicadoresOut(
        patients=patients,
        insured=insured,
        debug={"months": [m for _, m in months], "pPos": ppos, "sPos": spos},
    )


@dataclass
class SpecialtiesOut:
    months: List[str]
    series_by_specialty: Dict[str, pd.DataFrame]


def extract_medicos_por_especialidad(ws: Worksheet) -> SpecialtiesOut:
    aoa = pd.DataFrame(ws.values).fillna(value=pd.NA)

    header_row = None
    row_labels_col = None
    first_date_col = None

    scan_rows = min(50, len(aoa))
    for r in range(scan_rows):
        row = aoa.iloc[r].astype(object).tolist()
        for c, v in enumerate(row):
            if _norm(v).lower() == "row labels":
                for cc in range(c + 1, len(row)):
                    iso = _parse_month_to_iso(row[cc])
                    if iso:
                        header_row = r
                        row_labels_col = c
                        first_date_col = cc
                        break
        if header_row is not None:
            break

    if header_row is None or row_labels_col is None or first_date_col is None:
        raise ValueError('No encontré encabezado de pivot en "Medicos por Especialidad" (Row Labels + fechas).')

    header_vals = aoa.iloc[header_row].astype(object).tolist()
    months: List[Tuple[int, str]] = []
    for c in range(first_date_col, len(header_vals)):
        iso = _parse_month_to_iso(header_vals[c])
        if iso:
            months.append((c, iso))
    if not months:
        raise ValueError('No pude parsear meses en "Medicos por Especialidad".')

    series: Dict[str, pd.DataFrame] = {}
    for r in range(header_row + 1, len(aoa)):
        label = _norm(aoa.iat[r, row_labels_col])
        if not label:
            continue
        if "grand total" in label.lower():
            break

        vals = []
        for c, iso in months:
            v = aoa.iat[r, c] if c < aoa.shape[1] else pd.NA
            try:
                val = float(v) if v is not None and _norm(v) != "" and v is not pd.NA else None
            except Exception:
                val = None
            vals.append({"month": iso, "value": val})
        series[label] = pd.DataFrame(vals)

    return SpecialtiesOut(months=[m for _, m in months], series_by_specialty=series)


def series_value_at(df: pd.DataFrame, iso: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    hit = df.loc[df["month"] == iso, "value"]
    if hit.empty:
        return None
    v = hit.iloc[0]
    return float(v) if pd.notna(v) else None


def prev_month_iso(iso: str) -> str:
    d = pd.to_datetime(iso)
    d2 = d - relativedelta(months=1)
    return f"{d2.year}-{d2.month:02d}-01"


def prev_year_iso(iso: str) -> str:
    d = pd.to_datetime(iso)
    d2 = d - relativedelta(years=1)
    return f"{d2.year}-{d2.month:02d}-01"


def find_matching_specialty_label(all_labels: List[str], match_tokens: List[str]) -> Optional[str]:
    tokens = [t.lower() for t in match_tokens]
    for lbl in all_labels:
        L = lbl.lower()
        if any(t in L for t in tokens):
            return lbl
    return None


def compute_key_specialties_breakdown(spec_series_by_label: Dict[str, pd.DataFrame], iso: str) -> Dict[str, Any]:
    all_labels = list(spec_series_by_label.keys())
    labels, raw_values, values = [], [], []

    for s in KEY_SPECIALTIES:
        lbl = find_matching_specialty_label(all_labels, s["match"])
        labels.append(s["id"])
        if not lbl:
            raw_values.append(None)
            values.append(0.0)
            continue
        v = series_value_at(spec_series_by_label.get(lbl), iso)
        raw_values.append(v if v is not None else None)
        values.append(float(v) if v is not None else 0.0)

    return {"labels": labels, "values": values, "rawValues": raw_values}


@dataclass
class ExcelContext:
    iso: str
    patients: Optional[float]
    insured: Optional[float]
    insured_rate: Optional[float]
    patients_mom: Optional[float]
    patients_yoy: Optional[float]
    insured_mom: Optional[float]
    insured_yoy: Optional[float]
    key_breakdown: Dict[str, Any]


def build_context(ind_patients: pd.DataFrame, ind_insured: pd.DataFrame, spec_series: Dict[str, pd.DataFrame], iso: str) -> ExcelContext:
    p = series_value_at(ind_patients, iso)
    i = series_value_at(ind_insured, iso)
    rate = (i / p) if (i is not None and p is not None and p != 0) else None

    p_prev_m = series_value_at(ind_patients, prev_month_iso(iso))
    p_prev_y = series_value_at(ind_patients, prev_year_iso(iso))
    p_mom = ((p - p_prev_m) / p_prev_m) if (p is not None and p_prev_m not in (None, 0)) else None
    p_yoy = ((p - p_prev_y) / p_prev_y) if (p is not None and p_prev_y not in (None, 0)) else None

    i_prev_m = series_value_at(ind_insured, prev_month_iso(iso))
    i_prev_y = series_value_at(ind_insured, prev_year_iso(iso))
    i_mom = ((i - i_prev_m) / i_prev_m) if (i is not None and i_prev_m not in (None, 0)) else None
    i_yoy = ((i - i_prev_y) / i_prev_y) if (i is not None and i_prev_y not in (None, 0)) else None

    key_breakdown = compute_key_specialties_breakdown(spec_series, iso)

    return ExcelContext(
        iso=iso,
        patients=p,
        insured=i,
        insured_rate=rate,
        patients_mom=p_mom,
        patients_yoy=p_yoy,
        insured_mom=i_mom,
        insured_yoy=i_yoy,
        key_breakdown=key_breakdown,
    )


def get_actual(actual_source: str, ctx: ExcelContext) -> Optional[float]:
    if actual_source == "key_specialties_sum":
        raw = ctx.key_breakdown["rawValues"]
        s = sum(v for v in raw if isinstance(v, (int, float)))
        return s if s > 0 else None
    if actual_source == "indicadores_insured":
        return ctx.insured
    if actual_source == "indicadores_insured_rate":
        return ctx.insured_rate
    return None


def get_sheet_by_name(wb, wanted: str):
    for name in wb.sheetnames:
        if name.strip().lower() == wanted.strip().lower():
            return wb[name]
    return None


def build_kpi_table(eje_key: str, ctx: ExcelContext, targets_for_period: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in KPI_CONFIG[eje_key]:
        target = targets_for_period.get(item["id"], item["defaultTarget"])
        actual = get_actual(item["actualSource"], ctx)

        ratio = (actual / target) if (actual is not None and target not in (None, 0, "")) else None
        sem_text, sem_tone = traffic_light(ratio)

        compliance = fmt_pct(ratio) if ratio is not None else "—"

        actual_disp = fmt_pct(actual) if item["unit"] == "%" else fmt_int(actual)

        rows.append({
            "KPI_ID": item["id"],
            "Indicador": item["indicador"],
            "Estrategia": item["estrategia"],
            "Unidad": item["unit"],
            "Meta": float(target) if str(target).strip() != "" else None,
            "Real": actual_disp,
            "Cumplimiento": compliance,
            "Semáforo": sem_text,
            "_sem_tone": sem_tone,
            "_actual_is_missing": (actual is None and item["actualSource"] == "not_in_excel"),
        })
    return pd.DataFrame(rows)


def apply_targets_edits(df_editor: pd.DataFrame, targets_for_period: Dict[str, Any]) -> Dict[str, Any]:
    for _, r in df_editor.iterrows():
        kpi_id = r.get("KPI_ID")
        meta = r.get("Meta")
        if kpi_id:
            if meta is None or str(meta).strip() == "":
                targets_for_period.pop(kpi_id, None)
            else:
                targets_for_period[kpi_id] = float(meta)
    return targets_for_period


# =========================
# Sidebar (controls)
# =========================
st.sidebar.markdown("## CRMBI Comercial")
st.sidebar.caption("BI Dashboard · Excel (Indicadores + Médicos por Especialidad) · Metas editables por periodo")

today = pd.Timestamp.today()
months = list(range(1, 13))
month_labels = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

month = st.sidebar.selectbox(
    "Mes",
    options=months,
    index=int(today.month) - 1,
    format_func=lambda m: month_labels[m-1],
)
year = st.sidebar.selectbox("Año", options=list(range(today.year - 3, today.year + 2)), index=3)

period_key = f"{year}-{month:02d}"
iso = f"{year}-{month:02d}-01"

uploaded = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])

st.sidebar.divider()
st.sidebar.markdown("### Metas (Targets)")

cA, cB = st.sidebar.columns(2)
with cA:
    if st.button("Reset metas periodo", use_container_width=True):
        st.session_state.targets[period_key] = {}
        st.session_state.status = f"Metas reseteadas para {period_key}."

with cB:
    download_json_button("Export metas", st.session_state.targets, filename="crmbi_targets.json")

import_file = st.sidebar.file_uploader("Import metas (JSON)", type=["json"], key="import_json")
if import_file is not None:
    imported = read_uploaded_json(import_file)
    if isinstance(imported, dict):
        st.session_state.targets = imported
        st.session_state.status = "Metas importadas correctamente (JSON)."
    else:
        st.session_state.status = "No pude importar el JSON (formato inválido)."


# =========================
# Header
# =========================
l, r = st.columns([0.78, 0.22], vertical_alignment="center")
with l:
    st.title("CRMBI Comercial (Por EJE)")
    st.caption("Auto desde Excel: Indicadores + Médicos por Especialidad · Metas editables · Mes/Año")
with r:
    st.markdown("#### Estado")
    st.write(st.session_state.status)


# =========================
# Load Excel + compute
# =========================
if uploaded is None:
    badge("SIN EXCEL", "gray")
    st.info("Sube el Excel para habilitar KPIs y gráficos.")
    st.stop()

try:
    data = uploaded.read()
    wb = load_workbook(BytesIO(data), data_only=True)

    ws_ind = get_sheet_by_name(wb, "Indicadores")
    if ws_ind is None:
        raise ValueError('No existe hoja "Indicadores".')

    ws_spec = get_sheet_by_name(wb, "Medicos por Especialidad")
    if ws_spec is None:
        raise ValueError('No existe hoja "Medicos por Especialidad".')

    ind_out = extract_indicadores(ws_ind)
    spec_out = extract_medicos_por_especialidad(ws_spec)

    ctx = build_context(
        ind_patients=ind_out.patients,
        ind_insured=ind_out.insured,
        spec_series=spec_out.series_by_specialty,
        iso=iso,
    )

    badge("EXCEL CARGADO", "green")

except Exception as e:
    badge("ERROR", "red")
    st.error(str(e))
    with st.expander("Debug (error)"):
        st.code(repr(e))
    st.stop()


# =========================
# KPI Cards
# =========================
c1, c2, c3 = st.columns(3)
with c1:
    kpi_card(
        "Pacientes (Indicadores) · periodo",
        fmt_int(ctx.patients),
        f"MoM: {fmt_pct(ctx.patients_mom)} · YoY: {fmt_pct(ctx.patients_yoy)}",
    )
with c2:
    kpi_card(
        "Seguro (Indicadores) · periodo",
        fmt_int(ctx.insured),
        f"MoM: {fmt_pct(ctx.insured_mom)} · YoY: {fmt_pct(ctx.insured_yoy)}",
    )
with c3:
    kpi_card("% Seguro / Pacientes", fmt_pct(ctx.insured_rate), "Seguro ÷ Pacientes")

st.divider()


# =========================
# Charts
# =========================
ch1, ch2 = st.columns(2)
with ch1:
    st.subheader("Tendencia: Pacientes")
    dfp = ind_out.patients.copy()
    if dfp.empty:
        st.info("Sin serie de pacientes en el Excel.")
    else:
        fig = px.line(dfp, x="month", y="value", markers=True)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

with ch2:
    st.subheader("Tendencia: Seguro")
    dfi = ind_out.insured.copy()
    if dfi.empty:
        st.info("Sin serie de seguros en el Excel.")
    else:
        fig = px.line(dfi, x="month", y="value", markers=True)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Especialidades clave (Médicos por Especialidad)")
st.caption("Conteo automático de médicos por especialidad en el mes seleccionado (proxy de actividad).")

df_bar = pd.DataFrame({"Especialidad": ctx.key_breakdown["labels"], "Médicos": ctx.key_breakdown["values"]})
figb = px.bar(df_bar, x="Especialidad", y="Médicos")
figb.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
st.plotly_chart(figb, use_container_width=True)

st.divider()


# =========================
# Ejes in tabs
# =========================
targets_for_period = st.session_state.targets.get(period_key, {})

tabs = st.tabs([
    "EJE 1 · MÉDICOS Y PACIENTES",
    "EJE 2 · EMPRESAS Y COMPASS",
    "EJE 3 · SEGUROS Y BROKERS",
    "DEBUG",
])

eje_meta = {
    "eje1": "Objetivo: Aumentar captación, productividad y rentabilidad de médicos clave.",
    "eje2": "Objetivo: Generar convenios empresariales que alimenten flujo constante de pacientes.",
    "eje3": "Objetivo: Incrementar pacientes asegurados y relaciones con brokers.",
}

for idx, eje_key in enumerate(["eje1", "eje2", "eje3"]):
    with tabs[idx]:
        st.markdown(f"**{eje_meta[eje_key]}**")

        df = build_kpi_table(eje_key, ctx, targets_for_period)

        missing = df[df["_actual_is_missing"] == True]
        if len(missing) > 0:
            badge(f"{len(missing)} KPI(s) N/D (no existe dato en Excel)", "yellow")
            st.caption("Puedes usar metas, pero el 'Real' depende de que exista esa métrica en el Excel.")

        show = df[["KPI_ID", "Indicador", "Estrategia", "Unidad", "Meta", "Real", "Cumplimiento", "Semáforo"]].copy()

        edited = st.data_editor(
            show,
            use_container_width=True,
            hide_index=True,
            disabled=["KPI_ID", "Indicador", "Estrategia", "Unidad", "Real", "Cumplimiento", "Semáforo"],
            column_config={
                "Meta": st.column_config.NumberColumn(
                    "Meta (editable)",
                    help="Para %, usa decimal: 0.35 = 35%",
                    step=0.01,
                ),
                "Estrategia": st.column_config.TextColumn("Estrategia", width="large"),
            },
            key=f"editor_{eje_key}_{period_key}",
        )

        b1, b2 = st.columns([0.25, 0.75])
        with b1:
            if st.button("Guardar metas", use_container_width=True, key=f"save_{eje_key}_{period_key}"):
                targets_for_period = apply_targets_edits(edited, targets_for_period)
                st.session_state.targets[period_key] = targets_for_period
                st.session_state.status = f"Metas guardadas para {period_key}."
                st.success("Metas guardadas.")
        with b2:
            st.caption("Tip: Exporta metas (JSON) para conservarlas en Streamlit Cloud o reutilizarlas.")

with tabs[3]:
    st.subheader("Debug (Excel)")
    st.caption("Hojas detectadas, meses parseados, y muestra de especialidades.")
    st.json({
        "sheets": wb.sheetnames,
        "indicadores": ind_out.debug,
        "medicos_por_especialidad": {
            "months": spec_out.months[:12],
            "sample_specialties": list(spec_out.series_by_specialty.keys())[:20],
        },
        "selected_period": period_key,
        "iso": iso,
    })
