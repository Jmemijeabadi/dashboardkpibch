import io
import html
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NewCity Hospital BI | KO26",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TARGETS = {
    "annual_revenue": 268700000.0,
    "monthly_patients": 330,
    "monthly_insured": 55,
    "mix_pct": 60.0,
    "repeat_doctors_pct": 65.0,
}

KEY_SPECIALTIES = ["ORTOPEDIA", "TRAUMATOLOGIA", "CARDIO", "NEURO", "TORAC", "GENERAL"]

KPI_DEFINITIONS = {
    "revenue": {
        "title": "Ingresos del periodo",
        "goal": "Medir la facturación observable del periodo filtrado contra la meta ejecutiva.",
        "formula": "SUMA('Cuenta Ventas') del periodo seleccionado.",
        "meaning": "Representa el ingreso facturado del periodo. En semana y mes se estima además una proyección lineal de cierre.",
    },
    "mix": {
        "title": "Mix de especialidades clave",
        "goal": "Verificar si una parte suficiente del ingreso proviene de especialidades estratégicas.",
        "formula": "(Ingresos de especialidades clave / Ingresos totales) * 100",
        "meaning": "Ayuda a distinguir si el crecimiento viene del perfil de casos más estratégico y no solo del volumen total.",
    },
    "repeat_doctors": {
        "title": "Médicos con más de 1 paciente",
        "goal": "Monitorear recurrencia médica real en el periodo.",
        "formula": "(Médicos con >1 paciente único / Total de médicos activos) * 100",
        "meaning": "Es más confiable que contar filas repetidas, porque se calcula sobre pacientes únicos atendidos por médico.",
    },
    "patients": {
        "title": "Pacientes únicos",
        "goal": "Medir volumen real sin duplicar cargos del mismo caso.",
        "formula": "COUNT DISTINCT de Cuenta / # Cuenta",
        "meaning": "Cuenta cada expediente/cuenta una sola vez dentro del periodo filtrado.",
    },
}

SEGMENT_OPTIONS = ["Todos", "Privado", "Seguro", "Empresa"]
PERIOD_OPTIONS = ["week", "month", "year"]


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
        .main {
            background: #f8fafc;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .ga-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .ga-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px -20px rgba(15,23,42,0.18);
            min-height: 138px;
        }
        .kpi-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
            line-height: 1.05;
        }
        .kpi-sub {
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 0.3rem;
        }
        .trend {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
        }
        .trend-up {
            background: #dcfce7;
            color: #15803d;
        }
        .trend-down {
            background: #fee2e2;
            color: #b91c1c;
        }
        .trend-neutral {
            background: #f1f5f9;
            color: #64748b;
        }
        .badge-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            color: #334155;
        }
        .badge-ok {
            background: #ecfdf5;
            color: #047857;
            border-color: #bbf7d0;
        }
        .badge-info {
            background: #eff6ff;
            color: #1d4ed8;
            border-color: #bfdbfe;
        }
        .badge-warn {
            background: #fffbeb;
            color: #b45309;
            border-color: #fde68a;
        }
        .summary-box {
            background: linear-gradient(180deg,#ffffff 0%,#fbfdff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px -20px rgba(15,23,42,0.18);
        }
        .summary-item {
            background: #f8fafc;
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.65rem;
        }
        .summary-item-title {
            font-weight: 800;
            color: #0f172a;
            font-size: 0.88rem;
            margin-bottom: 0.12rem;
        }
        .summary-item-text {
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.4;
        }
        .panel-title {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .panel-subtitle {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.8rem;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }
        .metric-mini {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            min-height: 90px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def normalize_text(value) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


def clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_currency_compact(value: float) -> str:
    value = float(value or 0)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    return format_currency(value)


def format_num(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{float(value or 0):.1f}%"


def safe_percent_delta(curr: float, prev: Optional[float]) -> Optional[float]:
    if prev is None or pd.isna(prev):
        return None
    if prev == 0:
        return 0 if curr == 0 else None
    return ((curr - prev) / prev) * 100


def safe_point_delta(curr: float, prev: Optional[float]) -> Optional[float]:
    if prev is None or pd.isna(prev):
        return None
    return curr - prev


def trend_html(delta: Optional[float], unit: str = "%") -> str:
    if delta is None or pd.isna(delta):
        return '<span class="trend trend-neutral">Sin base</span>'
    if delta > 0.1:
        return f'<span class="trend trend-up">▲ {abs(delta):.1f}{unit} vs prev.</span>'
    if delta < -0.1:
        return f'<span class="trend trend-down">▼ {abs(delta):.1f}{unit} vs prev.</span>'
    return f'<span class="trend trend-neutral">• {abs(delta):.1f}{unit} vs prev.</span>'


def period_target(period_type: str, annual_value: float, monthly_value: float = 0) -> float:
    if period_type == "year":
        return annual_value
    if period_type == "month":
        return monthly_value or (annual_value / 12)
    if period_type == "week":
        return (monthly_value * 12 / 52) if monthly_value else (annual_value / 52)
    return annual_value


def parse_number_locale(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().replace("$", "").replace(" ", "")
    if not s:
        return 0.0

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma and not has_dot:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_dot and not has_comma:
        parts = s.split(".")
        if not (len(parts) == 2 and len(parts[1]) <= 2):
            s = s.replace(".", "")

    filtered = "".join(ch for ch in s if ch in "0123456789.-")
    try:
        return float(filtered)
    except Exception:
        return 0.0


def parse_excel_date(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float)) and 20_000 < float(value) < 60_000:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")

    try:
        d = pd.to_datetime(value, errors="coerce")
        return d.normalize() if pd.notna(d) else pd.NaT
    except Exception:
        return pd.NaT


def dedupe_headers(raw_headers: List) -> List[str]:
    counts = {}
    output = []
    for idx, h in enumerate(raw_headers):
        base = normalize_text(h) or f"col_{idx+1}"
        counts[base] = counts.get(base, 0) + 1
        output.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return output


def base_header(header: str) -> str:
    return header.split("__")[0]


def find_header(headers: List[str], candidates: List[str]) -> Optional[str]:
    normalized_candidates = [normalize_text(c) for c in candidates]
    for c in normalized_candidates:
        for h in headers:
            if base_header(h) == c:
                return h
    return None


def is_key_specialty(name: str) -> bool:
    s = clean_text(name).upper()
    return any(k in s for k in KEY_SPECIALTIES)


def classify_segment(type_value: str, insurance_group: str) -> str:
    t = clean_text(type_value).upper()
    s = clean_text(insurance_group).upper()

    if "EMPRESA" in t or "CONVENIO" in t:
        return "Empresa"
    if "SEGURO" in t or (s and s != "SIN SEGURO" and "PARTICULAR" not in s):
        return "Seguro"
    return "Privado"


def start_of_week(date_value: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(date_value):
        return pd.NaT
    return (date_value - pd.to_timedelta(date_value.weekday(), unit="D")).normalize()


def describe_period(period_type: str, year: int, sub_period) -> str:
    if period_type == "year":
        return f"Año {year}"
    if period_type == "month":
        month_name = pd.Timestamp(year=year, month=int(sub_period), day=1).strftime("%b").capitalize()
        return f"{month_name} {year}"
    if period_type == "week":
        dt = pd.to_datetime(sub_period)
        return f"Semana {dt.strftime('%d-%b-%Y')}"
    return "Periodo"


# =========================================================
# PARSEO WORKBOOK
# =========================================================
def detect_detail_sheet(df_raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(len(df_raw), 12)):
        values = [normalize_text(v) for v in df_raw.iloc[i].tolist()]
        score = sum(
            k in values
            for k in [
                "cuenta",
                "ingreso",
                "egreso",
                "cuenta ventas",
                "tipo",
                "especialidad grupo",
                "seguro grupo",
                "medico grupo",
            ]
        )
        if score >= 6:
            return i
    return None


def is_insurance_summary_sheet(df_raw: pd.DataFrame) -> bool:
    for i in range(min(len(df_raw), 10)):
        values = [normalize_text(v) for v in df_raw.iloc[i].tolist()]
        if "row labels" in values and "cuenta ventas" in values:
            return True
    return False


def parse_insurance_summary(df_raw: pd.DataFrame, sheet_name: str) -> Optional[dict]:
    header_idx = None
    for i in range(min(len(df_raw), 15)):
        values = [normalize_text(v) for v in df_raw.iloc[i].tolist()]
        if "row labels" in values and "cuenta ventas" in values:
            header_idx = i
            break

    if header_idx is None:
        return None

    raw_headers = dedupe_headers(df_raw.iloc[header_idx].tolist())
    body = df_raw.iloc[header_idx + 1 :].copy()
    body.columns = raw_headers

    label_col = find_header(raw_headers, ["row labels"])
    patients_col = find_header(raw_headers, ["pacientes"])
    full_col = find_header(raw_headers, ["cuenta full"])
    ventas_col = find_header(raw_headers, ["cuenta ventas"])

    if not label_col:
        return None

    body = body[body[label_col].astype(str).str.strip() != ""].copy()

    result_rows = []
    grand_total = None

    for _, row in body.iterrows():
        label = clean_text(row.get(label_col))
        item = {
            "label": label,
            "patients": parse_number_locale(row.get(patients_col)) if patients_col else 0,
            "account_full": parse_number_locale(row.get(full_col)) if full_col else 0,
            "account_ventas": parse_number_locale(row.get(ventas_col)) if ventas_col else 0,
        }
        if normalize_text(label) == "grand total":
            grand_total = item
            break
        result_rows.append(item)

    return {"sheet_name": sheet_name, "items": result_rows, "grand_total": grand_total}


def parse_detail_sheet(df_raw: pd.DataFrame, sheet_name: str, header_idx: int) -> pd.DataFrame:
    headers = dedupe_headers(df_raw.iloc[header_idx].tolist())
    body = df_raw.iloc[header_idx + 1 :].copy()
    body.columns = headers

    account_col = find_header(headers, ["# cuenta", "cuenta", "expediente"])
    admission_col = find_header(headers, ["ingreso", "fecha ingreso"])
    discharge_col = find_header(headers, ["egreso", "fecha egreso"])
    doctor_col = find_header(headers, ["medico grupo", "medico"])
    specialty_col = find_header(headers, ["especialidad grupo", "especialidad"])
    insurance_group_col = find_header(headers, ["seguro grupo", "seguro"])
    type_col = find_header(headers, ["tipo"])
    ventas_col = find_header(headers, ["cuenta ventas", "cuenta venta"])
    full_col = find_header(headers, ["cuenta full"])
    utility_col = find_header(headers, ["utilidad"])

    required = [account_col, admission_col, doctor_col, specialty_col, type_col, ventas_col]
    if any(col is None for col in required):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["source_sheet"] = sheet_name
    out["account"] = body[account_col].map(clean_text)
    out["date"] = body[admission_col].map(parse_excel_date)
    out["discharge_date"] = body[discharge_col].map(parse_excel_date) if discharge_col else pd.NaT
    out["doctor"] = body[doctor_col].map(clean_text).replace("", "SIN MÉDICO")
    out["specialty"] = body[specialty_col].map(clean_text).replace("", "SIN ESPECIALIDAD")
    out["insurance_group"] = body[insurance_group_col].map(clean_text) if insurance_group_col else "SIN SEGURO"
    out["type"] = body[type_col].map(clean_text).replace("", "Privado")
    out["account_ventas"] = body[ventas_col].map(parse_number_locale)
    out["account_full"] = body[full_col].map(parse_number_locale) if full_col else out["account_ventas"]
    out["utility"] = body[utility_col].map(parse_number_locale) if utility_col else None

    out = out[pd.notna(out["date"])].copy()

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["week_start"] = out["date"].map(start_of_week)
    out["week_key"] = out["week_start"].dt.strftime("%Y-%m-%d")
    out["segment"] = out.apply(lambda r: classify_segment(r["type"], r["insurance_group"]), axis=1)
    out["is_key_specialty"] = out["specialty"].map(is_key_specialty)

    out["length_of_stay"] = (
        (out["discharge_date"] - out["date"]).dt.days
        if "discharge_date" in out.columns
        else None
    )
    if "length_of_stay" in out.columns:
        out["length_of_stay"] = out["length_of_stay"].clip(lower=0)

    out["dedupe_key"] = (
        out["account"].astype(str)
        + "|"
        + out["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["doctor"].astype(str)
        + "|"
        + out["specialty"].astype(str)
        + "|"
        + out["account_ventas"].round(2).astype(str)
        + "|"
        + out["source_sheet"].astype(str)
    )

    out = out.drop_duplicates("dedupe_key").drop(columns=["dedupe_key"])
    return out.sort_values(["date", "source_sheet", "account"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_workbook(file_bytes: bytes) -> dict:
    excel = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, engine="openpyxl")

    detail_frames = []
    detail_sheets = []
    insurance_summary = None

    for sheet_name, df_raw in excel.items():
        detail_idx = detect_detail_sheet(df_raw)
        if detail_idx is not None:
            parsed = parse_detail_sheet(df_raw, sheet_name, detail_idx)
            if not parsed.empty:
                detail_frames.append(parsed)
                detail_sheets.append(sheet_name)
            continue

        if insurance_summary is None and is_insurance_summary_sheet(df_raw):
            insurance_summary = parse_insurance_summary(df_raw, sheet_name)

    if not detail_frames:
        raise ValueError("No se encontró ninguna hoja operativa con la estructura esperada.")

    detail_df = pd.concat(detail_frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    return {
        "rows": detail_df,
        "detail_sheets": detail_sheets,
        "all_sheets": list(excel.keys()),
        "insurance_summary": insurance_summary,
        "coverage_min": detail_df["date"].min(),
        "coverage_max": detail_df["date"].max(),
    }


# =========================================================
# FILTROS / MÉTRICAS
# =========================================================
def get_active_rows(df: pd.DataFrame, active_sheets: List[str], detail_sheets: List[str]) -> pd.DataFrame:
    effective_sheets = active_sheets if active_sheets else detail_sheets
    if not effective_sheets:
        return df.iloc[0:0].copy()
    return df[df["source_sheet"].isin(effective_sheets)].copy()


def get_available_years(df: pd.DataFrame) -> List[int]:
    return sorted(df["year"].dropna().astype(int).unique().tolist(), reverse=True)


def get_available_months(df: pd.DataFrame, year: int) -> List[int]:
    subset = df[df["year"] == year]
    return sorted(subset["month"].dropna().astype(int).unique().tolist())


def get_available_weeks(df: pd.DataFrame, year: int) -> List[Tuple[str, str]]:
    subset = df[df["year"] == year].copy()
    if subset.empty:
        return []

    weeks = (
        subset[["week_key", "week_start"]]
        .drop_duplicates()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    out = []
    for _, row in weeks.iterrows():
        start = row["week_start"]
        end = start + pd.Timedelta(days=6)
        label = f"{start.strftime('%d-%b-%Y')} - {end.strftime('%d-%b-%Y')}"
        out.append((row["week_key"], label))
    return out


def filter_rows(df: pd.DataFrame, segment: str, period_type: str, year: int, sub_period) -> pd.DataFrame:
    result = df.copy()

    if segment != "Todos":
        result = result[result["segment"] == segment]

    if period_type == "year":
        result = result[result["year"] == year]
    elif period_type == "month":
        result = result[(result["year"] == year) & (result["month"] == int(sub_period))]
    elif period_type == "week":
        result = result[result["week_key"] == sub_period]

    return result.copy()


def get_previous_period(period_type: str, year: int, sub_period):
    if period_type == "year":
        return {"period_type": "year", "year": year - 1, "sub_period": None}

    if period_type == "month":
        month = int(sub_period)
        if month == 1:
            return {"period_type": "month", "year": year - 1, "sub_period": 12}
        return {"period_type": "month", "year": year, "sub_period": month - 1}

    week_start = pd.to_datetime(sub_period)
    prev_week = week_start - pd.Timedelta(days=7)
    return {
        "period_type": "week",
        "year": prev_week.year,
        "sub_period": prev_week.strftime("%Y-%m-%d"),
    }


def aggregate_metrics(df: pd.DataFrame, period_type: str, year: int, sub_period) -> dict:
    if df.empty:
        return {
            "account_ventas": 0.0,
            "account_full": 0.0,
            "projected_ventas": 0.0,
            "patients": 0,
            "insured_patients": 0,
            "avg_ticket_ventas": 0.0,
            "avg_ticket_full": 0.0,
            "key_mix_pct": 0.0,
            "total_doctors": 0,
            "repeat_doctors": 0,
            "repeat_doctors_pct": 0.0,
            "avg_stay": 0.0,
            "specialty_ventas": {},
            "doctor_stats": pd.DataFrame(columns=["doctor", "patients", "account_full", "account_ventas"]),
        }

    patients = df["account"].nunique()
    insured_patients = df[df["segment"] == "Seguro"]["account"].nunique()

    doctor_stats = (
        df.groupby("doctor", dropna=False)
        .agg(
            patients=("account", "nunique"),
            account_full=("account_full", "sum"),
            account_ventas=("account_ventas", "sum"),
        )
        .reset_index()
    )

    repeat_doctors = int((doctor_stats["patients"] > 1).sum())
    total_doctors = int(len(doctor_stats))

    specialty_ventas = (
        df.groupby("specialty", dropna=False)["account_ventas"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    account_ventas = float(df["account_ventas"].sum())
    account_full = float(df["account_full"].sum())
    key_revenue = float(df.loc[df["is_key_specialty"], "account_ventas"].sum())
    key_mix_pct = (key_revenue / account_ventas * 100) if account_ventas > 0 else 0.0

    avg_stay = float(df["length_of_stay"].dropna().mean()) if "length_of_stay" in df.columns and df["length_of_stay"].notna().any() else 0.0

    projected_ventas = account_ventas
    if period_type == "month":
        max_day = int(df["day"].max())
        days_in_month = pd.Timestamp(year=year, month=int(sub_period), day=1).days_in_month
        if max_day > 0 and max_day < days_in_month:
            projected_ventas = (account_ventas / max_day) * days_in_month
    elif period_type == "week":
        observed_days = df["date"].dt.normalize().nunique()
        if observed_days > 0 and observed_days < 7:
            projected_ventas = (account_ventas / observed_days) * 7
    elif period_type == "year":
        max_month = int(df["month"].max())
        if max_month > 0 and max_month < 12:
            projected_ventas = (account_ventas / max_month) * 12

    return {
        "account_ventas": account_ventas,
        "account_full": account_full,
        "projected_ventas": projected_ventas,
        "patients": int(patients),
        "insured_patients": int(insured_patients),
        "avg_ticket_ventas": account_ventas / patients if patients else 0.0,
        "avg_ticket_full": account_full / patients if patients else 0.0,
        "key_mix_pct": key_mix_pct,
        "total_doctors": total_doctors,
        "repeat_doctors": repeat_doctors,
        "repeat_doctors_pct": (repeat_doctors / total_doctors * 100) if total_doctors else 0.0,
        "avg_stay": avg_stay,
        "specialty_ventas": specialty_ventas,
        "doctor_stats": doctor_stats.sort_values("account_ventas", ascending=False).reset_index(drop=True),
    }


# =========================================================
# CHARTS
# =========================================================
def build_main_chart(df_active: pd.DataFrame, segment: str, period_type: str, year: int, sub_period) -> go.Figure:
    labels = []
    revenue_values = []
    mix_values = []

    if period_type == "year":
        title = f"Tendencia mensual {year}"
        subtitle = "Cuenta Ventas vs Mix por mes"
        for month in range(1, 13):
            subset = filter_rows(df_active, segment, "month", year, month)
            metrics = aggregate_metrics(subset, "month", year, month)
            labels.append(pd.Timestamp(year=year, month=month, day=1).strftime("%b"))
            revenue_values.append(metrics["account_ventas"])
            mix_values.append(metrics["key_mix_pct"])

    elif period_type == "month":
        title = f"Tendencia diaria · {describe_period('month', year, sub_period)}"
        subtitle = "Cuenta Ventas vs Mix por día"
        days_in_month = pd.Timestamp(year=year, month=int(sub_period), day=1).days_in_month
        for day in range(1, days_in_month + 1):
            subset = df_active[
                (df_active["year"] == year)
                & (df_active["month"] == int(sub_period))
                & (df_active["day"] == day)
                & ((df_active["segment"] == segment) if segment != "Todos" else True)
            ]
            metrics = aggregate_metrics(subset, "month", year, sub_period)
            labels.append(str(day))
            revenue_values.append(metrics["account_ventas"])
            mix_values.append(metrics["key_mix_pct"])

    else:
        title = f"Tendencia diaria · {describe_period('week', year, sub_period)}"
        subtitle = "Cuenta Ventas vs Mix por día"
        subset = filter_rows(df_active, segment, "week", year, sub_period)
        grouped = subset.groupby(subset["date"].dt.normalize())
        for date_key, day_df in grouped:
            metrics = aggregate_metrics(day_df, "week", year, sub_period)
            labels.append(pd.to_datetime(date_key).strftime("%d-%b"))
            revenue_values.append(metrics["account_ventas"])
            mix_values.append(metrics["key_mix_pct"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="Cuenta Ventas", x=labels, y=revenue_values, marker_color="#0f172a"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Mix Clave %",
            x=labels,
            y=mix_values,
            mode="lines+markers",
            line=dict(color="#10b981", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        template="plotly_white",
        height=370,
        margin=dict(l=10, r=10, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Cuenta Ventas", secondary_y=False)
    fig.update_yaxes(title_text="Mix %", secondary_y=True)
    return fig


def build_specialty_chart(curr_metrics: dict, prev_metrics: Optional[dict]) -> go.Figure:
    items = sorted(curr_metrics["specialty_ventas"].items(), key=lambda x: x[1], reverse=True)[:5]

    labels = []
    values = []
    colors = []

    for name, value in items:
        prev_value = 0 if not prev_metrics else prev_metrics["specialty_ventas"].get(name, 0)
        labels.append(name)
        values.append(value)
        colors.append("#10b981" if value >= prev_value else "#ef4444")

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[format_currency(v) for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Top 5 especialidades · Cuenta Ventas",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=40, t=55, b=20),
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


# =========================================================
# INSIGHTS / BOARD
# =========================================================
def build_board_summary(metrics: dict, prev_metrics: Optional[dict], targets: dict, period_type: str) -> List[Tuple[str, str]]:
    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)

    bullets = []

    if metrics["account_ventas"] >= revenue_target:
        bullets.append((
            "Ingreso alineado o superior a meta",
            f"El periodo registra {format_currency(metrics['account_ventas'])} frente a una meta de {format_currency(revenue_target)}."
        ))
    elif period_type != "year" and metrics["projected_ventas"] >= revenue_target:
        bullets.append((
            "Meta recuperable por proyección",
            f"El ingreso actual es {format_currency(metrics['account_ventas'])}, pero la proyección de cierre alcanza {format_currency(metrics['projected_ventas'])}."
        ))
    else:
        bullets.append((
            "Brecha económica en el corte",
            f"Los ingresos actuales son {format_currency(metrics['account_ventas'])} contra una meta de {format_currency(revenue_target)}."
        ))

    bullets.append((
        "Composición estratégica",
        f"El mix de especialidades clave se ubica en {format_pct(metrics['key_mix_pct'])}."
    ))

    bullets.append((
        "Volumen observable",
        f"{format_num(metrics['patients'])} pacientes únicos en el periodo; {format_num(metrics['insured_patients'])} pertenecen al segmento seguro."
    ))

    bullets.append((
        "Recurrencia médica",
        f"{format_pct(metrics['repeat_doctors_pct'])} de los médicos activos atendió más de un paciente único."
    ))

    return bullets[:4]


def build_insights(metrics: dict, prev_metrics: Optional[dict], targets: dict, period_type: str) -> List[Tuple[str, str, str]]:
    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
    insured_target = period_target(period_type, 0, targets["monthly_insured"])

    insights = []

    if period_type != "year" and metrics["projected_ventas"] > metrics["account_ventas"] and metrics["projected_ventas"] < revenue_target * 0.95:
        insights.append(("error", "Proyección por debajo de meta", f"El periodo podría cerrar en {format_currency(metrics['projected_ventas'])} si el ritmo actual se mantiene."))

    if metrics["account_ventas"] < revenue_target * 0.9:
        insights.append(("error", "Brecha de ingreso", f"Los ingresos actuales son {format_currency(metrics['account_ventas'])} frente a meta de {format_currency(revenue_target)}."))

    if metrics["key_mix_pct"] < targets["mix_pct"]:
        insights.append(("warning", "Mix estratégico bajo", f"El mix de especialidades clave está en {format_pct(metrics['key_mix_pct'])} y la meta es {format_pct(targets['mix_pct'])}."))

    if metrics["repeat_doctors_pct"] < targets["repeat_doctors_pct"]:
        insights.append(("error", "Recurrencia médica insuficiente", f"Solo {format_pct(metrics['repeat_doctors_pct'])} de los médicos activos atendió más de un paciente único."))

    if metrics["insured_patients"] < insured_target * 0.6:
        insights.append(("info", "Bajo flujo de seguro", f"El volumen asegurado actual es {format_num(metrics['insured_patients'])} pacientes."))

    if metrics["specialty_ventas"]:
        top_specialty = sorted(metrics["specialty_ventas"].items(), key=lambda x: x[1], reverse=True)[0]
        insights.append(("success", "Especialidad tractora", f"{top_specialty[0]} lidera el periodo con {format_currency(top_specialty[1])} en Cuenta Ventas."))

    return insights


def build_detail_table(metrics: dict, targets: dict, period_type: str) -> pd.DataFrame:
    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
    patient_target = period_target(period_type, 0, targets["monthly_patients"])
    insured_target = period_target(period_type, 0, targets["monthly_insured"])
    ticket_target = (revenue_target / patient_target) if patient_target > 0 else 0

    rows = [
        ["Ingresos observados (Cuenta Ventas)", metrics["account_ventas"], revenue_target, "money", "Reforzar cierres de procedimientos de alto ticket y seguimiento comercial con médicos tractores."],
        ["Pacientes únicos", metrics["patients"], patient_target, "num", "Incrementar conversión, referencia médica y captación por canal."],
        ["Pacientes de seguro", metrics["insured_patients"], insured_target, "num", "Profundizar relación con brokers, convenios y aseguradoras de mayor volumen."],
        ["Ticket promedio", metrics["avg_ticket_ventas"], ticket_target, "money", "Revisar mezcla de casos y líneas de mayor valor por cuenta."],
        ["Mix especialidades clave", metrics["key_mix_pct"], targets["mix_pct"], "pct", "Activar ortopedia, cardio, neuro, torácica y cirugía general en agenda y vinculación médica."],
        ["Médicos con >1 paciente", metrics["repeat_doctors_pct"], targets["repeat_doctors_pct"], "pct", "Dar seguimiento a médicos de una sola cuenta y priorizar reactivación."],
    ]

    formatted = []
    for label, value, target, fmt, action in rows:
        ok = value >= target
        if fmt == "money":
            gap = value - target
            value_txt = format_currency(value)
            target_txt = format_currency(target)
            gap_txt = format_currency(gap)
        elif fmt == "pct":
            gap = value - target
            value_txt = format_pct(value)
            target_txt = format_pct(target)
            gap_txt = f"{gap:.1f}pp"
        else:
            gap = value - target
            value_txt = format_num(value)
            target_txt = format_num(target)
            gap_txt = format_num(gap)

        formatted.append({
            "Métrica": label,
            "Actual": value_txt,
            "Meta": target_txt,
            "Brecha": gap_txt,
            "Estado": "Cumple" if ok else "Brecha",
            "Acción sugerida": "Mantener disciplina operativa." if ok else action,
        })

    return pd.DataFrame(formatted)


def build_validation_badges(metrics: dict, insurance_summary: Optional[dict]) -> List[str]:
    badges = []

    if insurance_summary and insurance_summary.get("grand_total"):
        grand = insurance_summary["grand_total"]
        revenue_diff = abs(metrics["account_ventas"] - grand.get("account_ventas", 0))
        patient_diff = abs(metrics["patients"] - grand.get("patients", 0))

        if revenue_diff <= 1:
            badges.append('<span class="badge-pill badge-ok">Conciliación ingresos OK</span>')
        else:
            badges.append(f'<span class="badge-pill badge-warn">Diferencia ingresos: {html.escape(format_currency(revenue_diff))}</span>')

        if patient_diff <= 0.5:
            badges.append('<span class="badge-pill badge-ok">Conciliación pacientes OK</span>')
        else:
            badges.append(f'<span class="badge-pill badge-warn">Diferencia pacientes: {html.escape(format_num(patient_diff))}</span>')
    else:
        badges.append('<span class="badge-pill badge-info">Sin hoja resumen para conciliación</span>')

    return badges


# =========================================================
# EXPORTS
# =========================================================
def build_summary_text(
    workbook_name: str,
    period_label: str,
    segment: str,
    metrics: dict,
    detail_df: pd.DataFrame,
) -> str:
    lines = [
        "NEWCITY HOSPITAL | RESUMEN EJECUTIVO KO26",
        f"Workbook: {workbook_name}",
        f"Periodo: {period_label}",
        f"Segmento: {segment}",
        "",
        f"Ingresos observados: {format_currency(metrics['account_ventas'])}",
        f"Proyección: {format_currency(metrics['projected_ventas'])}",
        f"Pacientes únicos: {format_num(metrics['patients'])}",
        f"Pacientes seguro: {format_num(metrics['insured_patients'])}",
        f"Mix esp. clave: {format_pct(metrics['key_mix_pct'])}",
        f"Médicos con >1 paciente: {format_pct(metrics['repeat_doctors_pct'])}",
        f"Ticket promedio: {format_currency(metrics['avg_ticket_ventas'])}",
        "",
        "Detalle estratégico:",
    ]

    for _, row in detail_df.iterrows():
        lines.append(
            f"- {row['Métrica']}: {row['Actual']} | Meta {row['Meta']} | Estado {row['Estado']} | Acción: {row['Acción sugerida']}"
        )

    return "\n".join(lines)


def export_filtered_excel(filtered_df: pd.DataFrame, detail_table: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        filtered_df.to_excel(writer, sheet_name="Detalle_Filtrado", index=False)
        detail_table.to_excel(writer, sheet_name="Resumen_Estrategico", index=False)
    return output.getvalue()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar workbook Excel", type=["xlsx", "xls"])

with st.sidebar.expander("Metas ejecutivas", expanded=True):
    annual_revenue = st.number_input("Meta anual ingresos (MXN)", min_value=0.0, value=float(DEFAULT_TARGETS["annual_revenue"]), step=100000.0)
    monthly_patients = st.number_input("Meta mensual pacientes", min_value=0, value=int(DEFAULT_TARGETS["monthly_patients"]), step=1)
    monthly_insured = st.number_input("Meta mensual pacientes seguro", min_value=0, value=int(DEFAULT_TARGETS["monthly_insured"]), step=1)
    mix_pct = st.number_input("Meta mix esp. clave (%)", min_value=0.0, value=float(DEFAULT_TARGETS["mix_pct"]), step=0.5)
    repeat_doctors_pct = st.number_input("Meta médicos repetidores (%)", min_value=0.0, value=float(DEFAULT_TARGETS["repeat_doctors_pct"]), step=0.5)

targets = {
    "annual_revenue": annual_revenue,
    "monthly_patients": monthly_patients,
    "monthly_insured": monthly_insured,
    "mix_pct": mix_pct,
    "repeat_doctors_pct": repeat_doctors_pct,
}


# =========================================================
# MAIN
# =========================================================
st.markdown('<div class="ga-title">NewCity Hospital BI 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="ga-subtitle">Dashboard ejecutivo KO26 · consolidación de múltiples hojas · lectura fácil para dirección</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("Carga un workbook para activar el dashboard.")
    st.stop()

try:
    parsed = parse_workbook(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Error al procesar el archivo: {e}")
    st.stop()

df_all = parsed["rows"].copy()
detail_sheets = parsed["detail_sheets"]
insurance_summary = parsed["insurance_summary"]
coverage_min = parsed["coverage_min"]
coverage_max = parsed["coverage_max"]

st.sidebar.markdown("### Hojas detectadas")
if detail_sheets:
    st.sidebar.success(f"{len(detail_sheets)} hoja(s) operativa(s) detectada(s)")
    for s in detail_sheets:
        st.sidebar.caption(f"• {s}")
else:
    st.sidebar.error("No se detectaron hojas operativas")

# =========================================================
# HOJAS ACTIVAS - FIX SESSION STATE
# =========================================================
file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

if "last_file_signature" not in st.session_state:
    st.session_state.last_file_signature = file_signature

if st.session_state.last_file_signature != file_signature:
    st.session_state.last_file_signature = file_signature
    st.session_state.active_sheets = detail_sheets.copy()

if "active_sheets" not in st.session_state:
    st.session_state.active_sheets = detail_sheets.copy()

st.session_state.active_sheets = [
    s for s in st.session_state.active_sheets if s in detail_sheets
]

if not st.session_state.active_sheets and detail_sheets:
    st.session_state.active_sheets = detail_sheets.copy()

active_sheets = st.sidebar.multiselect(
    "Hojas operativas activas",
    options=detail_sheets,
    key="active_sheets",
)

effective_active_sheets = active_sheets if active_sheets else detail_sheets.copy()

df_active = get_active_rows(df_all, effective_active_sheets, detail_sheets)

if df_active.empty:
    st.error("Se detectaron hojas operativas, pero no se pudo construir un dataset activo. Revisa el parser o la estructura del archivo.")
    st.stop()

segment = st.sidebar.selectbox("Segmento", SEGMENT_OPTIONS, index=0)
period_type = st.sidebar.selectbox("Tipo de periodo", PERIOD_OPTIONS, format_func=lambda x: {"week": "Semanal", "month": "Mensual", "year": "Anual"}[x])

available_years = get_available_years(df_active)
year = st.sidebar.selectbox("Año", available_years)

sub_period = None
if period_type == "month":
    available_months = get_available_months(df_active, year)
    sub_period = st.sidebar.selectbox("Mes", available_months, format_func=lambda m: pd.Timestamp(year=year, month=int(m), day=1).strftime("%b").capitalize())
elif period_type == "week":
    available_weeks = get_available_weeks(df_active, year)
    week_map = {k: lbl for k, lbl in available_weeks}
    sub_period = st.sidebar.selectbox("Semana", list(week_map.keys()), format_func=lambda k: week_map[k])

filtered_df = filter_rows(df_active, segment, period_type, year, sub_period)
prev_period = get_previous_period(period_type, year, sub_period)
prev_df = filter_rows(df_active, segment, prev_period["period_type"], prev_period["year"], prev_period["sub_period"])

metrics = aggregate_metrics(filtered_df, period_type, year, sub_period)
prev_metrics = aggregate_metrics(prev_df, prev_period["period_type"], prev_period["year"], prev_period["sub_period"]) if not prev_df.empty else None

period_label = describe_period(period_type, year, sub_period)

# =========================================================
# HEADER META
# =========================================================
badges = [
    f'<span class="badge-pill badge-info">{len(parsed["all_sheets"])} hojas detectadas</span>',
    f'<span class="badge-pill badge-ok">{len(detail_sheets)} operativas</span>',
    f'<span class="badge-pill badge-info">{len(effective_active_sheets)} activas</span>',
]
badges.extend(build_validation_badges(metrics, insurance_summary))

st.markdown(
    f"""
    <div class="summary-box" style="margin-bottom: 1rem;">
        <div class="panel-title">Resumen ejecutivo del workbook</div>
        <div class="panel-subtitle">
            {html.escape(uploaded_file.name)} · cobertura {coverage_min.strftime('%d-%b-%Y')} a {coverage_max.strftime('%d-%b-%Y')} · {format_num(len(df_all))} registros consolidados
        </div>
        <div>{"".join(badges)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# BOARD SUMMARY
# =========================================================
board_bullets = build_board_summary(metrics, prev_metrics, targets, period_type)
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">Resumen ejecutivo board-ready</div>', unsafe_allow_html=True)
    for title, text in board_bullets:
        st.markdown(
            f"""
            <div class="summary-item">
                <div class="summary-item-title">{html.escape(title)}</div>
                <div class="summary-item-text">{html.escape(text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="panel-title">Contexto del filtro</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-mini">
            <div class="kpi-label">Periodo</div>
            <div class="kpi-value" style="font-size:1.2rem;">{html.escape(period_label)}</div>
            <div class="kpi-sub">Segmento: {html.escape(segment)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-mini">
            <div class="kpi-label">Hojas activas</div>
            <div class="kpi-value" style="font-size:1.2rem;">{format_num(len(effective_active_sheets))}</div>
            <div class="kpi-sub">{html.escape(', '.join(effective_active_sheets[:3]))}{'...' if len(effective_active_sheets) > 3 else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# KPI ROW
# =========================================================
revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
patients_target = period_target(period_type, 0, targets["monthly_patients"])

revenue_delta = safe_percent_delta(metrics["account_ventas"], prev_metrics["account_ventas"]) if prev_metrics else None
mix_delta = safe_point_delta(metrics["key_mix_pct"], prev_metrics["key_mix_pct"]) if prev_metrics else None
repeat_delta = safe_point_delta(metrics["repeat_doctors_pct"], prev_metrics["repeat_doctors_pct"]) if prev_metrics else None
patients_delta = safe_percent_delta(metrics["patients"], prev_metrics["patients"]) if prev_metrics else None

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Ingresos del periodo</div>
            <div class="kpi-value">{format_currency_compact(metrics["account_ventas"])}</div>
            <div class="kpi-sub">Meta: {format_currency_compact(revenue_target)}</div>
            <div class="kpi-sub">Proyección: {format_currency_compact(metrics["projected_ventas"])}</div>
            {trend_html(revenue_delta, "%")}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mix esp. clave</div>
            <div class="kpi-value" style="color:#10b981;">{format_pct(metrics["key_mix_pct"])}</div>
            <div class="kpi-sub">Meta: {format_pct(targets["mix_pct"])}</div>
            {trend_html(mix_delta, "pp")}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    color = "#10b981" if metrics["repeat_doctors_pct"] >= targets["repeat_doctors_pct"] else "#ef4444"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Médicos &gt; 1 paciente</div>
            <div class="kpi-value" style="color:{color};">{format_pct(metrics["repeat_doctors_pct"])}</div>
            <div class="kpi-sub">Meta: {format_pct(targets["repeat_doctors_pct"])}</div>
            {trend_html(repeat_delta, "pp")}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Pacientes únicos</div>
            <div class="kpi-value">{format_num(metrics["patients"])}</div>
            <div class="kpi-sub">Meta: {format_num(patients_target)} pacientes</div>
            <div class="kpi-sub">Ticket: {format_currency_compact(metrics["avg_ticket_ventas"])}</div>
            {trend_html(patients_delta, "%")}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# CHARTS
# =========================================================
left_chart, right_chart = st.columns([2, 1], gap="large")

with left_chart:
    fig_main = build_main_chart(df_active, segment, period_type, year, sub_period)
    st.plotly_chart(fig_main, use_container_width=True)

with right_chart:
    st.markdown('<div class="panel-title">Insights operativos</div>', unsafe_allow_html=True)
    insights = build_insights(metrics, prev_metrics, targets, period_type)
    if not insights:
        st.success("El periodo se observa estable bajo los criterios vigentes.")
    else:
        for level, title, text in insights:
            if level == "error":
                st.error(f"**{title}**\n\n{text}")
            elif level == "warning":
                st.warning(f"**{title}**\n\n{text}")
            elif level == "success":
                st.success(f"**{title}**\n\n{text}")
            else:
                st.info(f"**{title}**\n\n{text}")

# =========================================================
# SPECIALTIES + DOCTORS
# =========================================================
left_spec, right_spec = st.columns([2, 1], gap="large")

with left_spec:
    fig_spec = build_specialty_chart(metrics, prev_metrics)
    st.plotly_chart(fig_spec, use_container_width=True)

with right_spec:
    st.markdown('<div class="panel-title">Podio de facturación</div>', unsafe_allow_html=True)
    top_specialties = sorted(metrics["specialty_ventas"].items(), key=lambda x: x[1], reverse=True)[:3]
    if not top_specialties:
        st.info("Sin datos para el filtro actual.")
    else:
        medals = ["🥇", "🥈", "🥉"]
        for i, (name, value) in enumerate(top_specialties):
            prev_value = prev_metrics["specialty_ventas"].get(name, 0) if prev_metrics else None
            delta = safe_percent_delta(value, prev_value) if prev_metrics else None
            delta_txt = "Sin base" if delta is None else f"{'▲' if delta >= 0 else '▼'} {abs(delta):.1f}%"
            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-item-title">{medals[i]} {html.escape(name)}</div>
                    <div class="summary-item-text">{format_currency(value)} · {delta_txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("### Detalle por especialidad y médico")
specialty_options = sorted(filtered_df["specialty"].dropna().unique().tolist())
selected_specialty = st.selectbox("Especialidad", specialty_options) if specialty_options else None

if selected_specialty:
    doctor_df = (
        filtered_df[filtered_df["specialty"] == selected_specialty]
        .groupby("doctor", dropna=False)
        .agg(
            pacientes_unicos=("account", "nunique"),
            cuenta_full=("account_full", "sum"),
            cuenta_ventas=("account_ventas", "sum"),
        )
        .reset_index()
        .sort_values("cuenta_ventas", ascending=False)
    )

    doctor_df = doctor_df.rename(columns={"doctor": "Médico"})
    doctor_df["Cuenta Full"] = doctor_df["cuenta_full"].map(format_currency)
    doctor_df["Cuenta Ventas"] = doctor_df["cuenta_ventas"].map(format_currency)
    doctor_df["Pacientes Únicos"] = doctor_df["pacientes_unicos"].map(format_num)

    st.dataframe(
        doctor_df[["Médico", "Pacientes Únicos", "Cuenta Full", "Cuenta Ventas"]],
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# DETAIL TABLE
# =========================================================
st.markdown("### Detalle estratégico del periodo")
detail_table = build_detail_table(metrics, targets, period_type)
st.dataframe(detail_table, use_container_width=True, hide_index=True)

# =========================================================
# EXPORTS
# =========================================================
st.markdown("### Exportación")
export_col1, export_col2 = st.columns(2, gap="large")

summary_text = build_summary_text(
    workbook_name=uploaded_file.name,
    period_label=period_label,
    segment=segment,
    metrics=metrics,
    detail_df=detail_table,
)

export_excel_bytes = export_filtered_excel(filtered_df, detail_table)

with export_col1:
    st.download_button(
        label="Descargar resumen ejecutivo (.txt)",
        data=summary_text.encode("utf-8"),
        file_name=f"resumen_ejecutivo_{period_type}_{year}.txt",
        mime="text/plain",
        use_container_width=True,
    )

with export_col2:
    st.download_button(
        label="Descargar Excel filtrado (.xlsx)",
        data=export_excel_bytes,
        file_name=f"dashboard_filtrado_{period_type}_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# =========================================================
# KPI GLOSSARY
# =========================================================
with st.expander("Glosario de KPIs"):
    for _, item in KPI_DEFINITIONS.items():
        st.markdown(f"**{item['title']}**")
        st.markdown(f"- Objetivo: {item['goal']}")
        st.markdown(f"- Fórmula: `{item['formula']}`")
        st.markdown(f"- Interpretación: {item['meaning']}")
