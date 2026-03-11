import io
import html
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

SEGMENTS = ["Todos", "Privado", "Seguro", "Empresa"]
PERIODS = ["week", "month", "year"]
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
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        .ga-title {
            font-size: 1.7rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #0f172a;
            margin-bottom: 0.1rem;
        }
        .ga-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background: linear-gradient(180deg,#ffffff 0%,#fbfdff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 28px -24px rgba(15,23,42,0.18);
            min-height: 146px;
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
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
            line-height: 1.02;
        }
        .kpi-sub {
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 0.28rem;
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
            box-shadow: 0 14px 28px -24px rgba(15,23,42,0.18);
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
            font-size: 0.9rem;
            margin-bottom: 0.12rem;
        }
        .summary-item-text {
            color: #475569;
            font-size: 0.83rem;
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
def normalize_text(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def format_currency(value: float) -> str:
    return f"${float(value or 0):,.0f}"


def format_currency_compact(value: float) -> str:
    n = float(value or 0)
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.0f}k"
    return format_currency(n)


def format_num(value: float) -> str:
    return f"{float(value or 0):,.0f}"


def format_pct(value: float) -> str:
    return f"{float(value or 0):.1f}%"


def parse_number_locale(value: Any) -> float:
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


def parse_excel_date(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float)) and 20_000 < float(value) < 60_000:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.notna(parsed):
        return parsed.normalize()

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


def dedupe_headers(raw_headers: List[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    out: List[str] = []
    for idx, h in enumerate(raw_headers):
        base = normalize_text(h) or f"col_{idx+1}"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return out


def base_header(header: str) -> str:
    return header.split("__")[0]


def find_header(headers: List[str], candidates: List[str]) -> Optional[str]:
    norms = [normalize_text(c) for c in candidates]
    for candidate in norms:
        for h in headers:
            if base_header(h) == candidate:
                return h
    return None


def classify_segment(type_value: str, insurance_group: str) -> str:
    t = clean_text(type_value).upper()
    s = clean_text(insurance_group).upper()

    if "EMPRESA" in t or "CONVENIO" in t or "B2B" in t:
        return "Empresa"
    if "SEGURO" in t or (s and s not in {"SIN SEGURO", "PARTICULAR", "PRIVADO"}):
        return "Seguro"
    return "Privado"


def is_key_specialty(name: str) -> bool:
    s = clean_text(name).upper()
    return any(k in s for k in KEY_SPECIALTIES)


def start_of_week(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt):
        return pd.NaT
    return (dt - pd.to_timedelta(dt.weekday(), unit="D")).normalize()


def describe_period(period_type: str, year: int, sub_period: Optional[Any]) -> str:
    if period_type == "year":
        return f"Año {year}"
    if period_type == "month":
        return pd.Timestamp(year=year, month=int(sub_period), day=1).strftime("%b %Y").capitalize()
    if period_type == "week":
        dt = pd.to_datetime(sub_period)
        return f"Semana {dt.strftime('%d-%b-%Y')}"
    return "Periodo"


def period_target(period_type: str, annual_value: float, monthly_value: float = 0) -> float:
    if period_type == "year":
        return annual_value
    if period_type == "month":
        return monthly_value or annual_value / 12
    if period_type == "week":
        return (monthly_value * 12 / 52) if monthly_value else annual_value / 52
    return annual_value


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


# =========================================================
# PARSER DE WORKBOOK
# =========================================================
def detect_detail_sheet(df_raw: pd.DataFrame) -> Optional[int]:
    required_tokens = {
        "cuenta",
        "ingreso",
        "egreso",
        "cuenta ventas",
        "tipo",
        "especialidad grupo",
        "seguro grupo",
        "medico grupo",
    }
    for i in range(min(len(df_raw), 15)):
        values = [normalize_text(v) for v in df_raw.iloc[i].tolist()]
        score = sum(token in values for token in required_tokens)
        if score >= 5:
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

    headers = dedupe_headers(df_raw.iloc[header_idx].tolist())
    body = df_raw.iloc[header_idx + 1 :].copy()
    body.columns = headers

    label_col = find_header(headers, ["row labels"])
    patients_col = find_header(headers, ["pacientes"])
    full_col = find_header(headers, ["cuenta full"])
    ventas_col = find_header(headers, ["cuenta ventas"])

    if label_col is None:
        return None

    body = body[body[label_col].astype(str).str.strip() != ""].copy()

    rows: List[dict] = []
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
        rows.append(item)

    return {"sheet_name": sheet_name, "items": rows, "grand_total": grand_total}


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

    out = pd.DataFrame(index=body.index.copy())
    out["account"] = body[account_col].map(clean_text)
    out["date"] = body[admission_col].map(parse_excel_date)
    out["discharge_date"] = body[discharge_col].map(parse_excel_date) if discharge_col else pd.NaT
    out["doctor"] = body[doctor_col].map(clean_text).replace("", "SIN MÉDICO")
    out["specialty"] = body[specialty_col].map(clean_text).replace("", "SIN ESPECIALIDAD")
    out["insurance_group"] = body[insurance_group_col].map(clean_text) if insurance_group_col else "SIN SEGURO"
    out["type"] = body[type_col].map(clean_text).replace("", "Privado")
    out["account_ventas"] = body[ventas_col].map(parse_number_locale)
    out["account_full"] = body[full_col].map(parse_number_locale) if full_col else out["account_ventas"]
    out["utility"] = body[utility_col].map(parse_number_locale) if utility_col else pd.NA
    out["source_sheet"] = sheet_name

    out = out[pd.notna(out["date"])].copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["week_start"] = out["date"].map(start_of_week)
    out["week_key"] = out["week_start"].dt.strftime("%Y-%m-%d")
    out["segment"] = out.apply(lambda r: classify_segment(r["type"], r["insurance_group"]), axis=1)
    out["is_key_specialty"] = out["specialty"].map(is_key_specialty)

    out["length_of_stay"] = (
        (out["discharge_date"] - out["date"]).dt.days if "discharge_date" in out.columns else pd.NA
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

    detail_frames: List[pd.DataFrame] = []
    detail_sheets: List[str] = []
    insurance_summary = None

    for sheet_name, df_raw in excel.items():
        header_idx = detect_detail_sheet(df_raw)

        if header_idx is not None:
            parsed = parse_detail_sheet(df_raw, sheet_name, header_idx)
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
    return sorted(df.loc[df["year"] == year, "month"].dropna().astype(int).unique().tolist())


def get_available_weeks(df: pd.DataFrame, year: int) -> List[Tuple[str, str]]:
    subset = (
        df[df["year"] == year][["week_key", "week_start"]]
        .drop_duplicates()
        .sort_values("week_start")
    )
    rows: List[Tuple[str, str]] = []
    for _, row in subset.iterrows():
        start = row["week_start"]
        end = start + pd.Timedelta(days=6)
        rows.append((row["week_key"], f"{start.strftime('%d-%b-%Y')} - {end.strftime('%d-%b-%Y')}"))
    return rows


def filter_rows(df: pd.DataFrame, segment: str, period_type: str, year: int, sub_period: Optional[Any]) -> pd.DataFrame:
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


def get_previous_period(period_type: str, year: int, sub_period: Optional[Any]) -> Dict[str, Any]:
    if period_type == "year":
        return {"period_type": "year", "year": year - 1, "sub_period": None}

    if period_type == "month":
        month = int(sub_period)
        if month == 1:
            return {"period_type": "month", "year": year - 1, "sub_period": 12}
        return {"period_type": "month", "year": year, "sub_period": month - 1}

    dt = pd.to_datetime(sub_period) - pd.Timedelta(days=7)
    return {"period_type": "week", "year": dt.year, "sub_period": dt.strftime("%Y-%m-%d")}


def aggregate_metrics(df: pd.DataFrame, period_type: str, year: int, sub_period: Optional[Any]) -> Dict[str, Any]:
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
            "segment_ventas": {},
            "doctor_stats": pd.DataFrame(columns=["doctor", "patients", "account_full", "account_ventas"]),
        }

    patients = int(df["account"].nunique())
    insured_patients = int(df.loc[df["segment"] == "Seguro", "account"].nunique())

    doctor_stats = (
        df.groupby("doctor", dropna=False)
        .agg(
            patients=("account", "nunique"),
            account_full=("account_full", "sum"),
            account_ventas=("account_ventas", "sum"),
        )
        .reset_index()
        .sort_values("account_ventas", ascending=False)
        .reset_index(drop=True)
    )

    total_doctors = int(len(doctor_stats))
    repeat_doctors = int((doctor_stats["patients"] > 1).sum())

    account_ventas = float(df["account_ventas"].sum())
    account_full = float(df["account_full"].sum())
    key_revenue = float(df.loc[df["is_key_specialty"], "account_ventas"].sum())
    key_mix_pct = (key_revenue / account_ventas * 100) if account_ventas > 0 else 0.0

    avg_stay = (
        float(df["length_of_stay"].dropna().mean())
        if "length_of_stay" in df.columns and df["length_of_stay"].notna().any()
        else 0.0
    )

    projected_ventas = account_ventas
    if period_type == "month":
        max_day = int(df["day"].max())
        days_in_month = pd.Timestamp(year=year, month=int(sub_period), day=1).days_in_month
        if 0 < max_day < days_in_month:
            projected_ventas = (account_ventas / max_day) * days_in_month
    elif period_type == "week":
        observed_days = df["date"].dt.normalize().nunique()
        if 0 < observed_days < 7:
            projected_ventas = (account_ventas / observed_days) * 7
    elif period_type == "year":
        max_month = int(df["month"].max())
        if 0 < max_month < 12:
            projected_ventas = (account_ventas / max_month) * 12

    specialty_ventas = (
        df.groupby("specialty", dropna=False)["account_ventas"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    segment_ventas = (
        df.groupby("segment", dropna=False)["account_ventas"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "account_ventas": account_ventas,
        "account_full": account_full,
        "projected_ventas": projected_ventas,
        "patients": patients,
        "insured_patients": insured_patients,
        "avg_ticket_ventas": account_ventas / patients if patients else 0.0,
        "avg_ticket_full": account_full / patients if patients else 0.0,
        "key_mix_pct": key_mix_pct,
        "total_doctors": total_doctors,
        "repeat_doctors": repeat_doctors,
        "repeat_doctors_pct": (repeat_doctors / total_doctors * 100) if total_doctors else 0.0,
        "avg_stay": avg_stay,
        "specialty_ventas": specialty_ventas,
        "segment_ventas": segment_ventas,
        "doctor_stats": doctor_stats,
    }


# =========================================================
# CHARTS
# =========================================================
def build_main_chart(df_active: pd.DataFrame, segment: str, period_type: str, year: int, sub_period: Optional[Any]) -> go.Figure:
    labels: List[str] = []
    revs: List[float] = []
    mixes: List[float] = []

    if period_type == "year":
        title = f"Tendencia mensual {year}"
        subtitle = "Cuenta Ventas vs Mix por mes"
        for month in range(1, 13):
            subset = filter_rows(df_active, segment, "month", year, month)
            agg = aggregate_metrics(subset, "month", year, month)
            labels.append(pd.Timestamp(year=year, month=month, day=1).strftime("%b"))
            revs.append(agg["account_ventas"])
            mixes.append(agg["key_mix_pct"])

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
            agg = aggregate_metrics(subset, "month", year, sub_period)
            labels.append(str(day))
            revs.append(agg["account_ventas"])
            mixes.append(agg["key_mix_pct"])

    else:
        title = f"Tendencia diaria · {describe_period('week', year, sub_period)}"
        subtitle = "Cuenta Ventas vs Mix por día"
        subset = filter_rows(df_active, segment, "week", year, sub_period)
        for date_key, day_df in subset.groupby(subset["date"].dt.normalize()):
            agg = aggregate_metrics(day_df, "week", year, sub_period)
            labels.append(pd.to_datetime(date_key).strftime("%d-%b"))
            revs.append(agg["account_ventas"])
            mixes.append(agg["key_mix_pct"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="Cuenta Ventas", x=labels, y=revs, marker_color="#0f172a"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Mix Clave %",
            x=labels,
            y=mixes,
            mode="lines+markers",
            line=dict(color="#10b981", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        template="plotly_white",
        height=390,
        margin=dict(l=10, r=10, t=72, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Cuenta Ventas", secondary_y=False)
    fig.update_yaxes(title_text="Mix %", secondary_y=True)
    return fig


def build_specialty_chart(curr_metrics: Dict[str, Any], prev_metrics: Optional[Dict[str, Any]]) -> go.Figure:
    items = sorted(curr_metrics["specialty_ventas"].items(), key=lambda x: x[1], reverse=True)[:5]

    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []

    for name, value in items:
        prev_val = 0 if not prev_metrics else prev_metrics["specialty_ventas"].get(name, 0)
        labels.append(name)
        values.append(value)
        colors.append("#10b981" if value >= prev_val else "#ef4444")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[format_currency(v) for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top 5 especialidades · Cuenta Ventas",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=40, t=55, b=20),
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


def build_segment_donut(metrics: Dict[str, Any]) -> go.Figure:
    labels = list(metrics["segment_ventas"].keys())
    values = list(metrics["segment_ventas"].values())
    if not values:
        labels, values = ["Sin datos"], [1]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                marker=dict(colors=["#0f172a", "#3b82f6", "#10b981", "#f59e0b"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Composición por segmento",
        template="plotly_white",
        height=340,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    return fig


# =========================================================
# INSIGHTS / BOARD
# =========================================================
def build_board_summary(metrics: Dict[str, Any], prev_metrics: Optional[Dict[str, Any]], targets: Dict[str, float], period_type: str) -> List[Tuple[str, str]]:
    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
    bullets: List[Tuple[str, str]] = []

    if metrics["account_ventas"] >= revenue_target:
        bullets.append((
            "Ingreso alineado o superior a meta",
            f"El periodo registra {format_currency(metrics['account_ventas'])} frente a una meta de {format_currency(revenue_target)}.",
        ))
    elif period_type != "year" and metrics["projected_ventas"] >= revenue_target:
        bullets.append((
            "Meta recuperable por proyección",
            f"El ingreso actual es {format_currency(metrics['account_ventas'])}, pero la proyección de cierre alcanza {format_currency(metrics['projected_ventas'])}.",
        ))
    else:
        bullets.append((
            "Brecha económica en el corte",
            f"Los ingresos actuales son {format_currency(metrics['account_ventas'])} contra una meta de {format_currency(revenue_target)}.",
        ))

    bullets.append((
        "Composición estratégica",
        f"El mix de especialidades clave se ubica en {format_pct(metrics['key_mix_pct'])}.",
    ))
    bullets.append((
        "Volumen observable",
        f"{format_num(metrics['patients'])} pacientes únicos en el periodo; {format_num(metrics['insured_patients'])} pertenecen al segmento seguro.",
    ))
    bullets.append((
        "Recurrencia médica",
        f"{format_pct(metrics['repeat_doctors_pct'])} de los médicos activos atendió más de un paciente único.",
    ))
    return bullets[:4]


def build_insights(metrics: Dict[str, Any], prev_metrics: Optional[Dict[str, Any]], targets: Dict[str, float], period_type: str) -> List[Tuple[str, str, str]]:
    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
    insured_target = period_target(period_type, 0, targets["monthly_insured"])
    insights: List[Tuple[str, str, str]] = []

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


def build_detail_table(metrics: Dict[str, Any], targets: Dict[str, float], period_type: str) -> pd.DataFrame:
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

    formatted: List[Dict[str, str]] = []
    for label, value, target, fmt, action in rows:
        ok = value >= target

        if fmt == "money":
            gap_txt = format_currency(value - target)
            value_txt = format_currency(value)
            target_txt = format_currency(target)
        elif fmt == "pct":
            gap_txt = f"{value - target:.1f}pp"
            value_txt = format_pct(value)
            target_txt = format_pct(target)
        else:
            gap_txt = format_num(value - target)
            value_txt = format_num(value)
            target_txt = format_num(target)

        formatted.append(
            {
                "Métrica": label,
                "Actual": value_txt,
                "Meta": target_txt,
                "Brecha": gap_txt,
                "Estado": "Cumple" if ok else "Brecha",
                "Acción sugerida": "Mantener disciplina operativa." if ok else action,
            }
        )

    return pd.DataFrame(formatted)


def build_validation_badges(metrics: Dict[str, Any], insurance_summary: Optional[dict]) -> List[str]:
    badges: List[str] = []

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
def build_summary_text(workbook_name: str, period_label: str, segment: str, metrics: Dict[str, Any], detail_df: pd.DataFrame) -> str:
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
# EMPTY STATE
# =========================================================
st.markdown('<div class="ga-title">NewCity Hospital BI 2026</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ga-subtitle">Dashboard ejecutivo KO26 · Business Intelligence para ingresos, mezcla, recurrencia médica y volumen hospitalario</div>',
    unsafe_allow_html=True,
)

if not uploaded_file:
    st.info("Carga un workbook para activar el dashboard.")
    st.stop()


# =========================================================
# PARSE FILE
# =========================================================
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
# ACTIVE SHEETS
# =========================================================
file_signature = hashlib.md5(f"{uploaded_file.name}_{uploaded_file.size}".encode()).hexdigest()
sheet_widget_key = f"active_sheets_widget_{file_signature}"

default_active_sheets = detail_sheets.copy()
effective_active_sheets = st.sidebar.multiselect(
    "Hojas operativas activas",
    options=detail_sheets,
    default=default_active_sheets,
    key=sheet_widget_key,
)

if not effective_active_sheets:
    effective_active_sheets = detail_sheets.copy()

df_active = get_active_rows(df_all, effective_active_sheets, detail_sheets)

if df_active.empty:
    st.error("Se detectaron hojas operativas, pero no se pudo construir un dataset activo. Revisa el parser o la estructura del archivo.")
    st.stop()


# =========================================================
# FILTERS
# =========================================================
segment = st.sidebar.selectbox("Segmento", SEGMENTS, index=0)
period_type = st.sidebar.selectbox(
    "Tipo de periodo",
    PERIODS,
    format_func=lambda x: {"week": "Semanal", "month": "Mensual", "year": "Anual"}[x],
)

available_years = get_available_years(df_active)
year = st.sidebar.selectbox("Año", available_years)

sub_period = None
if period_type == "month":
    available_months = get_available_months(df_active, year)
    sub_period = st.sidebar.selectbox(
        "Mes",
        available_months,
        format_func=lambda m: pd.Timestamp(year=year, month=int(m), day=1).strftime("%b").capitalize(),
    )
elif period_type == "week":
    available_weeks = get_available_weeks(df_active, year)
    week_map = {k: lbl for k, lbl in available_weeks}
    sub_period = st.sidebar.selectbox(
        "Semana",
        list(week_map.keys()),
        format_func=lambda k: week_map[k],
    )


filtered_df = filter_rows(df_active, segment, period_type, year, sub_period)
prev_period = get_previous_period(period_type, year, sub_period)
prev_df = filter_rows(df_active, segment, prev_period["period_type"], prev_period["year"], prev_period["sub_period"])

metrics = aggregate_metrics(filtered_df, period_type, year, sub_period)
prev_metrics = (
    aggregate_metrics(prev_df, prev_period["period_type"], prev_period["year"], prev_period["sub_period"])
    if not prev_df.empty
    else None
)

period_label = describe_period(period_type, year, sub_period)


# =========================================================
# HEADER SUMMARY
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
# TABS
# =========================================================
tab_exec, tab_med, tab_data = st.tabs(
    ["Resumen Ejecutivo", "Especialidades y Médicos", "Calidad de Datos y Exportación"]
)

with tab_exec:
    board_bullets = build_board_summary(metrics, prev_metrics, targets, period_type)
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">Lectura board-ready</div>', unsafe_allow_html=True)
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

    revenue_target = period_target(period_type, targets["annual_revenue"], targets["annual_revenue"] / 12)
    patients_target = period_target(period_type, 0, targets["monthly_patients"])

    revenue_delta = safe_percent_delta(metrics["account_ventas"], prev_metrics["account_ventas"]) if prev_metrics else None
    mix_delta = safe_point_delta(metrics["key_mix_pct"], prev_metrics["key_mix_pct"]) if prev_metrics else None
    repeat_delta = safe_point_delta(metrics["repeat_doctors_pct"], prev_metrics["repeat_doctors_pct"]) if prev_metrics else None
    patients_delta = safe_percent_delta(metrics["patients"], prev_metrics["patients"]) if prev_metrics else None

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
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

    with c2:
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

    with c3:
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

    with c4:
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

    l1, l2 = st.columns([2, 1], gap="large")
    with l1:
        st.plotly_chart(build_main_chart(df_active, segment, period_type, year, sub_period), use_container_width=True)
    with l2:
        st.markdown('<div class="panel-title">Insights automáticos</div>', unsafe_allow_html=True)
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

    l3, l4 = st.columns([2, 1], gap="large")
    with l3:
        st.plotly_chart(build_specialty_chart(metrics, prev_metrics), use_container_width=True)
    with l4:
        st.plotly_chart(build_segment_donut(metrics), use_container_width=True)

    st.markdown("### Detalle estratégico del periodo")
    detail_table = build_detail_table(metrics, targets, period_type)
    st.dataframe(detail_table, use_container_width=True, hide_index=True)

with tab_med:
    st.markdown("### Especialidades, médicos y productividad")
    left_spec, right_spec = st.columns([1, 1], gap="large")

    with left_spec:
        top_specialties = sorted(metrics["specialty_ventas"].items(), key=lambda x: x[1], reverse=True)[:10]
        if top_specialties:
            specialty_df = pd.DataFrame(top_specialties, columns=["Especialidad", "Cuenta Ventas"])
            specialty_df["Cuenta Ventas"] = specialty_df["Cuenta Ventas"].map(format_currency)
            st.dataframe(specialty_df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin especialidades para el filtro actual.")

    with right_spec:
        top_doctors = metrics["doctor_stats"].copy()
        if not top_doctors.empty:
            top_doctors["Pacientes"] = top_doctors["patients"].map(format_num)
            top_doctors["Cuenta Full"] = top_doctors["account_full"].map(format_currency)
            top_doctors["Cuenta Ventas"] = top_doctors["account_ventas"].map(format_currency)
            st.dataframe(
                top_doctors.rename(columns={"doctor": "Médico"})[
                    ["Médico", "Pacientes", "Cuenta Full", "Cuenta Ventas"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Sin médicos para el filtro actual.")

    st.markdown("### Detalle por especialidad")
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
        doctor_df["Pacientes Únicos"] = doctor_df["pacientes_unicos"].map(format_num)
        doctor_df["Cuenta Full"] = doctor_df["cuenta_full"].map(format_currency)
        doctor_df["Cuenta Ventas"] = doctor_df["cuenta_ventas"].map(format_currency)

        st.dataframe(
            doctor_df[["Médico", "Pacientes Únicos", "Cuenta Full", "Cuenta Ventas"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_data:
    st.markdown("### Calidad de datos")
    q1, q2, q3, q4 = st.columns(4, gap="medium")

    with q1:
        st.metric("Registros visibles", format_num(len(filtered_df)))
    with q2:
        st.metric("Cuentas únicas visibles", format_num(filtered_df["account"].nunique() if not filtered_df.empty else 0))
    with q3:
        st.metric("Médicos visibles", format_num(filtered_df["doctor"].nunique() if not filtered_df.empty else 0))
    with q4:
        st.metric("Especialidades visibles", format_num(filtered_df["specialty"].nunique() if not filtered_df.empty else 0))

    st.markdown("### Muestra de datos filtrados")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.markdown("### Exportación")
    detail_table = build_detail_table(metrics, targets, period_type)
    summary_text = build_summary_text(
        workbook_name=uploaded_file.name,
        period_label=period_label,
        segment=segment,
        metrics=metrics,
        detail_df=detail_table,
    )
    export_excel_bytes = export_filtered_excel(filtered_df, detail_table)

    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.download_button(
            label="Descargar resumen ejecutivo (.txt)",
            data=summary_text.encode("utf-8"),
            file_name=f"resumen_ejecutivo_{period_type}_{year}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with e2:
        st.download_button(
            label="Descargar Excel filtrado (.xlsx)",
            data=export_excel_bytes,
            file_name=f"dashboard_filtrado_{period_type}_{year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander("Glosario de KPIs"):
        for _, item in KPI_DEFINITIONS.items():
            st.markdown(f"**{item['title']}**")
            st.markdown(f"- Objetivo: {item['goal']}")
            st.markdown(f"- Fórmula: `{item['formula']}`")
            st.markdown(f"- Interpretación: {item['meaning']}")
