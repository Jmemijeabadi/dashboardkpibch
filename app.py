import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN ESTRATÉGICA Y CONSTANTES
# ==========================================
st.set_page_config(page_title="NewCityHospital BI | KO26", layout="wide", page_icon="🏥")

KEYWORDS = {
    'DATE': ['mes ingreso', 'ingreso', 'fecha', 'mes'],
    'REVENUE': ['cuenta ventas', 'cuenta full', 'importe', 'total', 'ingreso'],
    'SPECIALTY': ['especialidad grupo', 'especialidad', 'servicio'],
    'DOCTOR': ['medico grupo', 'medico', 'doctor'],
    'PATIENT_ID': ['cuenta', '# cuenta', 'expediente', '#', 'paciente'],
    'TYPE': ['tipo'] 
}

TARGETS_ANNUAL = {
    'revenue': 268700000,           
    'rev_empresas': 36000000,       
    'rev_medicos_nuevos': 44000000, 
    'rev_medicos_act': 181000000,   
    'rev_fidelidad': 8000000,       
    'mezcla_pct': 60,               
    'retorno_pct': 65,              
    'volumen_mensual': 330,         
    'medicos_nuevos': 34,           
    'pacientes_seguro': 330         
}

ESP_CLAVES = ['ORTOPEDIA', 'TRAUMATOLOGIA', 'CARDIOLOGIA', 'NEUROCIRUGIA', 'TORÁCICA', 'TORACICA', 'GENERAL']

# ==========================================
# 2. ESTILOS VISUALES (CSS Personalizado)
# ==========================================
st.markdown("""
    <style>
    .kpi-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .kpi-label { font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { font-size: 2rem; color: #0f172a; font-weight: 700; margin-bottom: 5px; }
    .trend-up { color: #10b981; font-size: 0.85rem; font-weight: 600; background: #dcfce7; padding: 2px 8px; border-radius: 10px;}
    .trend-down { color: #ef4444; font-size: 0.85rem; font-weight: 600; background: #fee2e2; padding: 2px 8px; border-radius: 10px;}
    .trend-neutral { color: #64748b; font-size: 0.85rem; font-weight: 600; background: #f1f5f9; padding: 2px 8px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNCIONES DE PROCESAMIENTO
# ==========================================
def format_money(val):
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif abs(val) >= 1_000:
        return f"${val/1_000:.1f}k"
    return f"${val:,.0f}"

@st.cache_data
def process_data(file):
    df = pd.read_excel(file)
    
    # Mapeo inteligente de columnas
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for key, words in KEYWORDS.items():
            if any(w in col_lower for w in words) and key not in col_map:
                col_map[key] = col
                
    if 'REVENUE' not in col_map or 'DATE' not in col_map:
        st.error("No se encontraron las columnas clave (Fecha e Ingresos) en el Excel.")
        return pd.DataFrame()

    # Renombrar columnas para facilitar el manejo
    rename_dict = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename_dict)
    
    # Limpieza básica
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df['REVENUE'] = pd.to_numeric(df.get('REVENUE', 0).astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce').fillna(0)
    df['SPECIALTY'] = df.get('SPECIALTY', 'Otros').fillna('Otros').astype(str).str.upper()
    df['DOCTOR'] = df.get('DOCTOR', 'Desconocido').fillna('Desconocido').astype(str).str.upper()
    df['PATIENT_ID'] = df.get('PATIENT_ID', df.index).fillna(df.index).astype(str)
    df['TYPE'] = df.get('TYPE', 'Privado').fillna('Privado').astype(str).str.upper()

    # Columnas derivadas
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    df['isKey'] = df['SPECIALTY'].apply(lambda x: any(k in x for k in ESP_CLAVES))
    df['isEmpresa'] = df['TYPE'].apply(lambda x: any(k in x for k in ['EMPRESA', 'CONVENIO', 'COMPASS']))
    df['isSeguro'] = df['TYPE'].apply(lambda x: any(k in x for k in ['SEGURO', 'METLIFE', 'GNP']))
    
    def get_segment(row):
        if row['isEmpresa']: return 'Empresa'
        if row['isSeguro']: return 'Seguro'
        return 'Privado'
    df['segment'] = df.apply(get_segment, axis=1)

    return df

def aggregate_data(df, year, period_type, period_val, segment_filter):
    # Filtrar por segmento
    if segment_filter != 'Todos':
        df = df[df['segment'] == segment_filter]
        
    # Filtrar por año y periodo actual
    df_curr = df[df['year'] == year]
    df_prev = df[df['year'] == (year - 1)] # Default previous (Año anterior completo)

    if period_type == 'Mes':
        df_curr = df_curr[df_curr['month'] == period_val]
        df_prev = df[df['year'] == year][df['month'] == (period_val - 1 if period_val > 1 else 12)]
        if period_val == 1: df_prev = df[df['year'] == (year - 1)][df['month'] == 12]
    elif period_type == 'Semestre':
        months = [1, 2, 3, 4, 5, 6] if period_val == 1 else [7, 8, 9, 10, 11, 12]
        prev_months = [7, 8, 9, 10, 11, 12] if period_val == 1 else [1, 2, 3, 4, 5, 6]
        df_curr = df_curr[df_curr['month'].isin(months)]
        df_prev = df[df['year'] == (year - 1 if period_val == 1 else year)][df['month'].isin(prev_months)]

    def calc_metrics(d):
        if d.empty:
            return {'rev':0, 'mezcla':0, 'retorno':0, 'occ':0, 'pacientes':0, 'seguros':0, 'empresas':0, 'nuevos_rev':0, 'specialties': {}}
        
        rev = d['REVENUE'].sum()
        key_rev = d[d['isKey']]['REVENUE'].sum()
        mezcla = (key_rev / rev * 100) if rev > 0 else 0
        
        docs = d.groupby('DOCTOR')['PATIENT_ID'].nunique()
        retorno = (len(docs[docs > 1]) / len(docs) * 100) if len(docs) > 0 else 0
        
        pacientes = d['PATIENT_ID'].nunique()
        occ = (pacientes / 1100) * 100 # Regla de negocio del código original
        
        specialties = d.groupby('SPECIALTY')['REVENUE'].sum().sort_values(ascending=False).to_dict()
        
        return {
            'rev': rev, 'mezcla': mezcla, 'retorno': retorno, 'occ': occ, 
            'pacientes': pacientes, 'seguros': d[d['isSeguro']]['PATIENT_ID'].nunique(),
            'empresas': d[d['isEmpresa']]['REVENUE'].sum(), 'nuevos_rev': 0, # Simplificado
            'specialties': specialties,
            'raw': d
        }

    return calc_metrics(df_curr), calc_metrics(df_prev)

# ==========================================
# 4. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🏥 NewCityHospital BI 2026")
st.markdown("Estrategia KO26 - Sistema de Análisis Predictivo")

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls", "csv"])
    
    st.header("Filtros Estratégicos")
    segment = st.selectbox("Segmento", ["Todos", "Privado", "Seguro", "Empresa"])
    period_type = st.selectbox("Periodo", ["Mes", "Semestre", "Año"])
    
    if uploaded_file:
        df = process_data(uploaded_file)
        if not df.empty:
            years = sorted(df['year'].dropna().unique().tolist(), reverse=True)
            year = st.selectbox("Año", years)
            
            if period_type == "Mes":
                meses_str = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
                # Tomar el mes máximo disponible para ese año por defecto
                max_month = int(df[df['year'] == year]['month'].max())
                period_val_str = st.selectbox("Mes", meses_str, index=max_month-1)
                period_val = meses_str.index(period_val_str) + 1
            elif period_type == "Semestre":
                period_val = st.selectbox("Semestre", [1, 2], format_func=lambda x: f"Semestre {x}")
            else:
                period_val = None
    else:
        st.info("👈 Sube un archivo Excel para comenzar el análisis.")
        st.stop()

# --- CÁLCULO DE MÉTRICAS ---
curr, prev = aggregate_data(df, year, period_type, period_val, segment)

# Ajuste de metas según periodo
factor = 12 if period_type == 'Mes' else (2 if period_type == 'Semestre' else 1)
targets = {
    'rev': TARGETS_ANNUAL['revenue'] / factor,
    'mezcla': TARGETS_ANNUAL['mezcla_pct'],
    'retorno': TARGETS_ANNUAL['retorno_pct'],
    'occ': 50 if (period_type == 'Mes' and period_val >= 4) else 30 # Regla original
}

# --- TARJETAS KPI ---
col1, col2, col3, col4 = st.columns(4)

def kpi_html(title, val_str, target_str, delta, is_pct=False, reverse_colors=False):
    trend_class = "trend-neutral"
    icon = "➖"
    if delta > 0.1:
        trend_class = "trend-down" if reverse_colors else "trend-up"
        icon = "📈" if not reverse_colors else "📉"
    elif delta < -0.1:
        trend_class = "trend-up" if reverse_colors else "trend-down"
        icon = "📉" if not reverse_colors else "📈"
        
    delta_str = f"{abs(delta):.1f}{'pp' if is_pct else '%'}"
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{val_str}</div>
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:10px;">
            <div style="font-size:0.8rem; color:#64748b;">Meta: {target_str}</div>
            <div class="{trend_class}">{icon} {delta_str} vs Prev</div>
        </div>
    </div>
    """

with col1:
    delta_rev = ((curr['rev'] - prev['rev']) / prev['rev'] * 100) if prev['rev'] else 0
    st.markdown(kpi_html("Ingresos del Segmento", format_money(curr['rev']), format_money(targets['rev']), delta_rev), unsafe_allow_html=True)

with col2:
    delta_mix = curr['mezcla'] - prev['mezcla']
    st.markdown(kpi_html("Mezcla Esp. Clave", f"{curr['mezcla']:.1f}%", f"{targets['mezcla']}%", delta_mix, is_pct=True), unsafe_allow_html=True)

with col3:
    delta_ret = curr['retorno'] - prev['retorno']
    st.markdown(kpi_html("Retorno Médicos (>1 Cx)", f"{curr['retorno']:.1f}%", f"{targets['retorno']}%", delta_ret, is_pct=True), unsafe_allow_html=True)

with col4:
    delta_occ = curr['occ'] - prev['occ']
    st.markdown(kpi_html("Ocupación Camas", f"{curr['occ']:.1f}%", f"{targets['occ']}%", delta_occ, is_pct=True), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- GRÁFICAS PRINCIPALES ---
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.markdown("#### Tendencia Financiera vs Mezcla")
    # Generar datos mensuales para la gráfica principal
    df_year = df[(df['year'] == year) & (df['segment'] == segment if segment != 'Todos' else True)]
    monthly_data = df_year.groupby('month').apply(
        lambda x: pd.Series({
            'Ingresos': x['REVENUE'].sum(),
            'Mezcla': (x[x['isKey']]['REVENUE'].sum() / x['REVENUE'].sum() * 100) if x['REVENUE'].sum() > 0 else 0
        })
    ).reset_index()
    
    # Rellenar meses faltantes
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_data = pd.merge(all_months, monthly_data, on='month', how='left').fillna(0)
    meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    monthly_data['Mes'] = meses_nombres
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Bar(x=monthly_data['Mes'], y=monthly_data['Ingresos'], name="Ingresos ($)", marker_color='#0f172a', yaxis='y1'))
    fig_main.add_trace(go.Scatter(x=monthly_data['Mes'], y=monthly_data['Mezcla'], name="Mix Clave (%)", marker_color='#10b981', mode='lines+markers', yaxis='y2', line=dict(width=3)))
    
    fig_main.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Ingresos", side="left", showgrid=False),
        yaxis2=dict(title="Mix Clave (%)", side="right", overlaying="y", showgrid=False, range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_main, use_container_width=True)

with chart_col2:
    st.markdown("#### Ejes Rectores e Insights")
    # Smart Insights lógicos
    if curr['rev'] < (targets['rev'] * 0.9):
        st.error(f"**Ingreso Desfasado**: Facturación actual ({format_money(curr['rev'])}) muy por debajo de la meta del periodo.")
    else:
        st.success("**Ingresos Saludables**: Cumplimiento económico en línea con la meta.")
        
    if curr['occ'] < targets['occ']:
        st.warning(f"**Baja Ocupación**: {curr['occ']:.1f}%. Se requiere impulsar flujo hospitalario.")
        
    if curr['retorno'] < targets['retorno']:
        st.error("**Alerta de Retención Médica**: Alta cantidad de doctores realizando solo 1 procedimiento al mes.")
    
    st.info("**Pacientes Aseguradoras**: " + str(curr['seguros']) + f" captados en el periodo.")

st.markdown("<hr/>", unsafe_allow_html=True)

# --- ESPECIALIDADES Y TOP 3 ---
col_spec, col_top = st.columns([2, 1])

with col_spec:
    st.markdown("#### Top 5 Especialidades (Ingresos)")
    specs_df = pd.DataFrame(list(curr['specialties'].items()), columns=['Especialidad', 'Ingresos']).head(5)
    
    if not specs_df.empty:
        specs_df = specs_df.sort_values(by='Ingresos', ascending=True) # Ascending for Plotly horizontal bar
        fig_spec = px.bar(specs_df, x='Ingresos', y='Especialidad', orientation='h', text_auto='.2s')
        fig_spec.update_traces(marker_color='#3b82f6', textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig_spec.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis_title=None, margin=dict(l=0, r=50, t=0, b=0), height=300)
        st.plotly_chart(fig_spec, use_container_width=True)
    else:
        st.write("No hay datos de especialidades.")

with col_top:
    st.markdown("#### Podio de Rentabilidad (Top 3)")
    top3 = pd.DataFrame(list(curr['specialties'].items()), columns=['Especialidad', 'Ingresos']).head(3)
    for i, row_data in top3.iterrows():
        spec_name = row_data['Especialidad']
        rev_val = row_data['Ingresos']
        prev_val = prev['specialties'].get(spec_name, 0)
        delta = ((rev_val - prev_val) / prev_val * 100) if prev_val > 0 else 0
        color = "green" if delta >= 0 else "red"
        arrow = "▲" if delta >= 0 else "▼"
        
        st.markdown(f"""
        <div style="padding: 10px; border-bottom: 1px solid #eee; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <b style="font-size:1.1rem; color:#0f172a;">#{i+1} {spec_name[:20]}</b><br>
                <span style="color:#64748b;">{format_money(rev_val)}</span>
            </div>
            <div style="color:{color}; font-weight:bold; font-size:0.9rem; background:rgba({'0,200,0' if delta>=0 else '255,0,0'},0.1); padding:2px 8px; border-radius:5px;">
                {arrow} {abs(delta):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- DRILL DOWN DE MÉDICOS (Alternativa a Modal) ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔍 Ver Detalle de Médicos por Especialidad (Drill-down)", expanded=False):
    if not df_year.empty:
        selected_spec = st.selectbox("Selecciona una especialidad para ver sus médicos:", options=curr['specialties'].keys())
        
        doc_df = curr['raw'][curr['raw']['SPECIALTY'] == selected_spec]
        doc_summary = doc_df.groupby('DOCTOR').agg(
            Pacientes_Atendidos=('PATIENT_ID', 'nunique'),
            Ingresos_Generados=('REVENUE', 'sum')
        ).reset_index().sort_values(by='Ingresos_Generados', ascending=False)
        
        # Formato bonito para el dataframe
        st.dataframe(
            doc_summary.style.format({'Ingresos_Generados': '${:,.2f}'}),
            use_container_width=True,
            hide_index=True
        )
