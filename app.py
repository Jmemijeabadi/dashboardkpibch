import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="CRMBI Comercial", page_icon="📊")

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- DEFINICIÓN DE ESPECIALIDADES CLAVE ---
KEY_SPECIALTIES_MAP = {
    "ORTOPEDIA": ["ORTOPEDIA"],
    "CARDIOLOGIA": ["CARDIO"],
    "NEUROCIRUGIA": ["NEURO"],
    "CX TORAX": ["TORAX", "TÓRAX"],
    "CX GENERAL": ["CIRUGIA GENERAL", "CIRUGÍA GENERAL", "CX GENERAL"],
    "ONCOLOGIA": ["ONCO"]
}

# --- CONFIGURACIÓN DE KPIs ---
# Estructura base para inicializar datos
def get_kpi_config():
    return {
        "EJE 1": [
            {"id": "e1_medicos", "indicador": "Atracción médicos nuevos", "estrategia": "Suma Esp. Clave (Ort, Car, Neu...)", "unit": "médicos", "default": 25, "source": "calc_specialties"},
            {"id": "e1_eventos", "indicador": "Atracción médicos nuevos", "estrategia": "Eventos contacto 1:1", "unit": "eventos", "default": 2, "source": "manual"},
            {"id": "e1_atencion", "indicador": "Atención de médicos", "estrategia": "Atenciones 1:1 seguimiento", "unit": "atenciones", "default": 80, "source": "manual"},
            {"id": "e1_rentabilidad", "indicador": "Rentabilidad (proxy)", "estrategia": "% Seguro / Pacientes", "unit": "%", "default": 0.35, "source": "calc_rate"},
        ],
        "EJE 2": [
            {"id": "e2_prospectos", "indicador": "Nuevos convenios", "estrategia": "Prospectos empresas", "unit": "prospectos", "default": 6, "source": "manual"},
            {"id": "e2_cierres", "indicador": "Nuevos convenios", "estrategia": "Cierres convenio", "unit": "cierres", "default": 1, "source": "manual"},
            {"id": "e2_eventos_emp", "indicador": "Eventos", "estrategia": "Eventos empresariales", "unit": "eventos", "default": 1, "source": "manual"},
        ],
        "EJE 3": [
            {"id": "e3_pacientes_seg", "indicador": "Pacientes de seguros", "estrategia": "Pacientes con seguro (Excel)", "unit": "pacientes", "default": 33, "source": "excel_insured"},
            {"id": "e3_visitas", "indicador": "Pacientes de seguros", "estrategia": "Visitas brokers", "unit": "visitas", "default": 2, "source": "manual"},
        ]
    }

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---

@st.cache_data
def load_excel_data(uploaded_file):
    """Lee el Excel y extrae las DataFrames limpias"""
    xls = pd.ExcelFile(uploaded_file)
    
    # 1. PROCESAR HOJA "INDICADORES"
    # Buscamos la hoja flexiblemente (case insensitive)
    sheet_ind = next((s for s in xls.sheet_names if "indicadores" in s.lower()), None)
    df_ind = pd.DataFrame()
    
    if sheet_ind:
        # Leemos sin encabezado para buscar palabras clave
        raw = pd.read_excel(xls, sheet_name=sheet_ind, header=None)
        
        # Buscar fila de fechas (asumimos que la fila con fechas tiene muchos datetimes)
        # Una estrategia simple: buscar la celda "No. de Pacientes" y las fechas suelen estar en esa fila o la anterior
        # Aquí simplificamos buscando la fila donde empieza la data
        
        # En tu lógica JS buscabas "Estadistica", "No. de Pacientes", "Seguro"
        # Vamos a intentar localizar esas filas
        row_pacientes = raw[raw.apply(lambda row: row.astype(str).str.contains("No. de Pacientes", case=False).any(), axis=1)].index
        row_seguro = raw[raw.apply(lambda row: row.astype(str).str.contains("Seguro", case=False).any(), axis=1)].index
        
        if not row_pacientes.empty and not row_seguro.empty:
            idx_p = row_pacientes[0]
            idx_s = row_seguro[0]
            
            # Asumimos que la fila de encabezados (fechas) es la misma fila o una arriba
            # Para generalizar, buscaremos las columnas que son fechas en la fila de Pacientes (o la fila superior)
            # Extracción simple: Transponer y limpiar
            
            def extract_series(row_idx, raw_df):
                row_data = raw_df.iloc[row_idx]
                # Intentamos parsear fechas de las columnas
                # Frecuentemente los excel tienen headers arriba. Vamos a asumir estructura:
                # Col 0: Label, Col 1..N: Fechas
                vals = []
                dates = []
                
                # Buscamos la fila de fechas (Header)
                # Buscamos la fila que tiene "Estadistica" o dates
                header_row_idx = raw_df[raw_df.apply(lambda r: r.astype(str).str.contains("Estadistica", case=False).any(), axis=1)].index
                if header_row_idx.empty: return pd.Series()
                header_vals = raw_df.iloc[header_row_idx[0]]
                
                clean_data = {}
                for col in raw_df.columns:
                    val_header = header_vals[col]
                    # Intentar convertir header a fecha
                    try:
                        date_obj = pd.to_datetime(val_header)
                        if not pd.isna(date_obj):
                             # Normalizar al día 1 del mes
                            d_key = date_obj.replace(day=1)
                            val_data = pd.to_numeric(raw_df.iloc[row_idx, col], errors='coerce')
                            clean_data[d_key] = val_data
                    except:
                        continue
                return pd.Series(clean_data).sort_index()

            s_patients = extract_series(idx_p, raw)
            s_insured = extract_series(idx_s, raw)
            
            df_ind = pd.DataFrame({'pacientes': s_patients, 'seguro': s_insured})
            df_ind['rate'] = df_ind['seguro'] / df_ind['pacientes']

    # 2. PROCESAR HOJA "MEDICOS POR ESPECIALIDAD"
    sheet_spec = next((s for s in xls.sheet_names if "medicos" in s.lower() and "especialidad" in s.lower()), None)
    df_spec = pd.DataFrame()
    
    if sheet_spec:
        # Leemos buscando la estructura Pivot
        raw_s = pd.read_excel(xls, sheet_name=sheet_spec, header=None)
        # Buscar "Row Labels" o "Etiquetas de fila"
        start_row = raw_s[raw_s.apply(lambda r: r.astype(str).str.contains("Row Label|Etiquetas", case=False, regex=True).any(), axis=1)].index
        
        if not start_row.empty:
            idx_head = start_row[0]
            # Seteamos esa fila como header
            raw_s.columns = raw_s.iloc[idx_head]
            df_sliced = raw_s.iloc[idx_head+1:].copy()
            
            # La primera columna es la especialidad
            spec_col = df_sliced.columns[0]
            df_sliced = df_sliced.rename(columns={spec_col: 'Especialidad'})
            df_sliced = df_sliced.dropna(subset=['Especialidad'])
            
            # Filtrar columnas que no sean fecha
            date_cols = []
            for c in df_sliced.columns:
                try:
                    d = pd.to_datetime(c)
                    if not pd.isna(d): date_cols.append(c)
                except: pass
            
            # Unpivot (Melt) para tener formato tabla: Fecha, Especialidad, Cantidad
            df_melt = df_sliced.melt(id_vars=['Especialidad'], value_vars=date_cols, var_name='Fecha', value_name='Cantidad')
            df_melt['Fecha'] = pd.to_datetime(df_melt['Fecha']).apply(lambda x: x.replace(day=1))
            df_melt['Cantidad'] = pd.to_numeric(df_melt['Cantidad'], errors='coerce').fillna(0)
            df_spec = df_melt
            
    return df_ind, df_spec

def calculate_key_specialties_sum(df_spec, target_date):
    if df_spec.empty: return 0, {}
    
    # Filtrar por fecha
    mask_date = df_spec['Fecha'] == target_date
    df_period = df_spec[mask_date]
    
    total = 0
    details = {}
    
    for key, keywords in KEY_SPECIALTIES_MAP.items():
        # Buscar especialidades que coincidan con keywords
        # Regex: palabra contenida
        pattern = '|'.join([k.lower() for k in keywords])
        matches = df_period[df_period['Especialidad'].astype(str).str.lower().str.contains(pattern, regex=True)]
        val = matches['Cantidad'].sum()
        details[key] = val
        total += val
        
    return total, details

# --- INTERFAZ PRINCIPAL ---

# Sidebar para controles
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Cargar Excel (CRMBI)", type=['xlsx', 'xls'])
    
    # Selectores de Fecha
    col1, col2 = st.columns(2)
    today = datetime.now()
    current_year = today.year
    months = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
              7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    with col1:
        sel_year = st.selectbox("Año", range(current_year-2, current_year+2), index=2)
    with col2:
        sel_month_idx = st.selectbox("Mes", list(months.keys()), format_func=lambda x: months[x], index=today.month-1)

    target_date = datetime(sel_year, sel_month_idx, 1)
    
    st.divider()
    st.caption(f"Periodo seleccionado: {target_date.strftime('%B %Y')}")
    
    if st.button("Resetear Metas del Mes"):
        key_pattern = f"targets_{target_date.strftime('%Y-%m')}"
        if key_pattern in st.session_state:
            del st.session_state[key_pattern]
        st.rerun()

# --- LÓGICA DE DATOS ---

if uploaded_file:
    try:
        df_ind, df_spec = load_excel_data(uploaded_file)
        data_loaded = True
        st.toast("Datos cargados correctamente", icon="✅")
    except Exception as e:
        st.error(f"Error leyendo el Excel: {e}")
        data_loaded = False
        df_ind, df_spec = pd.DataFrame(), pd.DataFrame()
else:
    data_loaded = False
    df_ind, df_spec = pd.DataFrame(), pd.DataFrame()

# --- DASHBOARD HEADER ---

st.title(f"Dashboard Comercial: {months[sel_month_idx]} {sel_year}")

# --- TARJETAS SUPERIORES (KPIs MACRO) ---
col_k1, col_k2, col_k3 = st.columns(3)

if data_loaded and not df_ind.empty:
    # Datos del mes actual
    try:
        curr_row = df_ind.loc[target_date]
        val_pacientes = curr_row['pacientes']
        val_seguro = curr_row['seguro']
        val_rate = curr_row['rate']
    except KeyError:
        val_pacientes = val_seguro = val_rate = 0

    # Datos mes anterior (MoM)
    prev_date = (target_date - pd.DateOffset(months=1)).replace(day=1)
    try:
        prev_row = df_ind.loc[prev_date]
        delta_pacientes = val_pacientes - prev_row['pacientes']
        delta_seguro = val_seguro - prev_row['seguro']
        delta_rate = val_rate - prev_row['rate']
    except KeyError:
        delta_pacientes = delta_seguro = delta_rate = None

    with col_k1:
        st.metric("Pacientes Totales", f"{val_pacientes:,.0f}", delta=f"{delta_pacientes:,.0f}" if delta_pacientes else None)
    with col_k2:
        st.metric("Pacientes Seguro", f"{val_seguro:,.0f}", delta=f"{delta_seguro:,.0f}" if delta_seguro else None)
    with col_k3:
        st.metric("% Seguro / Pacientes", f"{val_rate:.1%}", delta=f"{delta_rate:.1%}" if delta_rate else None)
else:
    col_k1.metric("Pacientes", "—")
    col_k2.metric("Seguro", "—")
    col_k3.metric("Tasa Seguro", "—")

# --- GRÁFICOS DE TENDENCIA ---
if data_loaded and not df_ind.empty:
    with st.expander("Ver Tendencias y Gráficos", expanded=True):
        g1, g2 = st.columns(2)
        
        # Filtramos últimos 12 meses para que no sea enorme
        df_chart = df_ind.sort_index().tail(12).reset_index().rename(columns={'index':'Fecha'})
        
        fig_p = px.line(df_chart, x='Fecha', y=['pacientes', 'seguro'], markers=True, title="Tendencia Pacientes vs Seguro")
        fig_p.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None, height=300)
        g1.plotly_chart(fig_p, use_container_width=True)
        
        # Gráfico de Especialidades Clave (Mes actual)
        key_total, key_breakdown = calculate_key_specialties_sum(df_spec, target_date)
        df_breakdown = pd.DataFrame(list(key_breakdown.items()), columns=['Especialidad', 'Cantidad'])
        
        fig_b = px.bar(df_breakdown, x='Especialidad', y='Cantidad', title="Especialidades Clave (Mes Actual)", text_auto=True)
        fig_b.update_layout(height=300, xaxis_title=None)
        g2.plotly_chart(fig_b, use_container_width=True)

# --- TABLAS DETALLADAS POR EJE (EDITABLES) ---

# Gestión de estado para las metas
session_key = f"targets_{target_date.strftime('%Y-%m')}"
if session_key not in st.session_state:
    st.session_state[session_key] = {}

# Función para obtener valor Real dinámico
def get_actual_value(source_type):
    if not data_loaded: return 0
    if source_type == "manual": return 0 # O podría ser otro input manual
    if source_type == "calc_specialties":
        val, _ = calculate_key_specialties_sum(df_spec, target_date)
        return val
    if source_type == "excel_insured":
        return val_seguro if 'val_seguro' in locals() else 0
    if source_type == "calc_rate":
        return val_rate if 'val_rate' in locals() else 0
    return 0

st.write("---")
kpi_config = get_kpi_config()

# Iteramos por cada EJE
for eje_name, kpis in kpi_config.items():
    st.subheader(f"📌 {eje_name}")
    
    # Preparamos los datos para la tabla
    table_data = []
    
    # Recuperamos overrides guardados
    saved_targets = st.session_state[session_key]

    for k in kpis:
        real_val = get_actual_value(k['source'])
        
        # Meta: Check session state first, then default
        target_val = saved_targets.get(k['id'], k['default'])
        
        # Cumplimiento
        if target_val > 0:
            compliance = real_val / target_val
        else:
            compliance = 0
            
        # Semáforo (Emoji)
        if compliance >= 1: status = "🟢"
        elif compliance >= 0.8: status = "🟡"
        else: status = "🔴"
        
        # Formato bonito para el Real
        if k['unit'] == "%":
            real_fmt = f"{real_val:.1%}"
        else:
            real_fmt = f"{real_val:,.0f}"

        table_data.append({
            "ID": k['id'], # Hidden index
            "Indicador": k['indicador'],
            "Estrategia": k['estrategia'],
            "Meta (Editable)": float(target_val), # Float para que el editor permita decimales si hace falta
            "Real": real_fmt,
            "Cumplimiento": f"{compliance:.1%}",
            "Estado": status,
            "_raw_real": real_val # Para cálculos internos si hiciera falta
        })
    
    df_eje = pd.DataFrame(table_data)
    
    # Configuración del Data Editor
    # Permitimos editar SOLO la columna "Meta (Editable)"
    edited_df = st.data_editor(
        df_eje,
        column_config={
            "ID": None, # Ocultar
            "_raw_real": None, # Ocultar
            "Meta (Editable)": st.column_config.NumberColumn(
                "Meta",
                help="Modifica este valor y presiona Enter",
                min_value=0,
                step=1,
                format="%.2f" # Permite decimales visualmente
            ),
            "Estado": st.column_config.TextColumn("Status", width="small")
        },
        disabled=["Indicador", "Estrategia", "Real", "Cumplimiento", "Estado"],
        hide_index=True,
        key=f"editor_{eje_name}_{session_key}", # Clave única por mes para resetear el editor si cambia el mes
        use_container_width=True
    )
    
    # --- DETECTAR CAMBIOS Y GUARDAR ---
    # Comparamos el df original (targets calculados) con el editado
    # Si hay cambios en "Meta (Editable)", actualizamos session_state
    
    # Nota: st.data_editor devuelve el DF modificado. 
    # Extraemos las metas nuevas y las guardamos.
    for index, row in edited_df.iterrows():
        kpi_id = row['ID']
        new_target = row['Meta (Editable)']
        # Guardamos en el diccionario de sesión
        st.session_state[session_key][kpi_id] = new_target

# --- DEBUG OPTIONAL ---
with st.expander("🔧 Debug Data (Raw)"):
    st.write("Datos extraídos de Indicadores:")
    st.dataframe(df_ind.tail())
    st.write("Metas guardadas en sesión para este mes:")
    st.json(st.session_state.get(session_key, {}))
