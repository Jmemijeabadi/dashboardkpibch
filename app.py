import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA (ESTILO BI) ---
st.set_page_config(
    page_title="CRMBI Dashboard Comercial",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para dar look and feel de Dashboard Profesional
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    h1, h2, h3 { color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE BUSQUEDA (KEYWORDS) ---
# Esto ayuda a encontrar filas aunque el nombre cambie ligeramente
KEYWORDS = {
    "pacientes": ["pacientes", "no. de pacientes", "total pacientes"],
    "seguro": ["seguro", "pacientes con seguro", "asegurados"],
    "especialidades": ["row labels", "etiquetas de fila", "especialidad"]
}

ESPECIALIDADES_CLAVE = {
    "ORTOPEDIA": ["ortopedia", "trauma"],
    "CARDIOLOGIA": ["cardio"],
    "NEUROCIRUGIA": ["neuro"],
    "CX TORAX": ["torax", "tórax"],
    "CX GENERAL": ["general", "digestiva"],
    "ONCOLOGIA": ["onco"]
}

# --- 3. FUNCIONES DE PROCESAMIENTO ROBUSTO ---

def normalize_date(dt):
    """Convierte cualquier fecha al día 1 del mes para comparar fácil"""
    try:
        return pd.to_datetime(dt).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    except:
        return None

@st.cache_data
def load_excel_robust(file):
    """
    Lee el Excel buscando patrones en lugar de celdas fijas.
    Devuelve diccionarios con la data limpia: {fecha: valor}
    """
    xls = pd.ExcelFile(file)
    
    # --- A. HOJA INDICADORES ---
    # Busca hoja que contenga "ind" (insensitive)
    sheet_ind = next((s for s in xls.sheet_names if "ind" in s.lower()), None)
    
    data_indicadores = {"pacientes": {}, "seguro": {}}
    
    if sheet_ind:
        # Leemos todo como string primero para buscar
        df = pd.read_excel(xls, sheet_name=sheet_ind, header=None)
        
        # 1. Detectar fila de fechas (Header)
        # Buscamos la fila que tenga más de 3 columnas con formato fecha
        header_idx = -1
        for i, row in df.iterrows():
            dates_count = 0
            for val in row:
                if normalize_date(val) is not None:
                    dates_count += 1
            if dates_count > 2:
                header_idx = i
                break
        
        if header_idx != -1:
            fechas_row = df.iloc[header_idx]
            # Mapeamos índice de columna -> Fecha normalizada
            col_date_map = {}
            for col_idx, val in fechas_row.items():
                norm_d = normalize_date(val)
                if norm_d:
                    col_date_map[col_idx] = norm_d
            
            # 2. Buscar filas de datos (Pacientes y Seguro)
            # Iteramos filas buscando keywords en la primera columna (o segunda)
            for i, row in df.iterrows():
                if i <= header_idx: continue # Saltar header
                
                first_cell = str(row[0]).lower() + " " + str(row[1]).lower() # Unimos col 0 y 1 por si acaso
                
                # Buscar Pacientes
                if any(k in first_cell for k in KEYWORDS["pacientes"]):
                    for col_idx, date_val in col_date_map.items():
                        val = pd.to_numeric(row[col_idx], errors='coerce')
                        if not pd.isna(val): data_indicadores["pacientes"][date_val] = val
                
                # Buscar Seguro
                if any(k in first_cell for k in KEYWORDS["seguro"]) and "%" not in first_cell: # Evitar filas de porcentajes
                    for col_idx, date_val in col_date_map.items():
                        val = pd.to_numeric(row[col_idx], errors='coerce')
                        if not pd.isna(val): data_indicadores["seguro"][date_val] = val

    # --- B. HOJA ESPECIALIDADES (PIVOT) ---
    sheet_spec = next((s for s in xls.sheet_names if "medicos" in s.lower() or "esp" in s.lower()), None)
    df_specs_clean = pd.DataFrame()
    
    if sheet_spec:
        df_raw = pd.read_excel(xls, sheet_name=sheet_spec)
        # Buscar dónde empieza la tabla (fila con fechas)
        # Simplificación: Asumimos que pd.read_excel detecta el header si está en la fila 1
        # Si no, buscamos columna 'Row Labels'
        
        # Renombrar columna de etiquetas si tiene nombre raro
        for col in df_raw.columns:
            if any(k in str(col).lower() for k in KEYWORDS["especialidades"]):
                df_raw.rename(columns={col: "Especialidad"}, inplace=True)
                break
        
        if "Especialidad" in df_raw.columns:
            # Melt (Unpivot) para tener formato tabla
            # Filtramos columnas que parecen fechas
            date_cols = [c for c in df_raw.columns if normalize_date(c) is not None]
            if date_cols:
                df_specs_clean = df_raw.melt(id_vars=["Especialidad"], value_vars=date_cols, var_name="Fecha", value_name="Cantidad")
                df_specs_clean["Fecha"] = df_specs_clean["Fecha"].apply(normalize_date)
                df_specs_clean["Cantidad"] = pd.to_numeric(df_specs_clean["Cantidad"], errors='coerce').fillna(0)

    return data_indicadores, df_specs_clean

# --- 4. LÓGICA DE NEGOCIO ---

def get_kpis_config():
    """Define los KPIs base"""
    return [
        {"eje": "EJE 1 · MÉDICOS", "id": "kpi_1", "nombre": "Atracción Nuevos Médicos", "meta_def": 25, "tipo": "suma_esp", "unit": "médicos"},
        {"eje": "EJE 1 · MÉDICOS", "id": "kpi_2", "nombre": "Eventos Contacto 1:1", "meta_def": 2, "tipo": "manual", "unit": "eventos"},
        {"eje": "EJE 1 · MÉDICOS", "id": "kpi_3", "nombre": "Rentabilidad (% Seguro)", "meta_def": 0.35, "tipo": "calc_rate", "unit": "%"},
        
        {"eje": "EJE 2 · EMPRESAS", "id": "kpi_4", "nombre": "Nuevos Convenios", "meta_def": 1, "tipo": "manual", "unit": "convenios"},
        {"eje": "EJE 2 · EMPRESAS", "id": "kpi_5", "nombre": "Eventos Empresariales", "meta_def": 1, "tipo": "manual", "unit": "eventos"},
        
        {"eje": "EJE 3 · SEGUROS", "id": "kpi_6", "nombre": "Pacientes Seguro", "meta_def": 33, "tipo": "dato_seguro", "unit": "pacientes"},
        {"eje": "EJE 3 · SEGUROS", "id": "kpi_7", "nombre": "Visitas Brokers", "meta_def": 2, "tipo": "manual", "unit": "visitas"},
    ]

# --- 5. INTERFAZ PRINCIPAL ---

st.title("📊 CRMBI Comercial")
st.markdown("Sistema de seguimiento de metas comerciales dinámico.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])
    
    st.divider()
    
    st.header("2. Selector de Periodo")
    # Fechas inteligentes
    today = datetime.now()
    meses = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
             7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    col_y, col_m = st.columns(2)
    sel_year = col_y.number_input("Año", value=today.year, step=1)
    sel_month_name = col_m.selectbox("Mes", list(meses.values()), index=today.month-1)
    
    # Convertir nombre mes a número
    sel_month_num = list(meses.keys())[list(meses.values()).index(sel_month_name)]
    target_date = datetime(sel_year, sel_month_num, 1) # Fecha objetivo (día 1)
    
    st.info(f"Analizando: **{sel_month_name} {sel_year}**")
    
    if st.button("Resetear Metas"):
        st.session_state.clear()
        st.rerun()

# --- PROCESAMIENTO DATOS ---
data_ind = {"pacientes":{}, "seguro":{}}
df_specs = pd.DataFrame()
file_loaded = False

if uploaded_file:
    try:
        data_ind, df_specs = load_excel_robust(uploaded_file)
        file_loaded = True
        st.toast("Datos procesados correctamente", icon="✅")
    except Exception as e:
        st.error(f"Error procesando archivo: {e}")

# --- INSTRUCCIONES SI NO HAY DATOS ---
if not file_loaded:
    with st.expander("ℹ️ Instrucciones de Uso (¡Léeme primero!)", expanded=True):
        st.markdown("""
        **Para que el sistema detecte tus datos automáticamente:**
        
        1. **Hoja Indicadores:** Debe tener una fila con fechas (ej: `01/01/2024`) y filas con los textos "No. de Pacientes" y "Seguro".
        2. **Hoja Médicos:** Debe ser una tabla dinámica o tabla donde la primera columna sean las Especialidades y las columnas siguientes sean fechas.
        3. **Formato:** Asegúrate de que las fechas en Excel sean formato fecha y no texto.
        """)
        st.warning("⚠️ Esperando archivo Excel...")
        st.stop()

# --- CÁLCULO DE VALORES REALES ---
# Extraemos el valor para el mes seleccionado
val_pacientes = data_ind["pacientes"].get(target_date, 0)
val_seguro = data_ind["seguro"].get(target_date, 0)
val_rate = (val_seguro / val_pacientes) if val_pacientes > 0 else 0

# Cálculo especialidades
val_medicos_clave = 0
breakdown_specs = {}
if not df_specs.empty:
    # Filtrar fecha
    df_mes = df_specs[df_specs["Fecha"] == target_date]
    for grupo, keywords in ESPECIALIDADES_CLAVE.items():
        # Buscar regex
        pattern = '|'.join(keywords)
        matches = df_mes[df_mes["Especialidad"].astype(str).str.lower().str.contains(pattern)]
        suma = matches["Cantidad"].sum()
        val_medicos_clave += suma
        breakdown_specs[grupo] = suma

# --- PESTAÑAS DEL DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📈 Tablero de Control", "🔍 Análisis Detallado", "🛠️ Data Cruda (Debug)"])

with tab1:
    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Calcular Deltas (vs mes anterior)
    prev_date = (target_date - pd.DateOffset(months=1)).replace(day=1)
    prev_pac = data_ind["pacientes"].get(prev_date, 0)
    prev_seg = data_ind["seguro"].get(prev_date, 0)
    
    delta_pac = val_pacientes - prev_pac if prev_pac > 0 else 0
    delta_seg = val_seguro - prev_seg if prev_seg > 0 else 0
    
    col1.metric("Pacientes Totales", f"{val_pacientes:,.0f}", delta=f"{delta_pac:,.0f} vs mes ant")
    col2.metric("Pacientes Seguro", f"{val_seguro:,.0f}", delta=f"{delta_seg:,.0f}")
    col3.metric("% Rentabilidad", f"{val_rate:.1%}")
    col4.metric("Médicos Clave (Activos)", f"{val_medicos_clave:,.0f}")

    st.markdown("---")

    # --- TABLA DE GESTIÓN (EDITABLE) ---
    st.subheader("🎯 Gestión de Metas y Cumplimiento")
    st.caption("Edita la columna 'Meta' haciendo doble clic. Los cálculos se actualizan en tiempo real.")

    # Preparar DataFrame para el editor
    kpis = get_kpis_config()
    rows = []
    
    # Recuperar metas guardadas en session_state (Persistencia)
    session_key = f"metas_{target_date.strftime('%Y%m')}"
    if session_key not in st.session_state:
        st.session_state[session_key] = {}
        
    for k in kpis:
        # Obtener Real según tipo
        real = 0
        if k['tipo'] == 'suma_esp': real = val_medicos_clave
        elif k['tipo'] == 'dato_seguro': real = val_seguro
        elif k['tipo'] == 'calc_rate': real = val_rate
        # Manual se queda en 0 por ahora (o podrías hacerlo editable también)
        
        # Obtener Meta (Saved > Default)
        meta = st.session_state[session_key].get(k['id'], k['meta_def'])
        
        # Calcular cumplimiento
        cump = (real / meta) if meta > 0 else 0
        
        # Formateo visual
        is_pct = k['unit'] == "%"
        real_show = real * 100 if is_pct else real
        meta_show = meta * 100 if is_pct else meta
        
        rows.append({
            "Eje": k['eje'],
            "ID": k['id'], # Oculto
            "Indicador": k['nombre'],
            "Unidad": k['unit'],
            "Meta": float(meta_show), # Editable
            "Real (Auto)": float(real_show),
            "Cumplimiento": cump, # Barra progreso
            "Estado": "✅" if cump >= 1 else ("⚠️" if cump >= 0.8 else "🔻")
        })
    
    df_editor = pd.DataFrame(rows)
    
    # CONFIGURAR COLUMN CONFIG
    edited_df = st.data_editor(
        df_editor,
        column_config={
            "ID": None, # Ocultar
            "Meta": st.column_config.NumberColumn(
                "Meta Objetivo",
                help="Edita este valor",
                step=0.1,
                format="%.1f"
            ),
            "Real (Auto)": st.column_config.NumberColumn(
                "Real (Detectado)",
                format="%.1f",
                disabled=True # No editable
            ),
            "Cumplimiento": st.column_config.ProgressColumn(
                "% Logro",
                format="%.0f%%",
                min_value=0,
                max_value=1.5, # Cap visual en 150%
            ),
            "Eje": st.column_config.TextColumn("Categoría", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="editor_main"
    )
    
    # GUARDAR CAMBIOS
    # Si el usuario edita, actualizamos session_state
    for index, row in edited_df.iterrows():
        # Revertir porcentaje si es necesario para guardar el número puro
        is_pct = row['Unidad'] == "%"
        val_to_save = row['Meta'] / 100 if is_pct else row['Meta']
        st.session_state[session_key][row['ID']] = val_to_save

with tab2:
    st.subheader("Tendencias Históricas")
    
    # Crear DF histórico
    dates = sorted(list(data_ind["pacientes"].keys()))
    hist_data = []
    for d in dates:
        p = data_ind["pacientes"].get(d, 0)
        s = data_ind["seguro"].get(d, 0)
        hist_data.append({"Fecha": d, "Pacientes": p, "Seguro": s})
    
    if hist_data:
        df_hist = pd.DataFrame(hist_data)
        fig = px.line(df_hist, x="Fecha", y=["Pacientes", "Seguro"], markers=True, 
                      title="Evolución Pacientes vs Seguro", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay suficientes datos históricos para graficar.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Desglose Especialidades (Mes Actual)")
        if breakdown_specs:
            df_pie = pd.DataFrame(list(breakdown_specs.items()), columns=["Espec", "Cant"])
            fig_pie = px.pie(df_pie, names="Espec", values="Cant", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay datos de médicos para este mes.")

with tab3:
    st.error("Zona de Debug - Usa esto si los datos no coinciden")
    st.write("Datos extraídos de hoja 'Indicadores':")
    st.json({str(k): v for k, v in data_ind.items()}) # Convert dates to str for json
    
    st.write("Primeras filas de hoja 'Médicos':")
    st.dataframe(df_specs.head())
