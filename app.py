import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CRMBI Operativo - Grupo Abadi",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2E86C1;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    h3 { font-size: 1.2rem !important; font-weight: 600; }
    .stProgress > div > div > div > div { background-color: #2E86C1; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE MAPEO (ESPECIALIDADES) ---
# Mapeamos tus grupos de interés con los nombres reales que vimos en el Excel
SPECIALTY_GROUPS = {
    "ORTOPEDIA": ["ORTOPEDIA", "TRAUMATOLOGIA"],
    "CARDIOLOGIA": ["CARDIOLOGIA", "CARDIO", "HEMODINAMIA"],
    "NEUROCIRUGIA": ["NEUROCIRUGIA", "NEURO"],
    "CX TORAX": ["TORAX", "TÓRAX", "CARDIO TORAXICO"],
    "CX GENERAL": ["CIRUGIA GENERAL", "GASTRO", "DIGESTIVA", "PROCTOLOGIA", "COLOPROCTOLOGIA"],
    "ONCOLOGIA": ["ONCO", "ONCOLOGICA"]
}

# --- 3. MOTOR DE LECTURA DE EXCEL (A MEDIDA) ---

@st.cache_data
def load_hospital_data(file):
    xls = pd.ExcelFile(file)
    
    data_kpis = {"pacientes": {}, "seguro": {}}
    df_specs = pd.DataFrame()
    
    # ---------------------------------------------------------
    # A. PROCESAR HOJA "Indicadores"
    # ---------------------------------------------------------
    if "Indicadores" in xls.sheet_names:
        # Leemos sin header para buscar coordenadas exactas
        df_ind = pd.read_excel(xls, sheet_name="Indicadores", header=None)
        
        # 1. Buscar fila de encabezados (Fechas)
        # En tu scan, la fila 3 (índice 3) tiene "Estadistica" y fechas
        # Buscamos la fila que contenga "Estadistica"
        header_row_idx = -1
        for idx, row in df_ind.iterrows():
            row_str = row.astype(str).str.lower()
            if row_str.str.contains("estadistica").any():
                header_row_idx = idx
                break
        
        if header_row_idx != -1:
            # Extraer mapa de columnas de fechas
            date_map = {} # {col_index: datetime_obj}
            row_dates = df_ind.iloc[header_row_idx]
            
            for col_idx, val in row_dates.items():
                try:
                    dt = pd.to_datetime(val)
                    if not pd.isna(dt) and isinstance(dt, datetime):
                        # Normalizar al día 1
                        date_map[col_idx] = dt.replace(day=1, hour=0, minute=0, second=0)
                except:
                    continue
            
            # 2. Buscar filas de datos (Pacientes y Seguro)
            # Iteramos buscando los labels en la columna 1 (B) usualmente
            for idx, row in df_ind.iterrows():
                if idx <= header_row_idx: continue
                
                # Convertimos toda la fila a string para buscar texto clave
                row_text = " ".join(row.astype(str).tolist()).lower()
                
                series_key = None
                if "no. de pacientes" in row_text:
                    series_key = "pacientes"
                elif "seguro" in row_text and "latino" not in row_text and "%" not in row_text and "total" not in row_text:
                    # Ojo: tu excel tiene "Seguro", "Total Seguro sin Latino", "LA LATINO..."
                    # La fila 5 es solo "Seguro", asumimos esa es la meta global
                    # Para ser precisos, validamos que la celda clave sea exactamente "Seguro" o parecida
                    first_cells = [str(x).strip() for x in row.iloc[0:5].values]
                    if "Seguro" in first_cells: 
                        series_key = "seguro"
                
                if series_key:
                    for col_idx, date_obj in date_map.items():
                        val = pd.to_numeric(row[col_idx], errors='coerce')
                        if not pd.isna(val):
                            data_kpis[series_key][date_obj] = val

    # ---------------------------------------------------------
    # B. PROCESAR HOJA "Medicos por Especialidad"
    # ---------------------------------------------------------
    if "Medicos por Especialidad" in xls.sheet_names:
        # Leemos buscando la estructura Pivot
        df_raw = pd.read_excel(xls, sheet_name="Medicos por Especialidad", header=None)
        
        # Buscar fila con "Row Labels" (Tu scan dice fila 2)
        start_idx = -1
        for idx, row in df_raw.iterrows():
            if row.astype(str).str.contains("Row Labels").any():
                start_idx = idx
                break
        
        if start_idx != -1:
            # Definir esa fila como header
            df_raw.columns = df_raw.iloc[start_idx]
            df_spec_clean = df_raw.iloc[start_idx+1:].copy()
            
            # Renombrar columna índice
            col_label = [c for c in df_spec_clean.columns if "Row Labels" in str(c)]
            if col_label:
                df_spec_clean = df_spec_clean.rename(columns={col_label[0]: "Especialidad"})
                
                # Quedarnos solo con columnas de Fecha y Especialidad
                # Las fechas vendrán como columnas tipo datetime o string
                cols_to_keep = ["Especialidad"]
                date_cols = []
                
                for c in df_spec_clean.columns:
                    try:
                        dt = pd.to_datetime(c)
                        if not pd.isna(dt) and isinstance(dt, datetime):
                            cols_to_keep.append(c)
                            date_cols.append(c)
                    except:
                        pass
                
                df_final = df_spec_clean[cols_to_keep]
                
                # Unpivot (Melt) para tener tabla plana
                if date_cols:
                    df_melt = df_final.melt(id_vars=["Especialidad"], value_vars=date_cols, var_name="Fecha", value_name="Cantidad")
                    df_melt["Fecha"] = pd.to_datetime(df_melt["Fecha"]).apply(lambda x: x.replace(day=1))
                    df_melt["Cantidad"] = pd.to_numeric(df_melt["Cantidad"], errors='coerce').fillna(0)
                    df_specs = df_melt

    return data_kpis, df_specs

# --- 4. INTERFAZ Y LÓGICA ---

st.title("📊 Tablero de Control Comercial")
st.markdown("Monitor de Indicadores Hospitalarios")

# SIDEBAR
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Cargar Reporte Operativo (.xlsx)", type=["xlsx"])
    
    st.divider()
    
    # Fechas
    today = datetime.now()
    years = [today.year - 1, today.year, today.year + 1]
    months = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
              7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    col1, col2 = st.columns(2)
    sel_year = col1.selectbox("Año", years, index=1)
    sel_month_idx = col2.selectbox("Mes", list(months.keys()), format_func=lambda x: months[x], index=today.month-1)
    
    target_date = datetime(sel_year, sel_month_idx, 1)
    
    if st.button("Recargar Datos"):
        st.cache_data.clear()
        st.rerun()

# LOGICA PRINCIPAL
if uploaded_file:
    try:
        kpis, df_specs = load_hospital_data(uploaded_file)
        
        # --- Obtener valores del mes seleccionado ---
        val_pacientes = kpis["pacientes"].get(target_date, 0)
        val_seguro = kpis["seguro"].get(target_date, 0)
        val_tasa = (val_seguro / val_pacientes) if val_pacientes > 0 else 0
        
        # Calcular especialidades clave
        val_medicos_clave = 0
        if not df_specs.empty:
            df_mes = df_specs[df_specs["Fecha"] == target_date]
            # Sumar si coincide con keywords
            for _, row in df_mes.iterrows():
                spec_name = str(row["Especialidad"]).upper()
                count = row["Cantidad"]
                # Checar si pertenece a algun grupo clave
                for grp, keywords in SPECIALTY_GROUPS.items():
                    if any(k in spec_name for k in keywords):
                        val_medicos_clave += count
                        break # Solo contar una vez por fila
        
        # --- HEADER KPIS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pacientes Totales", f"{val_pacientes:,.0f}")
        c2.metric("Pacientes Seguro", f"{val_seguro:,.0f}")
        c3.metric("% Rentabilidad", f"{val_tasa:.1%}")
        c4.metric("Médicos Clave (Activos)", f"{val_medicos_clave:,.0f}")
        
        st.divider()
        
        # --- TABLA DE METAS EDITABLE ---
        st.subheader(f"🎯 Gestión de Metas - {months[sel_month_idx]} {sel_year}")
        
        # Definición de KPIs Base
        default_kpis = [
            {"Eje": "EJE 1 · MÉDICOS", "Indicador": "Atracción Médicos (Clave)", "Real": val_medicos_clave, "Meta": 25, "Unidad": "médicos"},
            {"Eje": "EJE 1 · MÉDICOS", "Indicador": "Rentabilidad (% Seguro)", "Real": val_tasa, "Meta": 0.35, "Unidad": "%"},
            {"Eje": "EJE 1 · MÉDICOS", "Indicador": "Eventos Contacto 1:1", "Real": 0, "Meta": 2, "Unidad": "eventos"},
            
            {"Eje": "EJE 2 · EMPRESAS", "Indicador": "Nuevos Convenios", "Real": 0, "Meta": 1, "Unidad": "convenios"},
            
            {"Eje": "EJE 3 · SEGUROS", "Indicador": "Pacientes Seguro (Volumen)", "Real": val_seguro, "Meta": 33, "Unidad": "pacientes"},
            {"Eje": "EJE 3 · SEGUROS", "Indicador": "Visitas Brokers", "Real": 0, "Meta": 2, "Unidad": "visitas"},
        ]
        
        # Crear DataFrame para editar
        # Usamos Session State para guardar cambios manuales si el usuario edita
        key_editor = f"editor_{sel_year}_{sel_month_idx}"
        
        if key_editor not in st.session_state:
            st.session_state[key_editor] = pd.DataFrame(default_kpis)
        
        # Actualizar los valores REALES automáticos (por si cambiamos de mes pero mantenemos metas editadas)
        # Solo actualizamos las filas que son automáticas, respetando las manuales
        curr_df = st.session_state[key_editor]
        # Actualizar fila Medicos
        curr_df.loc[curr_df["Indicador"]=="Atracción Médicos (Clave)", "Real"] = val_medicos_clave
        curr_df.loc[curr_df["Indicador"]=="Rentabilidad (% Seguro)", "Real"] = val_tasa
        curr_df.loc[curr_df["Indicador"]=="Pacientes Seguro (Volumen)", "Real"] = val_seguro
        
        # Calcular cumplimiento
        curr_df["Cumplimiento"] = curr_df.apply(lambda x: x["Real"]/x["Meta"] if x["Meta"]>0 else 0, axis=1)
        
        # Mostrar Editor
        edited_df = st.data_editor(
            curr_df,
            column_config={
                "Real": st.column_config.NumberColumn(format="%.2f"),
                "Meta": st.column_config.NumberColumn(format="%.2f"),
                "Cumplimiento": st.column_config.ProgressColumn(
                    format="%.0f%%",
                    min_value=0, max_value=1.5,
                    width="medium"
                ),
            },
            hide_index=True,
            use_container_width=True,
            key="main_editor"
        )
        
        # Guardar cambios
        st.session_state[key_editor] = edited_df
        
        # --- GRÁFICOS ---
        st.subheader("📈 Tendencias")
        t1, t2 = st.tabs(["Histórico Pacientes", "Desglose Especialidades"])
        
        with t1:
            # Crear DF Histórico
            if kpis["pacientes"]:
                hist_data = []
                for d, p in kpis["pacientes"].items():
                    s = kpis["seguro"].get(d, 0)
                    hist_data.append({"Fecha": d, "Pacientes": p, "Seguro": s})
                
                df_hist = pd.DataFrame(hist_data).sort_values("Fecha")
                fig = px.line(df_hist, x="Fecha", y=["Pacientes", "Seguro"], markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with t2:
            if not df_specs.empty:
                # Top 10 especialidades del mes
                top_specs = df_mes.groupby("Especialidad")["Cantidad"].sum().sort_values(ascending=False).head(10)
                fig2 = px.bar(top_specs, orientation='h', title="Top 10 Especialidades (Mes Actual)")
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        st.warning("Asegúrate de subir el archivo 'Grupo Abadi - Hospital Reporte Operativo.xlsx'")
        with st.expander("Ver detalle del error"):
            st.write(e)

else:
    # Pantalla de bienvenida
    st.info("👈 Por favor carga el archivo Excel en el menú lateral para comenzar.")
    st.markdown("""
    ### Estructura esperada del Excel:
    1. **Hoja Indicadores:** Debe contener la fila 'Estadistica' con fechas y las filas 'No. de Pacientes' y 'Seguro'.
    2. **Hoja Medicos por Especialidad:** Debe ser una tabla dinámica con 'Row Labels' y fechas en columnas.
    """)
