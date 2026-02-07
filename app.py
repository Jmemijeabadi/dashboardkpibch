import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(page_title="CRMBI Comercial", layout="wide")

# Estilos personalizados (CSS)
st.markdown("""
    <style>
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-value { font-size: 32px; font-weight: bold; color: #1E88E5; }
    .kpi-label { font-size: 14px; color: #666; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGICA DE PROCESAMIENTO ---
@st.cache_data
def load_data(file):
    # Cargar hojas - Se usa openpyxl como motor implícito
    try:
        df_ind = pd.read_excel(file, sheet_name="Indicadores", header=None)
        df_spec = pd.read_excel(file, sheet_name="Medicos por Especialidad")
        return df_ind, df_spec
    except Exception as e:
        return None, None

# --- SIDEBAR / FILTROS ---
st.sidebar.title("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu Excel", type=["xlsx"])
selected_year = st.sidebar.selectbox("Año", [2024, 2025, 2026], index=1)
selected_month = st.sidebar.selectbox("Mes", 
    ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"])

if uploaded_file:
    df_ind, df_spec = load_data(uploaded_file)
    
    if df_ind is not None:
        # --- DASHBOARD ---
        st.title(f"📊 Dashboard CRMBI - {selected_month} {selected_year}")
        
        # --- SECCIÓN KPI PRINCIPALES ---
        col1, col2, col3, col4 = st.columns(4)
        
        # Ejemplo de datos (PLACEHOLDER - Aquí conectarás tus celdas reales luego)
        val_pacientes = 1250 
        val_seguro = 450    
        pct_seguro = (val_seguro / val_pacientes) * 100

        with col1:
            st.metric("Total Pacientes", f"{val_pacientes}", "+5% MoM")
        with col2:
            st.metric("Pacientes Seguro", f"{val_seguro}", "-2% YoY")
        with col3:
            st.metric("% Penetración Seguro", f"{pct_seguro:.1f}%")
        with col4:
            st.metric("Meta Mensual", "85%", "En camino")

        st.divider()

        # --- GRÁFICAS ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📈 Tendencia de Pacientes")
            # Datos dummy para visualización
            fig_line = px.line(x=["Ene", "Feb", "Mar", "Abr"], y=[1100, 1200, 1150, 1250], markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        with c2:
            st.subheader("🩺 Médicos por Especialidad Clave")
            spec_labels = ["Ortopedia", "Cardio", "Neuro", "Tórax", "Onco"]
            spec_values = [12, 8, 5, 4, 7]
            fig_bar = px.bar(x=spec_labels, y=spec_values, color=spec_labels)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- TABLAS ESTRATÉGICAS (EJES) ---
        st.header("🎯 Seguimiento de Ejes Estratégicos")
        
        def render_eje_table(titulo, kpis):
            with st.expander(titulo, expanded=True):
                df_eje = pd.DataFrame(kpis)
                st.data_editor(
                    df_eje,
                    column_config={
                        "Meta": st.column_config.NumberColumn(help="Puedes editar la meta aquí"),
                        "Cumplimiento": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                        "Semaforo": st.column_config.TextColumn("Estado")
                    },
                    disabled=["Indicador", "Real"],
                    hide_index=True,
                    use_container_width=True
                )

        # Datos para Eje 1
        eje1_data = [
            {"Indicador": "Atracción Médicos Nuevos", "Meta": 25, "Real": 20, "Cumplimiento": 80, "Semaforo": "🟡"},
            {"Indicador": "Atención 1:1 Médicos", "Meta": 80, "Real": 85, "Cumplimiento": 100, "Semaforo": "🟢"},
        ]
        render_eje_table("EJE 1 · MÉDICOS Y PACIENTES", eje1_data)

        # Datos para Eje 3
        eje3_data = [
            {"Indicador": "Pacientes de Seguros", "Meta": 33, "Real": 15, "Cumplimiento": 45, "Semaforo": "🔴"},
        ]
        render_eje_table("EJE 3 · SEGUROS Y BROKERS", eje3_data)
    
    else:
        st.error("Error leyendo el Excel. Asegúrate de que tenga las hojas 'Indicadores' y 'Medicos por Especialidad'.")

else:
    st.info("
