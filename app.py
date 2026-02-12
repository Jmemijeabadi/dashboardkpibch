import streamlit as st
import pandas as pd

# Load the data (adjust paths as needed)
indicadores_data = pd.read_excel('Grupo Abadi - Hospital Reporte Operativo.xlsx', sheet_name='Indicadores')
medicos_por_especialidad_data = pd.read_excel('Grupo Abadi - Hospital Reporte Operativo.xlsx', sheet_name='Medicos por Especialidad')

# Clean data
indicadores_data_clean = indicadores_data.dropna(subset=['Indicator'])

# Streamlit UI for the dashboard
st.title('CRMBI Comercial - KPIs Dashboard')

# Input for selecting the month and year
month = st.selectbox('Select Month', ['2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01', '2024-02'], index=2)
year = st.selectbox('Select Year', ['2023', '2024'], index=1)

# Display the indicators data for the selected month
indicator_value = indicadores_data_clean[indicadores_data_clean['Indicator'] == 'No. de Pacientes'][month].values[0]
insured_value = indicadores_data_clean[indicadores_data_clean['Indicator'] == 'Seguro'][month].values[0]

# Display the KPIs for the selected month
st.subheader(f"KPIs for {month}-{year}")
st.write(f"**No. of Patients:** {indicator_value}")
st.write(f"**No. of Insured Patients:** {insured_value}")

# Interactive input for setting target values
target_patients = st.number_input('Set Target for Patients', min_value=0, value=int(indicator_value))
target_insured = st.number_input('Set Target for Insured Patients', min_value=0, value=int(insured_value))

# Calculate KPI ratios
insured_rate = target_insured / target_patients if target_patients > 0 else 0

# Display target ratios
st.write(f"**Target Insured Rate:** {insured_rate:.2%}")

# Data for key specialties (example data)
specialties = medicos_por_especialidad_data.dropna(subset=['Specialty'])
specialty_names = specialties['Specialty'].tolist()
specialty_counts = specialties['Count'].astype(int).tolist()

# Display a bar chart for doctors per specialty
st.subheader("Doctors per Specialty")
st.bar_chart(dict(zip(specialty_names, specialty_counts)))

# Add logic for saving the updated targets to local storage (in real use case, this would be saved to a backend or database)
st.write(f"Targets set for {month}-{year}:")
st.write(f"Patients: {target_patients}, Insured: {target_insured}")
