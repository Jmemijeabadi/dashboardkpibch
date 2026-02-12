import pandas as pd

# ---------------------------------------------------------
# CAMBIA ESTO POR EL NOMBRE DE TU ARCHIVO REAL
archivo_excel = "TU_ARCHIVO_GRANDE.xlsx" 
# ---------------------------------------------------------

try:
    print(f"🔍 Analizando estructura de: {archivo_excel}...\n")
    xls = pd.ExcelFile(archivo_excel)
    
    print(f"📂 Hojas encontradas: {xls.sheet_names}\n")

    for hoja in xls.sheet_names:
        print(f"{'='*40}")
        print(f"📑 HOJA: {hoja}")
        print(f"{'='*40}")
        
        # Leemos solo las primeras 15 filas para ver encabezados
        try:
            df = pd.read_excel(xls, sheet_name=hoja, nrows=15, header=None)
            
            # Imprimimos una vista previa limpia
            print("VISTA PREVIA DE LAS PRIMERAS 15 FILAS (Estructura):")
            print(df.to_string(index=True, header=False, na_rep="[VACIO]"))
            print("\n")
            
        except Exception as e:
            print(f"❌ Error leyendo hoja {hoja}: {e}")

except FileNotFoundError:
    print("❌ Error: No encuentro el archivo. Asegúrate de poner el nombre correcto y extensión (.xlsx).")
except Exception as e:
    print(f"❌ Error general: {e}")
