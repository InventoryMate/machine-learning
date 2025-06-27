import pandas as pd
import numpy as np
import re

raw = pd.read_csv('DATOS-VENTAS.csv', index_col=0)
print(f"[INICIO] Registros originales: {len(raw)}")

raw = raw.rename(columns={
    'Producto': 'producto',
    'Cant': 'cantidad',
    'Unidad': 'unidad',
    'Valor Unitario': 'valor_unitario',
    'Entrega Emision GRE': 'fecha'
})

raw['valor_unitario'] = raw['valor_unitario'].astype(str).str.replace(',', '.', regex=False)
raw['valor_unitario'] = pd.to_numeric(raw['valor_unitario'], errors='coerce')
raw['fecha'] = raw['fecha'].astype(str).str.strip().str.replace('"', '')
raw['fecha'] = pd.to_datetime(raw['fecha'], dayfirst=False, errors='coerce')

print(f"[FECHA] Fechas no parseadas (NaT): {raw['fecha'].isna().sum()}")

columnas_criticas = ['producto', 'cantidad', 'unidad', 'valor_unitario', 'fecha']

missing_por_columna = raw[columnas_criticas].isna().sum()
print("\n[FILTRO 1] Valores faltantes por columna antes de eliminar:")
print(missing_por_columna)

antes_filtro1 = len(raw)

raw = raw.dropna(subset=columnas_criticas)

despues_filtro1 = len(raw)
eliminados = antes_filtro1 - despues_filtro1
print(f"[FILTRO 1] Total eliminados por valores faltantes básicos: {eliminados} (quedan {despues_filtro1})\n")

raw['cantidad'] = pd.to_numeric(raw['cantidad'], errors='coerce')
raw = raw.dropna(subset=['cantidad'])
raw = raw.sort_values(by='fecha')


Q1 = raw['cantidad'].quantile(0.25)
Q3 = raw['cantidad'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

antes_cant = len(raw)
raw = raw[(raw['cantidad'] >= limite_inferior) & (raw['cantidad'] <= limite_superior)]
print(f"[FILTRO cantidad] Eliminados por valores atípicos fuera de [{limite_inferior:.2f}, {limite_superior:.2f}]: {antes_cant - len(raw)} (quedan {len(raw)})")
raw['dia_semana'] = raw['fecha'].dt.weekday.map({
    0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves',
    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
})

def obtener_estacion(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:
        return 'verano'
    elif mes in [3, 4, 5]:
        return 'otoño'
    elif mes in [6, 7, 8]:
        return 'invierno'
    else:
        return 'primavera'

raw['estacion'] = raw['fecha'].apply(obtener_estacion)
raw['cantidad_log'] = np.log1p(raw['cantidad'])

def extraer_unidades_y_limpio(nombre):
    patron = r"X\s*([\d.,]+)\s*(UNID|UND|KG|KGS|BOLSA|BOLSAS|BALDE|BALDES|CAJA|BOTELLA|BOTELLAS|LITRO|LITROS|LT|L)"
    matches = re.findall(patron, nombre, re.IGNORECASE)
    factor = 1.0
    tipos = []
    for unidades_str, tipo in matches:
        unidades_str = unidades_str.replace(',', '.')
        try:
            unidades = float(unidades_str)
        except ValueError:
            unidades = 1.0
        factor *= unidades
        tipos.append(tipo.upper())
    nombre_limpio = re.sub(patron, '', nombre, flags=re.IGNORECASE).strip()
    nombre_limpio = re.sub(r'\s*X\s*$', '', nombre_limpio).strip()
    tipo_final = ' x '.join(tipos) if tipos else ''
    return factor, tipo_final, nombre_limpio

raw[['unidades_producto', 'tipo_unidad', 'producto_limpio']] = raw['producto'].apply(lambda x: pd.Series(extraer_unidades_y_limpio(str(x))))
raw['cantidad_total'] = raw['cantidad'] * raw['unidades_producto']
raw['valor_unitario_normalizado'] = raw['valor_unitario'] / raw['unidades_producto']

raw['producto'] = raw['producto_limpio']

raw['valor_unitario'] = raw['valor_unitario_normalizado']
raw['cantidad'] = raw['cantidad_total']

conteos = raw['producto'].value_counts()
productos_validos = conteos[conteos >= 15].index
raw = raw[raw['producto'].isin(productos_validos)]


raw['cantidad_log'] = np.log1p(raw['cantidad'])

columnas_finales = ['producto', 'cantidad', 'valor_unitario', 'fecha', 'dia_semana', 'estacion', 'cantidad_log']
raw = raw[columnas_finales]

raw.to_csv('clean_DATOS-VENTAS.csv', index=False, encoding='utf-8')
print("✅ Dataset limpio guardado en 'clean_DATOS-VENTAS.csv'")

print(f"🟡 Filas finales: {len(raw)} | Columnas: {raw.shape[1]}")
print(f"🔹 Productos únicos: {raw['producto'].nunique()}")
