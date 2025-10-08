import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import random
import joblib
import matplotlib.pyplot as plt
import time

print("--- Iniciando Etapa 1: Resumen de productos ---")
df = pd.read_csv('clean_DATOS-VENTAS.csv')

valor_unitario_mas_frec = (
    df.groupby('producto')['valor_unitario']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .reset_index()
    .rename(columns={'valor_unitario': 'valor_unitario_mas_frec'})
)

apariciones = df['producto'].value_counts().reset_index()
apariciones.columns = ['producto', 'apariciones']

resumen = valor_unitario_mas_frec.merge(apariciones, on='producto')

resumen = resumen[['producto', 'valor_unitario_mas_frec', 'apariciones']]
print("✅ Resumen de productos calculado y mantenido en memoria.")
print("\n--- Iniciando Etapa 2: Creación de dataset completo ---")


df['fecha'] = pd.to_datetime(df['fecha'])
df_grouped = df.groupby(['producto', 'fecha']).agg(
    cantidad=('cantidad', 'sum'),
    valor_unitario=('valor_unitario', 'mean')
).reset_index()

df_grouped['cantidad_log'] = np.log1p(df_grouped['cantidad'])

productos_filtrados = resumen[resumen['apariciones'] > 275]['producto'].unique()

df_grouped['fecha'] = df_grouped['fecha'].dt.strftime('%Y-%m-%d')
fechas = pd.date_range(df_grouped['fecha'].min(), df_grouped['fecha'].max()).strftime('%Y-%m-%d')

combinaciones = pd.MultiIndex.from_product([productos_filtrados, fechas], names=['producto', 'fecha']).to_frame(index=False)

df_completo = combinaciones.merge(df_grouped, on=['producto', 'fecha'], how='left')

df_completo = df_completo.merge(resumen[['producto', 'valor_unitario_mas_frec']], on='producto', how='left')

df_completo['cantidad'] = df_completo['cantidad'].fillna(0)
df_completo['valor_unitario'] = df_completo['valor_unitario'].fillna(df_completo['valor_unitario_mas_frec'])
df_completo['cantidad_log'] = df_completo['cantidad_log'].fillna(0)

if 'dia_semana' not in df_completo or df_completo['dia_semana'].isnull().any():
    dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    fechas_dt = pd.to_datetime(df_completo['fecha'])
    df_completo['dia_semana'] = fechas_dt.dt.weekday.map(dias)
    def estacion(mes):
        if mes in [12, 1, 2]:
            return 'verano'
        elif mes in [3, 4, 5]:
            return 'otoño'
        elif mes in [6, 7, 8]:
            return 'invierno'
        else:
            return 'primavera'
    df_completo['estacion'] = fechas_dt.dt.month.map(estacion)

df_final_completo = df_completo[['producto', 'cantidad', 'valor_unitario', 'fecha', 'dia_semana', 'estacion', 'cantidad_log']]
df_final_completo = df_final_completo.sort_values(['fecha', 'producto'])
print("✅ Dataset completo generado y mantenido en memoria.")


print("\n--- Iniciando Etapa 3: Entrenamiento y evaluación del modelo ---")
random_state_used = random.randint(0, 1000000)
print(f"🌱 Semilla generada para esta ejecución: {random_state_used}")

df_entrenamiento = df_final_completo.copy()
df_entrenamiento['fecha'] = pd.to_datetime(df_entrenamiento['fecha'])
df_entrenamiento = df_entrenamiento.sort_values(['producto', 'fecha'])

fecha_corte = df_entrenamiento['fecha'].max() - pd.Timedelta(days=30)
train_df = df_entrenamiento[df_entrenamiento['fecha'] <= fecha_corte].copy()
val_df = df_entrenamiento[df_entrenamiento['fecha'] > fecha_corte].copy()

def crear_vars_lag_rolling(df, window=3):
    df = df.copy()
    df['cantidad_lag_1'] = df.groupby('producto')['cantidad'].shift(1)
    df['cantidad_rolling_mean_3'] = df.groupby('producto')['cantidad'].rolling(window=window).mean().reset_index(0, drop=True)
    df['cantidad_rolling_std_3'] = df.groupby('producto')['cantidad'].rolling(window=window).std().reset_index(0, drop=True)
    df['cantidad_log_lag_1'] = df.groupby('producto')['cantidad_log'].shift(1)
    df['cantidad_log_rolling_mean_3'] = df.groupby('producto')['cantidad_log'].rolling(window=window).mean().reset_index(0, drop=True)
    df['cantidad_log_rolling_std_3'] = df.groupby('producto')['cantidad_log'].rolling(window=window).std().reset_index(0, drop=True)
    return df

train_df = crear_vars_lag_rolling(train_df)
val_df = crear_vars_lag_rolling(val_df)

train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

cat_cols = ['producto', 'dia_semana', 'estacion']
num_cols = [
    'valor_unitario',
    'cantidad_lag_1',
    'cantidad_rolling_mean_3',
    'cantidad_rolling_std_3',
    'cantidad_log_lag_1',
    'cantidad_log_rolling_mean_3',
    'cantidad_log_rolling_std_3'
]

def preparar_df(df, base_columns=None):
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = df[num_cols].reset_index(drop=True)
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    if base_columns is not None:
        X = X.reindex(columns=base_columns, fill_value=0)
    return X

X_train = preparar_df(train_df)
X_val = preparar_df(val_df, base_columns=X_train.columns)

joblib.dump(X_train.columns, 'model_columns.pkl')
print('✅ Columnas del modelo guardadas como model_columns.pkl')

y_train = train_df['cantidad']
y_val = val_df['cantidad']

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': random_state_used
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=3)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state_used))
study.optimize(objective, n_trials=100)

print("\n✅ Mejores hiperparámetros encontrados:")
best_params = study.best_params
best_params['random_state'] = random_state_used
print(best_params)
print(f"Mejor R² promedio en CV: {study.best_value:.4f}")
print(f"🧬 Semilla utilizada para esta ejecución: {random_state_used}")

# --- ENTRENAMIENTO FINAL ---
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

joblib.dump(best_model, 'mejor_modelo.pkl')
print('✅ Modelo guardado como mejor_modelo.pkl')

# --- PREDICCIÓN Y VALIDACIÓN ---
y_pred_continuo = best_model.predict(X_val)
y_pred_continuo = np.maximum(0, y_pred_continuo)
y_pred = np.round(y_pred_continuo)

val_df = val_df.copy()
val_df['prediccion'] = y_pred_continuo
val_df.insert(val_df.columns.get_loc('prediccion'), 'prediccion_redondeada', y_pred)
val_df['error_abs'] = np.abs(val_df['cantidad'] - val_df['prediccion_redondeada'])
val_df['error_rel'] = val_df['error_abs'] / (val_df['cantidad'] + 1e-5)

umbral_error_abs = val_df['error_abs'].quantile(0.95)
umbral_error_rel = 0.5

def clasificar_error(row):
    if row['error_abs'] <= 1:
        return 'Dentro del margen de ±1 unidad'
    elif row['error_abs'] > umbral_error_abs:
        return 'Outlier por cantidad extrema'
    elif row['error_rel'] > umbral_error_rel:
        return 'Mala predicción (Error Relativo Alto)'
    else:
        return 'Predicción aceptable'

val_df['clasificacion_error'] = val_df.apply(clasificar_error, axis=1)

print(f"\n📊 Evaluación en Validación:")
print(f"R² estándar: {r2_score(y_val, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred):.3f}")

# --- MÉTRICAS ADICIONALES ---
from sklearn.metrics import mean_squared_error, median_absolute_error, explained_variance_score
smape = 100 * np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_val) + np.abs(y_pred) + 1e-8))
print(f"SMAPE: {smape:.2f}%")

# --- Análisis de reducción de variables redundantes ---
print("\n--- Iniciando análisis de reducción de variables ---")

from sklearn.metrics import mean_squared_error

variables_a_eliminar = [
    'cantidad_log_lag_1',
    'cantidad_log_rolling_mean_3',
    'cantidad_log_rolling_std_3'
]

# Verifica que existan antes de eliminarlas
variables_existentes = [v for v in variables_a_eliminar if v in X_train.columns]
if variables_existentes:
    print(f"🔍 Probando eliminar variables redundantes: {variables_existentes}")

    # Copias reducidas de los datasets
    X_train_reduced = X_train.drop(columns=variables_existentes)
    X_val_reduced = X_val.drop(columns=variables_existentes)

    # Reentrenar modelo con mismas configuraciones
    model_reduced = xgb.XGBRegressor(**best_params)
    model_reduced.fit(X_train_reduced, y_train)

    # Evaluación
    y_pred_reduced = model_reduced.predict(X_val_reduced)
    y_pred_reduced = np.maximum(0, y_pred_reduced)

    r2_reduced = r2_score(y_val, y_pred_reduced)
    rmse_reduced = np.sqrt(mean_squared_error(y_val, y_pred_reduced))
    mae_reduced = mean_absolute_error(y_val, y_pred_reduced)

    print("\n📉 Resultados con variables eliminadas:")
    print(f"R² reducido: {r2_reduced:.4f}")
    print(f"RMSE reducido: {rmse_reduced:.4f}")
    print(f"MAE reducido: {mae_reduced:.4f}")

    # Comparación directa
    r2_original = r2_score(y_val, y_pred)
    rmse_original = np.sqrt(mean_squared_error(y_val, y_pred))
    mae_original = mean_absolute_error(y_val, y_pred)

    print("\n📊 Comparación:")
    print(f"R² original: {r2_original:.4f}  →  R² reducido: {r2_reduced:.4f}")
    print(f"RMSE original: {rmse_original:.4f}  →  RMSE reducido: {rmse_reduced:.4f}")
    print(f"MAE original: {mae_original:.4f}  →  MAE reducido: {mae_reduced:.4f}")

    # Criterio simple de decisión
    if abs(r2_original - r2_reduced) < 0.01 and abs(rmse_original - rmse_reduced) < 2:
        print("✅ No hay pérdida significativa de desempeño. Puedes eliminar esas variables.")
    else:
        print("⚠️ Se detectó una pérdida apreciable de desempeño. Conviene mantenerlas.")
else:
    print("ℹ️ Las variables indicadas no existen en el dataset.")


# --- NUEVA SECCIÓN: IMPORTANCIA DE VARIABLES ---
print("\n--- Analizando Importancia de Variables ---")
importances = best_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Variable': list(importances.keys()),
    'Importancia': list(importances.values())
}).sort_values(by='Importancia', ascending=False)
print(importance_df)

xgb.plot_importance(best_model, importance_type='gain')
plt.title('Importancia de Variables en el Modelo XGBoost')
plt.tight_layout()
plt.savefig('importancia_variables.png')
plt.close()
print("✅ Gráfico de importancia guardado como 'importancia_variables.png'.")

# --- NUEVA SECCIÓN: ECUACIÓN INTERPRETATIVA ---
print("\n--- Generando Modelo Lineal Interpretativo ---")
from sklearn.linear_model import LinearRegression
y_pred_train = best_model.predict(X_train)
lin_model = LinearRegression().fit(X_train, y_pred_train)

coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': lin_model.coef_
}).sort_values(by='Coeficiente', ascending=False)
print(coef_df)

coef_df.to_csv('coeficientes_lineales.csv', index=False)
print("✅ Coeficientes guardados como 'coeficientes_lineales.csv'.")

# --- NUEVA SECCIÓN: INTERPRETACIÓN VISUAL CON SHAP ---
print("\n--- Generando interpretación visual con SHAP ---")
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()
print("✅ Gráfico SHAP guardado como 'shap_summary.png'.")

# --- VISUALIZACIÓN FINAL DE PREDICCIONES ---
print("\n--- Iniciando Etapa 4: Visualización de predicciones ---")
val_df_export = val_df[['fecha', 'producto', 'cantidad', 'prediccion_redondeada']].copy()
val_df_export.rename(columns={'prediccion_redondeada': 'predicho'}, inplace=True)

productos = val_df_export['producto'].unique()
if len(productos) > 0:
    fig, axs = plt.subplots(len(productos), 1, figsize=(14, 5 * len(productos)), sharex=True)
    if len(productos) == 1:
        axs = [axs]
    for i, producto in enumerate(productos):
        val_prod = val_df_export[val_df_export['producto'] == producto]
        axs[i].plot(val_prod['fecha'], val_prod['cantidad'], label='Cantidad real', marker='o')
        axs[i].plot(val_prod['fecha'], val_prod['predicho'], label='Cantidad predicha', marker='o')
        axs[i].set_title(f'{producto}')
        axs[i].set_ylabel('Cantidad')
        axs[i].legend()
        axs[i].grid(True)
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('predicciones_grafico.png')
    plt.close(fig)
    print("✅ Gráfico de predicciones guardado como 'predicciones_grafico.png'.")
else:
    print("ℹ️ No hay datos de validación para graficar.")
