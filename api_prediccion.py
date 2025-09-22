import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pymysql

best_model = joblib.load('mejor_modelo.pkl')

cat_cols = ['producto']
num_cols = [
    'valor_unitario',
    'cantidad_lag_1',
    'cantidad_rolling_mean_3',
    'cantidad_rolling_std_3',
    'cantidad_log_lag_1',
    'cantidad_log_rolling_mean_3',
    'cantidad_log_rolling_std_3'
]
try:
    input_columns = best_model.feature_names_in_.tolist()
except AttributeError:
    input_columns = num_cols 

def preparar_df(df, base_columns=None):
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = df[num_cols].reset_index(drop=True)
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    if base_columns is not None:
        X = X.reindex(columns=base_columns, fill_value=0)
    return X

def obtener_datos_mysql():
    engine = create_engine('mysql+pymysql://root:jiEpnqgeyQujFOHkuZwnnvgpidUMpiXq@hopper.proxy.rlwy.net:58023/railway')
    query = '''
    SELECT 
        od.id AS order_detail_id,
        o.order_date,
        p.product_name,
        od.quantity
    FROM 
        order_details od
    JOIN 
        `orders` o ON od.order_id = o.id
    JOIN 
        products p ON od.product_id = p.id
    ORDER BY 
        o.order_date ASC;
    '''
    df = pd.read_sql(query, engine)
    print('=== CONTENIDO REAL DEL DATAFRAME ===')
    print(df.head(20))
    print('=== TIPOS DE DATOS ===')
    print(df.dtypes)
    return df


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
print('✅ Flask app inicializada')

# Endpoint de prueba para verificar si la app responde
@app.route('/ping', methods=['GET'])
def ping():
    print('✅ Ping recibido')
    return 'pong', 200

@app.route('/api/predict', methods=['POST'])
def predecir():
    data = request.get_json()
    days = data.get('days', 1)
    products = data.get('products', [])
    start_date = data.get('date')
    season = data.get('season')
    results = []
    df_hist = obtener_datos_mysql()
    print("[API INPUT]", data)
    df_hist = obtener_datos_mysql()
    print(f"[MYSQL RESULT] shape: {df_hist.shape}, columns: {df_hist.columns.tolist()}")
    if not df_hist.empty:
        print(df_hist.head(10).to_string(index=False))
    else:
        print("[MYSQL RESULT] El DataFrame está vacío. No se encontraron datos.")
    df_hist = df_hist.sort_values(['product_name', 'order_date'])
    for prod in products:
        product_name = prod.get('productName')
        price = prod.get('price', 0)
        subdf = df_hist[df_hist['product_name'] == product_name].copy()
        if subdf.empty:
            continue
        subdf['cantidad_lag_1'] = subdf['quantity'].shift(1)
        subdf['cantidad_rolling_mean_3'] = subdf['quantity'].rolling(3, min_periods=1).mean()
        subdf['cantidad_rolling_std_3'] = subdf['quantity'].rolling(3, min_periods=1).std().fillna(0)
        subdf['cantidad_log_lag_1'] = np.log1p(subdf['cantidad_lag_1'].fillna(0))
        subdf['cantidad_log_rolling_mean_3'] = np.log1p(subdf['cantidad_rolling_mean_3'])
        subdf['cantidad_log_rolling_std_3'] = np.log1p(subdf['cantidad_rolling_std_3'])
        historial_pred = subdf.tail(3).copy()
        daily_predictions = []
        predicted_total = 0
        for i in range(days):
            current_date = pd.to_datetime(start_date) + pd.Timedelta(days=i+1)
            dia_semana_map = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
            dia_semana_str = dia_semana_map[current_date.weekday()]
            ultimos = historial_pred['quantity'].values[-3:] if len(historial_pred) >= 1 else [0, 0, 0]
            lag_1 = ultimos[-1] if len(ultimos) > 0 else 0
            rolling_mean_3 = np.mean(ultimos)
            rolling_std_3 = np.std(ultimos)
            log_lag_1 = np.log1p(lag_1)
            log_rolling_mean_3 = np.log1p(rolling_mean_3)
            log_rolling_std_3 = np.log1p(rolling_std_3) if rolling_std_3 > 0 else 0
            row = {
                'producto': product_name,
                'valor_unitario': price,
                'cantidad_lag_1': lag_1,
                'cantidad_rolling_mean_3': rolling_mean_3,
                'cantidad_rolling_std_3': rolling_std_3,
                'cantidad_log_lag_1': log_lag_1,
                'cantidad_log_rolling_mean_3': log_rolling_mean_3,
                'cantidad_log_rolling_std_3': log_rolling_std_3,
                'dia_semana': dia_semana_str,
                'estacion': season
            }
            df_input = pd.DataFrame([row])
            X_input = preparar_df(df_input, base_columns=input_columns)
            pred = best_model.predict(X_input)
            pred_redondeado = float(np.round(pred[0], 2))
            pred_redondeado = max(pred_redondeado, 0)
            pred_date = current_date.strftime('%Y-%m-%d')
            daily_predictions.append({
                'date': pred_date,
                'predicted_quantity': pred_redondeado
            })
            predicted_total += pred_redondeado
            historial_pred = pd.concat([
                historial_pred,
                pd.DataFrame({'quantity': [pred_redondeado]}, index=[current_date])
            ])
        results.append({
            'product_id': product_name,
            'predicted_total': float(np.round(predicted_total, 2)),
            'daily_predictions': daily_predictions
        })
    return jsonify({'predictions': results})

@app.route('/api/predict-mysql', methods=['GET'])
def predecir_mysql():
    df = obtener_datos_mysql()
    df = df.sort_values(['product_name', 'order_date'])
    df['cantidad_lag_1'] = df.groupby('product_name')['quantity'].shift(1)
    df['cantidad_rolling_mean_3'] = df.groupby('product_name')['quantity'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['cantidad_rolling_std_3'] = df.groupby('product_name')['quantity'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    df['cantidad_log_lag_1'] = np.log1p(df['cantidad_lag_1'].fillna(0))
    df['cantidad_log_rolling_mean_3'] = np.log1p(df['cantidad_rolling_mean_3'])
    df['cantidad_log_rolling_std_3'] = np.log1p(df['cantidad_rolling_std_3'])
    predicciones = []
    for prod, subdf in df.groupby('product_name'):
        row = subdf.iloc[-1]
        row_dict = {
            'producto': row['product_name'],
            'valor_unitario': 0, 
            'cantidad_lag_1': row['cantidad_lag_1'],
            'cantidad_rolling_mean_3': row['cantidad_rolling_mean_3'],
            'cantidad_rolling_std_3': row['cantidad_rolling_std_3'],
            'cantidad_log_lag_1': row['cantidad_log_lag_1'],
            'cantidad_log_rolling_mean_3': row['cantidad_log_rolling_mean_3'],
            'cantidad_log_rolling_std_3': row['cantidad_log_rolling_std_3']
        }
        df_input = pd.DataFrame([row_dict])
        X_input = preparar_df(df_input, base_columns=input_columns)
        pred = best_model.predict(X_input)
        predicciones.append({
            'product_name': prod,
            'predicted_quantity': float(np.round(pred[0], 2))
        })
    return jsonify({'predictions': predicciones})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)