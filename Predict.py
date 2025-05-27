import pandas as pd
# Cargar el archivo Excel
df = pd.read_excel("accesos.xlsx")

# Asegurar que la fecha esté en formato datetime
df['Fecha Completa'] = pd.to_datetime(df['Fecha Completa'])

# Crear variables temporales relevantes
df['Día'] = df['Fecha Completa'].dt.day
df['Mes'] = df['Fecha Completa'].dt.month
df['Año'] = df['Fecha Completa'].dt.year

# Codificar Jornada, Pregrado y Postgrado
df['Jornada'] = df['Jornada'].astype('category').cat.codes
df['Pregrado'] = df['Pregrado'].map({'SI': 1, 'NO': 0})
df['Postgrado'] = df['Postgrado'].map({'SI': 1, 'NO': 0})

# Crear una clave de fecha sin hora
df['Fecha'] = df['Fecha Completa'].dt.date

# Agrupar y contar accesos
daily_counts = df.groupby(['Fecha', 'Día', 'Mes', 'Año']).agg({
    'Jornada': 'mean',
    'Pregrado': 'mean',
    'Postgrado': 'mean',
    'Fecha Completa': 'count'  # esto cuenta los accesos
}).rename(columns={'Fecha Completa': 'Accesos'}).reset_index()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Variables predictoras y objetivo
X = daily_counts[['Día', 'Mes', 'Año', 'Jornada', 'Pregrado', 'Postgrado']]
y = daily_counts['Accesos']

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

predicciones = pd.DataFrame({
    'Fecha': X_test.index,
    'Prediccion_Accesos': y_pred
})
# Ordenar por menor cantidad de accesos predichos
dias_para_eventos = predicciones.sort_values('Prediccion_Accesos').head(10)
print(dias_para_eventos)
