# IA_Library_Proyect
Predicción de uso Biblioteca UDEC.

Este proyecto utiliza técnicas de aprendizaje automático para predecir el número de accesos diarios a la biblioteca de la Universidad de Concepción. El flujo principal incluye:

- **Preprocesamiento de datos:** Limpieza, agregación y generación de variables exógenas como vacaciones, medias móviles, variables cíclicas y lags temporales. Ver detalles en [`Archivos Auxiliares/Preprocesado_Dataset.ipynb`](Archivos Auxiliares/Preprocesado_Dataset.ipynb).
- **Modelado:** Se emplean modelos como Random Forest y XGBoost para la predicción, con ajuste de hiperparámetros y validación temporal. Ejemplo en [`LibraryUseXGB.ipynb`](LibraryUseXGB.ipynb) y [`LibraryUseRF.ipynb`](LibraryUseRF.ipynb).
- **Evaluación:** Se utilizan métricas como MSE, MAE y SMAPE, además de visualizaciones de resultados y análisis de importancia de variables.
- **Datos:** Los datasets utilizados están en la carpeta [`Datasets/`](Datasets), incluyendo accesos diarios y días de vacaciones.

El objetivo es entender los patrones de uso y mejorar la gestión de recursos en la biblioteca mediante modelos predictivos robustos.

### ¿Por qué se usan medias móviles, EWMA y variables cíclicas?

- **Medias móviles:** Permiten suavizar las fluctuaciones diarias y capturar tendencias a corto, mediano y largo plazo en los accesos. Ayudan al modelo a identificar patrones recurrentes y anomalías en el comportamiento de los usuarios.

- **EWMA (Media móvil exponencial):** Da mayor peso a los datos más recientes, lo que es útil para detectar cambios rápidos en la tendencia de accesos y mejorar la capacidad de reacción del modelo ante eventos recientes.

- **Variables cíclicas (seno y coseno de mes/día):** Representan la naturaleza periódica del uso de la biblioteca (por ejemplo, variaciones según el día de la semana o el mes). Esto permite que el modelo aprenda mejor los ciclos temporales y estacionales presentes en los datos.

Estas variables enriquecen el conjunto de características y mejoran la capacidad predictiva de los modelos al capturar tanto tendencias como estacionalidades y patrones temporales.