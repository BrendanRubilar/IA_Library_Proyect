# IA_Library_Proyect
Predicción de uso Biblioteca UDEC.

Este proyecto utiliza técnicas de aprendizaje automático para predecir el número de accesos diarios a la biblioteca de la Universidad de Concepción. El flujo principal incluye:

- **Preprocesamiento de datos:** Limpieza, agregación y generación de variables exógenas como vacaciones, medias móviles, variables cíclicas y lags temporales. Ver detalles en [`Archivos Auxiliares/Preprocesado_Dataset.ipynb`](Archivos Auxiliares/Preprocesado_Dataset.ipynb).
- **Modelado:** Se emplean modelos como Random Forest y XGBoost para la predicción, con ajuste de hiperparámetros y validación temporal. Ejemplo en [`LibraryUseXGB.ipynb`](LibraryUseXGB.ipynb) y [`LibraryUseRF.ipynb`](LibraryUseRF.ipynb).
- **Evaluación:** Se utilizan métricas como MSE, MAE y SMAPE, además de visualizaciones de resultados y análisis de importancia de variables.
- **Datos:** Los datasets utilizados están en la carpeta [`Datasets/`](Datasets), incluyendo accesos diarios y días de vacaciones.

El objetivo es entender los patrones de uso y mejorar la gestión de recursos en la biblioteca mediante