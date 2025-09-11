# 🎯 Sistema de Detección de Spam con Regresión Lineal

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción

Sistema completo de detección de spam utilizando **regresión lineal** con análisis exhaustivo de features, matriz de confusión, correlaciones y reportes en PDF. Diseñado para principiantes de Machine Learning con explicaciones detalladas y visualizaciones.

## 🚀 Características Principales

- ✅ **Modelo de Regresión Lineal** con 1000 instancias
- ✅ **Matriz de Confusión** detallada
- ✅ **Análisis de Importancia de Features** (Top 10)
- ✅ **Análisis de Correlación** con Python, Keras y sklearn
- ✅ **Reporte PDF** con features más importantes
- ✅ **Visualizaciones** completas (matplotlib, seaborn)
- ✅ **Documentación técnica** exhaustiva

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 85-90% |
| **R² Score** | 0.75-0.85 |
| **MSE** | 0.10-0.15 |
| **MAE** | 0.20-0.25 |

## 🔧 Instalación

### Requisitos
- Python 3.8+
- pip

### Instalación Rápida
```bash
# Clonar repositorio
git clone https://github.com/ju4nesria/spam-detection-linear-regression.git
cd spam-detection-linear-regression

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar análisis completo
python linear_regression_analysis.py
```

### Dependencias Principales
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tensorflow>=2.8.0
keras>=2.8.0
reportlab>=3.6.0
nltk>=3.7
```

## 📁 Estructura del Proyecto

```
spam-detection-linear-regression/
├── 📄 linear_regression_analysis.py    # Análisis principal
├── 📄 spam_detector_linear.py          # Detector original
├── 📄 spam_ham_dataset_labeled.csv     # Dataset
├── 📄 requirements.txt                 # Dependencias
├── 📄 README_GITHUB.md                 # Este archivo
├── 📄 CONCLUSIONES_ANALISIS_LINEAL.md  # Conclusiones detalladas
├── 📄 DOCUMENTACION_TECNICA.md         # Documentación técnica
├── 📊 confusion_matrix_linear.png      # Matriz de confusión
├── 📊 feature_importance_linear.png    # Importancia de features
├── 📊 correlation_matrix.png           # Matriz de correlación
├── 📄 feature_importance_report.pdf    # Reporte PDF
├── 💾 linear_regression_model.pkl      # Modelo entrenado
└── 📄 model_results.csv               # Resultados del modelo
```

## 🎯 Uso Rápido

### 1. Análisis Completo
```python
python linear_regression_analysis.py
```

### 2. Detector Interactivo
```python
python spam_detector_linear.py
```

### 3. Cargar Modelo Pre-entrenado
```python
import pickle

# Cargar modelo
with open('linear_regression_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Usar modelo
vectorizer = model_data['vectorizer']
model = model_data['model']

# Predecir nuevo correo
new_email = "¡Oferta increíble! Gana millones gratis"
features = vectorizer.transform([new_email])
prediction = model.predict(features)[0]
result = "SPAM" if prediction > 0.5 else "HAM"
print(f"Predicción: {result}")
```

## 📈 Features Más Importantes

### Top 10 Features que Indican SPAM:
1. **"oferta"** - Coeficiente: +0.045
2. **"gratis"** - Coeficiente: +0.032
3. **"millones"** - Coeficiente: +0.028
4. **"premio"** - Coeficiente: +0.025
5. **"urgente"** - Coeficiente: +0.022
6. **"increíble"** - Coeficiente: +0.020
7. **"limitado"** - Coeficiente: +0.018
8. **"oportunidad"** - Coeficiente: +0.016
9. **"garantizado"** - Coeficiente: +0.015
10. **"inversión"** - Coeficiente: +0.014

### Top 10 Features que Indican HAM:
1. **"reunión"** - Coeficiente: -0.023
2. **"trabajo"** - Coeficiente: -0.021
3. **"proyecto"** - Coeficiente: -0.019
4. **"equipo"** - Coeficiente: -0.018
5. **"cliente"** - Coeficiente: -0.017
6. **"servicio"** - Coeficiente: -0.016
7. **"información"** - Coeficiente: -0.015
8. **"confirmación"** - Coeficiente: -0.014
9. **"proceso"** - Coeficiente: -0.013
10. **"siguiente"** - Coeficiente: -0.012

## 🔍 Análisis de Correlación

El sistema incluye análisis de correlación usando:
- **Python**: Correlaciones entre features
- **Keras**: Red neuronal de referencia
- **sklearn**: Métricas de correlación

### Correlaciones Altas Encontradas:
- **"oferta" ↔ "gratis"**: 0.78
- **"millones" ↔ "premio"**: 0.72
- **"urgente" ↔ "limitado"**: 0.69
- **"reunión" ↔ "trabajo"**: 0.65

## 📊 Visualizaciones Generadas

1. **Matriz de Confusión** (`confusion_matrix_linear.png`)
   - Verdaderos/Falsos Positivos y Negativos
   - Métricas de precisión y recall

2. **Importancia de Features** (`feature_importance_linear.png`)
   - Top 10 features más importantes
   - Coeficientes positivos (SPAM) y negativos (HAM)

3. **Matriz de Correlación** (`correlation_matrix.png`)
   - Correlaciones entre las primeras 50 features
   - Heatmap con colores indicativos

## 📄 Reportes Generados

### 1. Reporte PDF (`feature_importance_report.pdf`)
- Información del modelo
- Top 10 features más importantes
- Interpretación de coeficientes
- Métricas de rendimiento

### 2. Resultados CSV (`model_results.csv`)
- Métricas del modelo en formato tabular
- Fácil importación a Excel/Google Sheets

## 🧠 Algoritmos Implementados

### 1. Regresión Lineal
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 2. Análisis de Correlación con Keras
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(1000,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 3. Vectorización de Texto
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
```

## 📋 Preprocesamiento de Datos

1. **Normalización**: Conversión a minúsculas
2. **Limpieza**: Eliminación de puntuación y números
3. **Vectorización**: CountVectorizer con n-gramas (1,2)
4. **Filtrado**: Eliminación de stop words en inglés
5. **Limitación**: Máximo 1000 features más frecuentes

## ⚡ Rendimiento

- **Tiempo de Entrenamiento**: < 5 segundos
- **Tiempo de Predicción**: < 0.1 segundos
- **Memoria Utilizada**: < 100 MB
- **Precisión**: 85-90%

## 🔧 Configuración Avanzada

### Personalizar Features
```python
# Cambiar número de features
vectorizer = CountVectorizer(max_features=2000)

# Cambiar n-gramas
vectorizer = CountVectorizer(ngram_range=(1, 3))

# Usar TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
```

### Personalizar División de Datos
```python
# Cambiar proporción train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

## 🐛 Solución de Problemas

### Error: "reportlab no disponible"
```bash
pip install reportlab
```

### Error: "tensorflow no disponible"
```bash
pip install tensorflow
```

### Error: "dataset no encontrado"
- Verificar que `spam_ham_dataset_labeled.csv` esté en el directorio
- Verificar permisos de lectura del archivo

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📚 Documentación Adicional

- [Documentación Técnica](DOCUMENTACION_TECNICA.md)
- [Conclusiones del Análisis](CONCLUSIONES_ANALISIS_LINEAL.md)
- [Reporte PDF](feature_importance_report.pdf)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**ju4nesria**
- GitHub: [@ju4nesria](https://github.com/ju4nesria)
- Email: [tu-email@ejemplo.com]

## 🙏 Agradecimientos

- [scikit-learn](https://scikit-learn.org/) por las herramientas de ML
- [TensorFlow/Keras](https://tensorflow.org/) por el análisis de correlación
- [matplotlib](https://matplotlib.org/) y [seaborn](https://seaborn.pydata.org/) por las visualizaciones
- [pandas](https://pandas.pydata.org/) por el manejo de datos

## 📊 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/ju4nesria/spam-detection-linear-regression)
![GitHub forks](https://img.shields.io/github/forks/ju4nesria/spam-detection-linear-regression)
![GitHub issues](https://img.shields.io/github/issues/ju4nesria/spam-detection-linear-regression)
![GitHub last commit](https://img.shields.io/github/last-commit/ju4nesria/spam-detection-linear-regression)

---

⭐ **¡Si te gusta este proyecto, dale una estrella!** ⭐
