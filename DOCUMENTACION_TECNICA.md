# 📚 Documentación Técnica - Sistema de Detección de Spam

## 🎯 Descripción General

Este sistema implementa un detector de spam usando **regresión lineal**, diseñado específicamente para principiantes de Machine Learning. El objetivo es clasificar correos electrónicos como SPAM (no deseado) o HAM (normal).

## 🏗️ Arquitectura del Sistema

### 1. **Preprocesamiento de Datos**
```python
def preprocess_text(self, text):
    # 1. Normalización: convertir a minúsculas
    text = text.lower()
    
    # 2. Limpieza: eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Filtrado: eliminar números
    text = re.sub(r'\d+', '', text)
    
    # 4. Normalización: limpiar espacios
    text = ' '.join(text.split())
```

**¿Por qué estos pasos?**
- **Minúsculas**: "SPAM" y "spam" deben tratarse igual
- **Sin puntuación**: "¡oferta!" y "oferta" son la misma palabra
- **Sin números**: "1000" no es relevante para detectar spam
- **Espacios limpios**: Evita problemas de formato

### 2. **Vectorización de Texto**
```python
CountVectorizer(
    max_features=1000,      # Máximo 1000 palabras únicas
    stop_words='english',   # Eliminar palabras comunes
    ngram_range=(1, 2)      # Palabras individuales y pares
)
```

**Proceso de Vectorización:**
```
Texto: "Hola mundo increíble oferta"
↓
Features: [1, 0, 1, 1, 0, 0, ...]
         [hola, mundo, increíble, oferta, ...]
```

**Parámetros Explicados:**
- `max_features=1000`: Limita el vocabulario a las 1000 palabras más frecuentes
- `stop_words='english'`: Elimina palabras como "the", "and", "or" que no aportan información
- `ngram_range=(1, 2)`: Considera palabras individuales y pares ("increíble oferta")

### 3. **Regresión Lineal**

**¿Por qué Regresión Lineal para Clasificación?**

Aunque la regresión lineal se usa típicamente para predicción continua, puede usarse para clasificación binaria:

```python
# El modelo predice un número entre 0 y 1
prediction = model.predict(features)[0]

# Convertir a clase
if prediction > 0.5:
    result = "SPAM"
else:
    result = "HAM"
```

**Ventajas para Principiantes:**
- ✅ **Simplicidad**: Fácil de entender y explicar
- ✅ **Interpretabilidad**: Puedes ver los coeficientes
- ✅ **Visualización**: Se puede graficar fácilmente
- ✅ **Fundamentos**: Te enseña conceptos básicos

## 📊 Conceptos Clave Explicados

### **Features (Características)**
Son las palabras que el modelo usa para tomar decisiones:
```python
# Ejemplo de features
features = {
    'oferta': 1,      # Aparece 1 vez
    'gratis': 0,      # No aparece
    'millones': 1,    # Aparece 1 vez
    'trabajo': 0      # No aparece
}
```

### **Target (Objetivo)**
Es lo que queremos predecir:
```python
# 0 = HAM (correo normal)
# 1 = SPAM (correo no deseado)
y = [0, 1, 0, 1, 0, ...]  # Etiquetas de entrenamiento
```

### **Coeficientes (Pesos)**
Cada palabra tiene un peso que indica su importancia:
```python
# Coeficientes del modelo
coefficients = {
    'oferta': +0.045,     # Indica SPAM
    'gratis': +0.032,     # Indica SPAM
    'trabajo': -0.023,    # Indica HAM
    'reunión': -0.019     # Indica HAM
}
```

### **Score de Predicción**
El modelo devuelve un número entre 0 y 1:
```python
# Interpretación del score
if score > 0.5:
    prediction = "SPAM"
    confidence = (score - 0.5) * 2  # Convertir a porcentaje
else:
    prediction = "HAM"
    confidence = (0.5 - score) * 2  # Convertir a porcentaje
```

## 🔍 Algoritmo de Regresión Lineal

### **Fórmula Matemática**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Donde:
- y = predicción (0 o 1)
- β₀ = intercepto
- βᵢ = coeficiente de la característica i
- xᵢ = valor de la característica i
```

### **Ejemplo Práctico**
```python
# Correo: "Hola oferta gratis millones"
# Features: [hola: 1, oferta: 1, gratis: 1, millones: 1]
# Coeficientes: [hola: -0.01, oferta: +0.045, gratis: +0.032, millones: +0.028]

score = -0.01 + 0.045 + 0.032 + 0.028 = 0.094
prediction = "SPAM" (porque 0.094 > 0.5)
```

## 📈 Métricas de Evaluación

### **1. Accuracy (Precisión)**
```python
accuracy = (predicciones_correctas / total_predicciones) * 100
```
- **Rango**: 0% a 100%
- **Interpretación**: Porcentaje de correos clasificados correctamente

### **2. MSE (Error Cuadrático Medio)**
```python
mse = Σ(y_real - y_pred)² / n
```
- **Rango**: 0 a ∞
- **Interpretación**: Menor = mejor
- **Uso**: Mide qué tan lejos están las predicciones del valor real

### **3. R² Score (Coeficiente de Determinación)**
```python
r2 = 1 - (mse_modelo / mse_baseline)
```
- **Rango**: 0 a 1
- **Interpretación**: 1 = perfecto, 0 = no mejor que línea horizontal
- **Uso**: Qué tan bien el modelo explica la variabilidad de los datos

## 🧪 Proceso de Entrenamiento

### **1. División de Datos**
```python
# 80% para entrenamiento, 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**¿Por qué dividir?**
- **Train**: Para que el modelo aprenda
- **Test**: Para evaluar el rendimiento real
- **stratify=y**: Mantener proporción de clases

### **2. Entrenamiento**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

**¿Qué hace fit()?**
- Calcula los mejores coeficientes βᵢ
- Minimiza el error cuadrático medio
- Encuentra la línea que mejor separa las clases

### **3. Evaluación**
```python
# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred > 0.5)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## 🔍 Interpretación de Resultados

### **Palabras Más Importantes**
```python
# Coeficientes ordenados por importancia absoluta
word_importance = pd.DataFrame({
    'word': feature_names,
    'coefficient': model.coef_,
    'abs_coefficient': abs(model.coef_)
})
```

**Interpretación:**
- **Coeficiente positivo**: Palabra indica SPAM
- **Coeficiente negativo**: Palabra indica HAM
- **Valor absoluto mayor**: Palabra más importante

### **Explicación de Predicciones**
```python
# Para cada palabra en el correo
contribution = count * coefficient

# Ejemplo
# Correo: "oferta gratis"
# oferta: 1 * (+0.045) = +0.045 (indica SPAM)
# gratis: 1 * (+0.032) = +0.032 (indica SPAM)
# Total: +0.077 → SPAM
```

## ⚠️ Limitaciones del Sistema

### **1. Simplicidad Excesiva**
- ❌ No captura relaciones complejas entre palabras
- ❌ Asume que las palabras son independientes
- ❌ Puede perder contexto

### **2. Sensibilidad a Palabras**
- ❌ Una sola palabra puede cambiar la predicción
- ❌ No considera el orden de las palabras
- ❌ Puede ser engañado por spam sofisticado

### **3. Overfitting**
- ❌ Puede memorizar los datos de entrenamiento
- ❌ No generaliza bien a nuevos correos
- ❌ Precisión muy alta puede indicar overfitting

## 🔮 Mejoras Posibles

### **1. Más Datos**
```python
# Agregar más correos de diferentes fuentes
# Balancear mejor las clases (50% spam, 50% ham)
# Incluir diferentes tipos de spam
```

### **2. Mejor Preprocesamiento**
```python
# Lematización: "running" → "run"
# Stemming: "running" → "run"
# Stopwords personalizadas
# Bigramas y trigramas
```

### **3. Algoritmos Más Avanzados**
```python
# Clasificación Logística (mejor para clasificación binaria)
from sklearn.linear_model import LogisticRegression

# Random Forest (más robusto)
from sklearn.ensemble import RandomForestClassifier

# Redes Neuronales (más complejo)
from sklearn.neural_network import MLPClassifier
```

## 📚 Recursos Adicionales

### **Conceptos de Machine Learning**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Linear Regression Tutorial](https://scikit-learn.org/stable/modules/linear_model.html)
- [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

### **Procesamiento de Texto**
- [NLTK Documentation](https://www.nltk.org/)
- [Regular Expressions](https://docs.python.org/3/library/re.html)
- [Text Preprocessing](https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a)

### **Regresión Lineal**
- [Linear Regression Explained](https://towardsdatascience.com/linear-regression-explained-1b36f97b7572)
- [Coefficient Interpretation](https://statisticsbyjim.com/regression/interpret-coefficients-p-values-regression/)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**¡Esta documentación te ayudará a entender completamente el sistema de detección de spam!** 🎉
