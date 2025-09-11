"""
🎯 SISTEMA DE DETECCIÓN DE SPAM CON REGRESIÓN LINEAL
===================================================

Sistema completo para detectar spam usando regresión lineal.
Diseñado para principiantes de Machine Learning.

¿QUÉ HACE ESTE SISTEMA?
- Entrena un modelo de regresión lineal con correos etiquetados
- Detecta si un correo es SPAM (no deseado) o HAM (normal)
- Explica por qué clasificó cada correo
- Permite probar correos personalizados

CONCEPTOS CLAVE:
1. Regresión Lineal: Algoritmo que encuentra la mejor línea para separar clases
2. Features: Palabras del correo convertidas en números
3. Target: 0 = HAM, 1 = SPAM
4. Vectorización: Proceso de convertir texto en números

INSTRUCCIONES:
1. Ejecutar: python spam_detector_linear.py
2. El sistema entrenará el modelo y te permitirá probar correos
"""

# =============================================================================
# IMPORTACIONES NECESARIAS
# =============================================================================

import pandas as pd          # Para manejar datos tabulares
import numpy as np           # Para operaciones matemáticas
import pickle               # Para guardar/cargar modelos
import re                   # Para expresiones regulares (limpieza de texto)
from sklearn.model_selection import train_test_split      # Dividir datos
from sklearn.feature_extraction.text import CountVectorizer  # Convertir texto a números
from sklearn.linear_model import LinearRegression         # Nuestro algoritmo
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # Métricas
import warnings
warnings.filterwarnings('ignore')  # Ocultar advertencias

# =============================================================================
# CLASE PRINCIPAL: SpamDetectorLinear
# =============================================================================

class SpamDetectorLinear:
    """
    Sistema de detección de spam usando regresión lineal.
    
    Esta clase contiene todo lo necesario para:
    1. Preprocesar texto de correos
    2. Convertir texto en características numéricas
    3. Entrenar un modelo de regresión lineal
    4. Predecir si un correo es spam o ham
    5. Explicar las predicciones
    """
    
    def __init__(self):
        """
        Inicializa el detector de spam.
        
        Atributos:
        - vectorizer: Convierte texto en números (CountVectorizer)
        - model: Nuestro modelo de regresión lineal
        - feature_names: Nombres de las características (palabras)
        """
        self.vectorizer = None      # Se inicializa en create_features()
        self.model = None           # Se inicializa en train_model()
        self.feature_names = []     # Lista de palabras únicas
        
    def preprocess_text(self, text):
        """
        Limpia el texto para el análisis de machine learning.
        
        Pasos de limpieza:
        1. Convertir a minúsculas (normalización)
        2. Eliminar puntuación (no aporta información)
        3. Eliminar números (no son relevantes para spam)
        4. Limpiar espacios extra
        
        Args:
            text (str): Texto del correo a limpiar
            
        Returns:
            str: Texto limpio y normalizado
        """
        # Convertir todo a minúsculas para normalización
        text = text.lower()
        
        # Eliminar puntuación usando expresiones regulares
        # [^\w\s] significa: cualquier carácter que NO sea palabra o espacio
        text = re.sub(r'[^\w\s]', '', text)
        
        # Eliminar números (no son relevantes para detectar spam)
        text = re.sub(r'\d+', '', text)
        
        # Eliminar espacios extra y normalizar
        text = ' '.join(text.split())
        
        return text
    
    def create_features(self, texts):
        """
        Convierte texto en características numéricas que el modelo puede entender.
        
        Usa CountVectorizer que:
        - Cuenta cuántas veces aparece cada palabra
        - Crea una matriz donde cada fila es un correo
        - Cada columna es una palabra diferente
        - max_features=1000: máximo 1000 palabras diferentes
        - stop_words='english': elimina palabras comunes (the, and, or, etc.)
        - ngram_range=(1, 2): palabras individuales y pares de palabras
        
        Args:
            texts (list): Lista de textos de correos
            
        Returns:
            scipy.sparse.csr_matrix: Matriz de características
        """
        print("🔧 Creando características del texto...")
        
        # Crear vectorizador (convierte texto en números)
        self.vectorizer = CountVectorizer(
            max_features=1000,      # Máximo 1000 palabras diferentes
            stop_words='english',   # Eliminar palabras comunes (the, and, or, etc.)
            ngram_range=(1, 2)      # Palabras individuales y pares de palabras
        )
        
        # Aplicar vectorización: convertir texto en matriz numérica
        features = self.vectorizer.fit_transform(texts)
        
        # Obtener nombres de características (palabras)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ Características creadas: {features.shape[1]} palabras únicas")
        print(f"📧 Correos procesados: {features.shape[0]}")
        
        return features
    
    def load_and_prepare_data(self, file_path):
        """
        Carga el dataset y prepara los datos para el entrenamiento.
        
        Pasos:
        1. Cargar datos desde CSV
        2. Preprocesar todos los textos
        3. Crear características numéricas
        4. Preparar etiquetas (target)
        
        Args:
            file_path (str): Ruta al archivo CSV con los datos
            
        Returns:
            tuple: (X, y) donde X son las características e y son las etiquetas
        """
        print("📂 Cargando dataset...")
        
        # Cargar datos desde archivo CSV
        df = pd.read_csv(file_path)
        
        # Preprocesar todos los textos
        print("🧹 Preprocesando textos...")
        df['clean_text'] = df['email_text'].apply(self.preprocess_text)
        
        # Crear características numéricas
        X = self.create_features(df['clean_text'])
        
        # Preparar etiquetas (target): 0 = HAM, 1 = SPAM
        y = (df['label'] == 'spam').astype(int)
        
        # Mostrar resumen de los datos
        print(f"\n📊 RESUMEN DE DATOS:")
        print(f"   Total de correos: {len(df)}")
        print(f"   Correos HAM: {sum(y == 0)}")
        print(f"   Correos SPAM: {sum(y == 1)}")
        print(f"   Porcentaje SPAM: {sum(y == 1)/len(y)*100:.1f}%")
        
        return X, y
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """
        Entrena el modelo de regresión lineal.
        
        ¿CÓMO FUNCIONA LA REGRESIÓN LINEAL?
        - Busca la mejor línea que separa SPAM de HAM
        - Cada palabra tiene un "peso" (coeficiente)
        - Palabras con peso positivo indican SPAM
        - Palabras con peso negativo indican HAM
        
        Args:
            X_train, X_test: Características de entrenamiento y prueba
            y_train, y_test: Etiquetas de entrenamiento y prueba
            
        Returns:
            dict: Diccionario con métricas de evaluación
        """
        print("\n🎯 ENTRENANDO MODELO DE REGRESIÓN LINEAL")
        print("=" * 50)
        
        # Crear y entrenar modelo de regresión lineal
        self.model = LinearRegression()
        print("🔄 Entrenando modelo...")
        self.model.fit(X_train, y_train)
        
        # Evaluar el modelo
        print("\n📈 EVALUANDO MODELO")
        print("-" * 30)
        
        # Hacer predicciones
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calcular métricas de regresión
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Mostrar métricas
        print(f"Error cuadrático medio (Train): {train_mse:.4f}")
        print(f"Error cuadrático medio (Test):  {test_mse:.4f}")
        print(f"R² Score (Train): {train_r2:.4f}")
        print(f"R² Score (Test):  {test_r2:.4f}")
        
        # Convertir predicciones continuas a clases (0 o 1)
        # Si predicción > 0.5 → SPAM, si < 0.5 → HAM
        y_pred_classes = (y_pred_test > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"Precisión de clasificación: {accuracy:.4f}")
        
        # Mostrar palabras más importantes
        self.show_important_words()
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'accuracy': accuracy
        }
    
    def show_important_words(self):
        """
        Muestra las palabras más importantes para detectar spam.
        
        Analiza los coeficientes del modelo:
        - Coeficiente positivo: palabra indica SPAM
        - Coeficiente negativo: palabra indica HAM
        - Valor absoluto mayor: palabra más importante
        """
        print(f"\n🔍 PALABRAS MÁS IMPORTANTES")
        print("=" * 40)
        
        # Obtener coeficientes del modelo
        coefficients = self.model.coef_
        
        # Crear DataFrame con palabras y sus coeficientes
        word_importance = pd.DataFrame({
            'word': self.feature_names,
            'coefficient': coefficients
        })
        
        # Ordenar por importancia absoluta (valor absoluto del coeficiente)
        word_importance['abs_coefficient'] = abs(word_importance['coefficient'])
        word_importance = word_importance.sort_values('abs_coefficient', ascending=False)
        
        # Mostrar palabras que indican SPAM (coeficiente positivo)
        print("📧 PALABRAS QUE INDICAN SPAM (coeficiente positivo):")
        spam_words = word_importance[word_importance['coefficient'] > 0].head(10)
        for _, row in spam_words.iterrows():
            print(f"   {row['word']:20} → +{row['coefficient']:.4f}")
        
        # Mostrar palabras que indican HAM (coeficiente negativo)
        print(f"\n✅ PALABRAS QUE INDICAN HAM (coeficiente negativo):")
        ham_words = word_importance[word_importance['coefficient'] < 0].head(10)
        for _, row in ham_words.iterrows():
            print(f"   {row['word']:20} → {row['coefficient']:.4f}")
    
    def predict_email(self, email_text):
        """
        Predice si un correo es spam o ham.
        
        Proceso:
        1. Preprocesar el texto del correo
        2. Convertir a características numéricas
        3. Hacer predicción con el modelo
        4. Convertir predicción continua a clase
        5. Calcular confianza
        
        Args:
            email_text (str): Texto del correo a clasificar
            
        Returns:
            dict: Diccionario con predicción, confianza, score y explicación
        """
        # Preprocesar texto
        clean_text = self.preprocess_text(email_text)
        
        # Convertir a características numéricas
        features = self.vectorizer.transform([clean_text])
        
        # Hacer predicción (número entre 0 y 1)
        prediction = self.model.predict(features)[0]
        
        # Convertir a clase: > 0.5 = SPAM, < 0.5 = HAM
        is_spam = prediction > 0.5
        
        # Calcular confianza (qué tan segura está la predicción)
        confidence = abs(prediction - 0.5) * 2  # Convertir a porcentaje
        
        return {
            'prediction': 'SPAM' if is_spam else 'HAM',
            'confidence': confidence,
            'raw_score': prediction,
            'explanation': self.explain_prediction(clean_text, prediction)
        }
    
    def explain_prediction(self, text, prediction):
        """
        Explica por qué el modelo hizo esa predicción.
        
        Analiza qué palabras del correo influyeron en la decisión:
        - Muestra las palabras presentes en el correo
        - Indica si cada palabra sugiere SPAM o HAM
        - Muestra el peso (coeficiente) de cada palabra
        
        Args:
            text (str): Texto limpio del correo
            prediction (float): Score de predicción del modelo
            
        Returns:
            str: Explicación de la predicción
        """
        # Convertir texto a características
        features = self.vectorizer.transform([text])
        feature_array = features.toarray()[0]  # Convertir a array normal
        
        # Obtener coeficientes del modelo
        coefficients = self.model.coef_
        
        # Encontrar palabras presentes en el texto
        present_words = []
        for i, count in enumerate(feature_array):
            if count > 0:  # Si la palabra está presente
                word = self.feature_names[i]
                weight = coefficients[i]
                present_words.append({
                    'word': word,
                    'weight': weight,
                    'contribution': count * weight  # Contribución total
                })
        
        # Ordenar por contribución absoluta (más importante primero)
        present_words.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Crear explicación
        explanation = f"El modelo predijo {'SPAM' if prediction > 0.5 else 'HAM'} "
        explanation += f"con un score de {prediction:.3f}.\n\n"
        
        # Mostrar palabras que influyeron
        if prediction > 0.5:
            explanation += "🔴 PALABRAS QUE INDICAN SPAM:\n"
        else:
            explanation += "🟢 PALABRAS QUE INDICAN HAM:\n"
        
        # Mostrar top 5 palabras más influyentes
        for word_info in present_words[:5]:
            if word_info['contribution'] > 0:
                explanation += f"   + {word_info['word']} (peso: +{word_info['weight']:.3f})\n"
            else:
                explanation += f"   - {word_info['word']} (peso: {word_info['weight']:.3f})\n"
        
        return explanation
    
    def save_model(self, filename='linear_spam_model.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        
        Guarda:
        - vectorizer: Para convertir texto a características
        - model: Modelo de regresión lineal entrenado
        - feature_names: Nombres de las características
        
        Args:
            filename (str): Nombre del archivo donde guardar
        """
        print(f"\n💾 Guardando modelo en {filename}...")
        
        # Preparar datos para guardar
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        # Guardar en archivo
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Modelo guardado exitosamente!")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def create_sample_emails():
    """
    Crea correos de ejemplo para probar el modelo.
    
    Incluye:
    - Correos HAM (normales): trabajo, universidad, servicios
    - Correos SPAM (no deseados): bancos falsos, premios falsos
    
    Returns:
        list: Lista de diccionarios con correos de ejemplo
    """
    return [
        {
            'from': 'juan.perez@empresa.com',
            'subject': 'Reunión de trabajo - Viernes 15:00',
            'body': 'Hola equipo, les recuerdo que tenemos una reunión importante este viernes a las 15:00 en la sala de conferencias. Por favor traigan sus reportes actualizados. Saludos, Juan Pérez - Gerente de Proyectos',
            'type': 'HAM'
        },
        {
            'from': 'info@banco-seguro.com',
            'subject': 'Verificación de cuenta bancaria requerida',
            'body': 'Estimado cliente, hemos detectado actividad sospechosa en su cuenta. Para proteger sus fondos, debe verificar su identidad inmediatamente. Haga clic en el enlace para confirmar sus datos bancarios. Su cuenta será bloqueada en 24 horas si no responde.',
            'type': 'SPAM'
        },
        {
            'from': 'maria.garcia@universidad.edu',
            'subject': 'Tarea de Machine Learning - Fecha límite',
            'body': 'Queridos estudiantes, les recuerdo que la tarea sobre regresión lineal debe entregarse el próximo lunes antes de las 23:59. Por favor revisen las instrucciones en el campus virtual. Cualquier duda, no duden en escribirme. Saludos, Prof. María García',
            'type': 'HAM'
        },
        {
            'from': 'premios@loteria-millonaria.com',
            'subject': '¡FELICIDADES! Ha ganado $1,000,000 USD',
            'body': '¡INCREÍBLE NOTICIA! Usted ha sido seleccionado como ganador de nuestro premio mayor de $1,000,000 USD. Para reclamar su premio, envíe inmediatamente sus datos personales y número de cuenta bancaria. ¡Esta oferta expira en 2 horas! Llame ahora al 1-800-PREMIOS.',
            'type': 'SPAM'
        },
        {
            'from': 'soporte@netflix.com',
            'subject': 'Confirmación de pago - Netflix',
            'body': 'Hola, hemos procesado su pago mensual de Netflix por $12.99. Su suscripción está activa hasta el próximo mes. Gracias por ser parte de Netflix. Si tiene alguna pregunta, visite nuestro centro de ayuda.',
            'type': 'HAM'
        }
    ]

def test_custom_emails(detector):
    """
    Permite al usuario probar correos personalizados.
    
    Funcionalidad:
    - Bucle infinito para ingresar correos
    - Validación de longitud mínima
    - Muestra predicción y explicación
    - Opción de salir
    
    Args:
        detector: Instancia de SpamDetectorLinear entrenada
    """
    print(f"\n{'='*60}")
    print("🎯 INGRESA TUS PROPIOS CORREOS")
    print(f"{'='*60}")
    
    while True:
        print(f"\n📝 Ingresa el texto de un correo (o 'salir' para terminar):")
        print("   (Puedes copiar y pegar un correo real)")
        
        # Obtener entrada del usuario
        user_input = input("\n   Texto del correo: ").strip()
        
        # Verificar si quiere salir
        if user_input.lower() in ['salir', 'exit', 'quit', '']:
            break
        
        # Validar longitud mínima
        if len(user_input) < 10:
            print("   ❌ El texto es muy corto. Ingresa un correo más largo.")
            continue
        
        # Hacer predicción
        result = detector.predict_email(user_input)
        
        # Mostrar resultados
        print(f"\n   🤖 RESULTADO:")
        print(f"      Predicción: {result['prediction']}")
        print(f"      Confianza: {result['confidence']:.1%}")
        print(f"      Score: {result['raw_score']:.3f}")
        
        print(f"\n   📝 EXPLICACIÓN:")
        print(f"      {result['explanation']}")
        
        print("-" * 80)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de detección de spam.
    
    Pasos:
    1. Inicializar detector
    2. Cargar y preparar datos
    3. Dividir en train/test
    4. Entrenar modelo
    5. Guardar modelo
    6. Probar con correos de ejemplo
    7. Permitir pruebas personalizadas
    8. Mostrar resumen final
    
    Returns:
        tuple: (detector, results) - detector entrenado y métricas
    """
    print("🚀 SISTEMA DE DETECCIÓN DE SPAM CON REGRESIÓN LINEAL")
    print("=" * 60)
    print("📚 Diseñado para principiantes de Machine Learning")
    print("=" * 60)
    
    # 1. Inicializar detector
    detector = SpamDetectorLinear()
    
    # 2. Cargar y preparar datos
    X, y = detector.load_and_prepare_data('spam_ham_dataset_labeled.csv')
    
    # 3. Dividir datos en train y test (80% train, 20% test)
    print(f"\n📊 Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Datos de entrenamiento: {X_train.shape[0]} correos")
    print(f"   Datos de prueba: {X_test.shape[0]} correos")
    
    # 4. Entrenar modelo
    results = detector.train_model(X_train, X_test, y_train, y_test)
    
    # 5. Guardar modelo
    detector.save_model()
    
    # 6. Probar con correos de ejemplo
    print(f"\n{'='*60}")
    print("🧪 PROBANDO CON CORREOS DE EJEMPLO")
    print(f"{'='*60}")
    
    sample_emails = create_sample_emails()
    
    for i, email in enumerate(sample_emails, 1):
        print(f"\n📧 CORREO {i}:")
        print(f"   De: {email['from']}")
        print(f"   Asunto: {email['subject']}")
        print(f"   Cuerpo: {email['body'][:100]}...")
        print(f"   Tipo real: {email['type']}")
        
        # Hacer predicción
        result = detector.predict_email(email['body'])
        
        print(f"\n   🤖 PREDICCIÓN DEL MODELO:")
        print(f"      Resultado: {result['prediction']}")
        print(f"      Confianza: {result['confidence']:.1%}")
        print(f"      Score: {result['raw_score']:.3f}")
        
        # Verificar si la predicción fue correcta
        correct = (result['prediction'] == email['type'])
        status = "✅ CORRECTO" if correct else "❌ INCORRECTO"
        print(f"      {status}")
        
        print(f"\n   📝 EXPLICACIÓN:")
        print(f"      {result['explanation']}")
        
        print("-" * 80)
    
    # 7. Permitir pruebas personalizadas
    test_custom_emails(detector)
    
    # 8. Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"✅ Modelo entrenado exitosamente")
    print(f"📈 Precisión en datos de prueba: {results['accuracy']:.1%}")
    print(f"📁 Modelo guardado como 'linear_spam_model.pkl'")
    print(f"🎯 Listo para detectar spam en nuevos correos!")
    print(f"{'='*60}")
    
    return detector, results

# =============================================================================
# EJECUCIÓN DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    # Ejecutar el sistema completo
    detector, results = main()
