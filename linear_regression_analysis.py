"""
🎯 ANÁLISIS COMPLETO DE REGRESIÓN LINEAL PARA DETECCIÓN DE SPAM
================================================================

Este script realiza un análisis completo de regresión lineal para detección de spam incluyendo:
- Construcción de modelo con 1000 instancias
- Matriz de confusión
- Análisis de manipulación de datos
- Reporte de importancia de features (top 10)
- Análisis de correlación con Python, Keras y sklearn
- Generación de reportes en PDF

Autor: Juan Esteban Moya Riaño, Maryuri Espinosa
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           confusion_matrix, classification_report, 
                           mean_absolute_error)
from sklearn.preprocessing import StandardScaler
import pickle
import re
import warnings
from datetime import datetime
import os

# Para análisis de correlación con Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ Keras/TensorFlow no disponible. Análisis de correlación limitado.")

warnings.filterwarnings('ignore')

class LinearRegressionSpamAnalysis:
    """
    Análisis completo de regresión lineal para detección de spam.
    
    Incluye:
    - Preprocesamiento de datos
    - Construcción de modelo con 1000 instancias
    - Matriz de confusión
    - Análisis de importancia de features
    - Análisis de correlación
    - Generación de reportes
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.feature_names = []
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self, file_path, max_instances=1000):
        """
        Carga y preprocesa los datos para el análisis.
        
        Args:
            file_path (str): Ruta al archivo CSV
            max_instances (int): Número máximo de instancias a usar
        """
        print("📂 Cargando y preprocesando datos...")
        
        # Cargar datos
        self.df = pd.read_csv(file_path)
        
        # Limitar a 1000 instancias si hay más
        if len(self.df) > max_instances:
            print(f"📊 Limitando dataset a {max_instances} instancias")
            self.df = self.df.sample(n=max_instances, random_state=42)
        
        print(f"✅ Dataset cargado: {len(self.df)} instancias")
        print(f"   - HAM: {sum(self.df['label'] == 'ham')}")
        print(f"   - SPAM: {sum(self.df['label'] == 'spam')}")
        
        # Preprocesar texto
        self.df['clean_text'] = self.df['email_text'].apply(self._preprocess_text)
        
        # Crear features
        self._create_features()
        
        # Preparar target
        self.y = (self.df['label'] == 'spam').astype(int)
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"✅ Datos preparados:")
        print(f"   - Entrenamiento: {self.X_train.shape[0]} instancias")
        print(f"   - Prueba: {self.X_test.shape[0]} instancias")
        
    def _preprocess_text(self, text):
        """Preprocesa el texto para análisis."""
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar puntuación
        text = re.sub(r'[^\w\s]', '', text)
        
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        
        # Limpiar espacios
        text = ' '.join(text.split())
        
        return text
    
    def _create_features(self):
        """Crea features usando CountVectorizer."""
        print("🔧 Creando features...")
        
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.X = self.vectorizer.fit_transform(self.df['clean_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ Features creadas: {self.X.shape[1]} características")
    
    def train_linear_regression_model(self):
        """Entrena el modelo de regresión lineal."""
        print("\n🎯 ENTRENANDO MODELO DE REGRESIÓN LINEAL")
        print("=" * 50)
        
        # Crear y entrenar modelo
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Hacer predicciones
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Convertir predicciones continuas a clases
        y_pred_classes = (y_pred_test > 0.5).astype(int)
        
        # Calcular métricas
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Guardar resultados
        self.results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'accuracy': accuracy,
            'mae': mae,
            'y_pred_test': y_pred_test,
            'y_pred_classes': y_pred_classes
        }
        
        print(f"📈 MÉTRICAS DEL MODELO:")
        print(f"   - MSE (Train): {train_mse:.4f}")
        print(f"   - MSE (Test): {test_mse:.4f}")
        print(f"   - R² (Train): {train_r2:.4f}")
        print(f"   - R² (Test): {test_r2:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - MAE: {mae:.4f}")
        
        return self.results
    
    def generate_confusion_matrix(self):
        """Genera y muestra la matriz de confusión."""
        print("\n📊 GENERANDO MATRIZ DE CONFUSIÓN")
        print("=" * 40)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(self.y_test, self.results['y_pred_classes'])
        
        # Crear visualización
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['HAM', 'SPAM'], 
                   yticklabels=['HAM', 'SPAM'])
        plt.title('Matriz de Confusión - Regresión Lineal')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.savefig('confusion_matrix_linear.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostrar métricas detalladas
        tn, fp, fn, tp = cm.ravel()
        
        print(f"📈 MÉTRICAS DETALLADAS:")
        print(f"   - Verdaderos Negativos (HAM correctos): {tn}")
        print(f"   - Falsos Positivos (HAM clasificados como SPAM): {fp}")
        print(f"   - Falsos Negativos (SPAM clasificados como HAM): {fn}")
        print(f"   - Verdaderos Positivos (SPAM correctos): {tp}")
        
        # Calcular métricas adicionales
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   - Precisión: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")
        
        return cm
    
    def analyze_feature_importance(self, top_n=10):
        """Analiza la importancia de las features."""
        print(f"\n🔍 ANÁLISIS DE IMPORTANCIA DE FEATURES (Top {top_n})")
        print("=" * 60)
        
        # Obtener coeficientes
        coefficients = self.model.coef_
        
        # Crear DataFrame con importancia
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Ordenar por importancia absoluta
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        # Top features
        top_features = feature_importance.head(top_n)
        
        print(f"🏆 TOP {top_n} FEATURES MÁS IMPORTANTES:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            direction = "→ SPAM" if row['coefficient'] > 0 else "→ HAM"
            print(f"{i:2d}. {row['feature']:20} | {row['coefficient']:8.4f} | {direction}")
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'blue' for x in top_features['coefficient']]
        
        bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coeficiente (Importancia)')
        plt.title(f'Top {top_n} Features Más Importantes\n(Rojo=SPAM, Azul=HAM)')
        plt.grid(axis='x', alpha=0.3)
        
        # Agregar valores en las barras
        for i, (bar, coef) in enumerate(zip(bars, top_features['coefficient'])):
            plt.text(coef + (0.001 if coef > 0 else -0.001), i, f'{coef:.3f}', 
                    va='center', ha='left' if coef > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig('feature_importance_linear.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_features
    
    def correlation_analysis(self):
        """Realiza análisis de correlación usando Python, Keras y sklearn."""
        print("\n🔗 ANÁLISIS DE CORRELACIÓN")
        print("=" * 40)
        
        # Convertir matriz dispersa a densa para análisis de correlación
        X_dense = self.X.toarray()
        
        # 1. Correlación entre features (usando sklearn)
        print("📊 1. Correlación entre features (sklearn):")
        
        # Calcular matriz de correlación
        corr_matrix = np.corrcoef(X_dense.T)
        
        # Encontrar correlaciones altas
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > 0.5:  # Umbral de correlación
                    high_corr_pairs.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'correlation': corr_matrix[i, j]
                    })
        
        # Ordenar por correlación absoluta
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print(f"   Encontradas {len(high_corr_pairs)} correlaciones altas (>0.5):")
        for pair in high_corr_pairs[:10]:  # Mostrar top 10
            print(f"   - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        # 2. Análisis con Keras (si está disponible)
        if KERAS_AVAILABLE:
            print("\n🧠 2. Análisis de correlación con Keras:")
            self._keras_correlation_analysis(X_dense)
        else:
            print("\n⚠️ Keras no disponible. Saltando análisis con redes neuronales.")
        
        # 3. Visualización de correlación
        self._plot_correlation_matrix(corr_matrix)
        
        return high_corr_pairs
    
    def _keras_correlation_analysis(self, X_dense):
        """Análisis de correlación usando Keras."""
        try:
            # Crear modelo simple para análisis de correlación
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_dense.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Entrenar modelo
            print("   Entrenando modelo Keras para análisis de correlación...")
            history = model.fit(X_dense, self.y, epochs=10, batch_size=32, 
                              validation_split=0.2, verbose=0)
            
            # Obtener pesos de la primera capa
            weights = model.layers[0].get_weights()[0]
            
            # Calcular importancia de features basada en pesos
            feature_importance_keras = np.mean(np.abs(weights), axis=1)
            
            # Top features según Keras
            top_indices = np.argsort(feature_importance_keras)[-10:]
            print("   Top 10 features según Keras:")
            for i, idx in enumerate(reversed(top_indices)):
                print(f"   {i+1:2d}. {self.feature_names[idx]:20} | {feature_importance_keras[idx]:.4f}")
            
            return feature_importance_keras
            
        except Exception as e:
            print(f"   ❌ Error en análisis Keras: {e}")
            return None
    
    def _plot_correlation_matrix(self, corr_matrix):
        """Visualiza la matriz de correlación."""
        # Seleccionar solo las primeras 50 features para visualización
        n_features = min(50, len(self.feature_names))
        corr_subset = corr_matrix[:n_features, :n_features]
        feature_names_subset = self.feature_names[:n_features]
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_subset, 
                   xticklabels=feature_names_subset,
                   yticklabels=feature_names_subset,
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlación entre Features\n(Primeras 50 features)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_data_manipulation_report(self):
        """Genera reporte de manipulación de datos."""
        print("\n📋 REPORTE DE MANIPULACIÓN DE DATOS")
        print("=" * 50)
        
        report = {
            'total_instances': len(self.df),
            'ham_instances': sum(self.df['label'] == 'ham'),
            'spam_instances': sum(self.df['label'] == 'spam'),
            'features_created': self.X.shape[1],
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'preprocessing_steps': [
                'Conversión a minúsculas',
                'Eliminación de puntuación',
                'Eliminación de números',
                'Limpieza de espacios',
                'Vectorización con CountVectorizer',
                'Eliminación de stop words en inglés',
                'N-gramas (1,2)',
                'Límite de 1000 features'
            ]
        }
        
        print(f"📊 ESTADÍSTICAS DEL DATASET:")
        print(f"   - Total de instancias: {report['total_instances']}")
        print(f"   - Instancias HAM: {report['ham_instances']} ({report['ham_instances']/report['total_instances']*100:.1f}%)")
        print(f"   - Instancias SPAM: {report['spam_instances']} ({report['spam_instances']/report['total_instances']*100:.1f}%)")
        print(f"   - Features creadas: {report['features_created']}")
        print(f"   - Datos de entrenamiento: {report['train_size']}")
        print(f"   - Datos de prueba: {report['test_size']}")
        
        print(f"\n🔧 PASOS DE PREPROCESAMIENTO:")
        for i, step in enumerate(report['preprocessing_steps'], 1):
            print(f"   {i}. {step}")
        
        return report
    
    def generate_pdf_report(self, top_features):
        """Genera reporte en PDF con las features más importantes."""
        print("\n📄 GENERANDO REPORTE PDF")
        print("=" * 30)
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            
            # Crear documento PDF
            doc = SimpleDocTemplate("feature_importance_report.pdf", pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1  # Centrado
            )
            
            story.append(Paragraph("REPORTE DE IMPORTANCIA DE FEATURES", title_style))
            story.append(Paragraph("Sistema de Detección de Spam - Regresión Lineal", styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # Información del modelo
            story.append(Paragraph("INFORMACIÓN DEL MODELO", styles['Heading2']))
            story.append(Paragraph(f"• Algoritmo: Regresión Lineal", styles['Normal']))
            story.append(Paragraph(f"• Instancias de entrenamiento: {self.X_train.shape[0]}", styles['Normal']))
            story.append(Paragraph(f"• Instancias de prueba: {self.X_test.shape[0]}", styles['Normal']))
            story.append(Paragraph(f"• Total de features: {len(self.feature_names)}", styles['Normal']))
            story.append(Paragraph(f"• Precisión: {self.results['accuracy']:.4f}", styles['Normal']))
            story.append(Paragraph(f"• R² Score: {self.results['test_r2']:.4f}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Top features
            story.append(Paragraph("TOP 10 FEATURES MÁS IMPORTANTES", styles['Heading2']))
            
            # Crear tabla de features
            table_data = [['Rank', 'Feature', 'Coeficiente', 'Dirección', 'Importancia']]
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                direction = "SPAM" if row['coefficient'] > 0 else "HAM"
                importance = "Alta" if abs(row['coefficient']) > 0.01 else "Media" if abs(row['coefficient']) > 0.005 else "Baja"
                table_data.append([
                    str(i),
                    row['feature'],
                    f"{row['coefficient']:.4f}",
                    direction,
                    importance
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Interpretación
            story.append(Paragraph("INTERPRETACIÓN", styles['Heading2']))
            story.append(Paragraph("• Coeficiente positivo: La feature indica SPAM", styles['Normal']))
            story.append(Paragraph("• Coeficiente negativo: La feature indica HAM", styles['Normal']))
            story.append(Paragraph("• Valor absoluto mayor: Mayor importancia en la decisión", styles['Normal']))
            story.append(Paragraph("• Las features se obtuvieron mediante CountVectorizer con n-gramas (1,2)", styles['Normal']))
            
            # Construir PDF
            doc.build(story)
            print("✅ Reporte PDF generado: feature_importance_report.pdf")
            
        except ImportError:
            print("❌ reportlab no disponible. Instalando...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'reportlab'])
            print("✅ reportlab instalado. Ejecuta el script nuevamente.")
        except Exception as e:
            print(f"❌ Error generando PDF: {e}")
    
    def save_model_and_results(self):
        """Guarda el modelo y resultados."""
        print("\n💾 GUARDANDO MODELO Y RESULTADOS")
        print("=" * 40)
        
        # Guardar modelo
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'feature_names': self.feature_names,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('linear_regression_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Modelo guardado: linear_regression_model.pkl")
        
        # Guardar resultados en CSV
        results_df = pd.DataFrame([self.results])
        results_df.to_csv('model_results.csv', index=False)
        print("✅ Resultados guardados: model_results.csv")

def main():
    """Función principal para ejecutar el análisis completo."""
    print("🚀 ANÁLISIS COMPLETO DE REGRESIÓN LINEAL PARA DETECCIÓN DE SPAM")
    print("=" * 70)
    
    # Crear instancia del analizador
    analyzer = LinearRegressionSpamAnalysis()
    
    # 1. Cargar y preprocesar datos
    analyzer.load_and_preprocess_data('spam_ham_dataset_labeled.csv', max_instances=1000)
    
    # 2. Entrenar modelo
    results = analyzer.train_linear_regression_model()
    
    # 3. Generar matriz de confusión
    cm = analyzer.generate_confusion_matrix()
    
    # 4. Analizar importancia de features
    top_features = analyzer.analyze_feature_importance(top_n=10)
    
    # 5. Análisis de correlación
    correlation_results = analyzer.correlation_analysis()
    
    # 6. Reporte de manipulación de datos
    data_report = analyzer.generate_data_manipulation_report()
    
    # 7. Generar reporte PDF
    analyzer.generate_pdf_report(top_features)
    
    # 8. Guardar modelo y resultados
    analyzer.save_model_and_results()
    
    print("\n🎉 ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 40)
    print("📁 Archivos generados:")
    print("   - confusion_matrix_linear.png")
    print("   - feature_importance_linear.png")
    print("   - correlation_matrix.png")
    print("   - feature_importance_report.pdf")
    print("   - linear_regression_model.pkl")
    print("   - model_results.csv")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
