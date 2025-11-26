import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class HistoricalTrainer:
    """
    Entrenador de modelos con datos históricos
    """
    
    def __init__(self, n_estimators=300, max_depth=None, min_samples_split=5, random_state=42):
        """
        Inicializa el entrenador
        
        Args:
            n_estimators: Número de árboles en el Random Forest
            max_depth: Profundidad máxima de los árboles
            min_samples_split: Mínimo de muestras para dividir un nodo
            random_state: Semilla para reproducibilidad
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.modelo = None
        
    def preparar_datos(self, df):
        """
        Prepara los datos para entrenamiento
        
        Args:
            df: DataFrame con features y target
            
        Returns:
            X, y: Features y target separados
        """
        try:
            if 'target' not in df.columns:
                logger.error("No se encuentra columna 'target' en el DataFrame")
                return None, None
            
            # Separar features y target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Eliminar columnas no numéricas
            X = X.select_dtypes(include=[np.number])
            
            logger.info(f"✅ Datos preparados: {X.shape[0]} filas, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos: {e}")
            return None, None
    
    def entrenar(self, df):
        """
        Entrena el modelo (método principal - llama a entrenar_modelo)
        
        Args:
            df: DataFrame con features y target
            
        Returns:
            Modelo entrenado o None si hay error
        """
        X, y = self.preparar_datos(df)
        if X is None or y is None:
            return None
        return self.entrenar_modelo(X, y)
    
    def entrenar_modelo(self, X, y):
        """
        Entrena el modelo Random Forest
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Modelo entrenado o None si hay error
        """
        try:
            if X is None or y is None:
                logger.error("Datos inválidos para entrenamiento")
                return None
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"📊 Distribución de datos:")
            logger.info(f"   Training: {len(X_train)} muestras")
            logger.info(f"   Testing: {len(X_test)} muestras")
            
            # Crear y entrenar modelo
            self.modelo = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            logger.info("🤖 Entrenando Random Forest...")
            self.modelo.fit(X_train, y_train)
            
            # Evaluar
            y_pred_train = self.modelo.predict(X_train)
            y_pred_test = self.modelo.predict(X_test)
            
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            
            logger.info(f"✅ Modelo entrenado exitosamente!")
            logger.info(f"   Accuracy Training: {acc_train:.4f}")
            logger.info(f"   Accuracy Testing: {acc_test:.4f}")
            
            # Reporte de clasificación
            logger.info("\n📈 Reporte de Clasificación:")
            print(classification_report(y_test, y_pred_test, 
                                       target_names=['Venta', 'Neutral', 'Compra']))
            
            # Matriz de confusión
            logger.info("\n🔢 Matriz de Confusión:")
            print(confusion_matrix(y_test, y_pred_test))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n🎯 Top 10 Features más importantes:")
            print(feature_importance.head(10).to_string(index=False))
            
            return self.modelo
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def guardar_modelo(self, ruta='models/modelo_historico.pkl'):
        """
        Guarda el modelo entrenado
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        try:
            if self.modelo is None:
                logger.error("No hay modelo para guardar")
                return False
            
            import os
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            
            joblib.dump(self.modelo, ruta)
            logger.info(f"✅ Modelo guardado en: {ruta}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def cargar_modelo(self, ruta='models/modelo_historico.pkl'):
        """
        Carga un modelo previamente guardado
        
        Args:
            ruta: Ruta del modelo a cargar
        """
        try:
            self.modelo = joblib.load(ruta)
            logger.info(f"✅ Modelo cargado desde: {ruta}")
            return self.modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
