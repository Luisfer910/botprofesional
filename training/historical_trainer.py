import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

class HybridTrainer:
    """
    Entrenador híbrido que combina modelo histórico con datos en tiempo real
    """
    
    def __init__(self, modelo_historico=None, peso_historico=0.7, n_estimators=200):
        """
        Inicializa el entrenador híbrido
        
        Args:
            modelo_historico: Modelo pre-entrenado con datos históricos
            peso_historico: Peso del modelo histórico en la predicción final (0-1)
            n_estimators: Número de árboles para el modelo de tiempo real
        """
        self.modelo_historico = modelo_historico
        self.peso_historico = peso_historico
        self.modelo_tiempo_real = None
        self.n_estimators = n_estimators
    
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
    
    def entrenar(self, df_historico, df_tiempo_real):
        """
        Método principal de entrenamiento (API compatible)
        
        Args:
            df_historico: DataFrame con datos históricos (con columna 'target')
            df_tiempo_real: DataFrame con datos recientes (con columna 'target')
            
        Returns:
            Modelo híbrido entrenado o None si hay error
        """
        return self.entrenar_modelo(df_historico, df_tiempo_real)
    
    def entrenar_modelo(self, df_historico, df_tiempo_real):
        """
        Entrena el modelo híbrido
        
        Args:
            df_historico: DataFrame con datos históricos
            df_tiempo_real: DataFrame con datos recientes
            
        Returns:
            Modelo híbrido o None si hay error
        """
        try:
            logger.info("🔄 Iniciando entrenamiento híbrido...")
            
            # Preparar datos de tiempo real
            X_rt, y_rt = self.preparar_datos(df_tiempo_real)
            
            if X_rt is None or y_rt is None:
                logger.error("Error preparando datos de tiempo real")
                return None
            
            if len(X_rt) < 100:
                logger.warning(f"⚠️ Pocos datos de tiempo real ({len(X_rt)}). Se recomienda al menos 100.")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_rt, y_rt, test_size=0.2, random_state=42, stratify=y_rt
            )
            
            logger.info(f"📊 Datos de entrenamiento híbrido:")
            logger.info(f"   Training: {len(X_train)} muestras")
            logger.info(f"   Testing: {len(X_test)} muestras")
            
            # Entrenar modelo de tiempo real
            self.modelo_tiempo_real = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("🤖 Entrenando modelo de tiempo real...")
            self.modelo_tiempo_real.fit(X_train, y_train)
            
            # Evaluar
            y_pred = self.modelo_tiempo_real.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ Modelo híbrido entrenado!")
            logger.info(f"   Accuracy: {acc:.4f}")
            
            # Si hay modelo histórico, mostrar configuración
            if self.modelo_historico is not None:
                logger.info(f"   Peso histórico: {self.peso_historico:.0%}")
                logger.info(f"   Peso tiempo real: {(1-self.peso_historico):.0%}")
            else:
                logger.warning("⚠️ No hay modelo histórico. Solo se usa modelo de tiempo real.")
            
            return self
            
        except Exception as e:
            logger.error(f"Error en entrenamiento híbrido: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predecir(self, X):
        """
        Realiza predicción híbrida combinando ambos modelos
        
        Args:
            X: Features para predicción
            
        Returns:
            Array con predicciones
        """
        try:
            if self.modelo_tiempo_real is None:
                logger.error("Modelo de tiempo real no entrenado")
                return None
            
            # Predicción del modelo de tiempo real
            pred_rt = self.modelo_tiempo_real.predict_proba(X)
            
            # Si no hay modelo histórico, retornar solo tiempo real
            if self.modelo_historico is None:
                return pred_rt
            
            # Predicción del modelo histórico
            pred_hist = self.modelo_historico.predict_proba(X)
            
            # Combinar predicciones
            pred_hibrida = (self.peso_historico * pred_hist + 
                           (1 - self.peso_historico) * pred_rt)
            
            return pred_hibrida
            
        except Exception as e:
            logger.error(f"Error en predicción híbrida: {e}")
            return None
    
    def guardar_modelo(self, ruta='models/modelo_hibrido.pkl'):
        """
        Guarda el modelo híbrido
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        try:
            if self.modelo_tiempo_real is None:
                logger.error("No hay modelo para guardar")
                return False
            
            import os
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            
            # Guardar todo el objeto HybridTrainer
            joblib.dump(self, ruta)
            logger.info(f"✅ Modelo híbrido guardado en: {ruta}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    @staticmethod
    def cargar_modelo(ruta='models/modelo_hibrido.pkl'):
        """
        Carga un modelo híbrido previamente guardado
        
        Args:
            ruta: Ruta del modelo a cargar
            
        Returns:
            Objeto HybridTrainer cargado
        """
        try:
            modelo = joblib.load(ruta)
            logger.info(f"✅ Modelo híbrido cargado desde: {ruta}")
            return modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
