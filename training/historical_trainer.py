import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class HistoricalTrainer:
    """
    Entrenador de modelo con datos históricos
    """
    
    def __init__(self, config=None):
        """
        Inicializa el entrenador histórico
        
        Args:
            config: Configuración del entrenador
        """
        self.config = config or {}
        
        # Parámetros del modelo
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 20)
        self.min_samples_split = self.config.get('min_samples_split', 10)
        self.random_state = self.config.get('random_state', 42)
        
        # Scaler
        self.scaler = StandardScaler()
        
        logger.info("✅ HistoricalTrainer inicializado")
    
    
    def preparar_datos(self, df):
        """
        Prepara los datos para entrenamiento
        
        Args:
            df: DataFrame con features
            
        Returns:
            tuple: (X, y) o (None, None) si hay error
        """
        try:
            if df is None or len(df) == 0:
                logger.error("DataFrame vacío")
                return None, None
            
            # Verificar que exista columna target
            if 'target' not in df.columns:
                logger.error("No se encontró columna 'target'")
                return None, None
            
            # Columnas a excluir
            columnas_excluir = [
                'time', 'target', 'label', 'future_return', 
                'precio_futuro', 'tick_volume', 'spread', 'real_volume'
            ]
            
            # Separar features y target
            X = df.copy()
            for col in columnas_excluir:
                if col in X.columns:
                    X = X.drop(columns=[col])
            
            y = df['target'].values
            
            # Eliminar NaN
            if X.isnull().any().any():
                logger.warning("Eliminando NaN...")
                mask = ~X.isnull().any(axis=1)
                X = X[mask]
                y = y[mask]
            
            logger.info(f"✅ Datos preparados: {len(X)} muestras, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    
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
            modelo = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            logger.info("🤖 Entrenando Random Forest...")
            modelo.fit(X_train, y_train)
            
            # Evaluar
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)
            
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
                'importance': modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n🎯 Top 10 Features más importantes:")
            print(feature_importance.head(10).to_string(index=False))
            
            return modelo
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def guardar_modelo(self, modelo, scaler=None, metadata=None):
        """
        Guarda el modelo entrenado
        
        Args:
            modelo: Modelo entrenado
            scaler: Scaler usado (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            tuple: (path, metadata) o None si falla
        """
        try:
            os.makedirs('models', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # ✅ NOMBRE CORRECTO: modelo_hibrido
            filename = f"modelo_hibrido_{timestamp}.pkl"
            path = os.path.join('models', filename)
            
            # Preparar metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'timestamp': timestamp,
                'tipo': 'hibrido',
                'modelo_clase': type(modelo).__name__
            })
            
            # Guardar modelo
            joblib.dump(modelo, path)
            
            # Verificar que se guardó
            if not os.path.exists(path):
                logger.error(f"❌ Archivo no creado: {path}")
                return None
            
            file_size = os.path.getsize(path) / 1024
            logger.info(f"✅ Modelo guardado: {path} ({file_size:.2f} KB)")
            
            # Guardar metadata
            metadata_path = path.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return (path, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def cargar_modelo(self, path):
        """
        Carga un modelo guardado
        
        Args:
            path: Ruta del modelo
            
        Returns:
            Modelo cargado o None si falla
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Archivo no encontrado: {path}")
                return None
            
            modelo = joblib.load(path)
            logger.info(f"✅ Modelo cargado: {path}")
            
            return modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
