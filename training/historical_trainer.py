import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import logging
from datetime import datetime

class HistoricalTrainer:
    def __init__(self, n_estimators=300, max_depth=None, min_samples_split=5, random_state=42):
        """
        Inicializa el entrenador de modelo histórico
        
        Args:
            n_estimators: Número de árboles en el Random Forest
            max_depth: Profundidad máxima de los árboles
            min_samples_split: Mínimo de muestras para dividir un nodo
            random_state: Semilla aleatoria
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.modelo = None
        self.scaler = None
        self.feature_cols = None
        
        logging.basicConfig(
            filename='logs/historical_trainer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def entrenar(self, df):
        """
        Entrena el modelo con datos históricos
        
        Args:
            df: DataFrame con features y columna 'target'
            
        Returns:
            Modelo entrenado o None si hay error
        """
        try:
            self.logger.info("🤖 Iniciando entrenamiento de modelo histórico...")
            
            # Verificar que tenga columna target
            if 'target' not in df.columns:
                self.logger.error("❌ No se encuentra columna 'target'")
                return None
            
            # Separar features y target
            excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                      'spread', 'real_volume', 'target', 'precio_futuro']
            self.feature_cols = [col for col in df.columns if col not in excluir]
            
            X = df[self.feature_cols]
            y = df['target']
            
            # Eliminar columnas no numéricas
            X = X.select_dtypes(include=[np.number])
            self.feature_cols = X.columns.tolist()
            
            self.logger.info(f"✅ Datos preparados: {len(X)} filas, {len(self.feature_cols)} features")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            self.logger.info(f"📊 Distribución de datos:")
            self.logger.info(f"   Training: {len(X_train)} muestras")
            self.logger.info(f"   Testing: {len(X_test)} muestras")
            
            # Normalizar
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar Random Forest
            self.logger.info("🤖 Entrenando Random Forest...")
            self.modelo = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            self.modelo.fit(X_train_scaled, y_train)
            
            # Evaluar
            y_pred_train = self.modelo.predict(X_train_scaled)
            y_pred_test = self.modelo.predict(X_test_scaled)
            
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            
            self.logger.info("✅ Modelo entrenado exitosamente!")
            self.logger.info(f"   Accuracy Training: {acc_train:.4f}")
            self.logger.info(f"   Accuracy Testing: {acc_test:.4f}")
            
            # Reporte de clasificación
            target_names = ['Venta', 'Neutral', 'Compra'] if len(np.unique(y)) == 3 else ['Venta', 'Compra']
            report = classification_report(y_test, y_pred_test, target_names=target_names)
            self.logger.info(f"\n📈 Reporte de Clasificación:\n{report}")
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_test)
            self.logger.info(f"\n🔢 Matriz de Confusión:\n{cm}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"\n🎯 Top 10 Features más importantes:\n{feature_importance.head(10).to_string(index=False)}")
            
            return self.modelo
            
        except Exception as e:
            self.logger.error(f"❌ Error en entrenamiento: {e}")
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
                self.logger.error("❌ No hay modelo para guardar")
                return False
            
            import os
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            
            # Guardar modelo completo
            modelo_data = {
                'modelo': self.modelo,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols
            }
            
            with open(ruta, 'wb') as f:
                pickle.dump(modelo_data, f)
            
            self.logger.info(f"✅ Modelo guardado en: {ruta}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando modelo: {e}")
            return False
    
    def cargar_modelo(self, ruta='models/modelo_historico.pkl'):
        """
        Carga un modelo previamente entrenado
        
        Args:
            ruta: Ruta del modelo a cargar
        """
        try:
            with open(ruta, 'rb') as f:
                modelo_data = pickle.load(f)
            
            self.modelo = modelo_data['modelo']
            self.scaler = modelo_data['scaler']
            self.feature_cols = modelo_data['feature_cols']
            
            self.logger.info(f"✅ Modelo cargado desde: {ruta}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo: {e}")
            return False
    
    def predecir(self, X):
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            X: DataFrame con features
            
        Returns:
            Predicciones y probabilidades
        """
        try:
            if self.modelo is None:
                self.logger.error("❌ Modelo no entrenado")
                return None, None
            
            # Seleccionar solo las features usadas en entrenamiento
            X_features = X[self.feature_cols]
            
            # Normalizar
            X_scaled = self.scaler.transform(X_features)
            
            # Predecir
            predicciones = self.modelo.predict(X_scaled)
            probabilidades = self.modelo.predict_proba(X_scaled)
            
            return predicciones, probabilidades
            
        except Exception as e:
            self.logger.error(f"❌ Error en predicción: {e}")
            return None, None
