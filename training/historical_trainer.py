import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import pickle
import json
import logging
from datetime import datetime

class HistoricalTrainer:
    def __init__(self, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.modelo = None
        self.scaler = None
        self.feature_importance = None
        self.metricas = {}
        
        logging.basicConfig(
            filename='logs/historical_trainer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def preparar_datos(self, df):
        """Prepara datos para entrenamiento"""
        self.logger.info("📊 Preparando datos para entrenamiento...")
        
        # Separar features y target
        feature_cols = [col for col in df.columns if col not in [
            'time', 'target', 'precio_futuro', 'open', 'high', 'low', 'close',
            'tick_volume', 'spread', 'real_volume'
        ]]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        self.logger.info(f"   Features: {len(feature_cols)}")
        self.logger.info(f"   Muestras: {len(X)}")
        self.logger.info(f"   Distribución target: CALL={np.sum(y==1)} ({np.mean(y)*100:.1f}%), PUT={np.sum(y==0)} ({(1-np.mean(y))*100:.1f}%)")
        
        return X, y, feature_cols
    
    def entrenar_modelo(self, X, y, feature_names):
        """Entrena modelo con validación temporal"""
        self.logger.info("🎓 Iniciando entrenamiento del modelo...")
        
        print(f"\n{'='*60}")
        print(f"🎓 ENTRENAMIENTO HISTÓRICO")
        print(f"{'='*60}")
        print(f"Total de muestras: {len(X)}")
        print(f"Total de features: {X.shape[1]}")
        print(f"{'='*60}\n")
        
        # Normalizar datos
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split temporal (80% train, 20% test)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"📊 División de datos:")
        print(f"   Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
        print(f"{'─'*60}\n")
        
        # Crear dataset de LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parámetros del modelo
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        # Entrenar
        print("🚀 Entrenando modelo LightGBM...")
        print(f"{'─'*60}\n")
        
        self.modelo = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        print(f"\n{'─'*60}")
        print("✅ Entrenamiento completado")
        print(f"{'─'*60}\n")
        
        # Predicciones
        y_pred_train = (self.modelo.predict(X_train) > 0.5).astype(int)
        y_pred_test = (self.modelo.predict(X_test) > 0.5).astype(int)
        
        # Métricas
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"📈 RESULTADOS:")
        print(f"{'─'*60}")
        print(f"   Accuracy Train: {train_acc*100:.2f}%")
        print(f"   Accuracy Test: {test_acc*100:.2f}%")
        print(f"{'─'*60}\n")
        
        # Reporte detallado
        print("📊 REPORTE DETALLADO (Test Set):")
        print(f"{'─'*60}")
        print(classification_report(y_test, y_pred_test, target_names=['PUT', 'CALL']))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\n📊 MATRIZ DE CONFUSIÓN:")
        print(f"{'─'*60}")
        print(f"                Predicho PUT    Predicho CALL")
        print(f"Real PUT        {cm[0][0]:>12}    {cm[0][1]:>13}")
        print(f"Real CALL       {cm[1][0]:>12}    {cm[1][1]:>13}")
        print(f"{'─'*60}\n")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.modelo.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"🔝 TOP 15 FEATURES MÁS IMPORTANTES:")
        print(f"{'─'*60}")
        for idx, row in self.feature_importance.head(15).iterrows():
            print(f"   {row['feature']:<30} {row['importance']:>10.2f}")
        print(f"{'─'*60}\n")
        
        # Guardar métricas
        self.metricas = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'num_features': int(X.shape[1]),
            'num_samples': int(len(X)),
            'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.logger.info(f"✅ Modelo entrenado - Test Accuracy: {test_acc*100:.2f}%")
        
        return self.modelo
    
    def guardar_modelo(self, nombre='historical_model'):
        """Guarda el modelo entrenado"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar modelo
            modelo_path = f'models/{nombre}_{timestamp}.pkl'
            with open(modelo_path, 'wb') as f:
                pickle.dump(self.modelo, f)
            
            # Guardar scaler
            scaler_path = f'models/{nombre}_scaler_{timestamp}.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Guardar feature importance
            importance_path = f'models/{nombre}_importance_{timestamp}.csv'
            self.feature_importance.to_csv(importance_path, index=False)
            
            # Guardar métricas
            metricas_path = f'models/{nombre}_metricas_{timestamp}.json'
            with open(metricas_path, 'w') as f:
                json.dump(self.metricas, f, indent=4)
            
            print(f"💾 MODELO GUARDADO:")
            print(f"{'─'*60}")
            print(f"   Modelo: {modelo_path}")
            print(f"   Scaler: {scaler_path}")
            print(f"   Importance: {importance_path}")
            print(f"   Métricas: {metricas_path}")
            print(f"{'='*60}\n")
            
            self.logger.info(f"✅ Modelo guardado: {modelo_path}")
            
            return modelo_path, scaler_path
            
        except Exception as e:
            self.logger.error(f"❌ Error al guardar modelo: {str(e)}")
            return None, None
    
    def cargar_modelo(self, modelo_path, scaler_path):
        """Carga un modelo previamente entrenado"""
        try:
            with open(modelo_path, 'rb') as f:
                self.modelo = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info(f"✅ Modelo cargado: {modelo_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error al cargar modelo: {str(e)}")
            return False
    
    def predecir(self, X):
        """Realiza predicciones"""
        if self.modelo is None or self.scaler is None:
            self.logger.error("❌ Modelo no entrenado")
            return None
        
        try:
            X_scaled = self.scaler.transform(X)
            probabilidades = self.modelo.predict(X_scaled)
            predicciones = (probabilidades > 0.5).astype(int)
            
            return predicciones, probabilidades
            
        except Exception as e:
            self.logger.error(f"❌ Error al predecir: {str(e)}")
            return None, None
