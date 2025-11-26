import os

# 1. CONTENIDO PARA training/historical_trainer.py
historical_trainer_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging

class HistoricalTrainer:
    def __init__(self, n_estimators=300, max_depth=None, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.modelo = None
        self.scaler = None
        self.feature_cols = None
        logging.basicConfig(filename='logs/historical_trainer.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def entrenar(self, df):
        try:
            self.logger.info("Iniciando entrenamiento...")
            if 'target' not in df.columns: return None
            excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
            self.feature_cols = [col for col in df.columns if col not in excluir]
            X = df[self.feature_cols].select_dtypes(include=[np.number])
            self.feature_cols = X.columns.tolist()
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.modelo = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=-1)
            self.modelo.fit(X_train_scaled, y_train)
            
            acc = accuracy_score(y_test, self.modelo.predict(X_test_scaled))
            self.logger.info(f"Modelo entrenado. Accuracy: {acc:.4f}")
            return self.modelo
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
    
    def guardar_modelo(self, ruta='models/modelo_historico.pkl'):
        try:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, 'wb') as f:
                pickle.dump({'modelo': self.modelo, 'scaler': self.scaler, 'feature_cols': self.feature_cols}, f)
            return True
        except: return False
"""

# 2. CONTENIDO PARA training/hybrid_trainer.py
hybrid_trainer_code = """
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import json
import logging
import os

class HybridTrainer:
    def __init__(self, config_path='config/xm_config.json'):
        self.modelo_historico = None
        self.modelo_live = None
        self.modelo_hibrido = None
        self.scaler = None
        logging.basicConfig(filename='logs/hybrid_trainer.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def entrenar(self, df_historico, df_tiempo_real):
        try:
            self.logger.info("Entrenando hibrido...")
            excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
            feature_cols = [col for col in df_tiempo_real.columns if col not in excluir]
            X = df_tiempo_real[feature_cols].select_dtypes(include=[np.number])
            y = df_tiempo_real['target']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            train_data = lgb.Dataset(X_scaled, label=y)
            
            params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'verbose': -1}
            self.modelo_live = lgb.train(params, train_data, num_boost_round=50)
            
            self.logger.info("Modelo hibrido entrenado")
            return self
        except Exception as e:
            self.logger.error(f"Error hibrido: {e}")
            return None

    def guardar_modelo(self, ruta='models/modelo_hibrido.pkl'):
        try:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            data = {'modelo': self.modelo_live, 'scaler': self.scaler, 'historico': self.modelo_historico}
            with open(ruta, 'wb') as f: pickle.dump(data, f)
            return True
        except: return False
"""

# 3. CONTENIDO PARA entrenar_completo.py
entrenar_completo_code = """
import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar módulos
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer
from core.feature_engineer import FeatureEngineer

def main():
    print("🚀 INICIANDO REPARACION Y ENTRENAMIENTO")
    
    # 1. CONEXION
    if not mt5.initialize():
        logger.error("Error MT5 init")
        return
        
    # USAMOS TUS CREDENCIALES DIRECTAMENTE PARA EVITAR ERROR DE CONFIG
    login = mt5.login(100464594, password="Fer101996-", server="XMGlobalSC-MT5 5")
    if not login:
        logger.error("Error Login")
        return
    logger.info("✅ Conectado a XM")

    # 2. DATOS
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 5000)
    if rates is None: return
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    logger.info(f"✅ {len(df)} velas descargadas")

    # 3. FEATURES
    fe = FeatureEngineer()
    df_features = fe.generar_todas_features(df)
    
    # Target simple para pruebas
    df_features['target'] = np.where(df_features['close'].shift(-1) > df_features['close'], 2, 0)
    df_features = df_features.dropna()
    
    # 4. ENTRENAMIENTO HISTORICO
    logger.info("🤖 Entrenando Historico...")
    trainer_hist = HistoricalTrainer()
    modelo_hist = trainer_hist.entrenar(df_features)
    trainer_hist.guardar_modelo('models/modelo_historico.pkl')
    
    # 5. ENTRENAMIENTO HIBRIDO
    logger.info("🔄 Entrenando Hibrido...")
    trainer_hib = HybridTrainer()
    trainer_hib.modelo_historico = modelo_hist
    trainer_hib.entrenar(df_features, df_features.tail(1000))
    trainer_hib.guardar_modelo('models/modelo_hibrido.pkl')
    
    logger.info("✅ TODO FINALIZADO CORRECTAMENTE")
    mt5.shutdown()

if __name__ == "__main__":
    main()
"""

# ESCRIBIR ARCHIVOS
print("REPARANDO ARCHIVOS...")

os.makedirs('training', exist_ok=True)

with open('training/historical_trainer.py', 'w', encoding='utf-8') as f:
    f.write(historical_trainer_code)
print(" - training/historical_trainer.py ... REPARADO")

with open('training/hybrid_trainer.py', 'w', encoding='utf-8') as f:
    f.write(hybrid_trainer_code)
print(" - training/hybrid_trainer.py ... REPARADO")

with open('entrenar_completo.py', 'w', encoding='utf-8') as f:
    f.write(entrenar_completo_code)
print(" - entrenar_completo.py ... REPARADO")

print("\n✅ LISTO. AHORA EJECUTA: python entrenar_completo.py")
