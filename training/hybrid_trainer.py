
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
