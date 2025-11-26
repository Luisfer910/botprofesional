
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
