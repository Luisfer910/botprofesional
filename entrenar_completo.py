import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import logging
import sys

# Importaciones de tus módulos (Asegúrate de que existan en las carpetas)
from config import CUENTA, PASSWORD, SERVIDOR, SYMBOL, TIMEFRAME
from core.data_handler import DataHandler
from core.feature_engineer import FeatureEngineer
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer # El que acabamos de arreglar

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("  🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO (CORREGIDO)")
    print("="*70 + "\n")
    
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -----------------------------------------------------------
    # PASO 1: CONEXIÓN A MT5
    # -----------------------------------------------------------
    print("\n" + "─"*70)
    print("  PASO 1: CONEXIÓN A MT5")
    print("─"*70 + "\n")
    
    if not mt5.initialize():
        logger.error("❌ Fallo al iniciar MT5")
        return

    authorized = mt5.login(CUENTA, password=PASSWORD, server=SERVIDOR)
    if authorized:
        logger.info("✅ Conectado exitosamente a XM")
        account_info = mt5.account_info()
        logger.info(f"💰 Balance: ${account_info.balance}")
    else:
        logger.error(f"❌ Fallo login: {mt5.last_error()}")
        return

    # -----------------------------------------------------------
    # PASO 2: DESCARGA DE DATOS
    # -----------------------------------------------------------
    print("\n" + "─"*70)
    print("  PASO 2: DESCARGA DE DATOS HISTÓRICOS")
    print("─"*70 + "\n")

    data_handler = DataHandler(symbol=SYMBOL, timeframe=TIMEFRAME)
    # Descargamos suficientes velas
    df_raw = data_handler.descargar_historico(n_velas=20000)
    
    if df_raw is None or df_raw.empty:
        logger.error("❌ No se descargaron datos.")
        return

    # -----------------------------------------------------------
    # PASO 3: FEATURE ENGINEERING
    # -----------------------------------------------------------
    print("\n" + "─"*70)
    print("  PASO 3: GENERACIÓN DE FEATURES")
    print("─"*70 + "\n")

    fe = FeatureEngineer()
    df = fe.generar_features(df_raw)
    
    # Definir Target (Ejemplo: Próxima vela cierra más arriba = 2, igual = 1, abajo = 0)
    # Ajusta esta lógica según tu estrategia real en FeatureEngineer
    if 'target' not in df.columns:
        logger.info("🎯 Creando variable target simple (Close vs Close previo)...")
        df['future_close'] = df['close'].shift(-1)
        df.dropna(inplace=True)
        
        def get_target(row):
            diff = row['future_close'] - row['close']
            if diff > 0.00010: return 2 # Compra
            if diff < -0.00010: return 0 # Venta
            return 1 # Neutral
            
        df['target'] = df.apply(get_target, axis=1)
        
    TARGET_COL = 'target'
    # Excluir columnas que no son features (fecha, target, etc)
    cols_to_exclude = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'future_close']
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    logger.info(f"✅ Features listas: {len(feature_cols)} columnas")
    logger.info(f"📊 Distribución target: \n{df[TARGET_COL].value_counts().sort_index()}")

    # -----------------------------------------------------------
    # PASO 4: MODELO HISTÓRICO (Random Forest)
    # -----------------------------------------------------------
    print("\n" + "─"*70)
    print("  PASO 4: ENTRENAMIENTO MODELO HISTÓRICO")
    print("─"*70 + "\n")

    trainer_historico = HistoricalTrainer(
        target_col=TARGET_COL,
        feature_cols=feature_cols
    )
    
    # Entrenar y guardar histórico
    modelo_historico = trainer_historico.entrenar(df)

    # -----------------------------------------------------------
    # PASO 5: MODELO HÍBRIDO (LightGBM)
    # -----------------------------------------------------------
    print("\n" + "─"*70)
    print("  PASO 5: ENTRENAMIENTO MODELO HÍBRIDO")
    print("─"*70 + "\n")
    
    # CORRECCIÓN AQUI: Instanciamos SIN pasar el modelo histórico todavía
    trainer_hibrido = HybridTrainer(
        target_col=TARGET_COL,
        feature_cols=feature_cols
    )
    
    # CORRECCIÓN AQUI: Pasamos el modelo histórico y el DF al método entrenar
    modelo_hibrido = trainer_hibrido.entrenar(
        df=df,
        modelo_historico=modelo_historico
    )
    
    print("\n" + "="*70)
    print("✅✅ ENTRENAMIENTO COMPLETO FINALIZADO CON ÉXITO")
    print("="*70 + "\n")

    mt5.shutdown()

if __name__ == "__main__":
    main()