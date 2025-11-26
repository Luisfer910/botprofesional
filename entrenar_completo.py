"""
Script de entrenamiento completo del bot
Versión: 3.0
Fecha: 2024
"""

import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos del proyecto
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer

def main():
    """
    Función principal de entrenamiento
    """
    print("\n" + "="*70)
    print("  🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print("="*70 + "\n")
    
    inicio = datetime.now()
    logger.info(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ──────────────────────────────────────────────────────────────
        # PASO 1: CONEXIÓN A MT5
        # ──────────────────────────────────────────────────────────────
        print("\n" + "─"*70)
        print("  PASO 1: CONEXIÓN A MT5")
        print("─"*70 + "\n")
        
        if not mt5.initialize():
            logger.error("❌ Error al inicializar MT5")
            return
        
        # Credenciales XM
        XM_LOGIN = 100464594
        XM_PASSWORD = "Fer101996-"
        XM_SERVER = "XMGlobalSC-MT5 5"
        
        login_ok = mt5.login(XM_LOGIN, password=XM_PASSWORD, server=XM_SERVER)
        if not login_ok:
            logger.error(f"❌ Error de login: {mt5.last_error()}")
            mt5.shutdown()
            return
        
        logger.info("✅ Conectado exitosamente a XM")
        
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"💰 Balance: ${account_info.balance:,.2f}")
            logger.info(f"📊 Equity: ${account_info.equity:,.2f}")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 2: DESCARGA DE DATOS HISTÓRICOS
        # ──────────────────────────────────────────────────────────────
        print("\n" + "─"*70)
        print("  PASO 2: DESCARGA DE DATOS HISTÓRICOS")
        print("─"*70 + "\n")
        
        ACTIVO = "EURUSD"
        TIMEFRAME = mt5.TIMEFRAME_M5
        CANT_VELAS = 20000
        
        mt5.symbol_select(ACTIVO, True)
        
        logger.info(f"📥 Descargando {CANT_VELAS:,} velas de {ACTIVO}...")
        rates = mt5.copy_rates_from_pos(ACTIVO, TIMEFRAME, 0, CANT_VELAS)
        
        if rates is None or len(rates) == 0:
            logger.error(f"❌ Error al descargar datos: {mt5.last_error()}")
            mt5.shutdown()
            return
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"✅ {len(df):,} velas descargadas")
        logger.info(f"📅 Desde: {df['time'].iloc[0]}")
        logger.info(f"📅 Hasta: {df['time'].iloc[-1]}")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 3: GENERACIÓN DE FEATURES
        # ──────────────────────────────────────────────────────────────
        print("\n" + "─"*70)
        print("  PASO 3: GENERACIÓN DE FEATURES")
        print("─"*70 + "\n")
        
        feature_engineer = FeatureEngineer()
        
        logger.info("🔧 Generando features técnicas...")
        # ✅ CAMBIO CRÍTICO: generar_todas_features en lugar de generar_features
        df_features = feature_engineer.generar_todas_features(df)
        
        if df_features is None or df_features.empty:
            logger.error("❌ Error generando features")
            return
        
        logger.info(f"✅ Features generadas: {df_features.shape[1]} columnas")
        logger.info(f"📊 Datos disponibles: {df_features.shape[0]} filas")
        
        # Crear target
        logger.info("🎯 Creando variable target...")
        HORIZON = 3
        df_features['target'] = np.where(
            df_features['close'].shift(-HORIZON) > df_features['close'],
            2,  # Compra
            np.where(
                df_features['close'].shift(-HORIZON) < df_features['close'],
                0,  # Venta
                1   # Neutral
            )
        )
        
        # Eliminar filas con NaN
        df_features = df_features.dropna()
        
        logger.info(f"✅ Target creado")
        logger.info(f"📊 Distribución del target:")
        logger.info(f"   Venta (0): {(df_features['target']==0).sum()}")
        logger.info(f"   Neutral (1): {(df_features['target']==1).sum()}")
        logger.info(f"   Compra (2): {(df_features['target']==2).sum()}")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 4: ENTRENAMIENTO MODELO HISTÓRICO
        # ──────────────────────────────────────────────────────────────
        print("\n" + "─"*70)
        print("  PASO 4: ENTRENAMIENTO MODELO HISTÓRICO")
        print("─"*70 + "\n")
        
        logger.info("🤖 Entrenando modelo histórico...")
        logger.info("   (Esto puede tomar 2-5 minutos)")
        
        trainer_historico = HistoricalTrainer(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            random_state=42
        )
        
        # Entrenar modelo
        modelo_historico = trainer_historico.entrenar(df_features)
        
        if modelo_historico is None:
            logger.error("❌ Error entrenando modelo histórico")
            return
        
        # Guardar modelo
        trainer_historico.guardar_modelo('models/modelo_historico.pkl')
        
        # ──────────────────────────────────────────────────────────────
        # PASO 5: ENTRENAMIENTO MODELO HÍBRIDO (OPCIONAL)
        # ──────────────────────────────────────────────────────────────
        print("\n" + "─"*70)
        print("  PASO 5: ENTRENAMIENTO MODELO HÍBRIDO")
        print("─"*70 + "\n")
        
        logger.info("🔄 Entrenando modelo híbrido...")
        
        # Cargar el modelo histórico recién entrenado
        trainer_hibrido = HybridTrainer(
            modelo_historico=modelo_historico,
            peso_historico=0.7,
            n_estimators=200
        )
        
        # Usar últimas 1000 velas como datos de "tiempo real"
        modelo_hibrido = trainer_hibrido.entrenar(
            df_historico=df_features,
            df_tiempo_real=df_features.tail(1000)
        )
        
        if modelo_hibrido:
            trainer_hibrido.guardar_modelo('models/modelo_hibrido.pkl')
            logger.info("✅ Modelo híbrido entrenado y guardado")
        else:
            logger.warning("⚠️ No se pudo entrenar el modelo híbrido")
        
        # ──────────────────────────────────────────────────────────────
        # FINALIZACIÓN
        # ──────────────────────────────────────────────────────────────
        mt5.shutdown()
        
        fin = datetime.now()
        duracion = (fin - inicio).total_seconds()
        
        print("\n" + "="*70)
        print("  ✅ ENTRENAMIENTO COMPLETADO")
        print("="*70)
        logger.info(f"⏱️  Tiempo total: {duracion:.1f} segundos")
        logger.info(f"📁 Modelos guardados en: ./models/")
        logger.info(f"\n🚀 SIGUIENTE PASO:")
        logger.info(f"   Ejecuta: python main.py")
        logger.info(f"   Para iniciar el bot en modo producción")
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso: {e}")
        import traceback
        traceback.print_exc()
        mt5.shutdown()

if __name__ == "__main__":
    main()
