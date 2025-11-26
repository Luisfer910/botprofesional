import MetaTrader5 as mt5
import pandas as pd
import time
import pickle
import logging
from datetime import datetime
from core.feature_engineer import FeatureEngineer

# Configuración directa (ya que config.py daba problemas)
CUENTA = 100464594
PASSWORD = "Fer101996-"
SERVIDOR = "XMGlobalSC-MT5 5"
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
VOLUME = 0.01

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("bot_trading.log"), logging.StreamHandler()]
)
logger = logging.getLogger()

def cargar_modelo():
    try:
        with open('models/modelo_hibrido.pkl', 'rb') as f:
            data = pickle.load(f)
            # El archivo guardado por el reparador tiene esta estructura:
            # {'modelo': self.modelo_live, 'scaler': self.scaler, 'historico': self.modelo_historico}
            return data['modelo'], data['scaler']
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return None, None

def conectar():
    if not mt5.initialize():
        logger.error("❌ Fallo al iniciar MT5")
        return False
    
    authorized = mt5.login(CUENTA, password=PASSWORD, server=SERVIDOR)
    if not authorized:
        logger.error(f"❌ Fallo al conectar a cuenta: {mt5.last_error()}")
        return False
    
    logger.info(f"✅ Conectado a: {CUENTA}")
    return True

def obtener_datos_live():
    # Necesitamos suficientes velas para generar los indicadores (RSI, MACD, etc.)
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Generar features
    fe = FeatureEngineer()
    df_features = fe.generar_todas_features(df)
    
    # Devolver solo la última fila (la vela actual/recién cerrada)
    return df_features.tail(1)

def ejecutar_bot():
    logger.info("🚀 INICIANDO BOT EN VIVO")
    
    if not conectar():
        return

    modelo, scaler = cargar_modelo()
    if modelo is None:
        return

    logger.info("✅ Modelo Híbrido cargado. Esperando velas...")

    while True:
        try:
            # Obtener datos actuales
            df_actual = obtener_datos_live()
            
            if df_actual is not None:
                # Preparar datos para el modelo
                excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
                cols_modelo = [c for c in df_actual.columns if c not in excluir]
                
                # Filtrar solo columnas numéricas que usa el modelo
                X_live = df_actual[cols_modelo].select_dtypes(include=['number'])
                
                # Escalar
                X_scaled = scaler.transform(X_live)
                
                # Predecir
                prediccion = modelo.predict(X_scaled)
                
                # Interpretación (depende de cómo entrenó LightGBM, usualmente devuelve array de probabilidades)
                # Si es multiclase: 0=Venta, 1=Neutral, 2=Compra
                import numpy as np
                clase = np.argmax(prediccion, axis=1)[0]
                
                precio_actual = mt5.symbol_info_tick(SYMBOL).ask
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                if clase == 2:
                    logger.info(f"⏰ {timestamp} | 🔮 Predicción: COMPRA 🟢 | Precio: {precio_actual}")
                    # Aquí iría la lógica de abrir orden (mt5.order_send)
                elif clase == 0:
                    logger.info(f"⏰ {timestamp} | 🔮 Predicción: VENTA 🔴 | Precio: {precio_actual}")
                else:
                    logger.info(f"⏰ {timestamp} | 🔮 Predicción: NEUTRAL ⚪")
            
            # Esperar a la siguiente vela (o 10 segundos para probar)
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por usuario")
            break
        except Exception as e:
            logger.error(f"❌ Error en bucle: {e}")
            time.sleep(5)

    mt5.shutdown()

if __name__ == "__main__":
    ejecutar_bot()
