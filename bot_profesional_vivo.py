import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from core.feature_engineer import FeatureEngineer
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from collections import deque

# ==============================================================================
# CONFIGURACIÓN PROFESIONAL
# ==============================================================================
CONFIG = {
    "login": 100464594,
    "password": "Fer101996-",
    "server": "XMGlobalSC-MT5 5",
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M5,
    "magic_number": 123456,
    "lot_size": 0.01,
    "history_size": 2000,     # Velas iniciales para el cerebro base
    "memory_size": 500,       # Cuántas velas recientes recuerda para re-entrenar rápido
    "min_confidence": 0.60    # Solo opera si la confianza es > 60%
}

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bot_profesional.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotProfesional")

# ==============================================================================
# CLASE: CEREBRO DE APRENDIZAJE CONTINUO
# ==============================================================================
class CerebroContinuo:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.memory = deque(maxlen=CONFIG["memory_size"]) # Memoria de corto plazo
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def inicializar_conocimiento(self, df_inicial):
        """Entrenamiento inicial con datos históricos"""
        logger.info("🧠 Inicializando cerebro con datos históricos...")
        
        # Generar features
        df_features = self.feature_engineer.generar_todas_features(df_inicial)
        df_features = self.crear_target(df_features)
        df_features.dropna(inplace=True)
        
        # Guardar en memoria
        self.memory.extend(df_features.to_dict('records'))
        
        # Preparar datos
        X, y = self.preparar_datos(df_features)
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo base (LightGBM es rápido y eficiente para esto)
        train_data = lgb.Dataset(X_scaled, label=y)
        params = {
            'objective': 'multiclass',
            'num_class': 3, # 0: Venta, 1: Neutral, 2: Compra
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=100)
        self.is_trained = True
        logger.info("🧠 Cerebro inicializado y listo.")

    def aprender_nueva_vela(self, nueva_vela_df):
        """
        APRENDIZAJE INCREMENTAL:
        Toma la vela que acaba de cerrar, verifica si la predicción anterior fue correcta,
        y ajusta el modelo.
        """
        if not self.is_trained: return

        # Generar features para esta nueva vela (unida al contexto anterior)
        # Necesitamos contexto para calcular indicadores, así que tomamos la memoria
        df_contexto = pd.DataFrame(list(self.memory))
        df_full = pd.concat([df_contexto, nueva_vela_df]).drop_duplicates(subset=['time'], keep='last')
        
        # Recalcular features
        df_features = self.feature_engineer.generar_todas_features(df_full)
        df_features = self.crear_target(df_features) # El target de la última vela será NaN (porque no conocemos el futuro)
        
        # Pero SÍ conocemos el target de la PENÚLTIMA vela ahora (porque ya cerró la última)
        # Tomamos los datos recientes válidos para re-entrenar
        datos_validos = df_features.dropna().tail(50) # Re-entrenamos con lo más reciente
        
        if len(datos_validos) > 10:
            X, y = self.preparar_datos(datos_validos)
            X_scaled = self.scaler.transform(X)
            
            # Actualizar modelo (Fine-tuning)
            train_data = lgb.Dataset(X_scaled, label=y)
            
            # Usamos el modelo anterior como base (init_model)
            self.model = lgb.train(
                {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.01}, 
                train_data, 
                num_boost_round=10, # Pocas iteraciones para ajuste rápido
                init_model=self.model,
                keep_training_booster=True
            )
            logger.info("🧠 Cerebro actualizado con la última vela.")
            
            # Actualizar memoria
            self.memory.append(nueva_vela_df.iloc[0].to_dict())

    def predecir(self, df_actual):
        """Analiza el mercado actual y da una predicción"""
        df_features = self.feature_engineer.generar_todas_features(df_actual)
        last_row = df_features.tail(1)
        
        cols_modelo = self.obtener_cols_modelo(last_row)
        X = last_row[cols_modelo].select_dtypes(include=[np.number])
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)[0]
        
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        return clase, confianza

    def crear_target(self, df):
        # Target: 2 si sube, 0 si baja, 1 si lateral
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 2, 
                       np.where(df['close'].shift(-1) < df['close'], 0, 1))
        return df

    def preparar_datos(self, df):
        cols = self.obtener_cols_modelo(df)
        return df[cols].select_dtypes(include=[np.number]), df['target']

    def obtener_cols_modelo(self, df):
        excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
        return [c for c in df.columns if c not in excluir]

# ==============================================================================
# MOTOR PRINCIPAL DEL BOT
# ==============================================================================
def main():
    print("="*60)
    print("🚀 BOT PROFESIONAL DE APRENDIZAJE CONTINUO V4.0")
    print("   Analiza ticks -> Cierra Vela -> Aprende -> Opera")
    print("="*60)

    # 1. Conexión
    if not mt5.initialize():
        logger.error("Error MT5 init")
        return
    
    if not mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"]):
        logger.error("Error Login")
        return
    
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")

    # 2. Inicialización del Cerebro
    cerebro = CerebroContinuo()
    
    # Descargar historia para el primer entrenamiento
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df_inicial = pd.DataFrame(rates)
    df_inicial['time'] = pd.to_datetime(df_inicial['time'], unit='s')
    
    cerebro.inicializar_conocimiento(df_inicial)

    # 3. Bucle Principal (Tick a Tick)
    last_candle_time = df_inicial['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre de vela actual ({last_candle_time})...")
    
    try:
        while True:
            # Obtener datos de la vela actual en formación
            rates_current = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            if rates_current is None: continue
            
            current_time = pd.to_datetime(rates_current[0]['time'], unit='s')
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            
            # Imprimir estado del tick (Visualización "Tick a Tick")
            print(f"\rTick: {tick.bid} / {tick.ask} | Vela: {current_time}", end="")

            # DETECTAR CIERRE DE VELA
            # Si el tiempo de la vela actual es MAYOR que la última registrada, significa que se abrió una nueva
            if current_time > last_candle_time:
                print("\n")
                logger.info(f"🕯️ CIERRE DE VELA DETECTADO: {last_candle_time}")
                
                # 1. Obtener la vela que ACABA de cerrarse completa (índice 1, no 0)
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                # 2. APRENDER: El bot estudia qué pasó en esa vela
                cerebro.aprender_nueva_vela(df_closed)
                
                # 3. PREDECIR: Analiza el contexto actual para la NUEVA vela
                # Necesitamos contexto histórico + la nueva vela abierta
                rates_context = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 100)
                df_context = pd.DataFrame(rates_context)
                df_context['time'] = pd.to_datetime(df_context['time'], unit='s')
                
                clase, confianza = cerebro.predecir(df_context)
                
                # 4. DECIDIR
                accion = "ESPERAR"
                if confianza > CONFIG["min_confidence"]:
                    if clase == 2: accion = "COMPRAR 🟢"
                    elif clase == 0: accion = "VENDER 🔴"
                
                logger.info(f"🎯 ANÁLISIS FINALIZADO: Predicción={accion} | Confianza={confianza:.2f}")
                
                # Actualizar referencia de tiempo
                last_candle_time = current_time
                
            time.sleep(0.1) # Pequeña pausa para no saturar CPU

    except KeyboardInterrupt:
        logger.info("🛑 Bot detenido.")
        mt5.shutdown()

if __name__ == "__main__":
    main()
