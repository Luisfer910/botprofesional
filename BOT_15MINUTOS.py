import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from core.feature_engineer import FeatureEngineer
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from collections import deque

# ==============================================================================
# CONFIGURACIÓN M15 (MÁS ESTABLE)
# ==============================================================================
CONFIG = {
    "login": 100464594,
    "password": "Fer101996-",
    "server": "XMGlobalSC-MT5 5",
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M15,  # CAMBIO A 15 MINUTOS
    "history_size": 3000,             # Más historia para mejor contexto
    "memory_size": 1000,              # Memoria más larga
    "min_confidence": 0.60            # Umbral de confianza
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler("bot_m15.log"), logging.StreamHandler()]
)
logger = logging.getLogger("BotM15")

class CerebroM15:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.ultima_prediccion = None # Para auditar si ganó o perdió
        self.precio_entrada_anterior = None
        
    def inicializar(self, df_inicial):
        logger.info("🧠 Entrenando cerebro base con 3000 velas de M15...")
        df_features = self.feature_engineer.generar_todas_features(df_inicial)
        df_features = self.crear_target(df_features)
        df_features.dropna(inplace=True)
        
        self.memory.extend(df_features.to_dict('records'))
        X, y = self.preparar_datos(df_features)
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y)
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'learning_rate': 0.03, 'num_leaves': 31
        }
        self.model = lgb.train(params, train_data, num_boost_round=150)
        self.is_trained = True
        logger.info("✅ Cerebro listo. Esperando cierre de vela...")

    def auditar_resultado(self, precio_cierre_actual):
        """Verifica si la predicción de hace 15 minutos fue correcta"""
        if self.ultima_prediccion is None or self.precio_entrada_anterior is None:
            return

        resultado = "NEUTRAL"
        pnl = 0
        
        if self.ultima_prediccion == 2: # COMPRA
            pnl = precio_cierre_actual - self.precio_entrada_anterior
            if pnl > 0: resultado = "✅ GANADA (Buy)"
            else: resultado = "❌ PERDIDA (Buy)"
            
        elif self.ultima_prediccion == 0: # VENTA
            pnl = self.precio_entrada_anterior - precio_cierre_actual
            if pnl > 0: resultado = "✅ GANADA (Sell)"
            else: resultado = "❌ PERDIDA (Sell)"
            
        if self.ultima_prediccion != 1: # Si no fue neutral
            logger.info(f"📊 AUDITORÍA: {resultado} | PnL: {pnl:.5f}")

    def aprender(self, nueva_vela_df):
        if not self.is_trained: return
        
        # Unir memoria con nueva vela para contexto
        df_contexto = pd.DataFrame(list(self.memory))
        df_full = pd.concat([df_contexto, nueva_vela_df]).drop_duplicates(subset=['time'], keep='last')
        
        # Generar features y target REAL (ahora sabemos qué pasó)
        df_features = self.feature_engineer.generar_todas_features(df_full)
        df_features = self.crear_target(df_features)
        
        # Tomar datos recientes para re-entrenar
        datos_validos = df_features.dropna().tail(100)
        
        if len(datos_validos) > 10:
            X, y = self.preparar_datos(datos_validos)
            X_scaled = self.scaler.transform(X)
            
            train_data = lgb.Dataset(X_scaled, label=y)
            self.model = lgb.train(
                {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.02}, 
                train_data, num_boost_round=20, init_model=self.model, keep_training_booster=True
            )
            logger.info("🧠 Conocimiento actualizado con nueva vela.")
            self.memory.append(nueva_vela_df.iloc[0].to_dict())

    def predecir(self, df_contexto):
        df_features = self.feature_engineer.generar_todas_features(df_contexto)
        last_row = df_features.tail(1)
        cols = self.obtener_cols_modelo(last_row)
        X = last_row[cols].select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict(X_scaled)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        # Guardar para auditar después
        self.ultima_prediccion = clase
        self.precio_entrada_anterior = df_contexto['close'].iloc[-1]
        
        return clase, confianza

    def crear_target(self, df):
        # Target simple: ¿La siguiente vela cierra más arriba o más abajo?
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 2, 
                       np.where(df['close'].shift(-1) < df['close'], 0, 1))
        return df

    def preparar_datos(self, df):
        cols = self.obtener_cols_modelo(df)
        return df[cols].select_dtypes(include=[np.number]), df['target']

    def obtener_cols_modelo(self, df):
        excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
        return [c for c in df.columns if c not in excluir]

def main():
    print("="*60)
    print("🚀 BOT M15 PROFESIONAL - AUDITORÍA EN VIVO")
    print("="*60)

    if not mt5.initialize(): return
    if not mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"]): return
    
    cerebro = CerebroM15()
    
    # Carga inicial
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df_inicial = pd.DataFrame(rates)
    df_inicial['time'] = pd.to_datetime(df_inicial['time'], unit='s')
    cerebro.inicializar(df_inicial)
    
    last_candle_time = df_inicial['time'].iloc[-1]
    logger.info(f"⏳ Última vela cerrada: {last_candle_time}")

    try:
        while True:
            # Monitoreo
            rates_current = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            current_time = pd.to_datetime(rates_current[0]['time'], unit='s')
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            
            print(f"\rTick: {tick.bid} | Vela Actual: {current_time}", end="")

            if current_time > last_candle_time:
                print("\n")
                logger.info(f"🕯️ CIERRE DE VELA M15: {last_candle_time}")
                
                # 1. Obtener vela cerrada
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                precio_cierre = df_closed['close'].iloc[0]

                # 2. AUDITAR: ¿Ganamos la anterior?
                cerebro.auditar_resultado(precio_cierre)

                # 3. APRENDER
                cerebro.aprender(df_closed)
                
                # 4. PREDECIR SIGUIENTE
                rates_context = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 150)
                df_context = pd.DataFrame(rates_context)
                df_context['time'] = pd.to_datetime(df_context['time'], unit='s')
                
                clase, confianza = cerebro.predecir(df_context)
                
                accion = "⚪ ESPERAR"
                if confianza > CONFIG["min_confidence"]:
                    if clase == 2: accion = "🟢 COMPRAR"
                    elif clase == 0: accion = "🔴 VENDER"
                
                logger.info(f"🔮 NUEVA PREDICCIÓN: {accion} (Confianza: {confianza:.1%})")
                
                last_candle_time = current_time
                
            time.sleep(0.5)

    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()
