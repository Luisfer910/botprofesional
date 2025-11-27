import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from collections import deque
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# ==============================================================================
# CONFIGURACIÓN PROFESIONAL
# ==============================================================================
CONFIG = {
    "login": 100464594,
    "password": "Fer101996-",
    "server": "XMGlobalSC-MT5 5",
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M5,
    "history_size": 10000,      # Más historia para mejor contexto
    "memory_size": 5000,
    "min_confidence": 0.60,     # Confianza base
    "max_drawdown_switch": 2    # Si pierde 2 seguidas, pasa a modo simulación
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BotProfesional_V10")

# ==============================================================================
# MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (CONTEXT AWARENESS)
# ==============================================================================
class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def calcular_datos(self, df):
        df = df.copy()
        
        # --- 1. INDICADORES BASE ---
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (Volatilidad)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = np.max(ranges, axis=1).rolling(14).mean()

        # --- 2. FEATURES DE CONTEXTO (LO QUE HACE AL BOT INTELIGENTE) ---
        
        # A. Trend Score (Puntuación de Tendencia)
        # Positivo = Alcista fuerte, Negativo = Bajista fuerte, 0 = Lateral
        df['trend_score'] = 0.0
        df.loc[df['close'] > df['ema_50'], 'trend_score'] += 1
        df.loc[df['ema_20'] > df['ema_50'], 'trend_score'] += 1
        df.loc[df['ema_50'] > df['ema_200'], 'trend_score'] += 1
        df.loc[df['close'] < df['ema_50'], 'trend_score'] -= 1
        df.loc[df['ema_20'] < df['ema_50'], 'trend_score'] -= 1
        df.loc[df['ema_50'] < df['ema_200'], 'trend_score'] -= 1
        
        # B. Distancia a la Media (Mean Reversion Potential)
        # Cuanto más lejos de la EMA 20, más probable es un rebote
        df['dist_ema20'] = (df['close'] - df['ema_20']) / df['atr']
        
        # C. Price Action: Soportes y Resistencias Dinámicos
        # Usamos ventanas rodantes para encontrar máximos/mínimos recientes
        df['rolling_high'] = df['high'].rolling(20).max()
        df['rolling_low'] = df['low'].rolling(20).min()
        
        # Distancia normalizada a S/R (0 = en el nivel, 1 = lejos)
        df['dist_resistance'] = (df['rolling_high'] - df['close']) / df['atr']
        df['dist_support'] = (df['close'] - df['rolling_low']) / df['atr']
        
        # D. Fuerza de la Vela (Candle Power)
        body = df['close'] - df['open']
        range_len = df['high'] - df['low']
        df['candle_power'] = body / range_len # 1 = Marubozu alcista, -1 = Marubozu bajista
        
        return df.dropna()

# ==============================================================================
# CEREBRO ARTIFICIAL (APRENDIZAJE PROFUNDO)
# ==============================================================================
class Brain:
    def __init__(self):
        self.model = None
        self.feature_engine = FeatureEngineering()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.is_trained = False
        
        # Variables de Estado
        self.racha_perdidas = 0
        self.modo_simulacion = False # Kill Switch
        self.historial = []
        self.ultima_prediccion = None
        self.precio_entrada = None
        
    def preparar_dataset(self, df, training=False):
        df_proc = self.feature_engine.calcular_datos(df)
        
        # Features seleccionadas (Contexto + Técnica)
        features = [
            'rsi', 'trend_score', 'dist_ema20', 'atr', 
            'dist_resistance', 'dist_support', 'candle_power'
        ]
        
        X = df_proc[features]
        
        if training:
            # Target Inteligente:
            # 2 = Subida significativa (> 1 ATR)
            # 0 = Bajada significativa (> 1 ATR)
            # 1 = Ruido / Lateral
            futuro = df_proc['close'].shift(-1)
            cambio = futuro - df_proc['close']
            umbral = df_proc['atr'] * 0.5 # Movimiento mínimo de medio ATR
            
            y = np.where(cambio > umbral, 2,
                np.where(cambio < -umbral, 0, 1))
            
            # Sample Weights: Dar más importancia a movimientos grandes
            weights = np.abs(cambio) / df_proc['atr']
            
            return X, y, weights
            
        return X, df_proc

    def entrenar_inicial(self, df):
        logger.info("🧠 Analizando historia del mercado (Contexto + Price Action)...")
        X, y, w = self.preparar_dataset(df, training=True)
        
        # Guardar en memoria
        self.memory.extend(df.to_dict('records'))
        
        # Entrenar LightGBM
        train_data = lgb.Dataset(X, label=y, weight=w)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=500)
        self.is_trained = True
        logger.info("✅ Cerebro entrenado. Listo para operar.")

    def reentrenar(self):
        """Aprendizaje Continuo: Se adapta con cada nueva vela"""
        if len(self.memory) < 500: return
        
        df = pd.DataFrame(list(self.memory))
        X, y, w = self.preparar_dataset(df, training=True)
        
        # Usar solo datos recientes para adaptación rápida
        X_recent = X.tail(1000)
        y_recent = y[-1000:]
        w_recent = w[-1000:]
        
        train_data = lgb.Dataset(X_recent, label=y_recent, weight=w_recent)
        
        # Actualizar modelo existente
        self.model = lgb.train(
            {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.05},
            train_data, num_boost_round=50, init_model=self.model, keep_training_booster=True
        )
        logger.info("🔄 Cerebro actualizado con nuevos patrones.")

    def predecir(self, df_live):
        X, df_proc = self.preparar_dataset(df_live, training=False)
        last_X = X.tail(1)
        
        # Predicción Pura (Sin filtros if/else)
        probs = self.model.predict(last_X)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        # Datos para log
        trend = df_proc['trend_score'].iloc[-1]
        
        self.ultima_prediccion = clase
        self.precio_entrada = df_live['close'].iloc[-1]
        
        return clase, confianza, trend

    def auditar_resultado(self, precio_cierre):
        if self.ultima_prediccion is None: return
        
        mov = precio_cierre - self.precio_entrada
        win = False
        
        if self.ultima_prediccion == 2: # Buy
            win = mov > 0
        elif self.ultima_prediccion == 0: # Sell
            win = mov < 0
        else:
            return # Esperar no cuenta
            
        # GESTIÓN DE ESTADO (SIMULACIÓN VS REAL)
        if win:
            res = "✅ WIN"
            self.racha_perdidas = 0
            if self.modo_simulacion:
                logger.info("🔰 El bot ha recuperado el ritmo. Volviendo a MODO REAL.")
                self.modo_simulacion = False
        else:
            res = "❌ LOSS"
            self.racha_perdidas += 1
            if self.racha_perdidas >= CONFIG['max_drawdown_switch'] and not self.modo_simulacion:
                logger.warning(f"⚠️ {self.racha_perdidas} Pérdidas seguidas. Activando MODO SIMULACIÓN para proteger capital.")
                self.modo_simulacion = True
                
        logger.info(f"📊 {res} | Mov: {mov:.5f} | Estado: {'SIMULADO' if self.modo_simulacion else 'REAL'}")
        self.historial.append({'win': win, 'mov': mov, 'sim': self.modo_simulacion})

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT PROFESIONAL V10 - CONTEXT AWARENESS & CAPITAL PROTECTION")
    print("   Sin filtros hardcodeados. Aprende del contexto (Tendencia/Rango).")
    print("="*80)
    
    if not mt5.initialize():
        logger.error("Error MT5 Init")
        return
        
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    logger.info(f"Conectado a {CONFIG['symbol']}")
    
    bot = Brain()
    
    # Carga inicial de datos
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    bot.entrenar_inicial(df)
    
    last_time = df['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre de vela... (Última: {last_time})")
    
    try:
        while True:
            # Datos en tiempo real
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            curr_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            print(f"\r💹 {CONFIG['symbol']} | {tick.bid:.5f} / {tick.ask:.5f} | {curr_time}", end="")
            
            # Lógica de Cierre de Vela
            if curr_time > last_time:
                print("\n")
                logger.info(f"🕯️ CIERRE: {last_time}")
                
                # 1. Obtener vela cerrada
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                # 2. Auditar resultado anterior
                bot.auditar_resultado(df_closed['close'].iloc[0])
                
                # 3. Aprender de lo sucedido
                bot.memory.append(df_closed.iloc[0].to_dict())
                bot.reentrenar()
                
                # 4. Predecir siguiente movimiento
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 500)
                df_ctx = pd.DataFrame(rates_ctx)
                df_ctx['time'] = pd.to_datetime(df_ctx['time'], unit='s')
                
                clase, conf, trend_score = bot.predecir(df_ctx)
                
                # Interpretar decisión
                accion = "⚪ ESPERAR"
                if conf > CONFIG["min_confidence"]:
                    if clase == 2: accion = "🟢 COMPRAR"
                    elif clase == 0: accion = "🔴 VENDER"
                
                # Etiqueta de estado
                estado_str = "[SIMULACIÓN]" if bot.modo_simulacion else "[REAL]"
                trend_str = "ALCISTA" if trend_score > 0 else "BAJISTA" if trend_score < 0 else "RANGO"
                
                logger.info(f"🔮 {estado_str} Predicción: {accion} | Conf: {conf:.1%} | Contexto: {trend_str} ({trend_score})")
                
                last_time = curr_time
                
                # Stats
                if len(bot.historial) > 0:
                    wins = sum(1 for h in bot.historial if h['win'])
                    total = len(bot.historial)
                    logger.info(f"📈 Winrate Global: {wins}/{total} ({(wins/total)*100:.1f}%)")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        mt5.shutdown()
        print("\n🛑 Bot detenido.")

if __name__ == "__main__":
    main()
