import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from collections import deque
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
CONFIG = {
    "login": 100464594,
    "password": "Fer101996-",
    "server": "XMGlobalSC-MT5 5",
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M5,
    "history_size": 10000,
    "memory_size": 5000,
    "min_confidence": 0.60  # Dejamos que el modelo decida con esta confianza
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("BotV11_DeepContext")

# ==============================================================================
# MOTOR DE CONTEXTO (LOS "OJOS" DEL BOT)
# ==============================================================================
class ContextEngine:
    def __init__(self):
        pass

    def calcular_features_avanzadas(self, df):
        df = df.copy()
        
        # 1. DATOS BÁSICOS
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # ATR para normalizar (hacer que los datos sean comparables siempre)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift()), 
                                         abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()

        # -------------------------------------------------------------------------
        # AQUÍ ESTÁ LA MAGIA: FEATURES DE CONTEXTO (NO REGLAS)
        # -------------------------------------------------------------------------
        
        # A. PENDIENTE DE LA TENDENCIA (Trend Slope)
        # El modelo aprenderá: Pendiente positiva alta = Tendencia fuerte = Comprar en retrocesos
        # Normalizamos por ATR para que funcione igual en volatilidad alta o baja
        df['slope_20'] = (df['ema_20'] - df['ema_20'].shift(5)) / df['atr']
        df['slope_50'] = (df['ema_50'] - df['ema_50'].shift(5)) / df['atr']
        
        # B. POSICIÓN RELATIVA (Z-Score)
        # ¿Dónde está el precio respecto a la media? 
        # El modelo aprenderá: Si está muy lejos (valor alto), toca reversión. Si está cerca, continuación.
        df['z_score_20'] = (df['close'] - df['ema_20']) / df['atr']
        df['z_score_200'] = (df['close'] - df['ema_200']) / df['atr']
        
        # C. ESTRUCTURA DE MERCADO (Soportes/Resistencias Dinámicos)
        # Distancia a máximos/mínimos recientes
        rolling_max = df['high'].rolling(20).max()
        rolling_min = df['low'].rolling(20).min()
        df['dist_to_high'] = (rolling_max - df['close']) / df['atr']
        df['dist_to_low'] = (df['close'] - rolling_min) / df['atr']
        
        # D. FUERZA DE LA VELA (Momentum Inmediato)
        df['candle_strength'] = (df['close'] - df['open']) / df['atr']
        df['wick_upper'] = (df['high'] - np.maximum(df['close'], df['open'])) / df['atr']
        df['wick_lower'] = (np.minimum(df['close'], df['open']) - df['low']) / df['atr']
        
        # E. VOLATILIDAD RELATIVA
        # ¿El mercado está acelerando o frenando?
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(50).mean()

        return df.dropna()

# ==============================================================================
# CEREBRO DE APRENDIZAJE PROFUNDO (SIN FILTROS)
# ==============================================================================
class DeepBrain:
    def __init__(self):
        self.model = None
        self.context_engine = ContextEngine()
        self.scaler = StandardScaler()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.is_trained = False
        
        # Auditoría
        self.ultima_prediccion = None
        self.precio_entrada = None
        self.historial = []

    def preparar_datos(self, df, training=False):
        df_ctx = self.context_engine.calcular_features_avanzadas(df)
        
        # Seleccionamos SOLO las features numéricas calculadas
        features = [
            'slope_20', 'slope_50',          # Contexto de Tendencia
            'z_score_20', 'z_score_200',     # Contexto de Sobrecompra/Sobreventa
            'dist_to_high', 'dist_to_low',   # Contexto de Estructura
            'candle_strength', 'wick_upper', 'wick_lower', # Price Action puro
            'volatility_ratio'               # Contexto de Volatilidad
        ]
        
        X = df_ctx[features]
        
        if training:
            # TARGET DINÁMICO:
            # No usamos pips fijos. Usamos ATR.
            # Si el precio se mueve +0.5 ATR en la siguiente vela -> COMPRA (2)
            # Si el precio se mueve -0.5 ATR en la siguiente vela -> VENTA (0)
            futuro = df_ctx['close'].shift(-1)
            movimiento = futuro - df_ctx['close']
            umbral = df_ctx['atr'] * 0.3 # Umbral dinámico
            
            y = np.where(movimiento > umbral, 2,
                np.where(movimiento < -umbral, 0, 1))
            
            # Pesos: Damos más importancia a los movimientos fuertes
            w = np.abs(movimiento)
            
            return X, y, w
            
        return X, df_ctx

    def entrenar(self, df):
        logger.info("🧠 Entrenando Red Neuronal de Contexto...")
        X, y, w = self.preparar_datos(df, training=True)
        
        self.memory.extend(df.to_dict('records'))
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y, weight=w)
        
        # Parámetros agresivos para aprender patrones complejos
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.03, # Aprendizaje fino
            'num_leaves': 50,      # Complejidad media-alta
            'feature_fraction': 0.8
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=600)
        self.is_trained = True
        logger.info("✅ Modelo listo. Entiende pendiente, estructura y volatilidad.")

    def reentrenar_rapido(self):
        if len(self.memory) < 500: return
        
        df = pd.DataFrame(list(self.memory))
        X, y, w = self.preparar_datos(df, training=True)
        
        # Aprender de lo reciente (últimas 800 velas)
        X_recent = X.tail(800)
        y_recent = y[-800:]
        w_recent = w[-800:]
        
        X_scaled = self.scaler.transform(X_recent)
        
        self.model = lgb.train(
            {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.05},
            lgb.Dataset(X_scaled, label=y_recent, weight=w_recent),
            num_boost_round=50, init_model=self.model, keep_training_booster=True
        )
        logger.info("🔄 Cerebro ajustado al mercado actual.")

    def predecir(self, df_live):
        X, _ = self.preparar_datos(df_live, training=False)
        last_X = X.tail(1)
        X_scaled = self.scaler.transform(last_X)
        
        # PREDICCIÓN PURA - SIN IF/ELSE
        # El modelo recibe la pendiente y la distancia. Él decide.
        probs = self.model.predict(X_scaled)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        self.ultima_prediccion = clase
        self.precio_entrada = df_live['close'].iloc[-1]
        
        return clase, confianza

    def auditar(self, precio_cierre):
        if self.ultima_prediccion is None: return
        
        mov = precio_cierre - self.precio_entrada
        win = False
        
        if self.ultima_prediccion == 2: # Buy
            win = mov > 0
            res = "✅ WIN (Buy)" if win else "❌ LOSS (Buy)"
        elif self.ultima_prediccion == 0: # Sell
            win = mov < 0
            res = "✅ WIN (Sell)" if win else "❌ LOSS (Sell)"
        else:
            return
            
        logger.info(f"📊 {res} | Mov: {mov:.5f}")
        self.historial.append({'win': win, 'mov': mov})
        
        # Stats
        if len(self.historial) > 0:
            wins = sum(1 for h in self.historial if h['win'])
            total = len(self.historial)
            logger.info(f"📈 Winrate: {wins}/{total} ({(wins/total)*100:.1f}%)")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT V11 - DEEP CONTEXT LEARNING (CERO FILTROS)")
    print("   El modelo recibe Pendiente, Z-Scores y Estructura. Él decide.")
    print("="*80)
    
    if not mt5.initialize(): return
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    
    bot = DeepBrain()
    
    # Carga inicial
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    bot.entrenar(df)
    
    last_time = df['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre... (Última: {last_time})")
    
    try:
        while True:
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            curr_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            print(f"\r💹 {tick.bid:.5f}/{tick.ask:.5f} | {curr_time}", end="")
            
            if curr_time > last_time:
                print("\n")
                logger.info(f"🕯️ CIERRE: {last_time}")
                
                # 1. Cerrar vela y auditar
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                bot.auditar(df_closed['close'].iloc[0])
                
                # 2. Aprender (Reentrenamiento dinámico)
                bot.memory.append(df_closed.iloc[0].to_dict())
                bot.reentrenar_rapido()
                
                # 3. Predecir
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 500)
                df_ctx = pd.DataFrame(rates_ctx)
                df_ctx['time'] = pd.to_datetime(df_ctx['time'], unit='s')
                
                clase, conf = bot.predecir(df_ctx)
                
                accion = "⚪ ESPERAR"
                if conf > CONFIG["min_confidence"]:
                    if clase == 2: accion = "🟢 COMPRAR"
                    elif clase == 0: accion = "🔴 VENDER"
                
                logger.info(f"🔮 PREDICCIÓN: {accion} | Confianza: {conf:.1%}")
                
                last_time = curr_time
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()
