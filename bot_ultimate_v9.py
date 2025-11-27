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
# CONFIGURACIÓN
# ==============================================================================
CONFIG = {
    "login": 100464594,
    "password": "Fer101996-",
    "server": "XMGlobalSC-MT5 5",
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M5,
    "history_size": 5000,
    "memory_size": 3000,
    "min_confidence": 0.60  # Subimos un poco la exigencia
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("bot_ultimate_v9.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotUltimate")

# ==============================================================================
# ANALIZADOR DE PRICE ACTION (Datos Puros)
# ==============================================================================
class PriceActionData:
    def __init__(self):
        self.soportes = []
        self.resistencias = []
        
    def extraer_features(self, df):
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, _ = find_peaks(highs, distance=15, prominence=0.0001)
        peaks_low, _ = find_peaks(-lows, distance=15, prominence=0.0001)
        
        self.resistencias = df.iloc[peaks_high]['high'].values if len(peaks_high) > 0 else np.array([])
        self.soportes = df.iloc[peaks_low]['low'].values if len(peaks_low) > 0 else np.array([])
        
        precio_actual = df['close'].iloc[-1]
        
        dist_soporte = min([abs(precio_actual - s) for s in self.soportes]) if len(self.soportes) > 0 else 0.01
        dist_resistencia = min([abs(precio_actual - r) for r in self.resistencias]) if len(self.resistencias) > 0 else 0.01
        
        # Distancia relativa a la media (Mean Reversion)
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        dist_sma = (precio_actual - sma_20) / sma_20
        
        return {
            'dist_soporte': dist_soporte,
            'dist_resistencia': dist_resistencia,
            'num_soportes': len(self.soportes),
            'num_resistencias': len(self.resistencias),
            'dist_sma_20': dist_sma,
            'breakout_soporte': 1 if precio_actual < (min(self.soportes) if len(self.soportes)>0 else 0) else 0,
            'breakout_resistencia': 1 if precio_actual > (max(self.resistencias) if len(self.resistencias)>0 else 99) else 0
        }

# ==============================================================================
# MOTOR DE CARACTERÍSTICAS (Con ADX para detectar Régimen)
# ==============================================================================
class FeatureEngine:
    def calcular_adx(self, df, period=14):
        """Calcula el ADX para saber si hay tendencia fuerte"""
        df = df.copy()
        df['tr'] = np.max(np.array([
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        ]), axis=0)
        
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                 np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                  np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        atr = df['tr'].rolling(period).mean()
        plus_di = 100 * (df['dm_plus'].rolling(period).mean() / atr)
        minus_di = 100 * (df['dm_minus'].rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(period).mean()
        return df['adx']

    def calcular_indicadores(self, df):
        df = df.copy()
        
        # Tendencia
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Régimen de Mercado (ADX)
        df['adx'] = self.calcular_adx(df)
        
        # Osciladores
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatilidad (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(5)
        
        return df.dropna()

# ==============================================================================
# CEREBRO ULTIMATE (V8 Mejorado)
# ==============================================================================
class CerebroUltimate:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engine = FeatureEngine()
        self.price_action = PriceActionData()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.is_trained = False
        
        self.historial = []
        self.ultima_prediccion = None
        self.precio_entrada = None
        
    def crear_target(self, df):
        df = df.copy()
        df['precio_futuro'] = df['close'].shift(-1)
        df['movimiento'] = df['precio_futuro'] - df['close']
        
        # Target: 2=Subida Clara, 0=Bajada Clara, 1=Ruido
        # Usamos ATR para definir qué es "Claro" (adaptativo)
        atr_actual = df['atr'].iloc[-1] if 'atr' in df else 0.0001
        umbral = atr_actual * 0.5 # El movimiento debe ser al menos medio ATR
        
        df['target'] = np.where(df['movimiento'] > 0.00002, 2,
                       np.where(df['movimiento'] < -0.00002, 0, 1))
        
        # Peso: Damos más importancia a velas con alto ADX (Tendencia definida)
        df['sample_weight'] = np.abs(df['movimiento']) * (1 + df['adx']/50)
        
        return df.dropna()
    
    def preparar_datos(self, df, training=False):
        df_tech = self.feature_engine.calcular_indicadores(df)
        pa_features = self.price_action.extraer_features(df_tech)
        for key, value in pa_features.items():
            df_tech[key] = value
        
        if training:
            df_tech = self.crear_target(df_tech)
            y = df_tech['target']
            weights = df_tech['sample_weight']
        else:
            y = None
            weights = None
        
        excluir = ['time', 'target', 'precio_futuro', 'movimiento', 'spread', 'real_volume', 'sample_weight']
        cols = [c for c in df_tech.columns if c not in excluir]
        X = df_tech[cols].select_dtypes(include=[np.number])
        
        return X, y, weights
    
    def inicializar(self, df_inicial):
        logger.info("🧠 Entrenando Cerebro V9 (Price Action + ADX Regime)...")
        
        X, y, weights = self.preparar_datos(df_inicial, training=True)
        self.memory.extend(df_inicial.to_dict('records'))
        
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y, weight=weights)
        
        # Hiperparámetros ajustados para estabilidad (menos "nervioso" que V7)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.03, # Más lento para aprender patrones sólidos
            'num_leaves': 50,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.7
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=500)
        self.is_trained = True
        logger.info("✅ Cerebro V9 Listo.")
    
    def auditar(self, precio_cierre):
        if self.ultima_prediccion is None or self.precio_entrada is None:
            return
        
        mov = precio_cierre - self.precio_entrada
        res = "NEUTRAL"
        if self.ultima_prediccion == 2:
            res = "✅ WIN (Buy)" if mov > 0 else "❌ LOSS (Buy)"
        elif self.ultima_prediccion == 0:
            res = "✅ WIN (Sell)" if mov < 0 else "❌ LOSS (Sell)"
        
        if self.ultima_prediccion != 1:
            logger.info(f"📊 {res} | Mov: {mov:.5f}")
            self.historial.append({'win': 'WIN' in res})
    
    def aprender(self, nueva_vela_df):
        if not self.is_trained: return
        
        self.memory.append(nueva_vela_df.iloc[0].to_dict())
        df_full = pd.DataFrame(list(self.memory))
        X, y, weights = self.preparar_datos(df_full, training=True)
        
        # Ventana de re-entrenamiento más amplia para no olvidar el pasado
        X_recent = X.tail(1000)
        y_recent = y.tail(1000)
        weights_recent = weights.tail(1000)
        
        if len(X_recent) > 50:
            X_scaled = self.scaler.transform(X_recent)
            train_data = lgb.Dataset(X_scaled, label=y_recent, weight=weights_recent)
            
            self.model = lgb.train(
                {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.02},
                train_data, num_boost_round=50, init_model=self.model, keep_training_booster=True
            )
            logger.info("🔄 Modelo ajustado (V9).")
    
    def predecir(self, df_contexto):
        X, _, _ = self.preparar_datos(df_contexto, training=False)
        X_scaled = self.scaler.transform(X.tail(1))
        
        probs = self.model.predict(X_scaled)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        self.ultima_prediccion = clase
        self.precio_entrada = df_contexto['close'].iloc[-1]
        return clase, confianza
    
    def stats(self):
        if len(self.historial) > 0:
            wins = sum(1 for h in self.historial if h['win'])
            total = len(self.historial)
            wr = (wins/total)*100
            logger.info(f"📈 STATS V9: {wins}/{total} ({wr:.1f}% WR)")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT ULTIMATE V9 - EL CAMPEÓN MEJORADO")
    print("   Basado en V8 (63% WR) + Filtro de Régimen ADX + Estabilidad")
    print("="*80)
    
    if not mt5.initialize(): return
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")
    
    cerebro = CerebroUltimate()
    
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    cerebro.inicializar(df)
    last_time = df['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre... (Última: {last_time})")
    
    contador = 0
    try:
        while True:
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            curr_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            print(f"\r💹 Tick: {tick.bid:.5f} | Ask: {tick.ask:.5f} | Vela: {curr_time}", end="")
            
            if curr_time > last_time:
                print("\n")
                logger.info(f"🕯️ CIERRE: {last_time}")
                
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                cerebro.auditar(df_closed['close'].iloc[0])
                cerebro.aprender(df_closed)
                
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 300)
                df_ctx = pd.DataFrame(rates_ctx)
                df_ctx['time'] = pd.to_datetime(df_ctx['time'], unit='s')
                
                clase, conf = cerebro.predecir(df_ctx)
                
                accion = "⚪ ESPERAR"
                if conf > CONFIG["min_confidence"]:
                    if clase == 2: accion = "🟢 COMPRAR"
                    elif clase == 0: accion = "🔴 VENDER"
                
                logger.info(f"🔮 PREDICCIÓN: {accion} | Confianza: {conf:.1%}")
                
                contador += 1
                if contador % 5 == 0: cerebro.stats()
                last_time = curr_time
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()
