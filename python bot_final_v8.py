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
    "memory_size": 2500,
    "min_confidence": 0.55
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("bot_final_v8.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotFinal")

# ==============================================================================
# ANALIZADOR DE PRICE ACTION (Como Datos, NO como Filtros)
# ==============================================================================
class PriceActionData:
    def __init__(self):
        self.soportes = []
        self.resistencias = []
        
    def extraer_features(self, df):
        """Extrae features de Price Action sin aplicar reglas"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Detectar niveles clave
        peaks_high, _ = find_peaks(highs, distance=20, prominence=0.0001)
        peaks_low, _ = find_peaks(-lows, distance=20, prominence=0.0001)
        
        self.resistencias = df.iloc[peaks_high]['high'].values if len(peaks_high) > 0 else np.array([])
        self.soportes = df.iloc[peaks_low]['low'].values if len(peaks_low) > 0 else np.array([])
        
        precio_actual = df['close'].iloc[-1]
        
        # Calcular distancias (el modelo decidirá si son importantes)
        dist_soporte = min([abs(precio_actual - s) for s in self.soportes]) if len(self.soportes) > 0 else 0.01
        dist_resistencia = min([abs(precio_actual - r) for r in self.resistencias]) if len(self.resistencias) > 0 else 0.01
        
        # Medias móviles
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        # Estructura de vela
        last = df.iloc[-1]
        body = last['close'] - last['open']
        total_range = last['high'] - last['low']
        body_ratio = body / total_range if total_range > 0 else 0
        
        # Retornar features sin interpretación
        return {
            'dist_soporte': dist_soporte,
            'dist_resistencia': dist_resistencia,
            'num_soportes': len(self.soportes),
            'num_resistencias': len(self.resistencias),
            'precio_vs_sma20': (precio_actual - sma_20) / precio_actual,
            'precio_vs_sma50': (precio_actual - sma_50) / precio_actual,
            'sma20_vs_sma50': (sma_20 - sma_50) / sma_20 if sma_20 > 0 else 0,
            'body_ratio': body_ratio,
            'upper_shadow': last['high'] - max(last['close'], last['open']),
            'lower_shadow': min(last['close'], last['open']) - last['low']
        }

# ==============================================================================
# MOTOR DE INDICADORES TÉCNICOS
# ==============================================================================
class FeatureEngine:
    def calcular_indicadores(self, df):
        df = df.copy()
        
        # Medias móviles
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(4)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Patrones de velas
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volumen
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        return df.dropna()

# ==============================================================================
# CEREBRO INTELIGENTE (Sin Filtros)
# ==============================================================================
class CerebroFinal:
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
        
        # Target simple sin umbrales artificiales
        df['target'] = np.where(df['movimiento'] > 0.00001, 2,
                       np.where(df['movimiento'] < -0.00001, 0, 1))
        
        # Peso basado en magnitud del movimiento
        df['sample_weight'] = np.abs(df['movimiento']) * 10000
        
        return df.dropna()
    
    def preparar_datos(self, df, training=False):
        # Indicadores técnicos
        df_tech = self.feature_engine.calcular_indicadores(df)
        
        # Price Action (como datos adicionales)
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
        logger.info("🧠 Entrenamiento inicial (Price Action como datos, NO como filtros)...")
        
        X, y, weights = self.preparar_datos(df_inicial, training=True)
        self.memory.extend(df_inicial.to_dict('records'))
        
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y, weight=weights)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 60,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=400)
        self.is_trained = True
        logger.info("✅ Cerebro inicializado.")
    
    def auditar(self, precio_cierre):
        if self.ultima_prediccion is None or self.precio_entrada is None:
            return
        
        mov = precio_cierre - self.precio_entrada
        res = "NEUTRAL"
        
        if self.ultima_prediccion == 2:
            res = "✅ GANADA (Buy)" if mov > 0 else "❌ PERDIDA (Buy)"
        elif self.ultima_prediccion == 0:
            res = "✅ GANADA (Sell)" if mov < 0 else "❌ PERDIDA (Sell)"
        
        if self.ultima_prediccion != 1:
            logger.info(f"📊 {res} | Movimiento: {mov:.5f}")
            self.historial.append({
                'pred': self.ultima_prediccion,
                'mov': mov,
                'win': 'GANADA' in res
            })
    
    def aprender(self, nueva_vela_df):
        if not self.is_trained:
            return
        
        self.memory.append(nueva_vela_df.iloc[0].to_dict())
        
        df_full = pd.DataFrame(list(self.memory))
        X, y, weights = self.preparar_datos(df_full, training=True)
        
        X_recent = X.tail(600)
        y_recent = y.tail(600)
        weights_recent = weights.tail(600)
        
        if len(X_recent) > 50:
            X_scaled = self.scaler.transform(X_recent)
            train_data = lgb.Dataset(X_scaled, label=y_recent, weight=weights_recent)
            
            self.model = lgb.train(
                {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': 0.03},
                train_data, num_boost_round=60, init_model=self.model, keep_training_booster=True
            )
            logger.info("🔄 Modelo actualizado con nueva información.")
    
    def predecir(self, df_contexto):
        X, _, _ = self.preparar_datos(df_contexto, training=False)
        
        last_X = X.tail(1)
        X_scaled = self.scaler.transform(last_X)
        
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
            logger.info(f"📈 ESTADÍSTICAS: {wins}/{total} operaciones ganadas ({wr:.1f}% winrate)")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT FINAL V8.0 - APRENDIZAJE PURO + PRICE ACTION")
    print("   Price Action = Datos (NO Filtros) | Aprendizaje Real")
    print("="*80)
    
    if not mt5.initialize():
        return
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")
    
    cerebro = CerebroFinal()
    
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    cerebro.inicializar(df)
    
    last_time = df['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre de vela... (Última: {last_time})")
    
    contador = 0
    
    try:
        while True:
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            curr_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            print(f"\r💹 Tick: {tick.bid:.5f} / {tick.ask:.5f} | Vela: {curr_time}", end="")
            
            if curr_time > last_time:
                print("\n")
                logger.info(f"🕯️ CIERRE DE VELA: {last_time}")
                
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                cerebro.auditar(df_closed['close'].iloc[0])
                cerebro.aprender(df_closed)
                
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 200)
                df_ctx = pd.DataFrame(rates_ctx)
                df_ctx['time'] = pd.to_datetime(df_ctx['time'], unit='s')
                
                clase, conf = cerebro.predecir(df_ctx)
                
                accion = "⚪ ESPERAR"
                if conf > CONFIG["min_confidence"]:
                    if clase == 2:
                        accion = "🟢 COMPRAR"
                    elif clase == 0:
                        accion = "🔴 VENDER"
                
                logger.info(f"🔮 PREDICCIÓN: {accion} | Confianza: {conf:.1%}")
                
                contador += 1
                if contador % 10 == 0:
                    cerebro.stats()
                
                last_time = curr_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot detenido")
        cerebro.stats()
        mt5.shutdown()

if __name__ == "__main__":
    main()
