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
    "min_confidence": 0.60,  # Subimos a 60% para ser más selectivos
    "max_atr_multiplier": 2.5  # Evitar operar en volatilidad extrema
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("bot_optimizado.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotOptimizado")

# ==============================================================================
# ANALIZADOR DE PRICE ACTION
# ==============================================================================
class PriceActionData:
    def __init__(self):
        self.soportes = []
        self.resistencias = []
        
    def extraer_features(self, df):
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, _ = find_peaks(highs, distance=20, prominence=0.0001)
        peaks_low, _ = find_peaks(-lows, distance=20, prominence=0.0001)
        
        self.resistencias = df.iloc[peaks_high]['high'].values if len(peaks_high) > 0 else np.array([])
        self.soportes = df.iloc[peaks_low]['low'].values if len(peaks_low) > 0 else np.array([])
        
        precio_actual = df['close'].iloc[-1]
        
        dist_soporte = min([abs(precio_actual - s) for s in self.soportes]) if len(self.soportes) > 0 else 0.01
        dist_resistencia = min([abs(precio_actual - r) for r in self.resistencias]) if len(self.resistencias) > 0 else 0.01
        
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        last = df.iloc[-1]
        body = last['close'] - last['open']
        total_range = last['high'] - last['low']
        body_ratio = body / total_range if total_range > 0 else 0
        
        # Calcular ATR para detectar volatilidad
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        atr_promedio = true_range.rolling(50).mean().iloc[-1]
        
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
            'lower_shadow': min(last['close'], last['open']) - last['low'],
            'atr': atr,
            'atr_ratio': atr / atr_promedio if atr_promedio > 0 else 1  # Detectar volatilidad anormal
        }

# ==============================================================================
# MOTOR DE INDICADORES TÉCNICOS
# ==============================================================================
class FeatureEngine:
    def calcular_indicadores(self, df):
        df = df.copy()
        
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['bb_middle'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        df['momentum'] = df['close'] - df['close'].shift(4)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        return df.dropna()

# ==============================================================================
# CEREBRO OPTIMIZADO
# ==============================================================================
class CerebroOptimizado:
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
        self.racha_perdidas = 0  # Contador de pérdidas consecutivas
        
    def crear_target(self, df):
        df = df.copy()
        df['precio_futuro'] = df['close'].shift(-1)
        df['movimiento'] = df['precio_futuro'] - df['close']
        
        df['target'] = np.where(df['movimiento'] > 0.00001, 2,
                       np.where(df['movimiento'] < -0.00001, 0, 1))
        
        # Peso adaptativo: penaliza más las pérdidas grandes
        df['sample_weight'] = np.abs(df['movimiento']) * 15000  # Aumentado para dar más importancia
        
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
        
        return X, y, weights, pa_features
    
    def inicializar(self, df_inicial):
        logger.info("🧠 Entrenamiento inicial optimizado...")
        
        X, y, weights, _ = self.preparar_datos(df_inicial, training=True)
        self.memory.extend(df_inicial.to_dict('records'))
        
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y, weight=weights)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.04,
            'num_leaves': 65,
            'min_data_in_leaf': 25,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'lambda_l1': 0.1,  # Regularización
            'lambda_l2': 0.1
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=450)
        self.is_trained = True
        logger.info("✅ Cerebro optimizado listo.")
    
    def auditar(self, precio_cierre):
        if self.ultima_prediccion is None or self.precio_entrada is None:
            return
        
        mov = precio_cierre - self.precio_entrada
        res = "NEUTRAL"
        
        if self.ultima_prediccion == 2:
            if mov > 0:
                res = "✅ GANADA (Buy)"
                self.racha_perdidas = 0
            else:
                res = "❌ PERDIDA (Buy)"
                self.racha_perdidas += 1
        elif self.ultima_prediccion == 0:
            if mov < 0:
                res = "✅ GANADA (Sell)"
                self.racha_perdidas = 0
            else:
                res = "❌ PERDIDA (Sell)"
                self.racha_perdidas += 1
        
        if self.ultima_prediccion != 1:
            logger.info(f"📊 {res} | Movimiento: {mov:.5f} | Racha pérdidas: {self.racha_perdidas}")
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
        X, y, weights, _ = self.preparar_datos(df_full, training=True)
        
        # Ventana adaptativa: más datos si está perdiendo
        ventana = 800 if self.racha_perdidas < 3 else 1200
        X_recent = X.tail(ventana)
        y_recent = y.tail(ventana)
        weights_recent = weights.tail(ventana)
        
        if len(X_recent) > 50:
            X_scaled = self.scaler.transform(X_recent)
            train_data = lgb.Dataset(X_scaled, label=y_recent, weight=weights_recent)
            
            # Learning rate adaptativo
            lr = 0.02 if self.racha_perdidas < 3 else 0.04
            
            self.model = lgb.train(
                {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, 'learning_rate': lr},
                train_data, num_boost_round=70, init_model=self.model, keep_training_booster=True
            )
            logger.info("🔄 Modelo actualizado.")
    
    def predecir(self, df_contexto):
        X, _, _, pa_features = self.preparar_datos(df_contexto, training=False)
        
        # PROTECCIÓN: No operar en volatilidad extrema
        if pa_features['atr_ratio'] > CONFIG['max_atr_multiplier']:
            logger.info(f"   ⚠️ Volatilidad extrema detectada (ATR Ratio: {pa_features['atr_ratio']:.2f}) - ESPERANDO")
            self.ultima_prediccion = 1
            self.precio_entrada = df_contexto['close'].iloc[-1]
            return 1, 0.5
        
        last_X = X.tail(1)
        X_scaled = self.scaler.transform(last_X)
        
        probs = self.model.predict(X_scaled)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        # Ajuste de confianza después de pérdidas consecutivas
        if self.racha_perdidas >= 3:
            confianza *= 0.8  # Reduce confianza un 20%
            logger.info(f"   ⚠️ Ajuste por racha de pérdidas: Confianza reducida a {confianza:.1%}")
        
        self.ultima_prediccion = clase
        self.precio_entrada = df_contexto['close'].iloc[-1]
        
        return clase, confianza
    
    def stats(self):
        if len(self.historial) > 0:
            wins = sum(1 for h in self.historial if h['win'])
            total = len(self.historial)
            wr = (wins/total)*100
            
            # Calcular profit neto
            profit_total = sum(h['mov'] for h in self.historial if h['win'])
            loss_total = abs(sum(h['mov'] for h in self.historial if not h['win']))
            
            logger.info(f"📈 STATS: {wins}/{total} ({wr:.1f}% WR) | Profit: {profit_total:.5f} | Loss: {loss_total:.5f}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT OPTIMIZADO V8.1 - GESTIÓN DE RIESGO MEJORADA")
    print("   Protección contra volatilidad extrema + Aprendizaje adaptativo")
    print("="*80)
    
    if not mt5.initialize():
        return
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")
    
    cerebro = CerebroOptimizado()
    
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
                
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 250)
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
