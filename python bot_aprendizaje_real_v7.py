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
    "min_confidence": 0.55  # Bajo para que opere más y aprenda más
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("bot_aprendizaje_real.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotInteligente")

# ==============================================================================
# ANALIZADOR DE PRICE ACTION (Sin Reglas, Solo Datos)
# ==============================================================================
class PriceActionAnalyzer:
    def __init__(self):
        self.soportes = []
        self.resistencias = []
        
    def detectar_niveles_clave(self, df, ventana=20):
        """Detecta soportes y resistencias usando picos"""
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, _ = find_peaks(highs, distance=ventana, prominence=0.0001)
        peaks_low, _ = find_peaks(-lows, distance=ventana, prominence=0.0001)
        
        self.resistencias = df.iloc[peaks_high]['high'].values
        self.soportes = df.iloc[peaks_low]['low'].values
        
        return self.soportes, self.resistencias
    
    def calcular_features_contexto(self, df):
        """
        Extrae TODOS los datos del mercado y deja que el modelo decida qué es importante.
        NO hay reglas aquí, solo información cruda.
        """
        self.detectar_niveles_clave(df)
        
        precio_actual = df['close'].iloc[-1]
        
        # Distancias a niveles clave
        dist_soporte_cercano = min([abs(precio_actual - s) for s in self.soportes]) if len(self.soportes) > 0 else 0.01
        dist_resistencia_cercana = min([abs(precio_actual - r) for r in self.resistencias]) if len(self.resistencias) > 0 else 0.01
        
        # Medias móviles (para detectar tendencia)
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        # Posición relativa del precio
        precio_vs_sma20 = (precio_actual - sma_20) / sma_20
        precio_vs_sma50 = (precio_actual - sma_50) / sma_50
        
        # Estructura de la vela actual
        last = df.iloc[-1]
        body = last['close'] - last['open']
        total_range = last['high'] - last['low']
        body_ratio = body / total_range if total_range > 0 else 0
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        
        # Volatilidad reciente
        volatilidad = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Momentum
        momentum_5 = df['close'].iloc[-1] - df['close'].iloc[-6] if len(df) > 5 else 0
        momentum_10 = df['close'].iloc[-1] - df['close'].iloc[-11] if len(df) > 10 else 0
        
        # Retornar TODAS las features (el modelo decidirá cuáles usar)
        features = {
            'dist_soporte': dist_soporte_cercano,
            'dist_resistencia': dist_resistencia_cercana,
            'num_soportes': len(self.soportes),
            'num_resistencias': len(self.resistencias),
            'precio_vs_sma20': precio_vs_sma20,
            'precio_vs_sma50': precio_vs_sma50,
            'body_ratio': body_ratio,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'volatilidad': volatilidad,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'sma20_vs_sma50': (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        }
        
        return features

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
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
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
        
        # ATR (Volatilidad)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Patrones de velas
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / (df['high'] - df['low'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volumen
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        return df.dropna()

# ==============================================================================
# CEREBRO CON APRENDIZAJE POR REFUERZO
# ==============================================================================
class CerebroInteligente:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engine = FeatureEngine()
        self.price_action = PriceActionAnalyzer()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.is_trained = False
        
        # Sistema de recompensas
        self.ultima_prediccion = None
        self.precio_entrada = None
        self.historial = []
        self.recompensas_acumuladas = 0
        
    def crear_target_con_recompensa(self, df):
        """
        Target basado en RECOMPENSA real (no solo dirección).
        El bot aprenderá a maximizar ganancias, no solo acertar dirección.
        """
        df = df.copy()
        df['precio_futuro'] = df['close'].shift(-1)
        df['movimiento'] = df['precio_futuro'] - df['close']
        
        # Target simple: 2=Sube, 0=Baja, 1=Lateral
        df['target'] = np.where(df['movimiento'] > 0, 2, 
                       np.where(df['movimiento'] < 0, 0, 1))
        
        # Peso de la muestra (samples con mayor movimiento son más importantes)
        df['sample_weight'] = np.abs(df['movimiento']) * 10000  # Escalar para que sea significativo
        
        return df.dropna()
    
    def preparar_datos(self, df, training=False):
        # Indicadores técnicos
        df_tech = self.feature_engine.calcular_indicadores(df)
        
        # Price Action
        contexto_pa = self.price_action.calcular_features_contexto(df_tech)
        
        # Agregar contexto de Price Action
        for key, value in contexto_pa.items():
            df_tech[key] = value
        
        if training:
            df_tech = self.crear_target_con_recompensa(df_tech)
            y = df_tech['target']
            weights = df_tech['sample_weight']
        else:
            y = None
            weights = None
        
        # Seleccionar features
        excluir = ['time', 'target', 'precio_futuro', 'movimiento', 'spread', 'real_volume', 'sample_weight']
        cols = [c for c in df_tech.columns if c not in excluir]
        X = df_tech[cols].select_dtypes(include=[np.number])
        
        return X, y, weights
    
    def inicializar(self, df_inicial):
        logger.info("🧠 Entrenamiento inicial (sin reglas, aprendizaje puro)...")
        
        X, y, weights = self.preparar_datos(df_inicial, training=True)
        self.memory.extend(df_inicial.to_dict('records'))
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar con pesos (muestras con mayor movimiento tienen más importancia)
        train_data = lgb.Dataset(X_scaled, label=y, weight=weights)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 70,
            'min_data_in_leaf': 15,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=500)
        self.is_trained = True
        logger.info("✅ Cerebro entrenado (aprendió de 5000 velas).")
    
    def auditar_y_aprender(self, precio_cierre):
        """
        Sistema de recompensas:
        - Si acertó → Recompensa positiva
        - Si falló → Recompensa negativa (penalización)
        """
        if self.ultima_prediccion is None or self.precio_entrada is None:
            return
        
        mov = precio_cierre - self.precio_entrada
        recompensa = 0
        res = "NEUTRAL"
        
        if self.ultima_prediccion == 2:  # Predijo COMPRA
            recompensa = mov * 10000  # Escalar para que sea significativo
            res = "✅ WIN (Buy)" if mov > 0 else "❌ LOSS (Buy)"
        elif self.ultima_prediccion == 0:  # Predijo VENTA
            recompensa = -mov * 10000
            res = "✅ WIN (Sell)" if mov < 0 else "❌ LOSS (Sell)"
        
        self.recompensas_acumuladas += recompensa
        
        if self.ultima_prediccion != 1:
            logger.info(f"📊 {res} | Mov: {mov:.5f} | Recompensa: {recompensa:.2f}")
            self.historial.append({
                'pred': self.ultima_prediccion,
                'mov': mov,
                'recompensa': recompensa,
                'win': 'WIN' in res
            })
    
    def aprender(self, nueva_vela_df):
        """
        Re-entrena el modelo con énfasis en los errores recientes.
        """
        if not self.is_trained:
            return
        
        self.memory.append(nueva_vela_df.iloc[0].to_dict())
        
        df_full = pd.DataFrame(list(self.memory))
        X, y, weights = self.preparar_datos(df_full, training=True)
        
        # Tomar ventana de aprendizaje reciente (últimas 1000 velas)
        X_recent = X.tail(1000)
        y_recent = y.tail(1000)
        weights_recent = weights.tail(1000)
        
        if len(X_recent) > 100:
            X_scaled = self.scaler.transform(X_recent)
            train_data = lgb.Dataset(X_scaled, label=y_recent, weight=weights_recent)
            
            # Re-entrenar (continuar desde modelo anterior)
            self.model = lgb.train(
                {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'verbosity': -1,
                    'learning_rate': 0.04  # Learning rate adaptativo
                },
                train_data,
                num_boost_round=100,
                init_model=self.model,
                keep_training_booster=True
            )
            logger.info("🔄 Modelo actualizado (aprendió de sus errores).")
    
    def predecir(self, df_contexto):
        """
        Predicción pura - SIN FILTROS.
        El modelo decide basándose en TODO lo que aprendió.
        """
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
            
            # Calcular profit factor
            ganancias = sum(h['recompensa'] for h in self.historial if h['recompensa'] > 0)
            perdidas = abs(sum(h['recompensa'] for h in self.historial if h['recompensa'] < 0))
            pf = ganancias / perdidas if perdidas > 0 else 0
            
            logger.info(f"📈 STATS: {wins}/{total} ({wr:.1f}% WR) | PF: {pf:.2f} | Recompensa Total: {self.recompensas_acumuladas:.2f}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("🚀 BOT V7.0 - APRENDIZAJE INTELIGENTE SIN REGLAS")
    print("   El bot descubre las reglas por sí mismo analizando el mercado")
    print("="*80)
    
    if not mt5.initialize():
        return
    mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"])
    
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")
    
    cerebro = CerebroInteligente()
    
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    cerebro.inicializar(df)
    
    last_time = df['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre de vela... (Última: {last_time})")
    
    contador = 0
    
    try:
        while True:
            # ANÁLISIS TICK POR TICK
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            curr_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            print(f"\r💹 Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | Vela: {curr_time}", end="")
            
            # CIERRE DE VELA
            if curr_time > last_time:
                print("\n")
                logger.info(f"🕯️ CIERRE: {last_time}")
                
                # Obtener vela cerrada
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                
                # AUDITAR Y CALCULAR RECOMPENSA
                cerebro.auditar_y_aprender(df_closed['close'].iloc[0])
                
                # APRENDER (actualizar modelo con nueva información)
                cerebro.aprender(df_closed)
                
                # PREDECIR (sin filtros, decisión pura del modelo)
                rates_ctx = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 300)
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
