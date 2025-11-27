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
    "history_size": 5000,
    "memory_size": 2000,        # Memoria más larga para aprender mejor
    "min_confidence": 0.55      # Bajamos para que opere más (y aprenda más)
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("BotPuro")

# ==============================================================================
# MINERÍA DE DATOS (Sin filtros, solo indicadores técnicos puros)
# ==============================================================================
class FeatureEngine:
    def calcular_indicadores(self, df):
        df = df.copy()
        
        # Medias móviles
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
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
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ATR (Volatilidad)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(4)
        
        # Patrones de velas
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volumen relativo
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        return df.dropna()

# ==============================================================================
# CEREBRO CON APRENDIZAJE CONTINUO REAL
# ==============================================================================
class CerebroAdaptativo:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engine = FeatureEngine()
        self.memory = deque(maxlen=CONFIG["memory_size"])
        self.is_trained = False
        
        # Para auditoría
        self.historial_operaciones = []
        self.ultima_prediccion = None
        self.precio_entrada = None
        
    def crear_target_real(self, df):
        """
        Target SIN márgenes artificiales.
        El bot aprenderá por sí mismo qué movimientos valen la pena.
        """
        df = df.copy()
        # Simple: ¿La siguiente vela cerró más arriba o más abajo?
        df['precio_futuro'] = df['close'].shift(-1)
        df['movimiento'] = df['precio_futuro'] - df['close']
        
        # Clasificación pura (sin umbrales):
        # 2 = Sube, 0 = Baja, 1 = Lateral (movimiento < 0.00001)
        df['target'] = np.where(df['movimiento'] > 0.00001, 2,
                       np.where(df['movimiento'] < -0.00001, 0, 1))
        
        return df.dropna()
    
    def preparar_datos(self, df, training=False):
        df_features = self.feature_engine.calcular_indicadores(df)
        
        if training:
            df_features = self.crear_target_real(df_features)
            y = df_features['target']
        else:
            y = None
        
        # Columnas para el modelo
        excluir = ['time', 'target', 'precio_futuro', 'movimiento', 'spread', 'real_volume']
        cols = [c for c in df_features.columns if c not in excluir]
        X = df_features[cols].select_dtypes(include=[np.number])
        
        return X, y, df_features
    
    def inicializar(self, df_inicial):
        logger.info("🧠 Entrenamiento inicial con 5000 velas...")
        
        X, y, _ = self.preparar_datos(df_inicial, training=True)
        self.memory.extend(df_inicial.to_dict('records'))
        
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'min_data_in_leaf': 20
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=300)
        self.is_trained = True
        logger.info("✅ Cerebro inicializado.")
    
    def auditar_operacion(self, precio_cierre_actual):
        """Verifica si la predicción anterior fue correcta"""
        if self.ultima_prediccion is None or self.precio_entrada is None:
            return
        
        movimiento_real = precio_cierre_actual - self.precio_entrada
        resultado = "NEUTRAL"
        
        if self.ultima_prediccion == 2:  # Predijo COMPRA
            if movimiento_real > 0:
                resultado = "✅ GANADA (Buy)"
            else:
                resultado = "❌ PERDIDA (Buy)"
        elif self.ultima_prediccion == 0:  # Predijo VENTA
            if movimiento_real < 0:
                resultado = "✅ GANADA (Sell)"
            else:
                resultado = "❌ PERDIDA (Sell)"
        
        if self.ultima_prediccion != 1:
            logger.info(f"📊 {resultado} | Movimiento: {movimiento_real:.5f}")
            self.historial_operaciones.append({
                'prediccion': self.ultima_prediccion,
                'movimiento': movimiento_real,
                'resultado': 'WIN' if 'GANADA' in resultado else 'LOSS'
            })
    
    def aprender_de_vela(self, nueva_vela_df):
        """Aprendizaje incremental: ajusta el modelo con cada nueva vela"""
        if not self.is_trained:
            return
        
        # Agregar a memoria
        self.memory.append(nueva_vela_df.iloc[0].to_dict())
        
        # Reconstruir dataset con memoria completa
        df_full = pd.DataFrame(list(self.memory))
        X, y, _ = self.preparar_datos(df_full, training=True)
        
        # Tomar las últimas 500 velas para re-entrenamiento rápido
        X_recent = X.tail(500)
        y_recent = y.tail(500)
        
        if len(X_recent) > 50:
            X_scaled = self.scaler.transform(X_recent)
            train_data = lgb.Dataset(X_scaled, label=y_recent)
            
            # Re-entrenar (continuar desde el modelo anterior)
            self.model = lgb.train(
                {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'verbosity': -1,
                    'learning_rate': 0.03  # Learning rate más alto para adaptarse rápido
                },
                train_data,
                num_boost_round=50,
                init_model=self.model,
                keep_training_booster=True
            )
            logger.info("🔄 Modelo actualizado con nueva información.")
    
    def predecir_siguiente(self, df_contexto):
        """Predice el movimiento de la siguiente vela"""
        X, _, df_features = self.preparar_datos(df_contexto, training=False)
        
        last_X = X.tail(1)
        X_scaled = self.scaler.transform(last_X)
        
        probs = self.model.predict(X_scaled)[0]
        clase = np.argmax(probs)
        confianza = probs[clase]
        
        # Guardar para auditoría posterior
        self.ultima_prediccion = clase
        self.precio_entrada = df_contexto['close'].iloc[-1]
        
        return clase, confianza
    
    def mostrar_estadisticas(self):
        """Muestra el rendimiento acumulado"""
        if len(self.historial_operaciones) > 0:
            wins = sum(1 for op in self.historial_operaciones if op['resultado'] == 'WIN')
            total = len(self.historial_operaciones)
            winrate = (wins / total) * 100
            logger.info(f"📈 ESTADÍSTICAS: {wins}/{total} operaciones ganadas ({winrate:.1f}% winrate)")

# ==============================================================================
# MOTOR PRINCIPAL
# ==============================================================================
def main():
    print("="*70)
    print("🚀 BOT DE APRENDIZAJE PURO - SIN FILTROS ARTIFICIALES")
    print("   Tick por Tick → Aprende de cada error → Se adapta")
    print("="*70)
    
    if not mt5.initialize():
        logger.error("Error al inicializar MT5")
        return
    
    if not mt5.login(CONFIG["login"], password=CONFIG["password"], server=CONFIG["server"]):
        logger.error("Error de login")
        return
    
    logger.info(f"✅ Conectado a {CONFIG['symbol']}")
    
    cerebro = CerebroAdaptativo()
    
    # Carga inicial
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, CONFIG["history_size"])
    df_inicial = pd.DataFrame(rates)
    df_inicial['time'] = pd.to_datetime(df_inicial['time'], unit='s')
    
    cerebro.inicializar(df_inicial)
    
    last_candle_time = df_inicial['time'].iloc[-1]
    logger.info(f"⏳ Esperando cierre de vela... (Última: {last_candle_time})")
    
    contador_velas = 0
    
    try:
        while True:
            # Obtener tick actual
            rates_current = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 1)
            if rates_current is None:
                time.sleep(0.1)
                continue
            
            current_time = pd.to_datetime(rates_current[0]['time'], unit='s')
            tick = mt5.symbol_info_tick(CONFIG["symbol"])
            
            # Mostrar tick en tiempo real
            print(f"\r💹 Tick: {tick.bid:.5f} / {tick.ask:.5f} | Vela: {current_time}", end="")
            
            # DETECTAR CIERRE DE VELA
            if current_time > last_candle_time:
                print("\n")
                logger.info(f"🕯️ CIERRE DE VELA: {last_candle_time}")
                
                # 1. Obtener vela cerrada
                rates_closed = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 1, 1)
                df_closed = pd.DataFrame(rates_closed)
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')
                precio_cierre = df_closed['close'].iloc[0]
                
                # 2. AUDITAR: ¿Acertó la predicción anterior?
                cerebro.auditar_operacion(precio_cierre)
                
                # 3. APRENDER: Actualizar modelo con la nueva información
                cerebro.aprender_de_vela(df_closed)
                
                # 4. PREDECIR: ¿Qué hará la siguiente vela?
                rates_context = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, 200)
                df_context = pd.DataFrame(rates_context)
                df_context['time'] = pd.to_datetime(df_context['time'], unit='s')
                
                clase, confianza = cerebro.predecir_siguiente(df_context)
                
                accion = "⚪ ESPERAR"
                if confianza > CONFIG["min_confidence"]:
                    if clase == 2:
                        accion = "🟢 COMPRAR"
                    elif clase == 0:
                        accion = "🔴 VENDER"
                
                logger.info(f"🔮 PREDICCIÓN: {accion} | Confianza: {confianza:.1%}")
                
                # Mostrar estadísticas cada 10 velas
                contador_velas += 1
                if contador_velas % 10 == 0:
                    cerebro.mostrar_estadisticas()
                
                last_candle_time = current_time
            
            time.sleep(0.1)  # Pequeña pausa para no saturar CPU
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot detenido por usuario")
        cerebro.mostrar_estadisticas()
        mt5.shutdown()

if __name__ == "__main__":
    main()
