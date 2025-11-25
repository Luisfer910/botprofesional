import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import logging
import json

class FeatureEngineer:
    def __init__(self, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.features_config = self.config['FEATURES']
        
        logging.basicConfig(
            filename='logs/feature_engineer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generar_features_basicas(self, df):
        """Genera features técnicas básicas"""
        df = df.copy()
        
        self.logger.info("🔧 Generando features básicas...")
        
        # Indicadores de tendencia
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX (fuerza de tendencia)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Indicadores de momentum
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ATR (volatilidad)
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # Volumen (usar real_volume si existe, sino tick_volume, sino 1)
        volume_col = df.get('real_volume', df.get('tick_volume', pd.Series([1] * len(df))))
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=volume_col).on_balance_volume()
        
        self.logger.info(f"✅ Features básicas generadas: {len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']])} features")
        
        return df
    
    def generar_features_precio(self, df):
        """Genera features basadas en acción del precio"""
        df = df.copy()
        
        self.logger.info("🔧 Generando features de precio...")
        
        # Patrones de velas
        df['body'] = df['close'] - df['open']
        df['body_abs'] = abs(df['body'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios
        df['body_ratio'] = df['body_abs'] / (df['total_range'] + 1e-10)
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-10)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-10)
        
        # Tipo de vela
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        # Patrones específicos
        df['is_hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['upper_shadow_ratio'] < 0.1)).astype(int)
        df['is_shooting_star'] = ((df['upper_shadow_ratio'] > 0.6) & (df['lower_shadow_ratio'] < 0.1)).astype(int)
        df['is_engulfing'] = ((df['body_abs'] > df['body_abs'].shift(1) * 1.5) & 
                               (df['is_bullish'] != df['is_bullish'].shift(1))).astype(int)
        
        # Cambios de precio
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = abs(df['price_change'])
        
        # Momentum de precio
        df['momentum_3'] = df['close'] - df['close'].shift(3)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Velocidad de cambio
        df['roc_3'] = df['close'].pct_change(3)
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)
        
        self.logger.info("✅ Features de precio generadas")
        
        return df
    
    def detectar_soportes_resistencias(self, df, window=20, num_niveles=5):
        """Detecta niveles de soporte y resistencia"""
        df = df.copy()
        
        if not self.features_config['USAR_SOPORTES_RESISTENCIAS']:
            return df
        
        self.logger.info("🔧 Detectando soportes y resistencias...")
        
        # Detectar máximos y mínimos locales
        df['is_pivot_high'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(-1)) &
            (df['high'] > df['high'].shift(2)) & 
            (df['high'] > df['high'].shift(-2))
        )
        
        df['is_pivot_low'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(-1)) &
            (df['low'] < df['low'].shift(2)) & 
            (df['low'] < df['low'].shift(-2))
        )
        
        # Extraer niveles
        resistencias = df[df['is_pivot_high']]['high'].tail(num_niveles).values
        soportes = df[df['is_pivot_low']]['low'].tail(num_niveles).values
        
        # Calcular distancia a niveles más cercanos
        if len(resistencias) > 0:
            df['dist_resistencia_cercana'] = df['close'].apply(
                lambda x: min([abs(x - r) for r in resistencias])
            )
            df['dist_resistencia_cercana_pct'] = df['dist_resistencia_cercana'] / df['close']
        else:
            df['dist_resistencia_cercana'] = 0
            df['dist_resistencia_cercana_pct'] = 0
        
        if len(soportes) > 0:
            df['dist_soporte_cercano'] = df['close'].apply(
                lambda x: min([abs(x - s) for s in soportes])
            )
            df['dist_soporte_cercano_pct'] = df['dist_soporte_cercano'] / df['close']
        else:
            df['dist_soporte_cercano'] = 0
            df['dist_soporte_cercano_pct'] = 0
        
        # En zona de soporte/resistencia
        threshold = df['atr'].iloc[-1] * 0.5 if 'atr' in df.columns else df['close'].iloc[-1] * 0.001
        df['en_resistencia'] = (df['dist_resistencia_cercana'] < threshold).astype(int)
        df['en_soporte'] = (df['dist_soporte_cercano'] < threshold).astype(int)
        
        self.logger.info(f"✅ Detectados {len(resistencias)} resistencias y {len(soportes)} soportes")
        
        return df
    
    def generar_features_impulsos(self, df):
        """Detecta impulsos y retrocesos"""
        df = df.copy()
        
        if not self.features_config['USAR_IMPULSOS']:
            return df
        
        self.logger.info("🔧 Generando features de impulsos...")
        
        # Detectar impulsos alcistas
        df['impulso_alcista'] = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2)) &
            (df['close'].shift(2) > df['close'].shift(3))
        ).astype(int)
        
        # Detectar impulsos bajistas
        df['impulso_bajista'] = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2)) &
            (df['close'].shift(2) < df['close'].shift(3))
        ).astype(int)
        
        # Fuerza del impulso
        df['fuerza_impulso'] = abs(df['close'] - df['close'].shift(3)) / (df['atr'] + 1e-10)
        
        # Retroceso después de impulso
        df['retroceso_alcista'] = (
            (df['impulso_alcista'].shift(1) == 1) &
            (df['close'] < df['close'].shift(1))
        ).astype(int)
        
        df['retroceso_bajista'] = (
            (df['impulso_bajista'].shift(1) == 1) &
            (df['close'] > df['close'].shift(1))
        ).astype(int)
        
        self.logger.info("✅ Features de impulsos generadas")
        
        return df
    
    def generar_features_volatilidad(self, df):
        """Genera features de volatilidad"""
        df = df.copy()
        
        if not self.features_config['USAR_VOLATILIDAD']:
            return df
        
        self.logger.info("🔧 Generando features de volatilidad...")
        
        # Volatilidad histórica
        df['volatilidad_5'] = df['close'].rolling(5).std()
        df['volatilidad_10'] = df['close'].rolling(10).std()
        df['volatilidad_20'] = df['close'].rolling(20).std()
        
        # Cambio en volatilidad
        df['volatilidad_cambio'] = df['volatilidad_10'] / (df['volatilidad_10'].shift(5) + 1e-10)
        
        # Rango verdadero normalizado
        df['atr_normalizado'] = df['atr'] / df['close']
        
        # Expansión/contracción de volatilidad
        df['volatilidad_expansion'] = (df['volatilidad_10'] > df['volatilidad_10'].shift(5)).astype(int)
        df['volatilidad_contraccion'] = (df['volatilidad_10'] < df['volatilidad_10'].shift(5)).astype(int)
        
        self.logger.info("✅ Features de volatilidad generadas")
        
        return df
    
    def generar_features_temporales(self, df):
        """Genera features temporales"""
        df = df.copy()
        
        self.logger.info("🔧 Generando features temporales...")
        
        # Extraer componentes de tiempo
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        
        # Sesiones de trading
        df['sesion_asiatica'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['sesion_europea'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['sesion_americana'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # Overlap de sesiones (mayor volatilidad)
        df['overlap_sesiones'] = ((df['hour'] >= 8) & (df['hour'] < 12)).astype(int)
        
        self.logger.info("✅ Features temporales generadas")
        
        return df
    
    def generar_features_intravela(self, df_ticks):
        """
        Genera features basadas en análisis tick-by-tick
        (formación de vela en tiempo real)
        """
        if df_ticks is None or len(df_ticks) == 0:
            return pd.DataFrame()
        
        self.logger.info("🔧 Generando features intravela...")
        
        # Agrupar por vela
        features_intravela = []
        
        for vela_time, group in df_ticks.groupby('vela_time'):
            if len(group) < 5:  # Mínimo 5 ticks
                continue
            
            # Presión compradora/vendedora
            presion_compradora = (group['presion'] == 1).sum() / len(group)
            presion_vendedora = (group['presion'] == -1).sum() / len(group)
            
            # Velocidad promedio
            velocidad_promedio = group['velocidad'].mean()
            velocidad_max = group['velocidad'].max()
            
            # Volatilidad intravela
            volatilidad_intravela = group['bid'].std()
            
            # Rango explorado
            rango_explorado = group['bid'].max() - group['bid'].min()
            
            # Posición final del precio
            posicion_final = group['posicion_precio'].iloc[-1]
            
            # Spread promedio
            spread_promedio = group['spread'].mean()
            spread_max = group['spread'].max()
            
            # Cambios de dirección
            cambios_direccion = (group['presion'].diff() != 0).sum()
            
            # Aceleración
            velocidades = group['velocidad'].values
            if len(velocidades) > 1:
                aceleracion = np.diff(velocidades).mean()
            else:
                aceleracion = 0
            
            # Momentum intravela
            momentum_intravela = group['bid'].iloc[-1] - group['bid'].iloc[0]
            
            features_intravela.append({
                'vela_time': vela_time,
                'presion_compradora': presion_compradora,
                'presion_vendedora': presion_vendedora,
                'presion_neta': presion_compradora - presion_vendedora,
                'velocidad_promedio': velocidad_promedio,
                'velocidad_max': velocidad_max,
                'volatilidad_intravela': volatilidad_intravela,
                'rango_explorado': rango_explorado,
                'posicion_final': posicion_final,
                'spread_promedio': spread_promedio,
                'spread_max': spread_max,
                'cambios_direccion': cambios_direccion,
                'aceleracion': aceleracion,
                'momentum_intravela': momentum_intravela,
                'num_ticks': len(group)
            })
        
        df_features_intravela = pd.DataFrame(features_intravela)
        
        self.logger.info(f"✅ Features intravela generadas: {len(df_features_intravela)} velas analizadas")
        
        return df_features_intravela
    
    def generar_target_clasificacion(self, df, horizonte=5, umbral=0.0001):
        """
        Genera TARGET para clasificación en 3 clases
        
        Target:
            1 = COMPRA (precio sube > umbral)
            0 = NEUTRAL (precio se mantiene)
           -1 = VENTA (precio baja > umbral)
        
        Args:
            df: DataFrame con OHLCV
            horizonte: Velas hacia adelante (default: 5)
            umbral: Umbral mínimo de movimiento (default: 0.01% = 0.0001)
            
        Returns:
            DataFrame con columna 'target'
        """
        try:
            if 'close' not in df.columns:
                self.logger.error("Columna 'close' no encontrada")
                return df
            
            self.logger.info(f"🎯 Generando target (horizonte: {horizonte}, umbral: {umbral*100:.2f}%)...")
            
            # Calcular retorno futuro
            df['future_return'] = df['close'].shift(-horizonte) / df['close'] - 1
            
            # Inicializar target como neutral
            df['target'] = 0
            
            # Clasificar según retorno
            df.loc[df['future_return'] > umbral, 'target'] = 1   # Compra
            df.loc[df['future_return'] < -umbral, 'target'] = -1  # Venta
            
            # Eliminar últimas filas sin futuro conocido
            df = df[:-horizonte].copy()
            
            # Estadísticas de distribución
            if len(df) > 0:
                total = len(df)
                compras = (df['target'] == 1).sum()
                ventas = (df['target'] == -1).sum()
                neutral = (df['target'] == 0).sum()
                
                self.logger.info(f"📊 Distribución del Target:")
                self.logger.info(f"   Compra (1):  {compras:5d} ({compras/total*100:5.1f}%)")
                self.logger.info(f"   Neutral (0): {neutral:5d} ({neutral/total*100:5.1f}%)")
                self.logger.info(f"   Venta (-1):  {ventas:5d} ({ventas/total*100:5.1f}%)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generando target: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def generar_todas_features(self, df, df_ticks=None):
        """Genera todas las features + TARGET"""
        self.logger.info("🚀 Iniciando generación completa de features...")
        
        # Features básicas
        df = self.generar_features_basicas(df)
        
        # Features de precio
        df = self.generar_features_precio(df)
        
        # Soportes y resistencias
        df = self.detectar_soportes_resistencias(df)
        
        # Impulsos
        df = self.generar_features_impulsos(df)
        
        # Volatilidad
        df = self.generar_features_volatilidad(df)
        
        # Temporales
        df = self.generar_features_temporales(df)
        
        # Features intravela (si hay datos)
        if df_ticks is not None and len(df_ticks) > 0:
            df_intravela = self.generar_features_intravela(df_ticks)
            if len(df_intravela) > 0:
                # Merge con datos históricos
                df = df.merge(df_intravela, left_on='time', right_on='vela_time', how='left')
                df = df.drop('vela_time', axis=1)
                
                # Rellenar NaN con 0 para features intravela
                intravela_cols = df_intravela.columns.drop('vela_time')
                df[intravela_cols] = df[intravela_cols].fillna(0)
        
        # ✅ GENERAR TARGET (CRÍTICO)
        df = self.generar_target_clasificacion(df)
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        self.logger.info(f"✅ Generación completa de features finalizada")
        self.logger.info(f"   Total de features: {len(df.columns)}")
        self.logger.info(f"   Total de muestras: {len(df)}")
        
        return df
    
    def crear_target(self, df, horizonte=1):
        """
        Crea la variable target (objetivo) - MÉTODO LEGACY
        Mantenido por compatibilidad, pero usa generar_target_clasificacion()
        
        1 = CALL (precio sube)
        0 = PUT (precio baja)
        """
        df = df.copy()
        
        # Precio futuro
        df['precio_futuro'] = df['close'].shift(-horizonte)
        
        # Target: 1 si sube, 0 si baja
        df['target'] = (df['precio_futuro'] > df['close']).astype(int)
        
        # Eliminar última fila (no tiene futuro)
        df = df[:-horizonte]
        
        self.logger.info(f"✅ Target creado (horizonte: {horizonte} velas)")
        self.logger.info(f"   CALL: {(df['target'] == 1).sum()} ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
        self.logger.info(f"   PUT: {(df['target'] == 0).sum()} ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
        
        return df
