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
        
        logging.basicConfig(
            filename='logs/feature_engineer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generar_features_tecnicas(self, df):
        """Genera features técnicas tradicionales"""
        df = df.copy()
        
        try:
            # Indicadores de tendencia
            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ADX
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['adx'] = adx.adx()
            
            # RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # Stochastic
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
            
            # ATR
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            
            # Momentum
            df['momentum_3'] = df['close'] - df['close'].shift(3)
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['roc_5'] = df['close'].pct_change(5)
            
            self.logger.info("✅ Features técnicas generadas")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error al generar features técnicas: {str(e)}")
            return df
    
    def agregar_features_intravela(self, df, features_intravela):
        """
        Agrega features intravela al DataFrame
        
        Args:
            df: DataFrame con velas
            features_intravela: dict con features intravela de la última vela
        """
        if features_intravela is None:
            # Si no hay features intravela, usar valores por defecto
            features_intravela = {
                'precio_cambio': 0.0,
                'precio_cambio_pct': 0.0,
                'rango_intravela': 0.0,
                'posicion_precio': 0.5,
                'volatilidad_intravela': 0.0,
                'volatilidad_normalizada': 0.0,
                'presion_compradora': 0.5,
                'presion_vendedora': 0.5,
                'presion_neta': 0.0,
                'velocidad': 0.0,
                'momentum_primera_mitad': 0.0,
                'momentum_segunda_mitad': 0.0,
                'spread_promedio': 0.0,
                'spread_max': 0.0,
                'spread_min': 0.0,
                'cambios_direccion': 0,
                'num_ticks': 0,
                'aceleracion': 0.0
            }
        
        # Agregar features intravela solo a la última fila
        for key, value in features_intravela.items():
            if key not in df.columns:
                df[key] = 0.0
            df.loc[df.index[-1], key] = value
        
        return df
    
    def generar_todas_features(self, df, features_intravela=None):
        """
        Genera todas las features (técnicas + intravela)
        
        Args:
            df: DataFrame con velas OHLC
            features_intravela: dict con features intravela (opcional)
        """
        try:
            # Features técnicas
            df = self.generar_features_tecnicas(df)
            
            # Features intravela
            if features_intravela is not None:
                df = self.agregar_features_intravela(df, features_intravela)
            else:
                # Si no hay features intravela, usar las simuladas del DataFrame
                # (ya vienen en el DataFrame si se usó obtener_datos_completos_con_ticks)
                pass
            
            # Eliminar NaN
            df = df.dropna()
            
            self.logger.info(f"✅ Features completas generadas: {len(df.columns)} columnas")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error al generar features: {str(e)}")
            return df
    
    def crear_target(self, df, horizonte=1):
        """
        Crea variable target para entrenamiento
        
        Args:
            horizonte: Velas hacia adelante para predecir
        """
        df = df.copy()
        
        # Target: 1 si el precio sube, 0 si baja
        df['precio_futuro'] = df['close'].shift(-horizonte)
        df['target'] = (df['precio_futuro'] > df['close']).astype(int)
        
        # Eliminar últimas filas sin target
        df = df[:-horizonte]
        
        return df
    
    def obtener_feature_columns(self, df):
        """Obtiene lista de columnas de features (excluye OHLC, time, target)"""
        excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                   'spread', 'real_volume', 'target', 'precio_futuro']
        
        feature_cols = [col for col in df.columns if col not in excluir]
        
        return feature_cols
