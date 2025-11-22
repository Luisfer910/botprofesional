"""
FEATURE ENGINEER - Ingeniería de Características Avanzada
Integrado con análisis de estructura, zonas y patrones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Importar módulos de strategy
from strategy.structure_analyzer import StructureAnalyzer
from strategy.zone_detector import ZoneDetector
from strategy.candle_pattern_analyzer import CandlePatternAnalyzer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Ingeniero de características que integra:
    - Análisis de estructura de mercado (BOS, CHoCH)
    - Detección de zonas de oferta/demanda
    - Patrones de velas avanzados
    - Indicadores técnicos tradicionales
    """
    
    def __init__(self):
        """Inicializar el ingeniero de características"""
        try:
            self.structure_analyzer = StructureAnalyzer()
            self.zone_detector = ZoneDetector()
            self.candle_analyzer = CandlePatternAnalyzer()
            logger.info("✅ FeatureEngineer inicializado con módulos avanzados")
        except Exception as e:
            logger.error(f"❌ Error inicializando FeatureEngineer: {e}")
            raise
    
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear todas las características para el modelo
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con todas las características
        """
        try:
            logger.info(f"🔧 Ingeniería de características para {len(df)} velas...")
            
            # Hacer copia para no modificar original
            df = df.copy()
            
            # Validar columnas necesarias
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Faltan columnas requeridas: {required_cols}")
            
            # 1. CARACTERÍSTICAS BÁSICAS
            df = self._add_basic_features(df)
            
            # 2. INDICADORES TÉCNICOS
            df = self._add_technical_indicators(df)
            
            # 3. ANÁLISIS DE ESTRUCTURA (BOS, CHoCH)
            df = self._add_structure_features(df)
            
            # 4. ZONAS DE OFERTA/DEMANDA
            df = self._add_zone_features(df)
            
            # 5. PATRONES DE VELAS
            df = self._add_candle_patterns(df)
            
            # 6. CARACTERÍSTICAS TEMPORALES
            df = self._add_temporal_features(df)
            
            # 7. CARACTERÍSTICAS DE VOLATILIDAD
            df = self._add_volatility_features(df)
            
            # 8. CARACTERÍSTICAS DE VOLUMEN
            df = self._add_volume_features(df)
            
            # Eliminar filas con NaN
            initial_len = len(df)
            df = df.dropna()
            final_len = len(df)
            
            if initial_len != final_len:
                logger.warning(f"⚠️ Se eliminaron {initial_len - final_len} filas con NaN")
            
            logger.info(f"✅ Características creadas: {len(df.columns)} columnas, {len(df)} filas")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error en ingeniería de características: {e}")
            raise
    
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características básicas de precio"""
        
        # Rango de la vela
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Ratios
        df['body_to_range'] = df['body'] / (df['range'] + 1e-10)
        df['upper_wick_ratio'] = df['upper_wick'] / (df['range'] + 1e-10)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['range'] + 1e-10)
        
        # Tipo de vela
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Cambios porcentuales
        df['price_change'] = df['close'].pct_change()
        df['high_change'] = df['high'].pct_change()
        df['low_change'] = df['low'].pct_change()
        
        return df
    
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir indicadores técnicos tradicionales"""
        
        # Medias móviles
        for period in [9, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # ADX
        df['adx_14'] = self._calculate_adx(df, 14)
        
        return df
    
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características de estructura de mercado"""
        
        try:
            # Analizar estructura
            structure_data = self.structure_analyzer.analyze_structure(df)
            
            # Añadir información de estructura
            df['market_structure'] = structure_data.get('current_structure', 'NEUTRAL')
            df['structure_strength'] = structure_data.get('structure_strength', 0.5)
            
            # Convertir estructura a valores numéricos
            structure_map = {'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}
            df['structure_numeric'] = df['market_structure'].map(structure_map).fillna(0)
            
            # BOS y CHoCH
            bos_data = structure_data.get('bos', [])
            choch_data = structure_data.get('choch', [])
            
            # Crear columnas para BOS y CHoCH
            df['has_bos'] = 0
            df['has_choch'] = 0
            df['bos_direction'] = 0
            df['choch_direction'] = 0
            
            # Marcar BOS
            for bos in bos_data:
                idx = bos.get('index', -1)
                if 0 <= idx < len(df):
                    df.iloc[idx, df.columns.get_loc('has_bos')] = 1
                    df.iloc[idx, df.columns.get_loc('bos_direction')] = 1 if bos.get('direction') == 'BULLISH' else -1
            
            # Marcar CHoCH
            for choch in choch_data:
                idx = choch.get('index', -1)
                if 0 <= idx < len(df):
                    df.iloc[idx, df.columns.get_loc('has_choch')] = 1
                    df.iloc[idx, df.columns.get_loc('choch_direction')] = 1 if choch.get('direction') == 'BULLISH' else -1
            
            # Distancia desde último BOS/CHoCH
            df['bars_since_bos'] = self._bars_since_event(df['has_bos'])
            df['bars_since_choch'] = self._bars_since_event(df['has_choch'])
            
            logger.info(f"✅ Características de estructura añadidas")
            
        except Exception as e:
            logger.error(f"❌ Error en características de estructura: {e}")
            # Crear columnas vacías si falla
            df['market_structure'] = 'NEUTRAL'
            df['structure_strength'] = 0.5
            df['structure_numeric'] = 0
            df['has_bos'] = 0
            df['has_choch'] = 0
            df['bos_direction'] = 0
            df['choch_direction'] = 0
            df['bars_since_bos'] = 999
            df['bars_since_choch'] = 999
        
        return df
    
    
    def _add_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características de zonas de oferta/demanda"""
        
        try:
            # Detectar zonas
            zones = self.zone_detector.detect_zones(df)
            
            # Crear columnas para zonas
            df['in_demand_zone'] = 0
            df['in_supply_zone'] = 0
            df['zone_strength'] = 0.0
            df['distance_to_nearest_zone'] = 999.0
            
            if len(zones) > 0:
                current_price = df['close'].iloc[-1]
                min_distance = float('inf')
                
                for zone in zones:
                    zone_type = zone.get('type', '')
                    top = zone.get('top', 0)
                    bottom = zone.get('bottom', 0)
                    strength = zone.get('strength', 0)
                    
                    # Calcular distancia a la zona
                    if current_price >= bottom and current_price <= top:
                        distance = 0
                    elif current_price < bottom:
                        distance = bottom - current_price
                    else:
                        distance = current_price - top
                    
                    min_distance = min(min_distance, distance)
                    
                    # Marcar si el precio está en la zona
                    mask = (df['close'] >= bottom) & (df['close'] <= top)
                    
                    if zone_type == 'DEMAND':
                        df.loc[mask, 'in_demand_zone'] = 1
                        df.loc[mask, 'zone_strength'] = df.loc[mask, 'zone_strength'].apply(lambda x: max(x, strength))
                    elif zone_type == 'SUPPLY':
                        df.loc[mask, 'in_supply_zone'] = 1
                        df.loc[mask, 'zone_strength'] = df.loc[mask, 'zone_strength'].apply(lambda x: max(x, strength))
                
                # Distancia normalizada a la zona más cercana
                if min_distance != float('inf'):
                    df['distance_to_nearest_zone'] = min_distance / (current_price + 1e-10)
            
            # Ratio de zonas
            df['zone_ratio'] = df['in_demand_zone'] - df['in_supply_zone']
            
            logger.info(f"✅ Características de zonas añadidas: {len(zones)} zonas detectadas")
            
        except Exception as e:
            logger.error(f"❌ Error en características de zonas: {e}")
            df['in_demand_zone'] = 0
            df['in_supply_zone'] = 0
            df['zone_strength'] = 0.0
            df['distance_to_nearest_zone'] = 999.0
            df['zone_ratio'] = 0
        
        return df
    
    
    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir patrones de velas"""
        
        try:
            # Analizar patrones
            patterns = self.candle_analyzer.analyze_patterns(df)
            
            # Crear columnas para patrones
            df['has_bullish_pattern'] = 0
            df['has_bearish_pattern'] = 0
            df['pattern_strength'] = 0.0
            df['pattern_type'] = 'NONE'
            
            for pattern in patterns:
                idx = pattern.get('index', -1)
                if 0 <= idx < len(df):
                    pattern_type = pattern.get('type', '')
                    strength = pattern.get('strength', 0)
                    direction = pattern.get('direction', '')
                    
                    df.iloc[idx, df.columns.get_loc('pattern_strength')] = strength
                    df.iloc[idx, df.columns.get_loc('pattern_type')] = pattern_type
                    
                    if direction == 'BULLISH':
                        df.iloc[idx, df.columns.get_loc('has_bullish_pattern')] = 1
                    elif direction == 'BEARISH':
                        df.iloc[idx, df.columns.get_loc('has_bearish_pattern')] = 1
            
            # Patrón neto
            df['pattern_net'] = df['has_bullish_pattern'] - df['has_bearish_pattern']
            
            # Barras desde último patrón
            df['bars_since_bullish_pattern'] = self._bars_since_event(df['has_bullish_pattern'])
            df['bars_since_bearish_pattern'] = self._bars_since_event(df['has_bearish_pattern'])
            
            logger.info(f"✅ Patrones de velas añadidos: {len(patterns)} patrones detectados")
            
        except Exception as e:
            logger.error(f"❌ Error en patrones de velas: {e}")
            df['has_bullish_pattern'] = 0
            df['has_bearish_pattern'] = 0
            df['pattern_strength'] = 0.0
            df['pattern_type'] = 'NONE'
            df['pattern_net'] = 0
            df['bars_since_bullish_pattern'] = 999
            df['bars_since_bearish_pattern'] = 999
        
        return df
    
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características temporales"""
        
        if 'time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['time']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
                df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
                df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
                df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            except:
                df['hour'] = 0
                df['day_of_week'] = 0
                df['is_london_session'] = 0
                df['is_ny_session'] = 0
                df['is_asian_session'] = 0
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
            df['is_london_session'] = 0
            df['is_ny_session'] = 0
            df['is_asian_session'] = 0
        
        return df
    
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características de volatilidad"""
        
        # Volatilidad histórica
        df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        
        # Ratio de volatilidad
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-10)
        
        # True Range
        df['true_range'] = self._calculate_true_range(df)
        
        # Rango promedio
        df['avg_range_10'] = df['range'].rolling(window=10).mean()
        df['range_ratio'] = df['range'] / (df['avg_range_10'] + 1e-10)
        
        return df
    
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir características de volumen"""
        
        if 'tick_volume' in df.columns:
            # Volumen promedio
            df['volume_sma_20'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / (df['volume_sma_20'] + 1e-10)
            
            # Volumen relativo
            df['volume_change'] = df['tick_volume'].pct_change()
            
            # OBV (On Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        else:
            df['volume_sma_20'] = 0
            df['volume_ratio'] = 1
            df['volume_change'] = 0
            df['obv'] = 0
        
        return df
    
    
    # ==================== FUNCIONES AUXILIARES ====================
    
    def _bars_since_event(self, event_series: pd.Series) -> pd.Series:
        """Calcular barras desde el último evento"""
        result = pd.Series(index=event_series.index, dtype=int)
        counter = 999
        
        for i in range(len(event_series)):
            if event_series.iloc[i] == 1:
                counter = 0
            else:
                counter += 1
            result.iloc[i] = counter
        
        return result
    
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calcular MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std: float = 2.0):
        """Calcular Bandas de Bollinger"""
        middle = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular ATR"""
        tr = self._calculate_true_range(df)
        atr = tr.rolling(window=period).mean()
        return atr
    
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calcular True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr
    
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth: int = 3):
        """Calcular Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        stoch_d = stoch_k.rolling(window=smooth).mean()
        return stoch_k, stoch_d
    
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular ADX"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = self._calculate_true_range(df)
        
        atr = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / (atr + 1e-10))
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx
