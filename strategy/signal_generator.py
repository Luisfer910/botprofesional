"""
SIGNAL GENERATOR - Generador de Señales Avanzado
Integra análisis técnico con estructura de mercado, zonas y patrones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generador de señales que combina:
    - Análisis técnico tradicional
    - Estructura de mercado (BOS, CHoCH)
    - Zonas de oferta/demanda
    - Patrones de velas
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar generador de señales
        
        Args:
            config: Configuración opcional
        """
        self.config = config or {}
        
        # Umbrales configurables
        self.min_signal_strength = self.config.get('min_signal_strength', 0.6)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.min_structure_strength = self.config.get('min_structure_strength', 0.5)
        self.min_zone_strength = self.config.get('min_zone_strength', 0.6)
        self.min_pattern_strength = self.config.get('min_pattern_strength', 0.7)
        
        logger.info("✅ SignalGenerator inicializado")
    
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generar señales de trading
        
        Args:
            df: DataFrame con características (del FeatureEngineer)
            
        Returns:
            DataFrame con señales añadidas
        """
        try:
            logger.info(f"🎯 Generando señales para {len(df)} velas...")
            
            # Hacer copia
            df = df.copy()
            
            # Inicializar columnas de señales
            df['signal'] = 0  # 1 = BUY, -1 = SELL, 0 = NEUTRAL
            df['signal_strength'] = 0.0
            df['signal_type'] = 'NONE'
            df['entry_reason'] = ''
            
            # 1. SEÑALES DE ESTRUCTURA
            structure_signals = self._generate_structure_signals(df)
            
            # 2. SEÑALES DE ZONAS
            zone_signals = self._generate_zone_signals(df)
            
            # 3. SEÑALES DE PATRONES
            pattern_signals = self._generate_pattern_signals(df)
            
            # 4. SEÑALES TÉCNICAS
            technical_signals = self._generate_technical_signals(df)
            
            # 5. COMBINAR SEÑALES
            df = self._combine_signals(df, structure_signals, zone_signals, 
                                      pattern_signals, technical_signals)
            
            # 6. FILTRAR SEÑALES DÉBILES
            df = self._filter_weak_signals(df)
            
            # Contar señales
            buy_signals = (df['signal'] == 1).sum()
            sell_signals = (df['signal'] == -1).sum()
            
            logger.info(f"✅ Señales generadas: {buy_signals} BUY, {sell_signals} SELL")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error generando señales: {e}")
            raise
    
    
    def _generate_structure_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generar señales basadas en estructura de mercado"""
        
        signals = pd.Series(0, index=df.index)
        
        try:
            # BOS Alcista + Estructura alcista = Señal de compra
            bullish_bos = (
                (df['has_bos'] == 1) &
                (df['bos_direction'] == 1) &
                (df['structure_numeric'] >= 0) &
                (df['structure_strength'] >= self.min_structure_strength)
            )
            
            # BOS Bajista + Estructura bajista = Señal de venta
            bearish_bos = (
                (df['has_bos'] == 1) &
                (df['bos_direction'] == -1) &
                (df['structure_numeric'] <= 0) &
                (df['structure_strength'] >= self.min_structure_strength)
            )
            
            # CHoCH indica cambio de estructura
            bullish_choch = (
                (df['has_choch'] == 1) &
                (df['choch_direction'] == 1) &
                (df['structure_strength'] >= self.min_structure_strength)
            )
            
            bearish_choch = (
                (df['has_choch'] == 1) &
                (df['choch_direction'] == -1) &
                (df['structure_strength'] >= self.min_structure_strength)
            )
            
            # Asignar señales
            signals[bullish_bos | bullish_choch] = 1
            signals[bearish_bos | bearish_choch] = -1
            
            logger.info(f"   📊 Señales de estructura: {(signals != 0).sum()}")
            
        except Exception as e:
            logger.error(f"   ❌ Error en señales de estructura: {e}")
        
        return signals
    
    
    def _generate_zone_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generar señales basadas en zonas de oferta/demanda"""
        
        signals = pd.Series(0, index=df.index)
        
        try:
            # Precio en zona de demanda + fuerza alta = Señal de compra
            demand_signal = (
                (df['in_demand_zone'] == 1) &
                (df['zone_strength'] >= self.min_zone_strength) &
                (df['structure_numeric'] >= 0)  # Confirmar con estructura
            )
            
            # Precio en zona de oferta + fuerza alta = Señal de venta
            supply_signal = (
                (df['in_supply_zone'] == 1) &
                (df['zone_strength'] >= self.min_zone_strength) &
                (df['structure_numeric'] <= 0)  # Confirmar con estructura
            )
            
            # Rebote desde zona
            price_bouncing_from_demand = (
                (df['in_demand_zone'].shift(1) == 1) &
                (df['close'] > df['open']) &  # Vela alcista
                (df['zone_strength'].shift(1) >= self.min_zone_strength)
            )
            
            price_bouncing_from_supply = (
                (df['in_supply_zone'].shift(1) == 1) &
                (df['close'] < df['open']) &  # Vela bajista
                (df['zone_strength'].shift(1) >= self.min_zone_strength)
            )
            
            # Asignar señales
            signals[demand_signal | price_bouncing_from_demand] = 1
            signals[supply_signal | price_bouncing_from_supply] = -1
            
            logger.info(f"   🎯 Señales de zonas: {(signals != 0).sum()}")
            
        except Exception as e:
            logger.error(f"   ❌ Error en señales de zonas: {e}")
        
        return signals
    
    
    def _generate_pattern_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generar señales basadas en patrones de velas"""
        
        signals = pd.Series(0, index=df.index)
        
        try:
            # Patrón alcista fuerte
            bullish_pattern = (
                (df['has_bullish_pattern'] == 1) &
                (df['pattern_strength'] >= self.min_pattern_strength)
            )
            
            # Patrón bajista fuerte
            bearish_pattern = (
                (df['has_bearish_pattern'] == 1) &
                (df['pattern_strength'] >= self.min_pattern_strength)
            )
            
            # Patrón + confirmación de estructura
            bullish_confirmed = (
                bullish_pattern &
                (df['structure_numeric'] >= 0)
            )
            
            bearish_confirmed = (
                bearish_pattern &
                (df['structure_numeric'] <= 0)
            )
            
            # Asignar señales
            signals[bullish_confirmed] = 1
            signals[bearish_confirmed] = -1
            
            logger.info(f"   📈 Señales de patrones: {(signals != 0).sum()}")
            
        except Exception as e:
            logger.error(f"   ❌ Error en señales de patrones: {e}")
        
        return signals
    
    
    def _generate_technical_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generar señales basadas en indicadores técnicos"""
        
        signals = pd.Series(0, index=df.index)
        
        try:
            # RSI sobreventa
            rsi_oversold = df['rsi_14'] < self.rsi_oversold
            
            # RSI sobrecompra
            rsi_overbought = df['rsi_14'] > self.rsi_overbought
            
            # MACD cruce alcista
            macd_bullish = (
                (df['macd'] > df['macd_signal']) &
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            )
            
            # MACD cruce bajista
            macd_bearish = (
                (df['macd'] < df['macd_signal']) &
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            )
            
            # Precio por debajo de BB inferior
            bb_oversold = df['close'] < df['bb_lower']
            
            # Precio por encima de BB superior
            bb_overbought = df['close'] > df['bb_upper']
            
            # Combinaciones alcistas
            bullish_technical = (
                (rsi_oversold | bb_oversold) &
                (macd_bullish | (df['macd_hist'] > 0))
            )
            
            # Combinaciones bajistas
            bearish_technical = (
                (rsi_overbought | bb_overbought) &
                (macd_bearish | (df['macd_hist'] < 0))
            )
            
            # Asignar señales
            signals[bullish_technical] = 1
            signals[bearish_technical] = -1
            
            logger.info(f"   📉 Señales técnicas: {(signals != 0).sum()}")
            
        except Exception as e:
            logger.error(f"   ❌ Error en señales técnicas: {e}")
        
        return signals
    
    
    def _combine_signals(self, df: pd.DataFrame, 
                        structure_signals: pd.Series,
                        zone_signals: pd.Series,
                        pattern_signals: pd.Series,
                        technical_signals: pd.Series) -> pd.DataFrame:
        """Combinar todas las señales con ponderación"""
        
        # Pesos para cada tipo de señal
        weights = {
            'structure': 0.35,  # Estructura es la más importante
            'zone': 0.30,       # Zonas son muy importantes
            'pattern': 0.20,    # Patrones confirman
            'technical': 0.15   # Técnicos apoyan
        }
        
        # Calcular señal combinada (promedio ponderado)
        combined_signal = (
            structure_signals * weights['structure'] +
            zone_signals * weights['zone'] +
            pattern_signals * weights['pattern'] +
            technical_signals * weights['technical']
        )
        
        # Calcular fuerza de la señal (0 a 1)
        signal_strength = abs(combined_signal)
        
        # Determinar dirección final
        df['signal'] = 0
        df['signal'][combined_signal > 0.3] = 1   # BUY
        df['signal'][combined_signal < -0.3] = -1  # SELL
        
        df['signal_strength'] = signal_strength
        
        # Determinar tipo de señal dominante
        df['signal_type'] = 'NONE'
        
        for idx in df.index:
            if df.loc[idx, 'signal'] != 0:
                # Encontrar la señal más fuerte
                signals_dict = {
                    'STRUCTURE': abs(structure_signals[idx]),
                    'ZONE': abs(zone_signals[idx]),
                    'PATTERN': abs(pattern_signals[idx]),
                    'TECHNICAL': abs(technical_signals[idx])
                }
                df.loc[idx, 'signal_type'] = max(signals_dict, key=signals_dict.get)
                
                # Crear razón de entrada
                reasons = []
                if structure_signals[idx] != 0:
                    reasons.append('BOS/CHoCH')
                if zone_signals[idx] != 0:
                    reasons.append('Zona O/D')
                if pattern_signals[idx] != 0:
                    reasons.append('Patrón')
                if technical_signals[idx] != 0:
                    reasons.append('Técnico')
                
                df.loc[idx, 'entry_reason'] = ' + '.join(reasons)
        
        return df
    
    
    def _filter_weak_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrar señales débiles"""
        
        # Eliminar señales con fuerza insuficiente
        weak_signals = df['signal_strength'] < self.min_signal_strength
        df.loc[weak_signals, 'signal'] = 0
        df.loc[weak_signals, 'signal_type'] = 'NONE'
        df.loc[weak_signals, 'entry_reason'] = ''
        
        # Eliminar señales contradictorias
        # (ej: señal de compra pero estructura bajista fuerte)
        contradictory_buy = (
            (df['signal'] == 1) &
            (df['structure_numeric'] < -0.5) &
            (df['in_supply_zone'] == 1)
        )
        
        contradictory_sell = (
            (df['signal'] == -1) &
            (df['structure_numeric'] > 0.5) &
            (df['in_demand_zone'] == 1)
        )
        
        df.loc[contradictory_buy | contradictory_sell, 'signal'] = 0
        df.loc[contradictory_buy | contradictory_sell, 'signal_type'] = 'FILTERED'
        
        return df
    
    
    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """
        Obtener la señal más reciente
        
        Returns:
            Dict con información de la señal
        """
        if len(df) == 0:
            return {
                'signal': 0,
                'strength': 0.0,
                'type': 'NONE',
                'reason': 'No data'
            }
        
        latest = df.iloc[-1]
        
        return {
            'signal': int(latest['signal']),
            'strength': float(latest['signal_strength']),
            'type': str(latest['signal_type']),
            'reason': str(latest['entry_reason']),
            'price': float(latest['close']),
            'time': latest.get('time', 'N/A')
        }
    
    
    def get_signal_summary(self, df: pd.DataFrame, last_n: int = 100) -> Dict:
        """
        Obtener resumen de señales recientes
        
        Args:
            df: DataFrame con señales
            last_n: Número de velas a analizar
            
        Returns:
            Dict con estadísticas
        """
        recent_df = df.tail(last_n)
        
        buy_signals = (recent_df['signal'] == 1).sum()
        sell_signals = (recent_df['signal'] == -1).sum()
        
        avg_buy_strength = recent_df[recent_df['signal'] == 1]['signal_strength'].mean() if buy_signals > 0 else 0
        avg_sell_strength = recent_df[recent_df['signal'] == -1]['signal_strength'].mean() if sell_signals > 0 else 0
        
        return {
            'total_signals': buy_signals + sell_signals,
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'avg_buy_strength': float(avg_buy_strength),
            'avg_sell_strength': float(avg_sell_strength),
            'signal_rate': float((buy_signals + sell_signals) / len(recent_df) * 100)
        }


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    # Crear datos de prueba con características
    test_df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'rsi_14': np.random.uniform(20, 80, 100),
        'macd': np.random.randn(100),
        'macd_signal': np.random.randn(100),
        'macd_hist': np.random.randn(100),
        'bb_upper': np.random.randn(100).cumsum() + 102,
        'bb_lower': np.random.randn(100).cumsum() + 98,
        'structure_numeric': np.random.choice([-1, 0, 1], 100),
        'structure_strength': np.random.uniform(0.3, 0.9, 100),
        'has_bos': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'bos_direction': np.random.choice([-1, 0, 1], 100),
        'has_choch': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'choch_direction': np.random.choice([-1, 0, 1], 100),
        'in_demand_zone': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'in_supply_zone': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'zone_strength': np.random.uniform(0.4, 0.9, 100),
        'has_bullish_pattern': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'has_bearish_pattern': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'pattern_strength': np.random.uniform(0.5, 0.9, 100),
    })
    
    # Crear generador
    generator = SignalGenerator()
    
    # Generar señales
    result = generator.generate_signals(test_df)
    
    # Mostrar resultados
    print(f"\n✅ Test completado:")
    print(f"   - Total velas: {len(result)}")
    print(f"   - Señales BUY: {(result['signal'] == 1).sum()}")
    print(f"   - Señales SELL: {(result['signal'] == -1).sum()}")
    
    # Última señal
    latest = generator.get_latest_signal(result)
    print(f"\n📊 Última señal: {latest}")
    
    # Resumen
    summary = generator.get_signal_summary(result)
    print(f"\n📈 Resumen: {summary}")
