"""
Signal Generator - Generador de señales de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generador de señales de trading basado en:
    - Estructura de mercado
    - Zonas de oferta/demanda
    - Patrones de velas
    - Indicadores técnicos
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar generador de señales
        
        Args:
            config: Configuración del generador
        """
        self.config = config or {}
        
        # Umbrales de señal
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        logger.info(f"✅ SignalGenerator inicializado (min_confidence={self.min_confidence})")
    
    
    def generate_signal(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """
        Generar señal de trading
        
        Args:
            df: DataFrame con datos OHLCV
            features: DataFrame con características
            
        Returns:
            Diccionario con la señal
        """
        try:
            if len(features) == 0:
                return self._no_signal()
            
            # Obtener última fila de características
            last_row = features.iloc[-1]
            
            # Calcular confianza alcista y bajista
            bullish_confidence = self._calculate_bullish_confidence(last_row)
            bearish_confidence = self._calculate_bearish_confidence(last_row)
            
            # Determinar señal
            if bullish_confidence > self.min_confidence and bullish_confidence > bearish_confidence:
                signal_type = 'BUY'
                confidence = bullish_confidence
            elif bearish_confidence > self.min_confidence and bearish_confidence > bullish_confidence:
                signal_type = 'SELL'
                confidence = bearish_confidence
            else:
                return self._no_signal()
            
            # Calcular niveles de entrada, SL y TP
            entry_price = df['close'].iloc[-1]
            atr = last_row.get('atr_14', 0.001)
            
            if signal_type == 'BUY':
                stop_loss = entry_price - (2 * atr)
                take_profit = entry_price + (3 * atr)
            else:  # SELL
                stop_loss = entry_price + (2 * atr)
                take_profit = entry_price - (3 * atr)
            
            # Crear señal
            signal = {
                'type': signal_type,
                'confidence': float(confidence),
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'timestamp': df['time'].iloc[-1] if 'time' in df.columns else pd.Timestamp.now(),
                'reasons': self._get_signal_reasons(last_row, signal_type)
            }
            
            logger.info(f"🎯 Señal generada: {signal_type} (confianza={confidence:.2%})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generando señal: {e}")
            return self._no_signal()
    
    
    def _calculate_bullish_confidence(self, row: pd.Series) -> float:
        """Calcular confianza alcista"""
        confidence = 0.0
        weight_sum = 0.0
        
        # Estructura alcista (peso: 0.25)
        if row.get('structure_numeric', 0) > 0:
            confidence += 0.25 * row.get('structure_strength', 0.5)
        weight_sum += 0.25
        
        # En zona de demanda (peso: 0.20)
        if row.get('in_demand_zone', 0) == 1:
            confidence += 0.20 * row.get('zone_strength', 0.5)
        weight_sum += 0.20
        
        # Patrón alcista (peso: 0.15)
        if row.get('has_bullish_pattern', 0) == 1:
            confidence += 0.15 * row.get('pattern_strength', 0.5)
        weight_sum += 0.15
        
        # RSI sobreventa (peso: 0.15)
        rsi = row.get('rsi_14', 50)
        if rsi < self.rsi_oversold:
            confidence += 0.15 * (1 - rsi / 100)
        weight_sum += 0.15
        
        # MACD alcista (peso: 0.10)
        if row.get('macd_hist', 0) > 0:
            confidence += 0.10
        weight_sum += 0.10
        
        # Precio sobre EMA 20 (peso: 0.10)
        if row.get('close', 0) > row.get('ema_20', 0):
            confidence += 0.10
        weight_sum += 0.10
        
        # BOS alcista reciente (peso: 0.05)
        if row.get('bars_since_bos', 999) < 10 and row.get('bos_direction', 0) > 0:
            confidence += 0.05
        weight_sum += 0.05
        
        return confidence / weight_sum if weight_sum > 0 else 0.0
    
    
    def _calculate_bearish_confidence(self, row: pd.Series) -> float:
        """Calcular confianza bajista"""
        confidence = 0.0
        weight_sum = 0.0
        
        # Estructura bajista (peso: 0.25)
        if row.get('structure_numeric', 0) < 0:
            confidence += 0.25 * row.get('structure_strength', 0.5)
        weight_sum += 0.25
        
        # En zona de oferta (peso: 0.20)
        if row.get('in_supply_zone', 0) == 1:
            confidence += 0.20 * row.get('zone_strength', 0.5)
        weight_sum += 0.20
        
        # Patrón bajista (peso: 0.15)
        if row.get('has_bearish_pattern', 0) == 1:
            confidence += 0.15 * row.get('pattern_strength', 0.5)
        weight_sum += 0.15
        
        # RSI sobrecompra (peso: 0.15)
        rsi = row.get('rsi_14', 50)
        if rsi > self.rsi_overbought:
            confidence += 0.15 * (rsi / 100)
        weight_sum += 0.15
        
        # MACD bajista (peso: 0.10)
        if row.get('macd_hist', 0) < 0:
            confidence += 0.10
        weight_sum += 0.10
        
        # Precio bajo EMA 20 (peso: 0.10)
        if row.get('close', 0) < row.get('ema_20', 0):
            confidence += 0.10
        weight_sum += 0.10
        
        # BOS bajista reciente (peso: 0.05)
        if row.get('bars_since_bos', 999) < 10 and row.get('bos_direction', 0) < 0:
            confidence += 0.05
        weight_sum += 0.05
        
        return confidence / weight_sum if weight_sum > 0 else 0.0
    
    
    def _get_signal_reasons(self, row: pd.Series, signal_type: str) -> List[str]:
        """Obtener razones de la señal"""
        reasons = []
        
        if signal_type == 'BUY':
            if row.get('structure_numeric', 0) > 0:
                reasons.append("Estructura alcista")
            if row.get('in_demand_zone', 0) == 1:
                reasons.append("En zona de demanda")
            if row.get('has_bullish_pattern', 0) == 1:
                reasons.append(f"Patrón alcista: {row.get('pattern_type', 'N/A')}")
            if row.get('rsi_14', 50) < self.rsi_oversold:
                reasons.append("RSI en sobreventa")
            if row.get('macd_hist', 0) > 0:
                reasons.append("MACD alcista")
        else:  # SELL
            if row.get('structure_numeric', 0) < 0:
                reasons.append("Estructura bajista")
            if row.get('in_supply_zone', 0) == 1:
                reasons.append("En zona de oferta")
            if row.get('has_bearish_pattern', 0) == 1:
                reasons.append(f"Patrón bajista: {row.get('pattern_type', 'N/A')}")
            if row.get('rsi_14', 50) > self.rsi_overbought:
                reasons.append("RSI en sobrecompra")
            if row.get('macd_hist', 0) < 0:
                reasons.append("MACD bajista")
        
        return reasons
    
    
    def _no_signal(self) -> Dict:
        """Retornar señal nula"""
        return {
            'type': 'NONE',
            'confidence': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reasons': []
        }
