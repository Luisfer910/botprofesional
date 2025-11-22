"""
CANDLE PATTERN ANALYZER - Analizador de Patrones de Velas
Detecta patrones de velas japonesas y analiza su contexto
"""

import pandas as pd
import numpy as np

class CandlePatternAnalyzer:
    """
    Detecta y analiza patrones de velas japonesas:
    - Patrones de reversión (engulfing, hammer, shooting star, etc.)
    - Patrones de continuación (marubozu, three white soldiers, etc.)
    - Análisis de contexto (ubicación en tendencia, volumen, etc.)
    """
    
    def __init__(self):
        self.patterns_detected = []
    
    def analyze_candle(self, df, index=-1):
        """
        Analiza la vela en el índice dado y detecta patrones
        
        Args:
            df: DataFrame con OHLC
            index: Índice de la vela a analizar (-1 = última vela)
            
        Returns:
            Dict con patrones detectados y análisis
        """
        if len(df) < 3:
            return {'patterns': [], 'analysis': 'Datos insuficientes'}
        
        # Obtener velas necesarias
        if index == -1:
            current = df.iloc[-1]
            prev1 = df.iloc[-2] if len(df) > 1 else None
            prev2 = df.iloc[-3] if len(df) > 2 else None
        else:
            current = df.iloc[index]
            prev1 = df.iloc[index-1] if index > 0 else None
            prev2 = df.iloc[index-2] if index > 1 else None
        
        patterns = []
        
        # Detectar patrones de 1 vela
        patterns.extend(self._detect_single_candle_patterns(current))
        
        # Detectar patrones de 2 velas
        if prev1 is not None:
            patterns.extend(self._detect_two_candle_patterns(prev1, current))
        
        # Detectar patrones de 3 velas
        if prev2 is not None and prev1 is not None:
            patterns.extend(self._detect_three_candle_patterns(prev2, prev1, current))
        
        # Analizar contexto
        context = self._analyze_context(df, index)
        
        # Calcular score de cada patrón basado en contexto
        for pattern in patterns:
            pattern['score'] = self._calculate_pattern_score(pattern, context)
        
        # Ordenar por score
        patterns.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'patterns': patterns,
            'context': context,
            'candle_info': self._get_candle_info(current)
        }
    
    def _detect_single_candle_patterns(self, candle):
        """Detecta patrones de una sola vela"""
        patterns = []
        
        body = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return patterns
        
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        # DOJI
        if body_ratio < 0.1:
            patterns.append({
                'name': 'Doji',
                'type': 'indecision',
                'direction': 'neutral',
                'strength': 'medium',
                'description': 'Indecisión del mercado'
            })
        
        # HAMMER (Martillo)
        if body_ratio > 0.2 and lower_wick > body * 2 and upper_wick < body * 0.3:
            patterns.append({
                'name': 'Hammer',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 'strong',
                'description': 'Posible reversión alcista'
            })
        
        # SHOOTING STAR (Estrella Fugaz)
        if body_ratio > 0.2 and upper_wick > body * 2 and lower_wick < body * 0.3:
            patterns.append({
                'name': 'Shooting Star',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 'strong',
                'description': 'Posible reversión bajista'
            })
        
        # MARUBOZU (Vela sin mechas)
        if body_ratio > 0.9:
            direction = 'bullish' if candle['close'] > candle['open'] else 'bearish'
            patterns.append({
                'name': 'Marubozu',
                'type': 'continuation',
                'direction': direction,
                'strength': 'strong',
                'description': f'Fuerte presión {direction}'
            })
        
        # SPINNING TOP (Peonza)
        if 0.1 < body_ratio < 0.3 and upper_wick > body and lower_wick > body:
            patterns.append({
                'name': 'Spinning Top',
                'type': 'indecision',
                'direction': 'neutral',
                'strength': 'weak',
                'description': 'Indecisión con volatilidad'
            })
        
        return patterns
    
    def _detect_two_candle_patterns(self, prev, current):
        """Detecta patrones de dos velas"""
        patterns = []
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(current['close'] - current['open'])
        
        prev_is_bullish = prev['close'] > prev['open']
        curr_is_bullish = current['close'] > current['open']
        
        # BULLISH ENGULFING (Envolvente Alcista)
        if (not prev_is_bullish and curr_is_bullish and
            current['open'] < prev['close'] and
            current['close'] > prev['open'] and
            curr_body > prev_body * 1.2):
            patterns.append({
                'name': 'Bullish Engulfing',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 'very_strong',
                'description': 'Fuerte reversión alcista'
            })
        
        # BEARISH ENGULFING (Envolvente Bajista)
        if (prev_is_bullish and not curr_is_bullish and
            current['open'] > prev['close'] and
            current['close'] < prev['open'] and
            curr_body > prev_body * 1.2):
            patterns.append({
                'name': 'Bearish Engulfing',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 'very_strong',
                'description': 'Fuerte reversión bajista'
            })
        
        # PIERCING PATTERN (Patrón Penetrante)
        if (not prev_is_bullish and curr_is_bullish and
            current['open'] < prev['low'] and
            current['close'] > (prev['open'] + prev['close']) / 2 and
            current['close'] < prev['open']):
            patterns.append({
                'name': 'Piercing Pattern',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 'strong',
                'description': 'Reversión alcista penetrante'
            })
        
        # DARK CLOUD COVER (Nube Oscura)
        if (prev_is_bullish and not curr_is_bullish and
            current['open'] > prev['high'] and
            current['close'] < (prev['open'] + prev['close']) / 2 and
            current['close'] > prev['open']):
            patterns.append({
                'name': 'Dark Cloud Cover',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 'strong',
                'description': 'Reversión bajista con nube oscura'
            })
        
        return patterns
    
    def _detect_three_candle_patterns(self, candle1, candle2, candle3):
        """Detecta patrones de tres velas"""
        patterns = []
        
        c1_bullish = candle1['close'] > candle1['open']
        c2_bullish = candle2['close'] > candle2['open']
        c3_bullish = candle3['close'] > candle3['open']
        
        # THREE WHITE SOLDIERS (Tres Soldados Blancos)
        if c1_bullish and c2_bullish and c3_bullish:
            if (candle2['close'] > candle1['close'] and
                candle3['close'] > candle2['close'] and
                candle2['open'] > candle1['open'] and
                candle3['open'] > candle2['open']):
                patterns.append({
                    'name': 'Three White Soldiers',
                    'type': 'continuation',
                    'direction': 'bullish',
                    'strength': 'very_strong',
                    'description': 'Fuerte continuación alcista'
                })
        
        # THREE BLACK CROWS (Tres Cuervos Negros)
        if not c1_bullish and not c2_bullish and not c3_bullish:
            if (candle2['close'] < candle1['close'] and
                candle3['close'] < candle2['close'] and
                candle2['open'] < candle1['open'] and
                candle3['open'] < candle2['open']):
                patterns.append({
                    'name': 'Three Black Crows',
                    'type': 'continuation',
                    'direction': 'bearish',
                    'strength': 'very_strong',
                    'description': 'Fuerte continuación bajista'
                })
        
        # MORNING STAR (Estrella de la Mañana)
        if (not c1_bullish and c3_bullish and
            abs(candle2['close'] - candle2['open']) < abs(candle1['close'] - candle1['open']) * 0.3):
            if (candle2['close'] < candle1['close'] and
                candle3['close'] > (candle1['open'] + candle1['close']) / 2):
                patterns.append({
                    'name': 'Morning Star',
                    'type': 'reversal',
                    'direction': 'bullish',
                    'strength': 'very_strong',
                    'description': 'Reversión alcista - Estrella de la mañana'
                })
        
        # EVENING STAR (Estrella de la Tarde)
        if (c1_bullish and not c3_bullish and
            abs(candle2['close'] - candle2['open']) < abs(candle1['close'] - candle1['open']) * 0.3):
            if (candle2['close'] > candle1['close'] and
                candle3['close'] < (candle1['open'] + candle1['close']) / 2):
                patterns.append({
                    'name': 'Evening Star',
                    'type': 'reversal',
                    'direction': 'bearish',
                    'strength': 'very_strong',
                    'description': 'Reversión bajista - Estrella de la tarde'
                })
        
        return patterns
    
    def _analyze_context(self, df, index):
        """Analiza el contexto de mercado"""
        if len(df) < 20:
            return {'trend': 'unknown', 'strength': 0}
        
        # Usar últimas 20 velas para determinar tendencia
        recent_df = df.tail(20) if index == -1 else df.iloc[max(0, index-19):index+1]
        
        # Calcular SMA simple
        sma_short = recent_df['close'].tail(5).mean()
        sma_long = recent_df['close'].mean()
        
        # Determinar tendencia
        if sma_short > sma_long * 1.001:
            trend = 'uptrend'
        elif sma_short < sma_long * 0.999:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Calcular fuerza de tendencia (basado en pendiente)
        prices = recent_df['close'].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        strength = abs(slope) / prices[-1] * 100  # Pendiente como % del precio
        
        return {
            'trend': trend,
            'strength': min(strength, 100),
            'sma_short': sma_short,
            'sma_long': sma_long
        }
    
    def _calculate_pattern_score(self, pattern, context):
        """Calcula el score de un patrón basado en el contexto"""
        score = 0
        
        # Score base por fuerza del patrón
        strength_scores = {
            'weak': 20,
            'medium': 40,
            'strong': 60,
            'very_strong': 80
        }
        score += strength_scores.get(pattern['strength'], 40)
        
        # Bonus si el patrón está alineado con la tendencia
        if pattern['type'] == 'continuation':
            if (pattern['direction'] == 'bullish' and context['trend'] == 'uptrend') or \
               (pattern['direction'] == 'bearish' and context['trend'] == 'downtrend'):
                score += 20
        
        # Bonus si es reversión contra tendencia fuerte
        if pattern['type'] == 'reversal':
            if (pattern['direction'] == 'bullish' and context['trend'] == 'downtrend') or \
               (pattern['direction'] == 'bearish' and context['trend'] == 'uptrend'):
                score += context['strength'] * 0.3
        
        return score
    
    def _get_candle_info(self, candle):
        """Obtiene información básica de la vela"""
        body = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return {
            'is_bullish': candle['close'] > candle['open'],
            'body_size': body,
            'total_range': candle_range,
            'body_ratio': body / candle_range if candle_range > 0 else 0,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'upper_wick_ratio': upper_wick / candle_range if candle_range > 0 else 0,
            'lower_wick_ratio': lower_wick / candle_range if candle_range > 0 else 0
        }
    
    def get_pattern_summary(self, analysis):
        """Genera un resumen legible del análisis"""
        if not analysis['patterns']:
            return "No se detectaron patrones significativos"
        
        summary = f"Contexto: {analysis['context']['trend'].upper()}\n"
        summary += f"Fuerza de tendencia: {analysis['context']['strength']:.1f}%\n\n"
        summary += "Patrones detectados:\n"
        
        for i, pattern in enumerate(analysis['patterns'][:3], 1):
            summary += f"  {i}. {pattern['name']} ({pattern['direction'].upper()})\n"
            summary += f"     Tipo: {pattern['type']} | Score: {pattern['score']:.0f}\n"
            summary += f"     {pattern['description']}\n"
        
        return summary
