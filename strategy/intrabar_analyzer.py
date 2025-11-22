"""
INTRABAR ANALYZER - Analizador Intrabarra
Analiza el comportamiento interno de las velas para detectar movimientos institucionales
"""

import pandas as pd
import numpy as np

class IntrabarAnalyzer:
    """
    Analiza el comportamiento intrabarra para detectar:
    - Manipulación de precio (stop hunting)
    - Absorción de liquidez
    - Rechazo de niveles
    - Fuerza institucional
    """
    
    def __init__(self):
        self.manipulation_events = []
    
    def analyze_intrabar(self, candle, prev_candle=None):
        """
        Analiza una vela individual en detalle
        
        Args:
            candle: Serie con OHLC de la vela actual
            prev_candle: Serie con OHLC de la vela anterior (opcional)
            
        Returns:
            Dict con análisis intrabarra
        """
        analysis = {
            'wick_analysis': self._analyze_wicks(candle),
            'body_analysis': self._analyze_body(candle),
            'rejection': self._detect_rejection(candle),
            'manipulation': self._detect_manipulation(candle, prev_candle),
            'institutional_activity': self._detect_institutional_activity(candle),
            'score': 0
        }
        
        # Calcular score general
        analysis['score'] = self._calculate_intrabar_score(analysis)
        
        return analysis
    
    def _analyze_wicks(self, candle):
        """Analiza las mechas de la vela"""
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return {
                'upper_wick': 0,
                'lower_wick': 0,
                'upper_ratio': 0,
                'lower_ratio': 0,
                'dominant_wick': 'none'
            }
        
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        upper_ratio = upper_wick / total_range
        lower_ratio = lower_wick / total_range
        
        # Determinar mecha dominante
        if upper_ratio > 0.5 and upper_ratio > lower_ratio * 2:
            dominant = 'upper'
        elif lower_ratio > 0.5 and lower_ratio > upper_ratio * 2:
            dominant = 'lower'
        elif abs(upper_ratio - lower_ratio) < 0.1:
            dominant = 'balanced'
        else:
            dominant = 'none'
        
        return {
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'upper_ratio': upper_ratio,
            'lower_ratio': lower_ratio,
            'dominant_wick': dominant,
            'total_range': total_range
        }
    
    def _analyze_body(self, candle):
        """Analiza el cuerpo de la vela"""
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return {
                'size': 0,
                'ratio': 0,
                'direction': 'neutral',
                'strength': 'none'
            }
        
        body_ratio = body / total_range
        is_bullish = candle['close'] > candle['open']
        
        # Clasificar fuerza del cuerpo
        if body_ratio > 0.8:
            strength = 'very_strong'
        elif body_ratio > 0.6:
            strength = 'strong'
        elif body_ratio > 0.4:
            strength = 'medium'
        elif body_ratio > 0.2:
            strength = 'weak'
        else:
            strength = 'very_weak'
        
        return {
            'size': body,
            'ratio': body_ratio,
            'direction': 'bullish' if is_bullish else 'bearish',
            'strength': strength,
            'open': candle['open'],
            'close': candle['close']
        }
    
    def _detect_rejection(self, candle):
        """
        Detecta rechazos fuertes de niveles
        Rechazo = Mecha larga con cierre lejos del extremo
        """
        wick_analysis = self._analyze_wicks(candle)
        body_analysis = self._analyze_body(candle)
        
        rejections = []
        
        # Rechazo alcista (mecha inferior larga)
        if wick_analysis['lower_ratio'] > 0.5:
            rejection_strength = wick_analysis['lower_ratio'] * 100
            
            # Bonus si el cuerpo es alcista
            if body_analysis['direction'] == 'bullish':
                rejection_strength += 20
            
            rejections.append({
                'type': 'bullish_rejection',
                'level': candle['low'],
                'strength': min(rejection_strength, 100),
                'description': 'Fuerte rechazo alcista desde mínimos'
            })
        
        # Rechazo bajista (mecha superior larga)
        if wick_analysis['upper_ratio'] > 0.5:
            rejection_strength = wick_analysis['upper_ratio'] * 100
            
            # Bonus si el cuerpo es bajista
            if body_analysis['direction'] == 'bearish':
                rejection_strength += 20
            
            rejections.append({
                'type': 'bearish_rejection',
                'level': candle['high'],
                'strength': min(rejection_strength, 100),
                'description': 'Fuerte rechazo bajista desde máximos'
            })
        
        return rejections
    
    def _detect_manipulation(self, candle, prev_candle):
        """
        Detecta manipulación de precio (stop hunting)
        Manipulación = Movimiento rápido seguido de reversión
        """
        if prev_candle is None:
            return None
        
        manipulations = []
        
        # Manipulación alcista (sweep de lows)
        # Precio baja por debajo del mínimo anterior pero cierra arriba
        if (candle['low'] < prev_candle['low'] and 
            candle['close'] > prev_candle['close']):
            
            sweep_distance = (prev_candle['low'] - candle['low']) / prev_candle['low'] * 100
            recovery = (candle['close'] - candle['low']) / (candle['high'] - candle['low'])
            
            if recovery > 0.7:  # Recuperó más del 70% del rango
                manipulations.append({
                    'type': 'bullish_manipulation',
                    'swept_level': prev_candle['low'],
                    'lowest_point': candle['low'],
                    'sweep_distance': sweep_distance,
                    'recovery_ratio': recovery,
                    'strength': min(recovery * 100, 100),
                    'description': 'Manipulación alcista - Barrida de stops vendedores'
                })
        
        # Manipulación bajista (sweep de highs)
        # Precio sube por encima del máximo anterior pero cierra abajo
        if (candle['high'] > prev_candle['high'] and 
            candle['close'] < prev_candle['close']):
            
            sweep_distance = (candle['high'] - prev_candle['high']) / prev_candle['high'] * 100
            recovery = (candle['high'] - candle['close']) / (candle['high'] - candle['low'])
            
            if recovery > 0.7:  # Recuperó más del 70% del rango
                manipulations.append({
                    'type': 'bearish_manipulation',
                    'swept_level': prev_candle['high'],
                    'highest_point': candle['high'],
                    'sweep_distance': sweep_distance,
                    'recovery_ratio': recovery,
                    'strength': min(recovery * 100, 100),
                    'description': 'Manipulación bajista - Barrida de stops compradores'
                })
        
        return manipulations if manipulations else None
    
    def _detect_institutional_activity(self, candle):
        """
        Detecta actividad institucional basada en características de la vela
        """
        wick_analysis = self._analyze_wicks(candle)
        body_analysis = self._analyze_body(candle)
        
        signals = []
        
        # 1. Absorción (cuerpo grande con mechas pequeñas)
        if body_analysis['ratio'] > 0.8:
            signals.append({
                'type': 'absorption',
                'direction': body_analysis['direction'],
                'strength': body_analysis['ratio'] * 100,
                'description': f'Fuerte absorción {body_analysis["direction"]}'
            })
        
        # 2. Indecisión institucional (mechas largas balanceadas)
        if (wick_analysis['dominant_wick'] == 'balanced' and 
            wick_analysis['upper_ratio'] > 0.3 and 
            wick_analysis['lower_ratio'] > 0.3):
            signals.append({
                'type': 'indecision',
                'direction': 'neutral',
                'strength': 60,
                'description': 'Indecisión institucional - Batalla entre compradores y vendedores'
            })
        
        # 3. Acumulación/Distribución (cuerpo pequeño con volumen)
        if body_analysis['ratio'] < 0.2:
            if 'volume' in candle and candle['volume'] > 0:
                signals.append({
                    'type': 'accumulation_distribution',
                    'direction': 'neutral',
                    'strength': 50,
                    'description': 'Posible acumulación o distribución institucional'
                })
        
        return signals if signals else None
    
    def _calculate_intrabar_score(self, analysis):
        """Calcula un score general del análisis intrabarra"""
        score = 50  # Score base
        
        # Bonus por rechazos fuertes
        if analysis['rejection']:
            max_rejection_strength = max([r['strength'] for r in analysis['rejection']])
            score += max_rejection_strength * 0.3
        
        # Bonus por manipulación detectada
        if analysis['manipulation']:
            max_manipulation_strength = max([m['strength'] for m in analysis['manipulation']])
            score += max_manipulation_strength * 0.4
        
        # Bonus por actividad institucional
        if analysis['institutional_activity']:
            score += len(analysis['institutional_activity']) * 10
        
        # Penalización por indecisión
        if analysis['body_analysis']['strength'] == 'very_weak':
            score -= 20
        
        return min(max(score, 0), 100)
    
    def get_intrabar_summary(self, analysis):
        """Genera un resumen legible del análisis intrabarra"""
        summary = "=== ANÁLISIS INTRABARRA ===\n\n"
        
        # Análisis de mechas
        wick = analysis['wick_analysis']
        summary += f"Mechas:\n"
        summary += f"  Superior: {wick['upper_ratio']*100:.1f}% del rango\n"
        summary += f"  Inferior: {wick['lower_ratio']*100:.1f}% del rango\n"
        summary += f"  Dominante: {wick['dominant_wick']}\n\n"
        
        # Análisis de cuerpo
        body = analysis['body_analysis']
        summary += f"Cuerpo:\n"
        summary += f"  Dirección: {body['direction'].upper()}\n"
        summary += f"  Fuerza: {body['strength']}\n"
        summary += f"  Ratio: {body['ratio']*100:.1f}%\n\n"
        
        # Rechazos
        if analysis['rejection']:
            summary += "Rechazos detectados:\n"
            for rej in analysis['rejection']:
                summary += f"  - {rej['type']}: {rej['description']} (Fuerza: {rej['strength']:.0f}%)\n"
            summary += "\n"
        
        # Manipulación
        if analysis['manipulation']:
            summary += "⚠️ MANIPULACIÓN DETECTADA:\n"
            for manip in analysis['manipulation']:
                summary += f"  - {manip['type']}: {manip['description']}\n"
                summary += f"    Distancia barrida: {manip['sweep_distance']:.2f}%\n"
                summary += f"    Recuperación: {manip['recovery_ratio']*100:.0f}%\n"
            summary += "\n"
        
        # Actividad institucional
        if analysis['institutional_activity']:
            summary += "🏦 Actividad Institucional:\n"
            for signal in analysis['institutional_activity']:
                summary += f"  - {signal['type']}: {signal['description']}\n"
            summary += "\n"
        
        summary += f"Score General: {analysis['score']:.0f}/100\n"
        
        return summary
