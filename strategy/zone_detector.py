
import pandas as pd
import numpy as np
from datetime import datetime

class ZoneDetector:
    """
    Detecta zonas de soporte y resistencia basándose en:
    - Múltiples toques del precio
    - Fuerza de rechazo (mechas largas)
    - Velas de impulso desde la zona
    """
    
    def __init__(self, tolerance_pips=5):
        """
        Args:
            tolerance_pips: Tolerancia en pips para considerar que el precio tocó una zona
        """
        self.tolerance_pips = tolerance_pips
        self.zones = []
        
    def detect_zones(self, df, lookback=200):
        """
        Detecta zonas de soporte y resistencia en el DataFrame
        
        Args:
            df: DataFrame con columnas ['open', 'high', 'low', 'close']
            lookback: Cantidad de velas a analizar hacia atrás
            
        Returns:
            Lista de zonas detectadas con información detallada
        """
        if len(df) < lookback:
            lookback = len(df)
        
        df_analysis = df.tail(lookback).copy()
        
        # Detectar máximos y mínimos locales
        swing_highs = self._find_swing_highs(df_analysis)
        swing_lows = self._find_swing_lows(df_analysis)
        
        # Agrupar niveles cercanos en zonas
        resistance_zones = self._group_into_zones(swing_highs, df_analysis)
        support_zones = self._group_into_zones(swing_lows, df_analysis)
        
        # Evaluar fuerza de cada zona
        for zone in resistance_zones:
            zone['type'] = 'resistance'
            zone['score'] = self._calculate_zone_strength(zone, df_analysis, 'resistance')
        
        for zone in support_zones:
            zone['type'] = 'support'
            zone['score'] = self._calculate_zone_strength(zone, df_analysis, 'support')
        
        # Combinar y ordenar por score
        all_zones = resistance_zones + support_zones
        all_zones.sort(key=lambda x: x['score'], reverse=True)
        
        self.zones = all_zones
        return all_zones
    
    def _find_swing_highs(self, df, window=5):
        """Encuentra máximos locales (swing highs)"""
        swing_highs = []
        
        for i in range(window, len(df) - window):
            current_high = df['high'].iloc[i]
            
            # Verificar si es máximo local
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'price': current_high,
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
        
        return swing_highs
    
    def _find_swing_lows(self, df, window=5):
        """Encuentra mínimos locales (swing lows)"""
        swing_lows = []
        
        for i in range(window, len(df) - window):
            current_low = df['low'].iloc[i]
            
            # Verificar si es mínimo local
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'price': current_low,
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
        
        return swing_lows
    
    def _group_into_zones(self, swing_points, df):
        """Agrupa swing points cercanos en zonas"""
        if not swing_points:
            return []
        
        # Ordenar por precio
        swing_points.sort(key=lambda x: x['price'])
        
        zones = []
        current_zone = [swing_points[0]]
        
        for i in range(1, len(swing_points)):
            # Calcular distancia en pips
            distance_pips = abs(swing_points[i]['price'] - current_zone[-1]['price']) * 10000
            
            if distance_pips <= self.tolerance_pips:
                # Agregar a zona actual
                current_zone.append(swing_points[i])
            else:
                # Crear nueva zona
                if len(current_zone) >= 2:  # Solo zonas con al menos 2 toques
                    zones.append(self._create_zone_from_points(current_zone, df))
                current_zone = [swing_points[i]]
        
        # Agregar última zona
        if len(current_zone) >= 2:
            zones.append(self._create_zone_from_points(current_zone, df))
        
        return zones
    
    def _create_zone_from_points(self, points, df):
        """Crea un objeto zona a partir de puntos"""
        prices = [p['price'] for p in points]
        
        return {
            'price_min': min(prices),
            'price_max': max(prices),
            'price_avg': np.mean(prices),
            'touches': len(points),
            'points': points,
            'first_touch': points[0]['timestamp'],
            'last_touch': points[-1]['timestamp']
        }
    
    def _calculate_zone_strength(self, zone, df, zone_type):
        """
        Calcula la fuerza de una zona basándose en:
        - Número de toques
        - Fuerza de rechazo (tamaño de mechas)
        - Velas de impulso desde la zona
        """
        score = 0
        
        # 1. Puntos por número de toques (más toques = más fuerte)
        score += zone['touches'] * 10
        
        # 2. Puntos por fuerza de rechazo
        rejection_strength = 0
        for point in zone['points']:
            idx = point['index']
            if idx < len(df):
                candle = df.iloc[idx]
                
                if zone_type == 'resistance':
                    # Mecha superior grande = rechazo fuerte
                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    body = abs(candle['close'] - candle['open'])
                    if body > 0:
                        wick_ratio = upper_wick / body
                        rejection_strength += min(wick_ratio * 5, 20)  # Max 20 puntos por vela
                
                elif zone_type == 'support':
                    # Mecha inferior grande = rechazo fuerte
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    body = abs(candle['close'] - candle['open'])
                    if body > 0:
                        wick_ratio = lower_wick / body
                        rejection_strength += min(wick_ratio * 5, 20)
        
        score += rejection_strength
        
        # 3. Puntos por velas de impulso
        impulse_count = 0
        for point in zone['points']:
            idx = point['index']
            if idx < len(df) - 1:
                # Verificar si la siguiente vela es de impulso
                next_candle = df.iloc[idx + 1]
                body = abs(next_candle['close'] - next_candle['open'])
                candle_range = next_candle['high'] - next_candle['low']
                
                if candle_range > 0:
                    body_ratio = body / candle_range
                    
                    # Vela de impulso: cuerpo > 70% del rango
                    if body_ratio > 0.7:
                        if zone_type == 'resistance' and next_candle['close'] < next_candle['open']:
                            impulse_count += 1
                        elif zone_type == 'support' and next_candle['close'] > next_candle['open']:
                            impulse_count += 1
        
        score += impulse_count * 15
        
        return score
    
    def get_nearest_zones(self, current_price, max_distance_pips=50):
        """
        Obtiene las zonas más cercanas al precio actual
        
        Args:
            current_price: Precio actual
            max_distance_pips: Distancia máxima en pips
            
        Returns:
            Dict con zonas de soporte y resistencia más cercanas
        """
        nearest_support = None
        nearest_resistance = None
        
        min_support_dist = float('inf')
        min_resistance_dist = float('inf')
        
        for zone in self.zones:
            distance_pips = abs(zone['price_avg'] - current_price) * 10000
            
            if distance_pips > max_distance_pips:
                continue
            
            if zone['type'] == 'support' and zone['price_avg'] < current_price:
                if distance_pips < min_support_dist:
                    min_support_dist = distance_pips
                    nearest_support = {
                        **zone,
                        'distance_pips': distance_pips
                    }
            
            elif zone['type'] == 'resistance' and zone['price_avg'] > current_price:
                if distance_pips < min_resistance_dist:
                    min_resistance_dist = distance_pips
                    nearest_resistance = {
                        **zone,
                        'distance_pips': distance_pips
                    }
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    
    def get_zone_summary(self):
        """Retorna un resumen de las zonas detectadas"""
        if not self.zones:
            return "No se han detectado zonas"
        
        resistances = [z for z in self.zones if z['type'] == 'resistance']
        supports = [z for z in self.zones if z['type'] == 'support']
        
        summary = f"Zonas detectadas:\\n"
        summary += f"  Resistencias: {len(resistances)}\\n"
        summary += f"  Soportes: {len(supports)}\\n\\n"
        
        summary += "Top 5 zonas más fuertes:\\n"
        for i, zone in enumerate(self.zones[:5], 1):
            summary += f"  {i}. {zone['type'].upper()} @ {zone['price_avg']:.5f} "
            summary += f"(Score: {zone['score']:.0f}, Toques: {zone['touches']})\\n"
        
        return summary
