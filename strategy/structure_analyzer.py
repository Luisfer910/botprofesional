"""
STRUCTURE ANALYZER - Analizador de Estructura de Mercado
Detecta BOS (Break of Structure), CHoCH (Change of Character) y Order Blocks
"""

import pandas as pd
import numpy as np

class StructureAnalyzer:
    """
    Analiza la estructura del mercado según Smart Money Concepts:
    - BOS (Break of Structure): Ruptura de estructura
    - CHoCH (Change of Character): Cambio de carácter
    - Order Blocks: Bloques de órdenes institucionales
    - Swing Highs/Lows: Máximos y mínimos significativos
    """
    
    def __init__(self, swing_length=5):
        """
        Args:
            swing_length: Número de velas a cada lado para confirmar un swing
        """
        self.swing_length = swing_length
        self.swing_highs = []
        self.swing_lows = []
        self.order_blocks = []
        self.structure_breaks = []
    
    def analyze_structure(self, df):
        """
        Analiza la estructura completa del mercado
        
        Args:
            df: DataFrame con OHLC
            
        Returns:
            Dict con análisis completo de estructura
        """
        if len(df) < self.swing_length * 2 + 1:
            return {
                'trend': 'unknown',
                'last_bos': None,
                'last_choch': None,
                'order_blocks': [],
                'swing_highs': [],
                'swing_lows': []
            }
        
        # 1. Detectar Swing Highs y Swing Lows
        self._detect_swings(df)
        
        # 2. Determinar tendencia actual
        trend = self._determine_trend()
        
        # 3. Detectar BOS y CHoCH
        last_bos = self._detect_bos(df)
        last_choch = self._detect_choch(df)
        
        # 4. Identificar Order Blocks
        self._detect_order_blocks(df)
        
        # 5. Analizar estructura actual
        current_structure = self._analyze_current_structure(df)
        
        return {
            'trend': trend,
            'last_bos': last_bos,
            'last_choch': last_choch,
            'order_blocks': self.order_blocks[-5:],  # Últimos 5 OB
            'swing_highs': self.swing_highs[-10:],   # Últimos 10 swing highs
            'swing_lows': self.swing_lows[-10:],     # Últimos 10 swing lows
            'current_structure': current_structure,
            'structure_strength': self._calculate_structure_strength()
        }
    
    def _detect_swings(self, df):
        """Detecta Swing Highs y Swing Lows"""
        self.swing_highs = []
        self.swing_lows = []
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            # Swing High: El high más alto en la ventana
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                self.swing_highs.append({
                    'index': i,
                    'price': current_high,
                    'time': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
            
            # Swing Low: El low más bajo en la ventana
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                self.swing_lows.append({
                    'index': i,
                    'price': current_low,
                    'time': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
    
    def _determine_trend(self):
        """Determina la tendencia basada en Higher Highs/Lower Lows"""
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return 'unknown'
        
        # Últimos 3 swing highs y lows
        recent_highs = self.swing_highs[-3:]
        recent_lows = self.swing_lows[-3:]
        
        # Contar Higher Highs
        higher_highs = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs[i]['price'] > recent_highs[i-1]['price'])
        
        # Contar Higher Lows
        higher_lows = sum(1 for i in range(1, len(recent_lows)) 
                         if recent_lows[i]['price'] > recent_lows[i-1]['price'])
        
        # Contar Lower Highs
        lower_highs = sum(1 for i in range(1, len(recent_highs)) 
                         if recent_highs[i]['price'] < recent_highs[i-1]['price'])
        
        # Contar Lower Lows
        lower_lows = sum(1 for i in range(1, len(recent_lows)) 
                        if recent_lows[i]['price'] < recent_lows[i-1]['price'])
        
        # Determinar tendencia
        bullish_score = higher_highs + higher_lows
        bearish_score = lower_highs + lower_lows
        
        if bullish_score > bearish_score + 1:
            return 'bullish'
        elif bearish_score > bullish_score + 1:
            return 'bearish'
        else:
            return 'ranging'
    
    def _detect_bos(self, df):
        """
        Detecta Break of Structure (BOS)
        BOS = Ruptura de un swing high/low en dirección de la tendencia
        """
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None
        
        current_price = df.iloc[-1]['close']
        
        # BOS Alcista: Precio rompe último swing high
        last_swing_high = self.swing_highs[-1]
        if current_price > last_swing_high['price']:
            return {
                'type': 'BOS',
                'direction': 'bullish',
                'level': last_swing_high['price'],
                'broken_at': len(df) - 1,
                'strength': self._calculate_break_strength(df, last_swing_high['price'], 'high')
            }
        
        # BOS Bajista: Precio rompe último swing low
        last_swing_low = self.swing_lows[-1]
        if current_price < last_swing_low['price']:
            return {
                'type': 'BOS',
                'direction': 'bearish',
                'level': last_swing_low['price'],
                'broken_at': len(df) - 1,
                'strength': self._calculate_break_strength(df, last_swing_low['price'], 'low')
            }
        
        return None
    
    def _detect_choch(self, df):
        """
        Detecta Change of Character (CHoCH)
        CHoCH = Ruptura de estructura en dirección OPUESTA a la tendencia
        """
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None
        
        trend = self._determine_trend()
        current_price = df.iloc[-1]['close']
        
        # CHoCH en tendencia alcista: Precio rompe swing low reciente
        if trend == 'bullish':
            last_swing_low = self.swing_lows[-1]
            if current_price < last_swing_low['price']:
                return {
                    'type': 'CHoCH',
                    'direction': 'bearish',
                    'previous_trend': 'bullish',
                    'level': last_swing_low['price'],
                    'broken_at': len(df) - 1,
                    'strength': self._calculate_break_strength(df, last_swing_low['price'], 'low')
                }
        
        # CHoCH en tendencia bajista: Precio rompe swing high reciente
        elif trend == 'bearish':
            last_swing_high = self.swing_highs[-1]
            if current_price > last_swing_high['price']:
                return {
                    'type': 'CHoCH',
                    'direction': 'bullish',
                    'previous_trend': 'bearish',
                    'level': last_swing_high['price'],
                    'broken_at': len(df) - 1,
                    'strength': self._calculate_break_strength(df, last_swing_high['price'], 'high')
                }
        
        return None
    
    def _detect_order_blocks(self, df):
        """
        Detecta Order Blocks (bloques de órdenes institucionales)
        OB = Última vela antes de un movimiento impulsivo
        """
        self.order_blocks = []
        
        if len(df) < 10:
            return
        
        for i in range(5, len(df) - 5):
            # Detectar movimiento impulsivo alcista
            if self._is_bullish_impulse(df, i):
                # Order Block = Vela bajista antes del impulso
                ob_candle = df.iloc[i-1]
                if ob_candle['close'] < ob_candle['open']:
                    self.order_blocks.append({
                        'type': 'bullish_ob',
                        'index': i-1,
                        'high': ob_candle['high'],
                        'low': ob_candle['low'],
                        'strength': self._calculate_ob_strength(df, i),
                        'tested': False
                    })
            
            # Detectar movimiento impulsivo bajista
            elif self._is_bearish_impulse(df, i):
                # Order Block = Vela alcista antes del impulso
                ob_candle = df.iloc[i-1]
                if ob_candle['close'] > ob_candle['open']:
                    self.order_blocks.append({
                        'type': 'bearish_ob',
                        'index': i-1,
                        'high': ob_candle['high'],
                        'low': ob_candle['low'],
                        'strength': self._calculate_ob_strength(df, i),
                        'tested': False
                    })
    
    def _is_bullish_impulse(self, df, index):
        """Detecta si hay un impulso alcista desde el índice dado"""
        if index + 3 >= len(df):
            return False
        
        # Verificar que las siguientes 3 velas sean alcistas y crecientes
        impulse_strength = 0
        for i in range(index, min(index + 3, len(df))):
            candle = df.iloc[i]
            if candle['close'] > candle['open']:
                impulse_strength += (candle['close'] - candle['open'])
            else:
                return False
        
        # El impulso debe ser significativo (>1% del precio)
        avg_price = df.iloc[index:index+3]['close'].mean()
        return (impulse_strength / avg_price) > 0.01
    
    def _is_bearish_impulse(self, df, index):
        """Detecta si hay un impulso bajista desde el índice dado"""
        if index + 3 >= len(df):
            return False
        
        # Verificar que las siguientes 3 velas sean bajistas y decrecientes
        impulse_strength = 0
        for i in range(index, min(index + 3, len(df))):
            candle = df.iloc[i]
            if candle['close'] < candle['open']:
                impulse_strength += (candle['open'] - candle['close'])
            else:
                return False
        
        # El impulso debe ser significativo (>1% del precio)
        avg_price = df.iloc[index:index+3]['close'].mean()
        return (impulse_strength / avg_price) > 0.01
    
    def _calculate_break_strength(self, df, level, break_type):
        """Calcula la fuerza de una ruptura"""
        current_price = df.iloc[-1]['close']
        distance = abs(current_price - level) / level * 100
        
        # Verificar volumen (si está disponible)
        volume_strength = 0
        if 'volume' in df.columns:
            recent_volume = df.iloc[-1]['volume']
            avg_volume = df['volume'].tail(20).mean()
            volume_strength = min((recent_volume / avg_volume - 1) * 50, 50)
        
        return min(distance * 10 + volume_strength, 100)
    
    def _calculate_ob_strength(self, df, index):
        """Calcula la fuerza de un Order Block"""
        # Basado en el tamaño del impulso posterior
        if index + 3 >= len(df):
            return 50
        
        impulse_move = abs(df.iloc[index+3]['close'] - df.iloc[index]['close'])
        avg_move = df['close'].diff().abs().tail(20).mean()
        
        strength = min((impulse_move / avg_move) * 20, 100)
        return strength
    
    def _analyze_current_structure(self, df):
        """Analiza la estructura actual del mercado"""
        current_price = df.iloc[-1]['close']
        
        # Encontrar niveles clave cercanos
        nearest_resistance = None
        nearest_support = None
        
        for sh in reversed(self.swing_highs):
            if sh['price'] > current_price:
                nearest_resistance = sh
                break
        
        for sl in reversed(self.swing_lows):
            if sl['price'] < current_price:
                nearest_support = sl
                break
        
        return {
            'current_price': current_price,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'distance_to_resistance': ((nearest_resistance['price'] - current_price) / current_price * 100) if nearest_resistance else None,
            'distance_to_support': ((current_price - nearest_support['price']) / current_price * 100) if nearest_support else None
        }
    
    def _calculate_structure_strength(self):
        """Calcula la fuerza general de la estructura"""
        if len(self.swing_highs) < 3 or len(self.swing_lows) < 3:
            return 0
        
        # Consistencia de swings
        trend = self._determine_trend()
        
        if trend == 'bullish':
            # Verificar Higher Highs y Higher Lows consistentes
            consistency = 0
            for i in range(1, min(3, len(self.swing_highs))):
                if self.swing_highs[-i]['price'] < self.swing_highs[-i-1]['price']:
                    consistency += 1
            return (3 - consistency) / 3 * 100
        
        elif trend == 'bearish':
            # Verificar Lower Highs y Lower Lows consistentes
            consistency = 0
            for i in range(1, min(3, len(self.swing_lows))):
                if self.swing_lows[-i]['price'] > self.swing_lows[-i-1]['price']:
                    consistency += 1
            return (3 - consistency) / 3 * 100
        
        return 50  # Ranging
    
    def get_structure_summary(self, analysis):
        """Genera un resumen legible del análisis de estructura"""
        summary = f"=== ANÁLISIS DE ESTRUCTURA ===\n\n"
        summary += f"Tendencia: {analysis['trend'].upper()}\n"
        summary += f"Fuerza de estructura: {analysis['structure_strength']:.1f}%\n\n"
        
        if analysis['last_bos']:
            summary += f"Último BOS: {analysis['last_bos']['direction'].upper()} "
            summary += f"en {analysis['last_bos']['level']:.5f} "
            summary += f"(Fuerza: {analysis['last_bos']['strength']:.0f}%)\n"
        
        if analysis['last_choch']:
            summary += f"Último CHoCH: {analysis['last_choch']['direction'].upper()} "
            summary += f"en {analysis['last_choch']['level']:.5f} "
            summary += f"(Cambio desde {analysis['last_choch']['previous_trend']})\n"
        
        summary += f"\nOrder Blocks activos: {len(analysis['order_blocks'])}\n"
        summary += f"Swing Highs detectados: {len(analysis['swing_highs'])}\n"
        summary += f"Swing Lows detectados: {len(analysis['swing_lows'])}\n"
        
        cs = analysis['current_structure']
        if cs['nearest_resistance']:
            summary += f"\nResistencia más cercana: {cs['nearest_resistance']['price']:.5f} "
            summary += f"({cs['distance_to_resistance']:.2f}% arriba)\n"
        
        if cs['nearest_support']:
            summary += f"Soporte más cercano: {cs['nearest_support']['price']:.5f} "
            summary += f"({cs['distance_to_support']:.2f}% abajo)\n"
        
        return summary
