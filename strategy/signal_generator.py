"""
Signal Generator - Generador de señales de trading
Versión híbrida: Soporta modelo ML + análisis técnico
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generador de señales de trading basado en:
    - Modelo de Machine Learning (si está disponible)
    - Estructura de mercado
    - Zonas de oferta/demanda
    - Patrones de velas
    - Indicadores técnicos
    """
    
    def __init__(self, modelo=None, feature_engineer=None, config: Optional[Dict] = None):
        """
        Inicializar generador de señales
        
        Args:
            modelo: Modelo de ML entrenado (opcional)
            feature_engineer: FeatureEngineer para generar features (opcional)
            config: Configuración del generador
        """
        self.modelo = modelo
        self.feature_engineer = feature_engineer
        self.config = config or {}
        
        # Umbrales de señal
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        # Modo de operación
        self.usar_ml = modelo is not None
        
        if self.usar_ml:
            logger.info(f"✅ SignalGenerator inicializado con MODELO ML")
        else:
            logger.info(f"✅ SignalGenerator inicializado con ANÁLISIS TÉCNICO")
        
        logger.info(f"   min_confidence={self.min_confidence}")
    
    
    def generar_señal(self, df: pd.DataFrame) -> Dict:
        """
        Generar señal de trading (método principal para main.py)
        
        Args:
            df: DataFrame con datos OHLCV y features
            
        Returns:
            Diccionario con la señal
        """
        # Si hay modelo ML, usarlo
        if self.usar_ml:
            return self._generar_señal_ml(df)
        else:
            # Fallback a análisis técnico
            return self._generar_señal_tecnica(df)
    
    
    def _generar_señal_ml(self, df: pd.DataFrame) -> Dict:
        """
        Generar señal usando modelo ML
        """
        try:
            if df is None or len(df) < 100:
                logger.warning("Datos insuficientes para generar señal")
                return self._no_signal()
            
            # Preparar features
            ultima_fila = df.iloc[-1:].copy()
            X = self._preparar_features(ultima_fila)
            
            if X is None or len(X) == 0:
                logger.error("Error al preparar features")
                return self._no_signal()
            
            # Hacer predicción
            prediccion = self.modelo.predict(X)[0]
            
            # Obtener probabilidades
            if hasattr(self.modelo, 'predict_proba'):
                probabilidades = self.modelo.predict_proba(X)[0]
                
                # Clasificación multi-clase: -1=SELL, 0=NEUTRAL, 1=BUY
                if len(probabilidades) == 3:
                    prob_sell = probabilidades[0]
                    prob_neutral = probabilidades[1]
                    prob_buy = probabilidades[2]
                # Clasificación binaria: 0=SELL, 1=BUY
                elif len(probabilidades) == 2:
                    prob_sell = probabilidades[0]
                    prob_buy = probabilidades[1]
                else:
                    prob_buy = probabilidades[int(prediccion)]
                    prob_sell = 1 - prob_buy
            else:
                # Sin probabilidades
                if prediccion == 1:
                    prob_buy = 0.8
                    prob_sell = 0.2
                elif prediccion == -1:
                    prob_buy = 0.2
                    prob_sell = 0.8
                else:
                    return self._no_signal()
            
            # Determinar tipo de señal
            if prob_buy > self.min_confidence and prob_buy > prob_sell:
                signal_type = 'CALL'  # Compatibilidad con main.py
                confidence = prob_buy
            elif prob_sell > self.min_confidence and prob_sell > prob_buy:
                signal_type = 'PUT'  # Compatibilidad con main.py
                confidence = prob_sell
            else:
                return self._no_signal()
            
            # Calcular niveles
            entry_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else entry_price * 0.001
            
            if signal_type == 'CALL':
                stop_loss = entry_price - (2 * atr)
                take_profit = entry_price + (3 * atr)
            else:  # PUT
                stop_loss = entry_price + (2 * atr)
                take_profit = entry_price - (3 * atr)
            
            # Análisis de contexto
            analisis = self._analizar_contexto(df, signal_type)
            
            # Construir señal (formato compatible con main.py)
            señal = {
                'tipo': signal_type,  # CALL o PUT
                'probabilidad': float(confidence),
                'confianza': 'ALTA' if confidence >= 0.75 else 'MEDIA',
                'precio_actual': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analisis': analisis,
                'features': X
            }
            
            logger.info(f"🎯 Señal ML generada: {signal_type} (prob={confidence:.3f})")
            
            return señal
            
        except Exception as e:
            logger.error(f"❌ Error generando señal ML: {e}")
            import traceback
            traceback.print_exc()
            return self._no_signal()
    
    
    def _generar_señal_tecnica(self, df: pd.DataFrame) -> Dict:
        """
        Generar señal usando análisis técnico puro
        """
        try:
            if len(df) == 0:
                return self._no_signal()
            
            last_row = df.iloc[-1]
            
            # Calcular confianza
            bullish_confidence = self._calculate_bullish_confidence(last_row)
            bearish_confidence = self._calculate_bearish_confidence(last_row)
            
            # Determinar señal
            if bullish_confidence > self.min_confidence and bullish_confidence > bearish_confidence:
                signal_type = 'CALL'
                confidence = bullish_confidence
            elif bearish_confidence > self.min_confidence and bearish_confidence > bullish_confidence:
                signal_type = 'PUT'
                confidence = bearish_confidence
            else:
                return self._no_signal()
            
            # Calcular niveles
            entry_price = float(df['close'].iloc[-1])
            atr = float(last_row.get('atr', entry_price * 0.001))
            
            if signal_type == 'CALL':
                stop_loss = entry_price - (2 * atr)
                take_profit = entry_price + (3 * atr)
            else:
                stop_loss = entry_price + (2 * atr)
                take_profit = entry_price - (3 * atr)
            
            # Razones
            reasons = self._get_signal_reasons(last_row, signal_type)
            
            señal = {
                'tipo': signal_type,
                'probabilidad': float(confidence),
                'confianza': 'ALTA' if confidence >= 0.75 else 'MEDIA',
                'precio_actual': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'analisis': reasons,
                'features': None
            }
            
            logger.info(f"🎯 Señal técnica generada: {signal_type} (conf={confidence:.2%})")
            
            return señal
            
        except Exception as e:
            logger.error(f"❌ Error generando señal técnica: {e}")
            return self._no_signal()
    
    
    def _preparar_features(self, df):
        """Preparar features para predicción ML"""
        try:
            columnas_excluir = ['time', 'tick_volume', 'spread', 'real_volume', 
                               'target', 'label', 'future_return', 'precio_futuro']
            
            X = df.copy()
            
            for col in columnas_excluir:
                if col in X.columns:
                    X = X.drop(columns=[col])
            
            if X.isnull().any().any():
                X = X.fillna(0)
            
            return X
            
        except Exception as e:
            logger.error(f"Error al preparar features: {e}")
            return None
    
    
    def _analizar_contexto(self, df, tipo_señal):
        """Analizar contexto del mercado"""
        analisis = []
        
        try:
            ultima = df.iloc[-1]
            
            # RSI
            if 'rsi' in df.columns:
                rsi = ultima['rsi']
                if rsi > 70:
                    analisis.append(f"RSI sobrecomprado ({rsi:.1f})")
                elif rsi < 30:
                    analisis.append(f"RSI sobrevendido ({rsi:.1f})")
            
            # Tendencia
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                if ultima['close'] > ultima['sma_20'] > ultima['sma_50']:
                    analisis.append("Tendencia alcista")
                elif ultima['close'] < ultima['sma_20'] < ultima['sma_50']:
                    analisis.append("Tendencia bajista")
            
            # MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                if ultima['macd'] > ultima['macd_signal']:
                    analisis.append("MACD alcista")
                else:
                    analisis.append("MACD bajista")
            
            # Volatilidad
            if 'atr_normalizado' in df.columns:
                if ultima['atr_normalizado'] > 0.002:
                    analisis.append("Alta volatilidad")
            
        except Exception as e:
            logger.error(f"Error en análisis: {e}")
        
        return analisis
    
    
    def _calculate_bullish_confidence(self, row: pd.Series) -> float:
        """Calcular confianza alcista"""
        confidence = 0.0
        
        # RSI
        rsi = row.get('rsi', 50)
        if rsi < self.rsi_oversold:
            confidence += 0.3
        
        # MACD
        if row.get('macd', 0) > row.get('macd_signal', 0):
            confidence += 0.2
        
        # Precio vs SMA
        if row.get('close', 0) > row.get('sma_20', 0):
            confidence += 0.2
        
        # Patrón alcista
        if row.get('is_hammer', 0) == 1:
            confidence += 0.15
        
        # Impulso alcista
        if row.get('impulso_alcista', 0) == 1:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    
    def _calculate_bearish_confidence(self, row: pd.Series) -> float:
        """Calcular confianza bajista"""
        confidence = 0.0
        
        # RSI
        rsi = row.get('rsi', 50)
        if rsi > self.rsi_overbought:
            confidence += 0.3
        
        # MACD
        if row.get('macd', 0) < row.get('macd_signal', 0):
            confidence += 0.2
        
        # Precio vs SMA
        if row.get('close', 0) < row.get('sma_20', 0):
            confidence += 0.2
        
        # Patrón bajista
        if row.get('is_shooting_star', 0) == 1:
            confidence += 0.15
        
        # Impulso bajista
        if row.get('impulso_bajista', 0) == 1:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    
    def _get_signal_reasons(self, row: pd.Series, signal_type: str) -> List[str]:
        """Obtener razones de la señal"""
        reasons = []
        
        if signal_type == 'CALL':
            if row.get('rsi', 50) < self.rsi_oversold:
                reasons.append("RSI en sobreventa")
            if row.get('macd', 0) > row.get('macd_signal', 0):
                reasons.append("MACD alcista")
            if row.get('is_hammer', 0) == 1:
                reasons.append("Patrón martillo")
        else:  # PUT
            if row.get('rsi', 50) > self.rsi_overbought:
                reasons.append("RSI en sobrecompra")
            if row.get('macd', 0) < row.get('macd_signal', 0):
                reasons.append("MACD bajista")
            if row.get('is_shooting_star', 0) == 1:
                reasons.append("Patrón estrella fugaz")
        
        return reasons
    
    
    def _no_signal(self) -> Dict:
        """Retornar señal nula (compatible con main.py)"""
        return {
            'tipo': 'NEUTRAL',
            'probabilidad': 0.5,
            'confianza': 'BAJA',
            'precio_actual': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'analisis': ['Sin señal clara'],
            'features': None
        }
    
    
    # Métodos de compatibilidad con código antiguo
    def generate_signal(self, df: pd.DataFrame, features: pd.DataFrame = None) -> Dict:
        """Método de compatibilidad (llama a generar_señal)"""
        if features is not None and len(features) > 0:
            return self.generar_señal(features)
        return self.generar_señal(df)
