import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generador de señales de trading basado en modelo de IA
    """
    
    def __init__(self, modelo, feature_engineer, config=None):
        """
        Inicializa el generador de señales
        
        Args:
            modelo: Modelo de ML entrenado
            feature_engineer: FeatureEngineer para procesar datos
            config: Configuración adicional
        """
        self.modelo = modelo
        self.feature_engineer = feature_engineer
        self.config = config or {}
        
        # Parámetros
        self.umbral_confianza = self.config.get('UMBRAL_CONFIANZA', 0.6)
        
        logger.info("✅ SignalGenerator inicializado")
    
    
    def generar_señal(self, df):
        """
        Genera señal de trading basada en el modelo
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            dict: Señal con tipo, fuerza, etc.
        """
        try:
            if df is None or len(df) == 0:
                logger.warning("DataFrame vacío")
                return self._señal_neutral()
            
            # ✅ GENERAR FEATURES COMPLETAS (CRÍTICO)
            logger.info("🔧 Generando features completas...")
            df = self.feature_engineer.generar_todas_features(df)
            
            if df is None or len(df) == 0:
                logger.error("Error generando features")
                return self._señal_neutral()
            
            # Preparar datos para predicción
            X = self._preparar_features(df)
            
            if X is None or len(X) == 0:
                logger.error("Error preparando features")
                return self._señal_neutral()
            
            # Predecir
            prediccion = self.modelo.predict(X)
            probabilidades = self.modelo.predict_proba(X)
            
            # Obtener última predicción
            accion_pred = prediccion[-1]
            probs = probabilidades[-1]
            
            # Mapear acción: -1=SELL, 0=HOLD, 1=BUY
            mapa_acciones = {
                -1: 'SELL',
                0: 'HOLD',
                1: 'BUY'
            }
            
            accion = mapa_acciones.get(accion_pred, 'HOLD')
            confianza = float(np.max(probs))
            
            # Verificar umbral de confianza
            if confianza < self.umbral_confianza:
                logger.info(f"Confianza baja ({confianza:.2%}). Sin señal.")
                return self._señal_neutral()
            
            # Solo operar BUY o SELL
            if accion == 'HOLD':
                return self._señal_neutral()
            
            # Obtener precio actual
            precio_actual = float(df['close'].iloc[-1])
            
            # Calcular ATR para stop loss y take profit
            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else precio_actual * 0.001
            
            # Calcular niveles
            if accion == 'BUY':
                stop_loss = precio_actual - (2 * atr)
                take_profit = precio_actual + (3 * atr)
            else:  # SELL
                stop_loss = precio_actual + (2 * atr)
                take_profit = precio_actual - (3 * atr)
            
            # Generar razón
            razon = self._generar_razon(df, accion, confianza)
            
            # ✅ FORMATO COMPATIBLE CON main.py
            señal = {
                'tipo': accion,                    # BUY, SELL o HOLD
                'fuerza': confianza,               # ✅ CAMPO REQUERIDO
                'confianza': confianza,
                'razon': razon,
                'timestamp': datetime.now(),
                'precio_actual': precio_actual,    # ✅ CAMPO REQUERIDO
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lote': 0.01
            }
            
            logger.info(f"✅ Señal generada: {accion} ({confianza:.2%})")
            
            return señal
            
        except Exception as e:
            logger.error(f"❌ Error generando señal: {e}")
            import traceback
            traceback.print_exc()
            return self._señal_neutral()
    
    
    def _preparar_features(self, df):
        """Prepara features para el modelo"""
        try:
            # Lista de features esperadas por el modelo
            feature_cols = [
                # Indicadores de tendencia
                'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21',
                'macd', 'macd_signal', 'macd_diff',
                'adx', 'adx_pos', 'adx_neg',
                
                # Indicadores de momentum
                'rsi', 'stoch_k', 'stoch_d',
                
                # Bollinger Bands
                'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_position',
                
                # Volatilidad
                'atr', 'atr_normalizado',
                
                # Volumen
                'obv',
                
                # Patrones de velas
                'body', 'body_abs', 'upper_shadow', 'lower_shadow',
                'total_range', 'body_ratio', 'upper_shadow_ratio',
                'lower_shadow_ratio', 'is_bullish', 'is_bearish',
                'is_doji', 'is_hammer', 'is_shooting_star', 'is_engulfing',
                
                # Cambios de precio
                'price_change', 'price_change_abs',
                'momentum_3', 'momentum_5', 'momentum_10',
                'roc_3', 'roc_5', 'roc_10'
            ]
            
            X = df.copy()
            
            # Verificar columnas faltantes
            missing_cols = [col for col in feature_cols if col not in X.columns]
            
            if missing_cols:
                logger.warning(f"⚠️  Columnas faltantes: {missing_cols[:5]}...")
                for col in missing_cols:
                    X[col] = 0.0
            
            # Seleccionar solo las features necesarias
            X = X[feature_cols]
            
            # Reemplazar NaN/inf
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _generar_razon(self, df, accion, confianza):
        """Genera explicación de la señal"""
        try:
            razones = []
            
            # Análisis de tendencia
            if 'sma_20' in df.columns and 'close' in df.columns:
                precio = df['close'].iloc[-1]
                sma20 = df['sma_20'].iloc[-1]
                
                if precio > sma20:
                    razones.append("Precio sobre SMA20")
                else:
                    razones.append("Precio bajo SMA20")
            
            # RSI
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    razones.append(f"RSI sobrecomprado ({rsi:.1f})")
                elif rsi < 30:
                    razones.append(f"RSI sobrevendido ({rsi:.1f})")
            
            # Confianza del modelo
            razones.append(f"Confianza: {confianza:.1%}")
            
            return " | ".join(razones) if razones else "Análisis del modelo"
            
        except Exception as e:
            logger.error(f"Error generando razón: {e}")
            return "Señal del modelo"
    
    
    def _señal_neutral(self):
        """Retorna señal neutral (HOLD)"""
        return {
            'tipo': 'HOLD',
            'fuerza': 0.0,
            'confianza': 0.0,
            'razon': 'Sin señal clara',
            'timestamp': datetime.now(),
            'precio_actual': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'lote': 0.0
        }