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
            df: DataFrame con features generadas
            
        Returns:
            dict: Señal con acción, confianza y razón, o None si no hay señal
        """
        try:
            if df is None or len(df) == 0:
                logger.warning("DataFrame vacío")
                return None
            
            # Preparar datos para predicción
            X = self._preparar_features(df)
            
            if X is None:
                return None
            
            # Predecir
            prediccion = self.modelo.predict(X)
            probabilidades = self.modelo.predict_proba(X)
            
            # Obtener última predicción
            accion_pred = prediccion[-1]
            probs = probabilidades[-1]
            
            # Mapear acción
            mapa_acciones = {
                -1: 'PUT',
                0: 'HOLD',
                1: 'CALL'
            }
            
            accion = mapa_acciones.get(accion_pred, 'HOLD')
            confianza = np.max(probs)
            
            # Verificar umbral de confianza
            if confianza < self.umbral_confianza:
                logger.info(f"Confianza baja ({confianza:.2%}). Sin señal.")
                return None
            
            # Solo operar CALL o PUT
            if accion == 'HOLD':
                return None
            
            # Generar razón
            razon = self._generar_razon(df, accion, confianza)
            
            señal = {
                'accion': accion,
                'confianza': confianza,
                'razon': razon,
                'timestamp': datetime.now(),
                'precio': df['close'].iloc[-1] if 'close' in df.columns else None
            }
            
            logger.info(f"✅ Señal generada: {accion} ({confianza:.2%})")
            
            return señal
            
        except Exception as e:
            logger.error(f"Error generando señal: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _preparar_features(self, df):
        """Prepara features para el modelo"""
        try:
            # Columnas a excluir
            columnas_excluir = [
                'time', 'target', 'label', 'future_return',
                'precio_futuro', 'tick_volume', 'spread', 'real_volume'
            ]
            
            X = df.copy()
            for col in columnas_excluir:
                if col in X.columns:
                    X = X.drop(columns=[col])
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
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
            razones.append(f"Confianza del modelo: {confianza:.1%}")
            
            return " | ".join(razones) if razones else "Análisis del modelo"
            
        except Exception as e:
            logger.error(f"Error generando razón: {e}")
            return "Señal del modelo"
