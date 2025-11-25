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
            
            # ✅ PASO 1: Generar TODAS las features (igual que en entrenamiento)
            logger.info("🔧 Generando features completas...")
            df_original = df.copy()
            
            # Llamar a generar_todas_features
            df = self.feature_engineer.generar_todas_features(df)
            
            if df is None or len(df) == 0:
                logger.error("Error generando features")
                return self._señal_neutral()
            
            # ✅ PASO 2: Generar features temporales (CRÍTICO)
            logger.info("🕐 Generando features temporales...")
            df = self.feature_engineer.generar_features_temporales(df)
            
            # ✅ PASO 3: Detectar soportes y resistencias (CRÍTICO)
            logger.info("📊 Detectando soportes y resistencias...")
            df = self.feature_engineer.detectar_soportes_resistencias(df)
            
            # ✅ PASO 4: Limpiar NaN e infinitos
            df = df.replace([np.inf, -np.inf], 0)
            df = df.fillna(0)
            
            logger.info(f"✅ Features generadas: {len(df.columns)} columnas")
            
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
            
            # Obtener precio actual (del DataFrame original)
            precio_actual = float(df_original['close'].iloc[-1])
            
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
        """
        Prepara features para el modelo
        ✅ EXCLUYE LAS MISMAS COLUMNAS QUE EN ENTRENAMIENTO
        """
        try:
            # ✅ COLUMNAS A EXCLUIR (IGUAL QUE EN historical_trainer.py)
            columnas_excluir = [
                'time',
                'target',
                'label',
                'future_return',      # ✅ CRÍTICO: Esta columna causaba el error
                'precio_futuro',
                'tick_volume',
                'spread',
                'real_volume',
                'target_clasificacion',
                'retorno_futuro'
            ]
            
            # Seleccionar todas las columnas excepto las excluidas
            feature_cols = [col for col in df.columns if col not in columnas_excluir]
            
            logger.info(f"📊 Usando {len(feature_cols)} features para predicción")
            
            # Tomar última fila
            X = df[feature_cols].tail(1).copy()
            
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