import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

class SignalGenerator:
    def __init__(self, modelo, feature_engineer, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.modelo = modelo
        self.feature_engineer = feature_engineer
        
        # Umbrales ajustados para generar más señales
        self.umbral_call = 0.55  # Bajado de 0.58
        self.umbral_put = 0.45   # Subido de 0.42
        
        logging.basicConfig(
            filename='logs/signal_generator.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generar_senal(self, df, features_intravela=None):
        """
        Genera señal de trading
        
        Args:
            df: DataFrame con velas OHLC
            features_intravela: dict con features intravela de la vela actual
            
        Returns:
            dict con señal
        """
        try:
            # Verificar que el modelo sea válido
            if not hasattr(self.modelo, 'predict'):
                error_msg = f"Modelo inválido: tipo {type(self.modelo).__name__}"
                self.logger.error(f"❌ {error_msg}")
                return self._senal_hold(error_msg)
            
            # Generar features completas
            df_features = self.feature_engineer.generar_todas_features(df, features_intravela)
            
            if len(df_features) == 0:
                return self._senal_hold("No hay datos suficientes")
            
            # Obtener última fila
            ultima_fila = df_features.iloc[-1]
            
            # Preparar features para predicción
            feature_cols = self.feature_engineer.obtener_feature_columns(df_features)
            
            # Verificar que existan las columnas necesarias
            feature_cols_disponibles = [col for col in feature_cols if col in df_features.columns]
            
            if len(feature_cols_disponibles) == 0:
                return self._senal_hold("No hay features disponibles")
            
            X = ultima_fila[feature_cols_disponibles].values.reshape(1, -1)
            
            # Predecir con el modelo
            if hasattr(self.modelo, 'predict_proba'):
                probabilidades = self.modelo.predict_proba(X)[0]
                prob_call = probabilidades[1]  # Probabilidad de clase 1 (CALL)
            else:
                # Si no tiene predict_proba, usar predict
                prediccion = self.modelo.predict(X)[0]
                # Normalizar predicción a rango 0-1
                if prediccion in [0, 1]:
                    prob_call = float(prediccion)
                else:
                    prob_call = 0.5
            
            # Determinar señal
            if prob_call >= self.umbral_call:
                tipo = 'CALL'
                fuerza = (prob_call - self.umbral_call) / (1.0 - self.umbral_call) * 100
            elif prob_call <= self.umbral_put:
                tipo = 'PUT'
                fuerza = (self.umbral_put - prob_call) / self.umbral_put * 100
            else:
                tipo = 'HOLD'
                fuerza = 0.0
            
            # Crear señal
            senal = {
                'tipo': tipo,
                'probabilidad': float(prob_call),
                'fuerza': float(fuerza),
                'precio': float(df['close'].iloc[-1]),
                'timestamp': datetime.now(),
                'razon': self._generar_razon(ultima_fila, tipo, prob_call)
            }
            
            if tipo != 'HOLD':
                self.logger.info(
                    f"🎯 SEÑAL: {tipo} | Prob: {prob_call:.3f} | "
                    f"Fuerza: {fuerza:.1f}% | Precio: {senal['precio']:.5f}"
                )
            
            return senal
            
        except Exception as e:
            self.logger.error(f"❌ Error al generar señal: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._senal_hold(f"Error: {str(e)}")

    
    def _generar_razon(self, fila, tipo, probabilidad):
        """Genera explicación de la señal"""
        razones = []
        
        try:
            # RSI
            if 'rsi' in fila:
                rsi = fila['rsi']
                if rsi > 70:
                    razones.append("RSI sobrecompra")
                elif rsi < 30:
                    razones.append("RSI sobreventa")
            
            # Tendencia
            if 'sma_20' in fila and 'close' in fila:
                if fila['close'] > fila['sma_20']:
                    razones.append("Precio sobre SMA20")
                else:
                    razones.append("Precio bajo SMA20")
            
            # Presión intravela
            if 'presion_neta' in fila:
                presion = fila['presion_neta']
                if presion > 0.2:
                    razones.append("Fuerte presión compradora")
                elif presion < -0.2:
                    razones.append("Fuerte presión vendedora")
            
            # Volatilidad
            if 'volatilidad_intravela' in fila:
                if fila['volatilidad_intravela'] > 0.001:
                    razones.append("Alta volatilidad intravela")
            
            if len(razones) == 0:
                razones.append(f"Modelo predice {tipo} con {probabilidad:.1%} confianza")
            
        except:
            razones = ["Análisis técnico"]
        
        return ", ".join(razones)
    
    def _senal_hold(self, razon):
        """Retorna señal HOLD"""
        return {
            'tipo': 'HOLD',
            'probabilidad': 0.5,
            'fuerza': 0.0,
            'precio': 0.0,
            'timestamp': datetime.now(),
            'razon': razon
        }
