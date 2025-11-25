"""
Continuous Learner - Aprendizaje continuo del modelo
Versión corregida v2.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Sistema de aprendizaje continuo que mejora el modelo
    con experiencias en tiempo real
    """
    
    def __init__(self, modelo_hibrido=None):
        """
        Inicializa el aprendizaje continuo
        
        Args:
            modelo_hibrido: Puede ser:
                - Dict con claves 'live' y 'historico'
                - Modelo directo (RandomForest, LightGBM, etc.)
                - None (modo observación)
        """
        self.modelo = None
        self.tipo_modelo = 'desconocido'
        
        # Manejar diferentes tipos de entrada
        if modelo_hibrido is None:
            logger.warning("⚠️ ContinuousLearner inicializado sin modelo (modo observación)")
            self.tipo_modelo = 'sin_modelo'
            
        elif isinstance(modelo_hibrido, dict):
            # Es un diccionario (modelo híbrido)
            if 'live' in modelo_hibrido:
                self.modelo = modelo_hibrido['live']
                self.tipo_modelo = 'hibrido_live'
                logger.info("✅ ContinuousLearner con modelo LIVE")
                
            elif 'historico' in modelo_hibrido:
                self.modelo = modelo_hibrido['historico']
                self.tipo_modelo = 'hibrido_historico'
                logger.info("✅ ContinuousLearner con modelo HISTÓRICO (será adaptado)")
                
            else:
                # Dict sin claves conocidas
                logger.error("❌ Dict no contiene claves 'live' ni 'historico'")
                self.tipo_modelo = 'dict_invalido'
        else:
            # Es un modelo directo (esto NO debería pasar en sistema live)
            self.modelo = modelo_hibrido
            self.tipo_modelo = 'directo'
            logger.warning("⚠️ ContinuousLearner recibió modelo directo (esperaba dict híbrido)")
        
        # Validar que el modelo tenga métodos necesarios
        if self.modelo is not None:
            if not hasattr(self.modelo, 'predict'):
                logger.error("❌ Modelo no tiene método 'predict'")
                self.modelo = None
            else:
                logger.info(f"   Tipo de modelo: {type(self.modelo).__name__}")
        
        # Buffer de experiencias
        self.experiencias = []
        self.max_experiencias = 1000
        
        # Estadísticas
        self.total_actualizaciones = 0
        self.ultima_actualizacion = None
        self.proximo_reentrenamiento = datetime.now() + timedelta(days=7)
        
        # Configuración
        self.min_experiencias_aprender = 50
        self.frecuencia_actualizacion = timedelta(hours=24)
        
        logger.info(f"   Buffer: {self.max_experiencias} experiencias")
        logger.info(f"   Min para aprender: {self.min_experiencias_aprender}")
        
        # Intentar cargar estado previo
        self.cargar_estado()

    
    
    def agregar_experiencia(self, features, prediccion, resultado_real):
        """
        Agrega una experiencia al buffer
        
        Args:
            features: Features del trade (array o DataFrame)
            prediccion: Predicción del modelo (0 o 1)
            resultado_real: Resultado real (0=perdió, 1=ganó)
        """
        try:
            # Convertir features a formato serializable
            if hasattr(features, 'values'):
                features_array = features.values
            elif isinstance(features, np.ndarray):
                features_array = features
            else:
                features_array = np.array(features)
            
            experiencia = {
                'features': features_array,
                'prediccion': int(prediccion),
                'resultado_real': int(resultado_real),
                'timestamp': datetime.now(),
                'correcto': int(prediccion) == int(resultado_real)
            }
            
            self.experiencias.append(experiencia)
            
            # Mantener solo las últimas N experiencias
            if len(self.experiencias) > self.max_experiencias:
                self.experiencias = self.experiencias[-self.max_experiencias:]
            
            logger.debug(f"Experiencia agregada (total: {len(self.experiencias)}, correcto: {experiencia['correcto']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error agregando experiencia: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def aprender_de_experiencias(self, min_experiencias=None):
        """
        Aprende de las experiencias acumuladas
        
        Args:
            min_experiencias: Mínimo de experiencias requeridas
            
        Returns:
            bool: True si se actualizó el modelo
        """
        try:
            if self.modelo is None:
                logger.warning("No hay modelo para aprender")
                return False
            
            min_exp = min_experiencias or self.min_experiencias_aprender
            
            if len(self.experiencias) < min_exp:
                logger.debug(f"Insuficientes experiencias ({len(self.experiencias)}/{min_exp})")
                return False
            
            # Verificar si es momento de actualizar
            if self.ultima_actualizacion:
                tiempo_desde_ultima = datetime.now() - self.ultima_actualizacion
                if tiempo_desde_ultima < self.frecuencia_actualizacion:
                    logger.debug("Aún no es momento de actualizar")
                    return False
            
            logger.info(f"🧠 Aprendiendo de {len(self.experiencias)} experiencias...")
            
            # Preparar datos
            X = []
            y = []
            
            for exp in self.experiencias:
                if exp['features'] is not None and len(exp['features']) > 0:
                    X.append(exp['features'].flatten() if exp['features'].ndim > 1 else exp['features'])
                    y.append(exp['resultado_real'])
            
            if len(X) == 0:
                logger.warning("No hay features válidas para aprender")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Validar dimensiones
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            # Calcular accuracy actual
            predicciones = [exp['prediccion'] for exp in self.experiencias if exp['features'] is not None]
            if len(predicciones) > 0:
                accuracy_actual = np.mean([p == r for p, r in zip(predicciones, y)])
                logger.info(f"   Accuracy actual: {accuracy_actual:.2%}")
            
            # Reentrenar modelo
            actualizado = False
            
            if hasattr(self.modelo, 'partial_fit'):
                # Aprendizaje incremental
                try:
                    self.modelo.partial_fit(X, y)
                    logger.info("   ✅ Modelo actualizado (partial_fit)")
                    actualizado = True
                except Exception as e:
                    logger.error(f"   Error en partial_fit: {e}")
                    
            elif hasattr(self.modelo, 'fit'):
                # Reentrenamiento completo
                try:
                    self.modelo.fit(X, y)
                    logger.info("   ✅ Modelo reentrenado (fit completo)")
                    actualizado = True
                except Exception as e:
                    logger.error(f"   Error en fit: {e}")
            else:
                logger.warning("   ⚠️ Modelo no soporta reentrenamiento")
            
            if actualizado:
                # Actualizar estadísticas
                self.total_actualizaciones += 1
                self.ultima_actualizacion = datetime.now()
                
                # Limpiar experiencias antiguas (mantener últimas 100)
                self.experiencias = self.experiencias[-100:]
                
                logger.info(f"   Total actualizaciones: {self.total_actualizaciones}")
                
                # Guardar estado
                self.guardar_estado()
            
            return actualizado
            
        except Exception as e:
            logger.error(f"Error aprendiendo de experiencias: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def necesita_reentrenamiento(self):
        """Verifica si es momento de reentrenamiento completo"""
        return datetime.now() >= self.proximo_reentrenamiento
    
    
    def programar_proximo_reentrenamiento(self, dias=7):
        """Programa el próximo reentrenamiento completo"""
        self.proximo_reentrenamiento = datetime.now() + timedelta(days=dias)
        logger.info(f"Próximo reentrenamiento: {self.proximo_reentrenamiento.strftime('%Y-%m-%d %H:%M')}")
    
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas del aprendizaje continuo"""
        if len(self.experiencias) == 0:
            win_rate = 0.0
        else:
            win_rate = sum(1 for exp in self.experiencias if exp['correcto']) / len(self.experiencias)
        
        return {
            'total_experiencias': len(self.experiencias),
            'win_rate_general': win_rate,
            'total_actualizaciones': self.total_actualizaciones,
            'ultima_actualizacion': self.ultima_actualizacion.strftime('%Y-%m-%d %H:%M:%S') if self.ultima_actualizacion else 'Nunca',
            'proximo_reentrenamiento': self.proximo_reentrenamiento.strftime('%Y-%m-%d %H:%M:%S'),
            'tipo_modelo': self.tipo_modelo
        }
    
    
    def guardar_estado(self, path='models/continuous_learner_state.pkl'):
        """Guarda el estado del learner"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            estado = {
                'experiencias': self.experiencias,
                'total_actualizaciones': self.total_actualizaciones,
                'ultima_actualizacion': self.ultima_actualizacion,
                'proximo_reentrenamiento': self.proximo_reentrenamiento,
                'tipo_modelo': self.tipo_modelo
            }
            
            joblib.dump(estado, path)
            logger.info(f"💾 Estado guardado: {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
            return False
    
    
    def cargar_estado(self, path='models/continuous_learner_state.pkl'):
        """Carga el estado del learner"""
        try:
            if not os.path.exists(path):
                logger.debug("No hay estado previo para cargar")
                return False
            
            estado = joblib.load(path)
            
            self.experiencias = estado.get('experiencias', [])
            self.total_actualizaciones = estado.get('total_actualizaciones', 0)
            self.ultima_actualizacion = estado.get('ultima_actualizacion', None)
            self.proximo_reentrenamiento = estado.get('proximo_reentrenamiento', datetime.now() + timedelta(days=7))
            
            logger.info(f"📂 Estado cargado: {path}")
            logger.info(f"   Experiencias: {len(self.experiencias)}")
            logger.info(f"   Actualizaciones: {self.total_actualizaciones}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando estado: {e}")
            return False
