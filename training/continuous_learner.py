"""
Continuous Learner - Aprendizaje continuo en tiempo real
Versión 3.0 - Con retroalimentación real
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime, timedelta
from collections import deque

class ContinuousLearner:
    def __init__(self, modelo, feature_engineer, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.modelo = modelo
        self.feature_engineer = feature_engineer
        
        # Buffer de experiencias
        self.experiencias = deque(maxlen=500)
        self.actualizaciones = 0
        self.ultima_actualizacion = None
        
        logging.basicConfig(
            filename='logs/continuous_learner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🧠 Continuous Learner inicializado")
    
    def agregar_experiencia(self, senal, resultado_trade):
        """
        Agrega una experiencia al buffer
        
        Args:
            senal: dict con la señal generada
            resultado_trade: dict con resultado del trade
        """
        try:
            experiencia = {
                'timestamp': datetime.now(),
                'senal': senal,
                'resultado': resultado_trade,
                'ganancia': resultado_trade.get('profit', 0),
                'exito': resultado_trade.get('profit', 0) > 0
            }
            
            self.experiencias.append(experiencia)
            
            self.logger.info(
                f"📝 Experiencia agregada: {senal['tipo']} | "
                f"Profit: ${experiencia['ganancia']:.2f} | "
                f"Éxito: {experiencia['exito']}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error al agregar experiencia: {str(e)}")
    
    def aprender(self, min_experiencias=20):
        """
        Aprende de las experiencias acumuladas
        
        Args:
            min_experiencias: Mínimo de experiencias para aprender
        """
        if len(self.experiencias) < min_experiencias:
            self.logger.info(
                f"⏳ Esperando más experiencias "
                f"({len(self.experiencias)}/{min_experiencias})"
            )
            return False
        
        try:
            self.logger.info(f"🧠 Aprendiendo de {len(self.experiencias)} experiencias...")
            
            print(f"\n{'='*70}")
            print(f"  🧠 APRENDIZAJE CONTINUO")
            print(f"{'='*70}")
            print(f"Experiencias: {len(self.experiencias)}")
            
            # Calcular métricas
            total = len(self.experiencias)
            exitos = sum(1 for exp in self.experiencias if exp['exito'])
            win_rate = exitos / total * 100
            
            profit_total = sum(exp['ganancia'] for exp in self.experiencias)
            
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Profit Total: ${profit_total:.2f}")
            
            # Aquí podrías implementar reentrenamiento incremental
            # Por ahora solo registramos las métricas
            
            self.actualizaciones += 1
            self.ultima_actualizacion = datetime.now()
            
            print(f"Actualizaciones totales: {self.actualizaciones}")
            print(f"{'='*70}\n")
            
            self.logger.info(
                f"✅ Aprendizaje completado | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Profit: ${profit_total:.2f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error en aprendizaje: {str(e)}")
            return False
    
    def guardar_estado(self):
        """Guarda el estado del learner"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            estado = {
                'experiencias': list(self.experiencias),
                'actualizaciones': self.actualizaciones,
                'ultima_actualizacion': self.ultima_actualizacion
            }
            
            path = f'models/learner_estado_{timestamp}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(estado, f)
            
            self.logger.info(f"💾 Estado guardado: {path}")
            print(f"💾 Estado del learner guardado")
            
        except Exception as e:
            self.logger.error(f"❌ Error al guardar estado: {str(e)}")
