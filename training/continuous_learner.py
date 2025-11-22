import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import deque
import pickle
import json
import logging
from datetime import datetime, timedelta

class ContinuousLearner:
    def __init__(self, modelo_hibrido, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.modelo = modelo_hibrido['live']  # Usar modelo live como base
        self.scaler = modelo_hibrido['scaler']
        
        # Buffer de experiencias (últimos N trades)
        self.buffer_size = 500
        self.experiencias = deque(maxlen=self.buffer_size)
        
        # Métricas de aprendizaje
        self.total_actualizaciones = 0
        self.ultima_actualizacion = None
        self.metricas_aprendizaje = []
        
        # Configuración de reentrenamiento
        self.horas_reentrenamiento = self.config['MODELO']['REENTRENAMIENTO_HORAS']
        self.proximo_reentrenamiento = datetime.now() + timedelta(hours=self.horas_reentrenamiento)
        
        logging.basicConfig(
            filename='logs/continuous_learner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🧠 Continuous Learner inicializado")
    
    def agregar_experiencia(self, features, prediccion, resultado_real):
        """
        Agrega una nueva experiencia al buffer
        
        features: array con las features de la predicción
        prediccion: 0 (PUT) o 1 (CALL)
        resultado_real: 0 (perdió) o 1 (ganó)
        """
        experiencia = {
            'features': features,
            'prediccion': prediccion,
            'resultado_real': resultado_real,
            'timestamp': datetime.now()
        }
        
        self.experiencias.append(experiencia)
        
        self.logger.info(f"📝 Experiencia agregada: Pred={prediccion}, Real={resultado_real}")
    
    def aprender_de_experiencias(self, min_experiencias=50):
        """
        Aprende de las experiencias acumuladas
        (Actualización incremental del modelo)
        """
        if len(self.experiencias) < min_experiencias:
            self.logger.info(f"⏳ Esperando más experiencias ({len(self.experiencias)}/{min_experiencias})")
            return False
        
        self.logger.info(f"🧠 Aprendiendo de {len(self.experiencias)} experiencias...")
        
        print(f"\n{'='*60}")
        print(f"🧠 APRENDIZAJE CONTINUO")
        print(f"{'='*60}")
        print(f"Experiencias acumuladas: {len(self.experiencias)}")
        
        try:
            # Extraer datos del buffer
            X_buffer = np.array([exp['features'] for exp in self.experiencias])
            y_buffer = np.array([exp['resultado_real'] for exp in self.experiencias])
            
            # Normalizar
            X_buffer_scaled = self.scaler.transform(X_buffer)
            
            # Calcular accuracy actual
            predicciones_actuales = (self.modelo.predict(X_buffer_scaled) > 0.5).astype(int)
            accuracy_antes = np.mean(predicciones_actuales == y_buffer)
            
            print(f"Accuracy antes: {accuracy_antes*100:.2f}%")
            
            # Crear dataset
            buffer_data = lgb.Dataset(X_buffer_scaled, label=y_buffer)
            
            # Parámetros para actualización incremental
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.005,  # Muy bajo para ajuste suave
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 7,
                'min_data_in_leaf': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
            
            # Actualizar modelo (continuar entrenamiento)
            self.modelo = lgb.train(
                params,
                buffer_data,
                num_boost_round=20,  # Pocas iteraciones
                init_model=self.modelo,
                valid_sets=[buffer_data],
                callbacks=[lgb.log_evaluation(period=10)]
            )
            
            # Calcular accuracy después
            predicciones_nuevas = (self.modelo.predict(X_buffer_scaled) > 0.5).astype(int)
            accuracy_despues = np.mean(predicciones_nuevas == y_buffer)
            
            print(f"Accuracy después: {accuracy_despues*100:.2f}%")
            print(f"Mejora: {(accuracy_despues-accuracy_antes)*100:+.2f}%")
            print(f"{'='*60}\n")
            
            # Guardar métricas
            self.metricas_aprendizaje.append({
                'timestamp': datetime.now(),
                'num_experiencias': len(self.experiencias),
                'accuracy_antes': float(accuracy_antes),
                'accuracy_despues': float(accuracy_despues),
                'mejora': float(accuracy_despues - accuracy_antes)
            })
            
            self.total_actualizaciones += 1
            self.ultima_actualizacion = datetime.now()
            
            self.logger.info(f"✅ Aprendizaje completado - Accuracy: {accuracy_despues*100:.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error en aprendizaje continuo: {str(e)}")
            return False
    
    def necesita_reentrenamiento(self):
        """Verifica si es momento de reentrenar completamente"""
        return datetime.now() >= self.proximo_reentrenamiento
    
    def programar_proximo_reentrenamiento(self):
        """Programa el próximo reentrenamiento automático"""
        self.proximo_reentrenamiento = datetime.now() + timedelta(hours=self.horas_reentrenamiento)
        self.logger.info(f"⏰ Próximo reentrenamiento: {self.proximo_reentrenamiento.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas del aprendizaje continuo"""
        if len(self.experiencias) == 0:
            return None
        
        # Calcular win rate reciente
        resultados = [exp['resultado_real'] for exp in self.experiencias]
        win_rate = np.mean(resultados)
        
        # Calcular win rate por tipo de señal
        calls = [exp for exp in self.experiencias if exp['prediccion'] == 1]
        puts = [exp for exp in self.experiencias if exp['prediccion'] == 0]
        
        win_rate_calls = np.mean([exp['resultado_real'] for exp in calls]) if len(calls) > 0 else 0
        win_rate_puts = np.mean([exp['resultado_real'] for exp in puts]) if len(puts) > 0 else 0
        
        stats = {
            'total_experiencias': len(self.experiencias),
            'win_rate_general': float(win_rate),
            'win_rate_calls': float(win_rate_calls),
            'win_rate_puts': float(win_rate_puts),
            'total_calls': len(calls),
            'total_puts': len(puts),
            'total_actualizaciones': self.total_actualizaciones,
            'ultima_actualizacion': self.ultima_actualizacion.strftime('%Y-%m-%d %H:%M:%S') if self.ultima_actualizacion else 'Nunca',
            'proximo_reentrenamiento': self.proximo_reentrenamiento.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return stats
    
    def guardar_estado(self, nombre='continuous_learner'):
        """Guarda el estado del learner"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            estado = {
                'modelo': self.modelo,
                'scaler': self.scaler,
                'experiencias': list(self.experiencias),
                'metricas_aprendizaje': self.metricas_aprendizaje,
                'total_actualizaciones': self.total_actualizaciones,
                'ultima_actualizacion': self.ultima_actualizacion,
                'proximo_reentrenamiento': self.proximo_reentrenamiento
            }
            
            estado_path = f'models/{nombre}_{timestamp}.pkl'
            with open(estado_path, 'wb') as f:
                pickle.dump(estado, f)
            
            self.logger.info(f"✅ Estado guardado: {estado_path}")
            
            return estado_path
            
        except Exception as e:
            self.logger.error(f"❌ Error al guardar estado: {str(e)}")
            return None
    
    def cargar_estado(self, estado_path):
        """Carga el estado del learner"""
        try:
            with open(estado_path, 'rb') as f:
                estado = pickle.load(f)
            
            self.modelo = estado['modelo']
            self.scaler = estado['scaler']
            self.experiencias = deque(estado['experiencias'], maxlen=self.buffer_size)
            self.metricas_aprendizaje = estado['metricas_aprendizaje']
            self.total_actualizaciones = estado['total_actualizaciones']
            self.ultima_actualizacion = estado['ultima_actualizacion']
            self.proximo_reentrenamiento = estado['proximo_reentrenamiento']
            
            self.logger.info(f"✅ Estado cargado: {estado_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error al cargar estado: {str(e)}")
            return False
