import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import json
import logging
from datetime import datetime

class HybridTrainer:
    def __init__(self, config_path='config/xm_config.json', modelo_historico=None, peso_historico=0.7, n_estimators=200):
        """
        Inicializa el HybridTrainer
        
        Args:
            config_path: Ruta al archivo de configuración
            modelo_historico: Modelo pre-entrenado (opcional)
            peso_historico: Peso del modelo histórico (no usado en esta versión)
            n_estimators: Número de estimadores (no usado en esta versión)
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {}
        
        self.modelo_historico = modelo_historico
        self.modelo_live = None
        self.modelo_hibrido = None
        self.scaler = None
        self.peso_historico = peso_historico
        self.peso_live = 1 - peso_historico
        
        logging.basicConfig(
            filename='logs/hybrid_trainer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def entrenar(self, df_historico, df_tiempo_real):
        """
        Método principal de entrenamiento (compatible con entrenar_completo.py)
        
        Args:
            df_historico: DataFrame con datos históricos (no usado en esta versión)
            df_tiempo_real: DataFrame con datos recientes para entrenar
            
        Returns:
            self si exitoso, None si hay error
        """
        try:
            self.logger.info("🔄 Iniciando entrenamiento híbrido...")
            print(f"\n{'='*70}")
            print(f"🔄 ENTRENAMIENTO MODELO HÍBRIDO")
            print(f"{'='*70}")
            
            # Verificar que tenga columna target
            if 'target' not in df_tiempo_real.columns:
                self.logger.error("❌ No se encuentra columna 'target'")
                print("❌ Error: DataFrame no tiene columna 'target'")
                return None
            
            # Separar features y target
            excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                      'spread', 'real_volume', 'target', 'precio_futuro']
            feature_cols = [col for col in df_tiempo_real.columns if col not in excluir]
            
            X_live = df_tiempo_real[feature_cols]
            y_live = df_tiempo_real['target']
            
            # Eliminar columnas no numéricas
            X_live = X_live.select_dtypes(include=[np.number])
            feature_cols = X_live.columns.tolist()
            
            print(f"📊 Datos de entrenamiento:")
            print(f"   Muestras: {len(X_live)}")
            print(f"   Features: {len(feature_cols)}")
            print(f"{'─'*70}\n")
            
            if len(X_live) < 50:
                self.logger.warning(f"⚠️ Pocos datos ({len(X_live)}). Se recomienda al menos 50.")
                print(f"⚠️ Advertencia: Solo {len(X_live)} muestras disponibles")
            
            # Crear scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_live)
            
            # Crear dataset de LightGBM
            train_data = lgb.Dataset(X_scaled, label=y_live)
            
            # Parámetros para LightGBM
            # Detectar si es binario o multiclase
            num_classes = len(np.unique(y_live))
            
            if num_classes == 2:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'max_depth': 7,
                    'min_data_in_leaf': 5
                }
            else:
                params = {
                    'objective': 'multiclass',
                    'num_class': num_classes,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'max_depth': 7,
                    'min_data_in_leaf': 5
                }
            
            print("🤖 Entrenando modelo híbrido...")
            print(f"   Tipo: {'Binario' if num_classes == 2 else f'Multiclase ({num_classes} clases)'}")
            print(f"{'─'*70}\n")
            
            # Entrenar modelo
            self.modelo_live = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            # Evaluar
            if num_classes == 2:
                y_pred_prob = self.modelo_live.predict(X_scaled)
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred_prob = self.modelo_live.predict(X_scaled)
                y_pred = np.argmax(y_pred_prob, axis=1)
            
            acc = accuracy_score(y_live, y_pred)
            
            print(f"✅ Modelo híbrido entrenado exitosamente!")
            print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"{'='*70}\n")
            
            self.logger.info(f"✅ Modelo híbrido entrenado - Accuracy: {acc:.4f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"❌ Error en entrenamiento híbrido: {e}")
            print(f"❌ Error en entrenamiento híbrido: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def guardar_modelo(self, ruta='models/modelo_hibrido.pkl'):
        """
        Guarda el modelo híbrido
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        try:
            if self.modelo_live is None:
                self.logger.error("❌ No hay modelo para guardar")
                return False
            
            import os
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            
            # Guardar modelo completo
            modelo_data = {
                'modelo': self.modelo_live,
                'scaler': self.scaler,
                'modelo_historico': self.modelo_historico
            }
            
            with open(ruta, 'wb') as f:
                pickle.dump(modelo_data, f)
            
            self.logger.info(f"✅ Modelo guardado en: {ruta}")
            print(f"✅ Modelo híbrido guardado: {ruta}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando modelo: {e}")
            print(f"❌ Error guardando modelo: {e}")
            return False
    
    def cargar_modelo_historico(self, modelo_path, scaler_path):
        """Carga el modelo entrenado con datos históricos"""
        try:
            with open(modelo_path, 'rb') as f:
                self.modelo_historico = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info(f"✅ Modelo histórico cargado")
            print(f"✅ Modelo histórico cargado: {modelo_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error al cargar modelo histórico: {str(e)}")
            return False

    def refinar_con_datos_live(self, df_live, feature_cols):
        """
        Refina el modelo histórico con datos observados en vivo
        (Fine-tuning con datos recientes)
        """
        self.logger.info("🔧 Refinando modelo con datos live...")
        print(f"\n{'='*60}")
        print(f"🔧 REFINAMIENTO CON DATOS LIVE")
        print(f"{'='*60}")
        
        if self.modelo_historico is None:
            print("❌ Error: Modelo histórico no cargado")
            return None
        
        # Preparar datos live
        X_live = df_live[feature_cols].values
        y_live = df_live['target'].values
        
        print(f"Muestras live: {len(X_live)}")
        print(f"Features: {len(feature_cols)}")
        print(f"{'─'*60}\n")
        
        # Normalizar con el mismo scaler
        X_live_scaled = self.scaler.transform(X_live)
        
        # Crear dataset de LightGBM
        live_data = lgb.Dataset(X_live_scaled, label=y_live)
        
        # Parámetros para fine-tuning (learning rate más bajo)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,  # Más bajo para ajuste fino
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_data_in_leaf': 5,  # Más bajo por menos datos
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        print("🚀 Refinando modelo con datos live...")
        print(f"{'─'*60}\n")
        
        # Continuar entrenamiento desde el modelo histórico
        self.modelo_live = lgb.train(
            params,
            live_data,
            num_boost_round=100,  # Menos iteraciones
            init_model=self.modelo_historico,  # Partir del modelo histórico
            valid_sets=[live_data],
            valid_names=['live'],
            callbacks=[
                lgb.log_evaluation(period=20)
            ]
        )
        
        print(f"\n{'─'*60}")
        print("✅ Refinamiento completado")
        print(f"{'─'*60}\n")
        
        # Evaluar mejora
        y_pred_historico = (self.modelo_historico.predict(X_live_scaled) > 0.5).astype(int)
        y_pred_live = (self.modelo_live.predict(X_live_scaled) > 0.5).astype(int)
        
        acc_historico = accuracy_score(y_live, y_pred_historico)
        acc_live = accuracy_score(y_live, y_pred_live)
        
        print(f"📊 COMPARACIÓN DE MODELOS:")
        print(f"{'─'*60}")
        print(f"   Modelo Histórico: {acc_historico*100:.2f}%")
        print(f"   Modelo Refinado: {acc_live*100:.2f}%")
        print(f"   Mejora: {(acc_live-acc_historico)*100:+.2f}%")
        print(f"{'─'*60}\n")
        
        self.logger.info(f"✅ Modelo refinado - Accuracy: {acc_live*100:.2f}%")
        
        return self.modelo_live

    def crear_modelo_hibrido(self, peso_historico=0.6, peso_live=0.4):
        """
        Crea un modelo híbrido que combina predicciones del modelo histórico
        y el modelo live
        """
        self.logger.info("🔀 Creando modelo híbrido...")
        print(f"\n{'='*60}")
        print(f"🔀 CREACIÓN DE MODELO HÍBRIDO")
        print(f"{'='*60}")
        print(f"Peso Histórico: {peso_historico*100:.0f}%")
        print(f"Peso Live: {peso_live*100:.0f}%")
        print(f"{'='*60}\n")
        
        if self.modelo_historico is None or self.modelo_live is None:
            print("❌ Error: Faltan modelos para crear híbrido")
            return None
        
        # Guardar pesos
        self.peso_historico = peso_historico
        self.peso_live = peso_live
        
        # El modelo híbrido es una combinación de ambos
        self.modelo_hibrido = {
            'historico': self.modelo_historico,
            'live': self.modelo_live,
            'peso_historico': peso_historico,
            'peso_live': peso_live,
            'scaler': self.scaler
        }
        
        print("✅ Modelo híbrido creado exitosamente\n")
        self.logger.info("✅ Modelo híbrido creado")
        
        return self.modelo_hibrido

    def predecir_hibrido(self, X):
        """Realiza predicción con el modelo híbrido"""
        if self.modelo_hibrido is None:
            self.logger.error("❌ Modelo híbrido no creado")
            return None, None
        
        try:
            # Normalizar
            X_scaled = self.scaler.transform(X)
            
            # Predicciones de cada modelo
            prob_historico = self.modelo_historico.predict(X_scaled)
            prob_live = self.modelo_live.predict(X_scaled)
            
            # Combinar predicciones con pesos
            prob_hibrido = (
                self.peso_historico * prob_historico +
                self.peso_live * prob_live
            )
            
            # Convertir a clase
            pred_hibrido = (prob_hibrido > 0.5).astype(int)
            
            return pred_hibrido, prob_hibrido
            
        except Exception as e:
            self.logger.error(f"❌ Error al predecir: {str(e)}")
            return None, None

    def guardar_modelo_hibrido(self, nombre='hybrid_model'):
        """Guarda el modelo híbrido"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar modelo completo
            modelo_path = f'models/{nombre}_{timestamp}.pkl'
            with open(modelo_path, 'wb') as f:
                pickle.dump(self.modelo_hibrido, f)
            
            print(f"💾 MODELO HÍBRIDO GUARDADO:")
            print(f"{'─'*60}")
            print(f"   {modelo_path}")
            print(f"{'='*60}\n")
            
            self.logger.info(f"✅ Modelo híbrido guardado: {modelo_path}")
            
            return modelo_path
            
        except Exception as e:
            self.logger.error(f"❌ Error al guardar modelo híbrido: {str(e)}")
            return None

    def cargar_modelo_hibrido(self, modelo_path):
        """Carga un modelo híbrido"""
        try:
            with open(modelo_path, 'rb') as f:
                self.modelo_hibrido = pickle.load(f)
            
            self.modelo_historico = self.modelo_hibrido['historico']
            self.modelo_live = self.modelo_hibrido['live']
            self.peso_historico = self.modelo_hibrido['peso_historico']
            self.peso_live = self.modelo_hibrido['peso_live']
            self.scaler = self.modelo_hibrido['scaler']
            
            self.logger.info(f"✅ Modelo híbrido cargado: {modelo_path}")
            print(f"✅ Modelo híbrido cargado: {modelo_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error al cargar modelo híbrido: {str(e)}")
            return False
