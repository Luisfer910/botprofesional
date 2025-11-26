import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import os

class HybridTrainer:
    def __init__(self, target_col, feature_cols):
        """
        Inicializa el entrenador híbrido.
        """
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.model = None
        
        # Configuración de LightGBM (Optimizada para clasificación)
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,  # 0: Venta, 1: Neutral, 2: Compra
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Crear directorio de modelos si no existe
        os.makedirs('models', exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def entrenar(self, df, modelo_historico):
        """
        Entrena el modelo híbrido (LightGBM) usando las predicciones del modelo histórico como input extra.
        """
        self.logger.info("🚀 Iniciando entrenamiento del Modelo Híbrido...")

        # 1. Generar predicciones del modelo histórico (Random Forest)
        # Usamos predict_proba para dar más información al LightGBM (confianza del RF)
        try:
            # Aseguramos que solo usamos las columnas con las que se entrenó el histórico
            features_historicas = df[self.feature_cols]
            
            # Obtenemos probabilidades: [Prob_Venta, Prob_Neutral, Prob_Compra]
            probs_historicas = modelo_historico.predict_proba(features_historicas)
            
            # Añadimos estas probabilidades como NUEVAS features para el LightGBM
            df_hibrido = df.copy()
            df_hibrido['rf_prob_sell'] = probs_historicas[:, 0]
            df_hibrido['rf_prob_neutral'] = probs_historicas[:, 1]
            df_hibrido['rf_prob_buy'] = probs_historicas[:, 2]
            
            # Actualizamos la lista de features para incluir las nuevas
            features_hibridas = self.feature_cols + ['rf_prob_sell', 'rf_prob_neutral', 'rf_prob_buy']
            
        except Exception as e:
            self.logger.error(f"❌ Error generando inputs del modelo histórico: {e}")
            raise e

        # 2. Preparar datos para LightGBM
        X = df_hibrido[features_hibridas]
        y = df_hibrido[self.target_col]

        # Split temporal (respetando el orden del tiempo)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Crear Datasets de LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 3. Entrenar LightGBM
        self.logger.info("🔄 Entrenando LightGBM...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # 4. Evaluar
        y_pred_prob = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = [np.argmax(line) for line in y_pred_prob]

        acc = accuracy_score(y_test, y_pred)
        self.logger.info(f"✅ Modelo Híbrido Entrenado. Accuracy Test: {acc:.4f}")
        self.logger.info("\n" + classification_report(y_test, y_pred))

        # 5. Guardar
        path = 'models/modelo_hibrido.txt' # LightGBM se guarda mejor como txt o json
        self.model.save_model(path)
        self.logger.info(f"💾 Modelo guardado en {path}")
        
        return self.model