
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging

class HistoricalTrainer:
    def __init__(self, n_estimators=300, max_depth=None, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.modelo = None
        self.scaler = None
        self.feature_cols = None
        logging.basicConfig(filename='logs/historical_trainer.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def entrenar(self, df):
        try:
            self.logger.info("Iniciando entrenamiento...")
            if 'target' not in df.columns: return None
            excluir = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'target', 'precio_futuro']
            self.feature_cols = [col for col in df.columns if col not in excluir]
            X = df[self.feature_cols].select_dtypes(include=[np.number])
            self.feature_cols = X.columns.tolist()
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.modelo = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=-1)
            self.modelo.fit(X_train_scaled, y_train)
            
            acc = accuracy_score(y_test, self.modelo.predict(X_test_scaled))
            self.logger.info(f"Modelo entrenado. Accuracy: {acc:.4f}")
            return self.modelo
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
    
    def guardar_modelo(self, ruta='models/modelo_historico.pkl'):
        try:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, 'wb') as f:
                pickle.dump({'modelo': self.modelo, 'scaler': self.scaler, 'feature_cols': self.feature_cols}, f)
            return True
        except: return False
