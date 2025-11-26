from typing import Tuple, Optional, Callable, Dict
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

class HistoricalTrainer:
    def __init__(self, log_fn: Optional[Callable[[str], None]] = None, default_train_frac: float = 0.85):
        self.log = log_fn if log_fn is not None else self._default_log
        self.default_train_frac = default_train_frac

    @staticmethod
    def _default_log(msg: str):
        print(msg)

    @staticmethod
    def _validate_inputs(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Entradas de entrenamiento/prueba no pueden ser None.")
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("X_train o X_test están vacíos.")
        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("y_train o y_test están vacíos.")
        if len(X_train) != len(y_train):
            raise ValueError(f"Longitud inconsistente: X_train={len(X_train)} vs y_train={len(y_train)}")
        if len(X_test) != len(y_test):
            raise ValueError(f"Longitud inconsistente: X_test={len(X_test)} vs y_test={len(y_test)}")

    def _numeric_view(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_original = list(df.columns)
        df_num = df.select_dtypes(include=['number']).copy()
        excluidas = [c for c in cols_original if c not in df_num.columns]
        if excluidas:
            self.log(f"ℹ️ Columnas no numéricas excluidas del entrenamiento: {excluidas}")
        if df_num.empty:
            raise ValueError("No hay columnas numéricas disponibles para el modelo.")
        for c in df_num.columns:
            if pd.api.types.is_bool_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.int64)
            elif pd.api.types.is_integer_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.int64)
            elif pd.api.types.is_float_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.float64)
        return df_num

    def preparar_split(self, X: pd.DataFrame, y: pd.Series, train_frac: float = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        tf = self.default_train_frac if train_frac is None else train_frac
        if not 0.0 < tf < 1.0:
            raise ValueError("train_frac debe estar entre 0 y 1.")
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud.")
        split_idx = int(len(X) * tf)
        X_train = X.iloc[:split_idx].copy()
        y_train = y.iloc[:split_idx].copy()
        X_test  = X.iloc[split_idx:].copy()
        y_test  = y.iloc[split_idx:].copy()
        return X_train, y_train, X_test, y_test

    # API COMPLETA (recomendada)
    def entrenar_modelo_full(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             modelo) -> Tuple[object, Dict[str, float]]:
        self._validate_inputs(X_train, y_train, X_test, y_test)
        self.log("🔧 Preparando datos para entrenamiento (solo columnas numéricas)...")
        X_train_model = self._numeric_view(X_train)
        X_test_model  = self._numeric_view(X_test)
        self.log("🤖 Entrenando modelo con datos históricos...")
        modelo.fit(X_train_model, y_train)
        self.log("📈 Generando predicciones en test...")
        y_pred_proba = None
        y_pred_binary = None
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_test_model)
            y_pred_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.squeeze()
        else:
            y_pred_binary = modelo.predict(X_test_model)
            y_pred_proba = y_pred_binary.astype(float)
            self.log("ℹ️ Modelo sin predict_proba: usando predicción binaria como probabilidad proxy.")
        metricas: Dict[str, float] = {}
        if _HAS_SKLEARN:
            try:
                metricas['auc'] = float(roc_auc_score(y_test, y_pred_proba))
                self.log(f"📊 ROC-AUC: {metricas['auc']:.4f}")
            except Exception as e:
                self.log(f"ℹ️ No se pudo calcular AUC: {e}")
            try:
                if y_pred_binary is None:
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                from sklearn.metrics import accuracy_score
                metricas['accuracy'] = float(accuracy_score(y_test, y_pred_binary))
                self.log(f"📊 Accuracy: {metricas['accuracy']:.4f}")
            except Exception as e:
                self.log(f"ℹ️ No se pudo calcular Accuracy: {e}")
        self.log("✅ Entrenamiento y evaluación completados.")
        return modelo, metricas

    # API CORTA (compatibilidad con tu llamada actual)
    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Tuple[object, Dict[str, float]]:
        """
        Compatibilidad con llamadas existentes: trainer.entrenar_modelo(X, y)
        - Hace split interno
        - Usa RandomForest por defecto (ajusta si prefieres otro)
        - Llama a entrenar_modelo_full y retorna (modelo, métricas)
        """
        self.log("ℹ️ Usando API corta entrenar_modelo(X, y): se hará split interno y modelo por defecto.")
        X_train, y_train, X_test, y_test = self.preparar_split(X, y, train_frac=self.default_train_frac)

        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn no disponible para construir el modelo por defecto.")

        modelo = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        return self.entrenar_modelo_full(X_train, y_train, X_test, y_test, modelo)