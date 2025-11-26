# =============================================================================
#  HISTORICAL TRAINER - Clase con filtro numérico no destructivo
# =============================================================================
#  Propósito:
#    - Proveer la clase HistoricalTrainer esperada por entrenar_completo.py.
#    - Evitar DTypePromotionError filtrando columnas no numéricas SOLO para el modelo.
#    - No modificar DataFrames originales; usar vistas temporales.
#    - Retornar (modelo, métricas) para evitar NoneType en unpack.
# =============================================================================

from typing import Tuple, Optional, Callable, Dict
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

class HistoricalTrainer:
    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self.log = log_fn if log_fn is not None else self._default_log

    # -------------------------
    # Utilidades internas
    # -------------------------
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
        """
        Retorna una vista SOLO con columnas numéricas sin modificar df original.
        Loguea columnas excluidas para trazabilidad.
        """
        cols_original = list(df.columns)
        df_num = df.select_dtypes(include=['number']).copy()

        excluidas = [c for c in cols_original if c not in df_num.columns]
        if excluidas:
            self.log(f"ℹ️ Columnas no numéricas excluidas del entrenamiento: {excluidas}")

        if df_num.empty:
            raise ValueError("No hay columnas numéricas disponibles para el modelo.")

        # Log de primeras columnas para visibilidad
        preview_cols = list(df_num.columns)[:10]
        suffix = "..." if len(df_num.columns) > 10 else ""
        self.log(f"✅ Columnas usadas para el modelo ({len(df_num.columns)}): {preview_cols}{suffix}")

        # Normalizar tipos (evitar bool/object)
        for c in df_num.columns:
            if pd.api.types.is_bool_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.int64)
            elif pd.api.types.is_integer_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.int64)
            elif pd.api.types.is_float_dtype(df_num[c]):
                df_num[c] = df_num[c].astype(np.float64)

        return df_num

    # -------------------------
    # API pública esperada
    # -------------------------
    def preparar_split(self, X: pd.DataFrame, y: pd.Series, train_frac: float = 0.85) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepara un split temporal simple (sin shuffle) respetando el orden.
        Retorna X_train, y_train, X_test, y_test.
        """
        if not 0.0 < train_frac < 1.0:
            raise ValueError("train_frac debe estar entre 0 y 1.")

        n = len(X)
        if n != len(y):
            raise ValueError("X e y deben tener la misma longitud.")

        split_idx = int(n * train_frac)
        X_train = X.iloc[:split_idx].copy()
        y_train = y.iloc[:split_idx].copy()
        X_test  = X.iloc[split_idx:].copy()
        y_test  = y.iloc[split_idx:].copy()

        return X_train, y_train, X_test, y_test

    def entrenar_modelo(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        modelo) -> Tuple[object, Dict[str, float]]:
        """
        Entrena el modelo con datos filtrando columnas no numéricas SOLO para el modelo.

        Retorna:
          - modelo entrenado
          - metricas (dict): {'auc': float, 'accuracy': float} cuando es posible.
        """
        # Validaciones
        self._validate_inputs(X_train, y_train, X_test, y_test)

        self.log("🔧 Preparando datos para entrenamiento (solo columnas numéricas)...")
        X_train_model = self._numeric_view(X_train)
        X_test_model  = self._numeric_view(X_test)

        # Entrenamiento
        self.log("🤖 Entrenando modelo con datos históricos...")
        try:
            modelo.fit(X_train_model, y_train)
        except Exception as e:
            self.log(f"❌ Error al entrenar el modelo: {e}")
            raise

        # Predicción de probabilidades
        self.log("📈 Generando predicciones en test...")
        y_pred_proba: Optional[np.ndarray] = None
        y_pred_binary: Optional[np.ndarray] = None

        try:
            if hasattr(modelo, "predict_proba"):
                proba = modelo.predict_proba(X_test_model)
                if isinstance(proba, np.ndarray):
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_pred_proba = proba[:, 1]
                    else:
                        y_pred_proba = proba.squeeze()
                else:
                    y_pred_proba = np.array(proba)
            else:
                # proxy si no hay predict_proba
                y_pred_binary = modelo.predict(X_test_model)
                y_pred_proba = y_pred_binary.astype(float)
                self.log("ℹ️ Modelo sin predict_proba: usando predicción binaria como probabilidad proxy.")
        except Exception as e:
            self.log(f"❌ Error al predecir: {e}")
            raise

        # Calcular métricas (si sklearn disponible)
        metricas: Dict[str, float] = {}
        if _HAS_SKLEARN:
            try:
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    metricas['auc'] = float(auc)
                    self.log(f"📊 ROC-AUC: {auc:.4f}")
            except Exception as e:
                self.log(f"ℹ️ No se pudo calcular AUC: {e}")

            try:
                if y_pred_binary is None and y_pred_proba is not None:
                    # Generar binario desde proba con umbral 0.5
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                if y_pred_binary is not None:
                    acc = accuracy_score(y_test, y_pred_binary)
                    metricas['accuracy'] = float(acc)
                    self.log(f"📊 Accuracy: {acc:.4f}")
            except Exception as e:
                self.log(f"ℹ️ No se pudo calcular Accuracy: {e}")

        self.log("✅ Entrenamiento y evaluación completados.")
        return modelo, metricas