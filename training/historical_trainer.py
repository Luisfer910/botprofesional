import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class HistoricalTrainer:
    """
    Entrenador de modelo con datos históricos
    """
    
    def __init__(self, config=None):
        """
        Inicializa el entrenador histórico
        
        Args:
            config: Configuración del entrenador
        """
        self.config = config or {}
        
        # Parámetros del modelo
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 20)
        self.min_samples_split = self.config.get('min_samples_split', 10)
        self.random_state = self.config.get('random_state', 42)
        
        # Scaler
        self.scaler = StandardScaler()
        
        logger.info("✅ HistoricalTrainer inicializado")
    
    
    def preparar_datos(self, df):
        """
        Prepara los datos para entrenamiento
        
        Args:
            df: DataFrame con features
            
        Returns:
            tuple: (X, y) o (None, None) si hay error
        """
        try:
            if df is None or len(df) == 0:
                logger.error("DataFrame vacío")
                return None, None
            
            # Verificar que exista columna target
            if 'target' not in df.columns:
                logger.error("No se encontró columna 'target'")
                return None, None
            
            # Columnas a excluir
            columnas_excluir = [
                'time', 'target', 'label', 'future_return', 
                'precio_futuro', 'tick_volume', 'spread', 'real_volume'
            ]
            
            # Separar features y target
            X = df.copy()
            for col in columnas_excluir:
                if col in X.columns:
                    X = X.drop(columns=[col])
            
            y = df['target'].values
            
            # Eliminar NaN
            if X.isnull().any().any():
                logger.warning("Eliminando NaN...")
                mask = ~X.isnull().any(axis=1)
                X = X[mask]
                y = y[mask]
            
            logger.info(f"✅ Datos preparados: {len(X)} muestras, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    
    def entrenar_modelo(self, X, y):
        """
        Entrena el modelo Random Forest
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Modelo entrenado o None si hay error
        """
        try:
            if X is None or y is None:
                logger.error("Datos inválidos para entrenamiento")
                return None
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"📊 Distribución de datos:")
            logger.info(f"   Training: {len(X_train)} muestras")
            logger.info(f"   Testing: {len(X_test)} muestras")
            
            # Crear y entrenar modelo
            modelo = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            logger.info("🤖 Entrenando Random Forest...")
            modelo.fit(X_train, y_train)
            
            # Evaluar
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)
            
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            
            logger.info(f"✅ Modelo entrenado exitosamente!")
            logger.info(f"   Accuracy Training: {acc_train:.4f}")
            logger.info(f"   Accuracy Testing: {acc_test:.4f}")
            
            # Reporte de clasificación
            logger.info("\n📈 Reporte de Clasificación:")
            print(classification_report(y_test, y_pred_test, 
                                       target_names=['Venta', 'Neutral', 'Compra']))
            
            # Matriz de confusión
            logger.info("\n🔢 Matriz de Confusión:")
            print(confusion_matrix(y_test, y_pred_test))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n🎯 Top 10 Features más importantes:")
            print(feature_importance.head(10).to_string(index=False))
            
            return modelo
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def guardar_modelo(self, modelo, scaler=None, metadata=None):
        """
        Guarda el modelo entrenado
        
        Args:
            modelo: Modelo entrenado
            scaler: Scaler usado (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            tuple: (path, metadata) o None si falla
        """
        try:
            os.makedirs('models', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # ✅ NOMBRE CORRECTO: modelo_hibrido
            filename = f"modelo_hibrido_{timestamp}.pkl"
            path = os.path.join('models', filename)
            
            # Preparar metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'timestamp': timestamp,
                'tipo': 'hibrido',
                'modelo_clase': type(modelo).__name__
            })
            
            # Guardar modelo
            joblib.dump(modelo, path)
            
            # Verificar que se guardó
            if not os.path.exists(path):
                logger.error(f"❌ Archivo no creado: {path}")
                return None
            
            file_size = os.path.getsize(path) / 1024
            logger.info(f"✅ Modelo guardado: {path} ({file_size:.2f} KB)")
            
            # Guardar metadata
            metadata_path = path.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return (path, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def cargar_modelo(self, path):
        """
        Carga un modelo guardado
        
        Args:
            path: Ruta del modelo
            
        Returns:
            Modelo cargado o None si falla
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Archivo no encontrado: {path}")
                return None
            
            modelo = joblib.load(path)
            logger.info(f"✅ Modelo cargado: {path}")
            
            return modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
# =============================================================================
#  HISTORICAL TRAINER - Entrenamiento con filtro numérico no destructivo
# =============================================================================
#  Propósito:
#    - Entrenar modelos con datos históricos evitando errores de dtype.
#    - NO elimina columnas del DataFrame original; filtra numéricas SOLO para el modelo.
#    - Devuelve el modelo entrenado y las probabilidades de test para métricas externas.
#
#  Uso esperado:
#    - Importado por entrenar_completo.py u otros módulos del proyecto.
#
#  Cambios clave:
#    - Filtro de columnas numéricas antes de .fit() y .predict()/predict_proba().
#    - Manejo de errores y logs informativos.
#
#  Compatibilidad:
#    - sklearn (RandomForest, etc.)
#    - lightgbm (si estuviera en uso)
# =============================================================================

from typing import Tuple, Optional, Callable
import numpy as np
import pandas as pd

# Métricas pueden calcularse fuera (en el script principal)
# Se deja import opcional aquí si es necesario:
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

def _default_log(msg: str):
    """Logger por defecto si no se provee log_fn."""
    print(msg)

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

def _numeric_view(df: pd.DataFrame, log_fn: Callable = _default_log) -> pd.DataFrame:
    """
    Retorna una vista SOLO con columnas numéricas sin modificar df original.
    Loguea columnas excluidas para trazabilidad.
    """
    cols_original = list(df.columns)
    df_num = df.select_dtypes(include=['number']).copy()

    excluidas = [c for c in cols_original if c not in df_num.columns]
    if excluidas:
        log_fn(f"ℹ️ Columnas no numéricas excluidas del entrenamiento: {excluidas}")

    if df_num.empty:
        raise ValueError("No hay columnas numéricas disponibles para el modelo.")

    # Log de primeras columnas para visibilidad
    preview_cols = list(df_num.columns)[:10]
    suffix = "..." if len(df_num.columns) > 10 else ""
    log_fn(f"✅ Columnas usadas para el modelo ({len(df_num.columns)}): {preview_cols}{suffix}")

    # Asegurar tipos float64/int64 uniformes (evitar object/bool residuales)
    for c in df_num.columns:
        if pd.api.types.is_bool_dtype(df_num[c]):
            df_num[c] = df_num[c].astype(np.int64)
        elif pd.api.types.is_integer_dtype(df_num[c]):
            df_num[c] = df_num[c].astype(np.int64)
        elif pd.api.types.is_float_dtype(df_num[c]):
            df_num[c] = df_num[c].astype(np.float64)
        # otros tipos numéricos se mantienen

    return df_num

def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    modelo,
    *,
    log_fn: Callable = _default_log
) -> Tuple[object, np.ndarray]:
    """
    Entrena el modelo con datos históricos filtrando columnas no numéricas SOLO para el modelo.
    
    Parámetros:
      - X_train, y_train, X_test, y_test: conjuntos ya preparados y alineados
      - modelo: instancia del modelo (RandomForest, LGBM, etc.)
      - log_fn: función de logging (por defecto print)
    
    Retorna:
      - modelo entrenado
      - y_pred_proba (np.ndarray): probabilidades del conjunto de test para métricas externas
    """
    # Validaciones básicas
    _validate_inputs(X_train, y_train, X_test, y_test)

    log_fn("🔧 Preparando datos para entrenamiento (solo columnas numéricas)...")
    X_train_model = _numeric_view(X_train, log_fn=log_fn)
    X_test_model  = _numeric_view(X_test, log_fn=log_fn)

    # Entrenamiento
    log_fn("🤖 Entrenando modelo con datos históricos...")
    try:
        modelo.fit(X_train_model, y_train)
    except Exception as e:
        log_fn(f"❌ Error al entrenar el modelo: {e}")
        raise

    # Predicción de probabilidades
    log_fn("📈 Generando predicciones en test...")
    y_pred_proba: Optional[np.ndarray] = None
    try:
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_test_model)
            # Modelos binarios: usar la columna de proba de clase 1
            if isinstance(proba, np.ndarray):
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_pred_proba = proba[:, 1]
                elif proba.ndim == 1:
                    y_pred_proba = proba
                else:
                    # fallback: si devuelve form raro
                    y_pred_proba = proba.squeeze()
            else:
                # si no es ndarray, intenta convertir
                y_pred_proba = np.array(proba)
        else:
            # Si no hay predict_proba, usa predicción binaria como proxy 0/1
            y_pred = modelo.predict(X_test_model)
            y_pred_proba = np.array(y_pred, dtype=float)
            log_fn("ℹ️ Modelo sin predict_proba: usando predicción binaria como probabilidad proxy.")
    except Exception as e:
        log_fn(f"❌ Error al predecir: {e}")
        raise

    # Log opcional de AUC si sklearn está disponible y el target es binario
    if _HAS_SKLEARN:
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            log_fn(f"📊 ROC-AUC (interno): {auc:.4f}")
        except Exception as e:
            log_fn(f"ℹ️ No se pudo calcular AUC interno: {e}")

    log_fn("✅ Entrenamiento y predicción completados.")
    return modelo, y_pred_proba

# =============================================================================
#  Funciones auxiliares opcionales para preparar splits (si se desea usar aquí)
# =============================================================================

def preparar_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.85):
    """
    Prepara un split temporal simple (sin shuffle) respetando orden temporal.
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