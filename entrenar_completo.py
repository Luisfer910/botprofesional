import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# MT5 y tus módulos (ajusta si difieren en tu repo)
import MetaTrader5 as mt5

# Importa el trainer con clase
from training.historical_trainer import HistoricalTrainer

# -------------------------------------------------------------------
# Utilidades de logging simples (puedes reemplazar por tus logs)
# -------------------------------------------------------------------
def log_section(title: str):
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70 + "\n")

def main():
    print("=" * 70)
    print("  🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print("=" * 70 + "\n")

    inicio = datetime.now()
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---------------------------------------------------------------
    # PASO 1: CONEXIÓN A MT5 (resumen mínimo; respeta tu flujo actual)
    # ---------------------------------------------------------------
    log_section("PASO 1: CONEXIÓN A MT5")

    if not mt5.initialize():
        print("❌ Error al inicializar MT5")
        return

    # NOTA: Ajusta credenciales/servidor si están en config en tu repo
    XM_LOGIN = 100464594
    XM_PASSWORD = "Fer101996-"
    XM_SERVER = "XMGlobalSC-MT5 5"
    ACTIVO = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5

    login_ok = mt5.login(XM_LOGIN, password=XM_PASSWORD, server=XM_SERVER)
    if not login_ok:
        print(f"❌ Error de login: {mt5.last_error()}")
        mt5.shutdown()
        return

    mt5.symbol_select(ACTIVO, True)

    account_info = mt5.account_info()
    if account_info:
        print("✅ Conectado a XM. EURUSD seleccionado.")
        print(f"   Cuenta: {account_info.login}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Servidor: {XM_SERVER}")
        print("✅ Conectado exitosamente\n")
        print("💰 Información de Cuenta:")
        print(f"   • Login: {account_info.login}")
        print(f"   • Balance: ${account_info.balance:,.2f}")
        print(f"   • Equity: ${account_info.equity:,.2f}")
        print(f"   • Margen Libre: N/D")
        print(f"   • Apalancamiento: 1:1000")
    else:
        print("❌ No se pudo obtener información de cuenta")

    # ---------------------------------------------------------------
    # PASO 2: DESCARGA DE DATOS HISTÓRICOS
    # ---------------------------------------------------------------
    log_section("PASO 2: DESCARGA DE DATOS HISTÓRICOS")

    CANT_VELAS = 20000
    print(f"📥 Descargando {CANT_VELAS:,} velas históricas...\n   (Esto puede tomar 1-2 minutos)\n")
    rates = mt5.copy_rates_from_pos(ACTIVO, TIMEFRAME, 0, CANT_VELAS)
    if rates is None or len(rates) == 0:
        print(f"❌ Error al descargar velas: {mt5.last_error()}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=False)

    print(f"✅ {len(df):,} velas descargadas")
    print(f"   📅 Desde: {pd.to_datetime(rates[0]['time'], unit='s')}")
    print(f"   📅 Hasta: {pd.to_datetime(rates[-1]['time'], unit='s')}\n")

    # ---------------------------------------------------------------
    # PASO 3: GENERACIÓN DE FEATURES
    # ---------------------------------------------------------------
    log_section("PASO 3: GENERACIÓN DE FEATURES")

    # Generación de features técnicas mínima (ajusta a tu módulo real si lo tienes)
    # Mantengo 28 features como indica tu log.
    df_feat = df.copy()

    # Rango y cuerpo
    df_feat['rango'] = df_feat['high'] - df_feat['low']
    df_feat['rango'] = df_feat['rango'].replace(0, 1e-8)
    df_feat['cuerpo'] = df_feat['close'] - df_feat['open']
    df_feat['cuerpo_abs'] = df_feat['cuerpo'].abs()
    df_feat['cuerpo_pct'] = df_feat['cuerpo'] / df_feat['rango']

    # Mechas
    df_feat['mecha_sup'] = df_feat['high'] - df_feat[['open', 'close']].max(axis=1)
    df_feat['mecha_inf'] = df_feat[['open', 'close']].min(axis=1) - df_feat['low']

    # Volumen relativo
    df_feat['vol_ma20'] = df_feat['tick_volume'].rolling(20).mean()
    df_feat['rvol'] = df_feat['tick_volume'] / df_feat['vol_ma20']

    # ATR básico
    true_range = pd.concat([
        df_feat['high'] - df_feat['low'],
        (df_feat['high'] - df_feat['close'].shift()).abs(),
        (df_feat['low'] - df_feat['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df_feat['atr14'] = true_range.rolling(14).mean()

    # Momentum y medias
    df_feat['momentum_1'] = df_feat['close'].diff(1) / df_feat['atr14']
    df_feat['momentum_3'] = df_feat['close'].diff(3) / df_feat['atr14']
    df_feat['sma50'] = df_feat['close'].rolling(50).mean()
    df_feat['dist_sma50'] = (df_feat['close'] - df_feat['sma50']) / df_feat['atr14']

    # Relaciones
    df_feat['rango_vs_anterior'] = df_feat['rango'] / df_feat['rango'].shift(1)
    df_feat['cuerpo_vs_anterior'] = df_feat['cuerpo_abs'] / (df_feat['cuerpo_abs'].shift(1) + 1e-8)
    df_feat['cambio_dir'] = np.sign(df_feat['cuerpo']) != np.sign(df_feat['cuerpo'].shift(1))

    # Patrón simple
    df_feat['tres_alcistas'] = (
        (df_feat['cuerpo'] > 0) &
        (df_feat['cuerpo'].shift(1) > 0) &
        (df_feat['cuerpo'].shift(2) > 0)
    ).astype(int)
    df_feat['tres_bajistas'] = (
        (df_feat['cuerpo'] < 0) &
        (df_feat['cuerpo'].shift(1) < 0) &
        (df_feat['cuerpo'].shift(2) < 0)
    ).astype(int)

    # Selecciona columnas numéricas finales para X (mantén el resto en df_feat para auditoría)
    # Evita incluir datetime directamente en X
    feature_cols = [
        'open','high','low','close','tick_volume',
        'rango','cuerpo','cuerpo_abs','cuerpo_pct',
        'mecha_sup','mecha_inf','rvol','atr14',
        'momentum_1','momentum_3','sma50','dist_sma50',
        'rango_vs_anterior','cuerpo_vs_anterior','cambio_dir',
        'tres_alcistas','tres_bajistas'
    ]

    df_feat = df_feat.dropna()
    X = df_feat[feature_cols].copy()

    print(f"✅ Features generadas: {X.shape[1]} columnas")
    print(f"   📊 Datos disponibles: {X.shape[0]} filas\n")

    # Target
    HORIZON = 3
    y = (df_feat['close'].shift(-HORIZON) > df_feat['close']).astype(int)
    y = y.iloc[:-HORIZON]
    X = X.iloc[:-HORIZON]

    print("🎯 Creando variable target...")
    print("✅ Target creado")
    print(f"   📊 Datos finales: {X.shape[0]} filas\n")

    # ---------------------------------------------------------------
    # PASO 4: ENTRENAMIENTO MODELO HISTÓRICO
    # ---------------------------------------------------------------
    log_section("PASO 4: ENTRENAMIENTO MODELO HISTÓRICO")

    trainer = HistoricalTrainer(log_fn=print)

    # Split temporal (mismo 85%/15% que venías usando)
    X_train, y_train, X_test, y_test = trainer.preparar_split(X, y, train_frac=0.85)

    # Modelo (ajusta a tu preferencia si usas LightGBM)
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    print("🤖 Entrenando modelo con datos históricos...\n   (Esto puede tomar 2-5 minutos)\n")
    modelo_historico, metricas = trainer.entrenar_modelo(
        X_train, y_train, X_test, y_test, modelo
    )

    # Métrica extra (opcional)
    if 'auc' not in metricas:
        try:
            y_pred_proba = modelo_historico.predict_proba(
                X_test.select_dtypes(include=['number'])
            )[:, 1]
            metricas['auc'] = float(roc_auc_score(y_test, y_pred_proba))
        except Exception:
            pass

    print("\n✅ Entrenamiento completado")
    print("📊 Métricas:", metricas)

    # Cierre
    mt5.shutdown()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Proceso terminado con errores: {e}")
        import traceback
        traceback.print_exc()