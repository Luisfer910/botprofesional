"""
Script de entrenamiento completo del bot de trading XM
Incluye: descarga histórica + observación live + entrenamiento
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import MetaTrader5 as mt5

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from training.historical_trainer import HistoricalTrainer
from config.config import CONFIG

def print_header(texto):
    """Imprime un encabezado formateado"""
    print("\n" + "─" * 70)
    print(f"  {texto}")
    print("─" * 70 + "\n")


def observar_mercado_live(connector, duracion_minutos=60, intervalo_segundos=1):
    """
    Observa el mercado en vivo capturando ticks
    
    Args:
        connector: MT5Connector
        duracion_minutos: Duración de la observación
        intervalo_segundos: Intervalo entre capturas
        
    Returns:
        DataFrame con datos OHLCV de las velas formadas
    """
    print(f"👁️  Observando mercado en vivo por {duracion_minutos} minutos...")
    print(f"   Capturando ticks cada {intervalo_segundos} segundo(s)")
    print(f"   Presiona Ctrl+C para detener antes\n")
    
    ticks_data = []
    inicio = time.time()
    duracion_segundos = duracion_minutos * 60
    
    try:
        while (time.time() - inicio) < duracion_segundos:
            # Obtener tick actual
            tick = mt5.symbol_info_tick(connector.symbol)
            
            if tick is not None:
                ticks_data.append({
                    'time': datetime.fromtimestamp(tick.time),
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume
                })
            
            # Mostrar progreso cada minuto
            tiempo_transcurrido = (time.time() - inicio) / 60
            if len(ticks_data) % 60 == 0:
                print(f"   📊 {len(ticks_data)} ticks capturados ({tiempo_transcurrido:.1f} min)")
            
            time.sleep(intervalo_segundos)
            
    except KeyboardInterrupt:
        print("\n   ⚠️  Observación interrumpida por el usuario")
    
    duracion_real = (time.time() - inicio) / 60
    
    print(f"\n   ✅ Observación completada")
    print(f"   📊 Total ticks capturados: {len(ticks_data)}")
    print(f"   ⏱️  Duración real: {duracion_real:.1f} minutos")
    
    if len(ticks_data) == 0:
        return None
    
    # ✅ CONVERTIR TICKS A VELAS OHLCV
    df_ticks = pd.DataFrame(ticks_data)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'])
    df_ticks.set_index('time', inplace=True)
    
    # Usar el precio 'last' (o 'bid' si last no existe) como precio de cierre
    df_ticks['price'] = df_ticks['last'].fillna(df_ticks['bid'])
    
    # ✅ RESAMPLE A VELAS DE 5 MINUTOS (M5)
    df_ohlcv = pd.DataFrame({
        'open': df_ticks['price'].resample('5T').first(),
        'high': df_ticks['price'].resample('5T').max(),
        'low': df_ticks['price'].resample('5T').min(),
        'close': df_ticks['price'].resample('5T').last(),
        'tick_volume': df_ticks['volume'].resample('5T').sum()
    })
    
    # Eliminar filas con NaN
    df_ohlcv = df_ohlcv.dropna()
    
    # Reset index para tener 'time' como columna
    df_ohlcv.reset_index(inplace=True)
    
    print(f"\n   ✅ Convertidos a {len(df_ohlcv)} velas OHLCV (M5)")
    
    return df_ohlcv


def main():
    """Función principal del entrenamiento completo"""
    
    print("=" * 70)
    print("  🤖 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\nInicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # ──────────────────────────────────────────────────────────────
        # PASO 1: CONEXIÓN A MT5
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 1: CONEXIÓN A MT5")
        
        connector = MT5Connector(CONFIG)
        if not connector.conectar():
            print("❌ Error al conectar con MT5")
            return
        
        print("✅ Conectado exitosamente")
        
        # Mostrar info de cuenta
        info = mt5.account_info()
        if info:
            print(f"\n💰 Información de Cuenta:")
            print(f"   • Login: {info.login}")
            print(f"   • Balance: ${info.balance:,.2f}")
            print(f"   • Equity: ${info.equity:,.2f}")
            print(f"   • Margen Libre: ${info.margin_free:,.2f}")
            print(f"   • Apalancamiento: 1:{info.leverage}")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 2: DESCARGA DE DATOS HISTÓRICOS
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 2: DESCARGA DE DATOS HISTÓRICOS")
        
        data_manager = DataManager(connector, CONFIG)
        
        print(f"📥 Descargando {CONFIG['VELAS_HISTORICAS']:,} velas históricas...")
        print(f"   (Esto puede tomar 1-2 minutos)\n")
        
        df_historico = data_manager.obtener_datos_historicos(
            num_velas=CONFIG['VELAS_HISTORICAS']
        )
        
        if df_historico is None or len(df_historico) == 0:
            print("❌ Error al obtener datos históricos")
            connector.desconectar()
            return
        
        print(f"✅ Datos históricos cargados: {len(df_historico)} velas")
        print(f"   Período: {df_historico['time'].min()} a {df_historico['time'].max()}\n")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 3: OBSERVACIÓN LIVE
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 3: OBSERVACIÓN LIVE")
        
        print("🔴 OBSERVACIÓN EN VIVO")
        print("   Esta fase observa el mercado tick-by-tick durante 1 hora")
        print("   para capturar datos de formación de velas en tiempo real.\n")
        
        respuesta = input("¿Deseas realizar la observación live? (s/n): ").lower()
        
        df_live = None
        if respuesta == 's':
            duracion = 60  # minutos
            print(f"\n⏱️  Iniciando observación live por {duracion} minutos...")
            print(f"   Puedes detener con Ctrl+C si lo deseas\n")
            
            df_live = observar_mercado_live(
                connector, 
                duracion_minutos=duracion,
                intervalo_segundos=1
            )
            
            if df_live is not None and len(df_live) > 0:
                print(f"✅ Observación completada: {len(df_live)} velas capturadas")
            else:
                print("⚠️  No se capturaron datos live suficientes")
        else:
            print("⏭️  Saltando observación live")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 4: GENERACIÓN DE FEATURES
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 4: GENERACIÓN DE FEATURES")
        
        feature_engineer = FeatureEngineer(CONFIG)
        
        print("🔧 Generando features para datos históricos...")
        print("   - Indicadores técnicos (RSI, MACD, ADX, etc.)")
        print("   - Patrones de velas")
        print("   - Soportes y resistencias")
        print("   - Impulsos y retrocesos")
        print("   - Análisis de volatilidad\n")
        
        df_historico_features = feature_engineer.generar_todas_features(df_historico)
        
        if df_historico_features is None or len(df_historico_features) == 0:
            print("❌ Error al generar features históricas")
            connector.desconectar()
            return
        
        print(f"✅ Features generadas: {len(df_historico_features.columns)} columnas")
        print(f"   Datos válidos: {len(df_historico_features)} filas\n")
        
        # Generar features para datos live si existen
        df_live_features = None
        if df_live is not None and len(df_live) > 0:
            print("🔧 Generando features para datos live...\n")
            df_live_features = feature_engineer.generar_todas_features(df_live)
            
            if df_live_features is not None and len(df_live_features) > 0:
                print(f"✅ Features live generadas: {len(df_live_features)} filas\n")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 5: ENTRENAMIENTO DEL MODELO
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 5: ENTRENAMIENTO DEL MODELO")
        
        trainer = HistoricalTrainer(CONFIG)
        
        print("🤖 Entrenando modelo de Machine Learning...")
        print("   - Algoritmo: Random Forest + XGBoost")
        print("   - Validación cruzada temporal")
        print("   - Optimización de hiperparámetros\n")
        
        # Combinar datos históricos y live si existen
        if df_live_features is not None and len(df_live_features) > 0:
            print("📊 Combinando datos históricos + live...\n")
            df_completo = pd.concat([df_historico_features, df_live_features], ignore_index=True)
        else:
            df_completo = df_historico_features
        
        # Entrenar
        modelo, metricas = trainer.entrenar(df_completo)
        
        if modelo is None:
            print("❌ Error en el entrenamiento")
            connector.desconectar()
            return
        
        print("\n✅ Modelo entrenado exitosamente")
        print(f"\n📊 Métricas del modelo:")
        print(f"   • Accuracy: {metricas.get('accuracy', 0):.2%}")
        print(f"   • Precision: {metricas.get('precision', 0):.2%}")
        print(f"   • Recall: {metricas.get('recall', 0):.2%}")
        print(f"   • F1-Score: {metricas.get('f1', 0):.2%}")
        
        # ──────────────────────────────────────────────────────────────
        # PASO 6: GUARDAR MODELO
        # ──────────────────────────────────────────────────────────────
        print_header("PASO 6: GUARDAR MODELO")
        
        ruta_modelo = trainer.guardar_modelo(modelo)
        
        if ruta_modelo:
            print(f"✅ Modelo guardado en: {ruta_modelo}")
        else:
            print("⚠️  Error al guardar el modelo")
        
        # ──────────────────────────────────────────────────────────────
        # FINALIZACIÓN
        # ──────────────────────────────────────────────────────────────
        connector.desconectar()
        
        print("\n" + "=" * 70)
        print("  ✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n💡 Ahora puedes ejecutar 'python main.py' para usar el bot\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        if 'connector' in locals():
            connector.desconectar()
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        if 'connector' in locals():
            connector.desconectar()
        print("❌ Proceso terminado con errores")


if __name__ == "__main__":
    main()