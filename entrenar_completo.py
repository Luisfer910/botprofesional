"""
Script de entrenamiento completo del bot
Versión: 2.0
"""

import sys
import os
from datetime import datetime
import pandas as pd
import json

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer


def print_header(text, char="─"):
    """Imprime un encabezado formateado"""
    width = 70
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width + "\n")


def main():
    """
    Función principal de entrenamiento
    """
    print("\n" + "=" * 70)
    print("  🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\nInicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # ══════════════════════════════════════════════════════════════
        # PASO 1: CONEXIÓN A MT5
        # ══════════════════════════════════════════════════════════════
        print_header("PASO 1: CONEXIÓN A MT5")
        
        mt5_connector = MT5Connector()
        
        if not mt5_connector.conectar():
            print("❌ Error al conectar con MT5")
            return
        
        print("✅ Conectado exitosamente\n")
        
        # Mostrar información de cuenta
        info = mt5_connector.obtener_info_cuenta()
        print("💰 Información de Cuenta:")
        print(f"   • Login: {info['login']}")
        print(f"   • Balance: ${info['balance']:,.2f}")
        print(f"   • Equity: ${info['equity']:,.2f}")
        print(f"   • Margen Libre: ${info['margin_free']:,.2f}")
        print(f"   • Apalancamiento: 1:{info['leverage']}")
        
        # ══════════════════════════════════════════════════════════════
        # PASO 2: DESCARGA DE DATOS HISTÓRICOS
        # ══════════════════════════════════════════════════════════════
        print_header("PASO 2: DESCARGA DE DATOS HISTÓRICOS")
        
        data_manager = DataManager(mt5_connector)
        
        print("📥 Descargando 20,000 velas históricas...")
        print("   (Esto puede tomar 1-2 minutos)\n")
        
        df_historico = data_manager.cargar_datos_historicos(cantidad=20000)
        
        if df_historico is None or len(df_historico) == 0:
            print("❌ Error al descargar datos históricos")
            return
        
        print(f"✅ Datos históricos cargados: {len(df_historico)} velas")
        print(f"   Período: {df_historico['time'].iloc[0]} a {df_historico['time'].iloc[-1]}\n")
        
        # ══════════════════════════════════════════════════════════════
        # PASO 3: OBSERVACIÓN LIVE
        # ══════════════════════════════════════════════════════════════
        print_header("PASO 3: OBSERVACIÓN LIVE")
        
        print("🔴 OBSERVACIÓN EN VIVO")
        print("   Esta fase observa el mercado tick-by-tick durante 1 hora")
        print("   para capturar datos de formación de velas en tiempo real.\n")
        
        respuesta = input("¿Deseas realizar la observación live? (s/n): ").lower()
        
        df_live = None
        
        if respuesta == 's':
            print("\n⏱️  Iniciando observación live por 60 minutos...")
            print("   Puedes detener con Ctrl+C si lo deseas\n")
            
            # Observar mercado en vivo (ticks)
            df_live_ticks = data_manager.observar_mercado_live(
                duracion_minutos=3,  # Cambia esto según necesites
                intervalo_segundos=1
            )
            
            if df_live_ticks is not None and len(df_live_ticks) > 0:
                print(f"✅ Observación completada: {len(df_live_ticks)} ticks capturados\n")
                
                # ✅ FIX: Resamplear ticks a velas ANTES de pasar a features
                print("🔄 Convirtiendo ticks a velas...\n")
                df_combinado = data_manager.agregar_datos_live_a_velas(
                    df_historico.iloc[-500:],  # Últimas 500 velas históricas para contexto
                    df_live_ticks
                )
                
                # Extraer solo las velas live (las últimas agregadas)
                num_velas_live = max(1, len(df_live_ticks) // 60)  # Aproximación: 1 vela cada 60 ticks
                df_live = df_combinado.iloc[-num_velas_live:].copy()
                
                print(f"✅ Velas live generadas: {len(df_live)}\n")
                
                # DEBUG: Verificar columnas
                print("--- DEBUG: Columnas de df_live ---")
                print(df_live.columns.tolist())
                print(df_live.head())
                print("---\n")
                
            else:
                print("⚠️  No se capturaron datos live\n")
        else:
            print("⏭️  Observación live omitida\n")
        
        # ══════════════════════════════════════════════════════════════
        # PASO 4: GENERACIÓN DE FEATURES
        # ══════════════════════════════════════════════════════════════
        print_header("PASO 4: GENERACIÓN DE FEATURES")
        
        feature_engineer = FeatureEngineer()
        
        # Generar features para datos históricos
        print("🔧 Generando features para datos históricos...")
        print("   - Indicadores técnicos (RSI, MACD, ADX, etc.)")
        print("   - Patrones de velas")
        print("   - Soportes y resistencias")
        print("   - Impulsos y retrocesos")
        print("   - Análisis de volatilidad\n")
        
        df_historico_features = feature_engineer.generar_todas_features(df_historico)
        
        if df_historico_features is None:
            print("❌ Error al generar features históricas")
            return
        
        print(f"✅ Features generadas: {len(df_historico_features.columns)} columnas")
        print(f"   Datos válidos: {len(df_historico_features)} filas\n")
        
        # Generar features para datos live (si existen)
        df_live_features = None
        
        if df_live is not None and len(df_live) > 0:
            print("🔧 Generando features para datos live...\n")
            
            # ✅ Verificar que tenga la columna 'close' antes de pasar a features
            if 'close' not in df_live.columns:
                print("⚠️  Falta columna 'close'. Intentando corregir...")
                if 'last' in df_live.columns:
                    df_live['close'] = df_live['last']
                    print("✅ Columna 'close' creada desde 'last'\n")
                else:
                    print("❌ No se puede crear 'close'. Omitiendo features live.\n")
                    df_live = None
            
            if df_live is not None:
                df_live_features = feature_engineer.generar_todas_features(df_live)
                
                if df_live_features is not None:
                    print(f"✅ Features live generadas: {len(df_live_features)} filas\n")
                else:
                    print("⚠️  No se pudieron generar features live\n")
        
        # ══════════════════════════════════════════════════════════════
        # PASO 5: ENTRENAMIENTO
        # ══════════════════════════════════════════════════════════════
        print_header("PASO 5: ENTRENAMIENTO DE MODELOS")
        
        print("📊 Selecciona el tipo de entrenamiento:\n")
        print("   1. Solo datos históricos (más rápido)")
        print("   2. Híbrido (históricos + live, más preciso)")
        print("   3. Ambos (recomendado)\n")
        
        opcion = input("Opción (1/2/3): ").strip()
        
        if opcion in ['1', '3']:
            print("\n🎯 Entrenando con datos históricos...\n")
            
            trainer_historico = HistoricalTrainer()
            modelo_historico = trainer_historico.entrenar(df_historico_features)
            
            if modelo_historico:
                print("✅ Modelo histórico entrenado exitosamente\n")
            else:
                print("❌ Error en entrenamiento histórico\n")
        
        if opcion in ['2', '3'] and df_live_features is not None:
            print("\n🎯 Entrenando con datos híbridos...\n")
            
            trainer_hibrido = HybridTrainer()
            modelo_hibrido = trainer_hibrido.entrenar(
                df_historico_features,
                df_live_features
            )
            
            if modelo_hibrido:
                print("✅ Modelo híbrido entrenado exitosamente\n")
            else:
                print("❌ Error en entrenamiento híbrido\n")
        
        # ══════════════════════════════════════════════════════════════
        # FINALIZACIÓN
        # ══════════════════════════════════════════════════════════════
        print_header("ENTRENAMIENTO COMPLETADO", "=")
        
        print("✅ Proceso finalizado exitosamente")
        print(f"⏱️  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Desconectar MT5
        mt5_connector.desconectar()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        print("❌ Proceso terminado con errores\n")


if __name__ == "__main__":
    main()