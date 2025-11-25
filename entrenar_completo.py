"""
Script de entrenamiento completo del bot
Versión corregida v3.4 - Fix config
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import traceback

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from core.mt5_connector import MT5Connector
from core.feature_engineer import FeatureEngineer
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer
from core.data_manager import DataManager

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(numero, texto):
    """Imprime un paso del proceso"""
    print("\n" + "─" * 70)
    print(f"  PASO {numero}: {texto}")
    print("─" * 70 + "\n")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"✅ {text}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"❌ {text}")

def print_info(text):
    """Imprime mensaje informativo"""
    print(f"ℹ️  {text}")

def main():
    """Función principal de entrenamiento"""
    
    print_header("🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # ============================================================
        # PASO 1: CONEXIÓN A MT5
        # ============================================================
        print_step(1, "CONEXIÓN A MT5")
        
        mt5 = MT5Connector(config_path='config/xm_config.json')
        
        if not mt5.conectar():
            print_error("No se pudo conectar a MT5")
            return False
        
        print_success("Conectado exitosamente")
        
        # Obtener información de la cuenta
        info = mt5.obtener_info_cuenta()
        if info:
            print(f"\n💰 Información de Cuenta:")
            print(f"   • Login: {info.get('login', 'N/A')}")
            print(f"   • Balance: ${info['balance']:,.2f}")
            print(f"   • Equity: ${info['equity']:,.2f}")
            print(f"   • Margen Libre: ${info.get('margin_free', 0):,.2f}")
            print(f"   • Apalancamiento: 1:{info.get('leverage', 'N/A')}")
        
        # ============================================================
        # PASO 2: DESCARGA DE DATOS HISTÓRICOS
        # ============================================================
        print_step(2, "DESCARGA DE DATOS HISTÓRICOS")
        
        data_manager = DataManager(mt5)
        
        print("📥 Descargando 20,000 velas históricas...")
        print("   (Esto puede tomar 1-2 minutos)\n")
        
        df_historico = data_manager.cargar_datos_historicos(cantidad=20000)
        
        if df_historico is None or len(df_historico) == 0:
            print_error("No se pudieron cargar datos históricos")
            mt5.desconectar() 
            return False
        
        print_success(f"Datos históricos cargados: {len(df_historico)} velas")
        print(f"   Período: {df_historico['time'].iloc[0]} a {df_historico['time'].iloc[-1]}\n")
        
        # ============================================================
        # PASO 3: OBSERVACIÓN LIVE (OPCIONAL)
        # ============================================================
        print_step(3, "OBSERVACIÓN LIVE")
        
        print("🔴 OBSERVACIÓN EN VIVO")
        print("   Esta fase observa el mercado tick-by-tick durante 1 hora")
        print("   para capturar datos de formación de velas en tiempo real.\n")
        
        respuesta = input("¿Deseas realizar la observación live? (s/n): ")
        
        df_live = None
        
        if respuesta.lower() == 's':
            print("\n⏱️  Iniciando observación live por 60 minutos...")
            print("   Puedes detener con Ctrl+C si lo deseas\n")
            
            try:
                df_live = data_manager.observar_mercado_live(duracion_minutos=60)
                
                if df_live is not None and len(df_live) > 0:
                    print_success(f"Observación completada: {len(df_live)} ticks capturados")
                else:
                    print_info("No se capturaron datos live")
                    df_live = None
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Observación interrumpida por el usuario")
                df_live = data_manager.obtener_datos_live()
                
                if df_live is not None and len(df_live) > 0:
                    print(f"   Datos parciales capturados: {len(df_live)} ticks")
                else:
                    df_live = None
        else:
            print_info("Observación live omitida")
            print("   El modelo se entrenará solo con datos históricos\n")
        
        # ============================================================
        # PASO 4: GENERACIÓN DE FEATURES
        # ============================================================
        print_step(4, "GENERACIÓN DE FEATURES")
        
        feature_engineer = FeatureEngineer()
        
        print("🔧 Generando features para datos históricos...")
        print("   - Indicadores técnicos (RSI, MACD, ADX, etc.)")
        print("   - Patrones de velas")
        print("   - Soportes y resistencias")
        print("   - Impulsos y retrocesos")
        print("   - Análisis de volatilidad\n")
        
        df_historico_features = feature_engineer.generar_todas_features(df_historico)
        
        if df_historico_features is None or len(df_historico_features) == 0:
            print_error("No se pudieron generar features")
            mt5.desconectar() 
            return False
        
        print_success(f"Features generadas: {len(df_historico_features.columns)} columnas")
        print(f"   Datos válidos: {len(df_historico_features)} filas\n")
        
        # Features para datos live (si existen)
        df_live_features = None
        
        if df_live is not None:
            print("🔧 Generando features para datos live...")
            df_live_features = feature_engineer.generar_todas_features(df_live)
            
            if df_live_features is not None and len(df_live_features) > 0:
                print_success(f"Features live generadas: {len(df_live_features)} filas\n")
            else:
                print_info("No se pudieron generar features live\n")
                df_live_features = None
        
        # ============================================================
        # PASO 5: ENTRENAMIENTO MODELO HISTÓRICO
        # ============================================================
        print_step(5, "ENTRENAMIENTO MODELO HISTÓRICO")
        
        # >>> FIX: Pasar config={} al constructor <<<
        historical_trainer = HistoricalTrainer(config={})
        
        print("🧠 Entrenando modelo con Random Forest...")
        print("   (Esto puede tomar 2-5 minutos)\n")
        
        # Preparar datos
        X, y = historical_trainer.preparar_datos(df_historico_features)
        
        if X is None or y is None:
            print_error("No se pudieron preparar los datos")
            mt5.desconectar()
            return False
        
        # Entrenar modelo
        modelo_historico = historical_trainer.entrenar_modelo(X, y)
        
        if modelo_historico is None:
            print_error("No se pudo entrenar el modelo histórico")
            mt5.desconectar() 
            return False
        
        print_success("Modelo histórico entrenado exitosamente\n")
        
        # ============================================================
        # PASO 6: GUARDADO DEL MODELO
        # ============================================================
        print_step(6, "GUARDADO DEL MODELO")
        
        print("💾 Guardando modelo en carpeta 'models/'...\n")
        
        try:
            resultado = historical_trainer.guardar_modelo(modelo_historico)
            
            if resultado is not None:
                if isinstance(resultado, tuple):
                    path, metadata = resultado
                    if path and os.path.exists(path):
                        print_success(f"Modelo guardado: {path}")
                    else:
                        print_error("Error: archivo no creado")
                else:
                    print_error("Error: formato de resultado inválido")
            else:
                print_error("Error al guardar el modelo")
                
        except Exception as e:
            print_error(f"Excepción al guardar: {str(e)}")
            traceback.print_exc()
        
        # ============================================================
        # RESUMEN FINAL
        # ============================================================
        print_header("✅ ENTRENAMIENTO COMPLETADO")
        
        print("📊 RESUMEN:")
        print(f"   Datos históricos:  {len(df_historico)} velas")
        
        if df_live is not None:
            print(f"   Datos live:        {len(df_live)} ticks")
        
        print(f"   Features:          {len(df_historico_features.columns)}")
        print(f"   Modelo:            Entrenado con {len(X)} muestras")
        
        print(f"\n🎯 PRÓXIMOS PASOS:")
        print(f"   1. python inicio_rapido.py  → Verificar instalación")
        print(f"   2. python main.py           → Iniciar bot de trading")
        print(f"   3. Selecciona modo automático")
        print(f"   4. ¡Deja que el bot opere!\n")
        
        print("=" * 70 + "\n")
        
        # Cerrar conexión
        mt5.desconectar() 
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        return False
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        exito = main()
        if exito:
            print("✅ Proceso completado exitosamente")
        else:
            print("❌ Proceso terminado con errores")
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        traceback.print_exc()
