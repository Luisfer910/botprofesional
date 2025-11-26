"""
Script de entrenamiento completo del bot
Versión: 2.0
"""

import sys
import os
from datetime import datetime
import json

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from training.historical_trainer import HistoricalTrainer
from training.hybrid_trainer import HybridTrainer

def print_header(texto):
    print(f"\n{'─'*70}")
    print(f"  {texto}")
    print(f"{'─'*70}\n")

def main():
    print(f"\n{'='*70}")
    print(f"  🚀 BOT DE TRADING XM - ENTRENAMIENTO COMPLETO")
    print(f"{'='*70}\n")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # =====================================================================
        # PASO 1: CONEXIÓN A MT5
        # =====================================================================
        print_header("PASO 1: CONEXIÓN A MT5")

        mt5 = MT5Connector()
        if not mt5.conectar():
            print("❌ No se pudo conectar a MT5")
            return

        print("✅ Conectado exitosamente\n")

        # Mostrar info de cuenta
        info = mt5.obtener_info_cuenta()
        if info:
            print("💰 Información de Cuenta:")
            print(f"   • Login: {info.get('login', 'N/D')}")
            print(f"   • Balance: ${info.get('balance', 0):,.2f}")
            print(f"   • Equity: ${info.get('equity', 0):,.2f}")
            # --- CORRECCIÓN ---
            if 'margin_libre' in info:
                print(f"   • Margen Libre: ${info['margin_libre']:,.2f}")
            else:
                print(f"   • Margen Libre: N/D")
            print(f"   • Apalancamiento: 1:{info.get('leverage', 'N/D')}")
        # =====================================================================
        # PASO 2: DESCARGA DE DATOS HISTÓRICOS
        # =====================================================================
        print_header("PASO 2: DESCARGA DE DATOS HISTÓRICOS")

        print("📥 Descargando 20,000 velas históricas...")
        print("   (Esto puede tomar 1-2 minutos)\n")

        data_manager = DataManager(mt5)

        # --- CORRECCIÓN ---
        df_historico = data_manager.obtener_datos_historicos(cantidad=20000)

        if df_historico is None or len(df_historico) == 0:
            print("❌ No se pudieron obtener datos históricos")
            # --- CORRECCIÓN: método cerrar puede no existir ---
            # try:
            #     mt5.cerrar()
            # except AttributeError:
            #     pass
            return

        print(f"✅ {len(df_historico)} velas descargadas")
        print(f"   📅 Desde: {df_historico['time'].iloc[0]}")
        print(f"   📅 Hasta: {df_historico['time'].iloc[-1]}")

        # =====================================================================
        # PASO 3: GENERACIÓN DE FEATURES
        # =====================================================================
        print_header("PASO 3: GENERACIÓN DE FEATURES")

        print("🔧 Generando features técnicas...")

        feature_engineer = FeatureEngineer()

        df_features = feature_engineer.generar_todas_features(df_historico)

        if df_features is None or len(df_features) == 0:
            print("❌ No se pudieron generar features")
            # --- CORRECCIÓN: método cerrar puede no existir ---
            # try:
            #     mt5.cerrar()
            # except AttributeError:
            #     pass
            return

        print(f"✅ Features generadas: {len(df_features.columns)} columnas")
        print(f"   📊 Datos disponibles: {len(df_features)} filas")

        print("\n🎯 Creando variable target...")
        df_features = feature_engineer.crear_target(df_features, horizonte=1)

        print(f"✅ Target creado")
        print(f"   📊 Datos finales: {len(df_features)} filas")

        # =====================================================================
        # PASO 4: ENTRENAMIENTO MODELO HISTÓRICO
        # =====================================================================
        print_header("PASO 4: ENTRENAMIENTO MODELO HISTÓRICO")

        print("🤖 Entrenando modelo con datos históricos...")
        print("   (Esto puede tomar 2-5 minutos)\n")

        trainer = HistoricalTrainer()

        
        X = df_features.drop(columns=["target"])
        y = df_features["target"]
        modelo_historico, metricas = trainer.entrenar_modelo(X, y)

        if modelo_historico is None:
            print("❌ Error en el entrenamiento")
            # --- CORRECCIÓN: método cerrar puede no existir ---
            # try:
            #     mt5.cerrar()
            # except AttributeError:
            #     pass
            return

        print("\n✅ Modelo histórico entrenado exitosamente")
        print(f"\n📊 MÉTRICAS DEL MODELO:")
        print(f"   • Accuracy: {metricas['accuracy']:.2%}")
        print(f"   • Precision: {metricas['precision']:.2%}")
        print(f"   • Recall: {metricas['recall']:.2%}")
        print(f"   • F1-Score: {metricas['f1']:.2%}")

        print("\n💾 Guardando modelo histórico...")
        modelo_path = trainer.guardar_modelo(modelo_historico)
        print(f"   ✅ Guardado en: {modelo_path}")

        # =====================================================================
        # PASO 5: OBSERVACIÓN EN VIVO (OPCIONAL)
        # =====================================================================
        print_header("PASO 5: OBSERVACIÓN EN VIVO (OPCIONAL)")

        print("¿Deseas observar el mercado en vivo para refinar el modelo?")
        print("(Recomendado: 30-60 minutos)")
        print("\nOpciones:")
        print("  1. Sí, observar 30 minutos")
        print("  2. Sí, observar 60 minutos")
        print("  3. No, usar solo modelo histórico")

        opcion = input("\nSelecciona opción (1-3): ")

        df_live = None

        if opcion == '1':
            print("\n🎯 Observando mercado por 30 minutos...")
            print("   (Capturando ticks en tiempo real)\n")
            df_ticks = data_manager.capturar_ticks_tiempo_real(duracion_segundos=1800)
            if df_ticks is not None and len(df_ticks) > 0:
                print(f"\n✅ {len(df_ticks)} ticks capturados")
                df_live = data_manager.obtener_datos_historicos(cantidad=100)
        elif opcion == '2':
            print("\n🎯 Observando mercado por 60 minutos...")
            print("   (Capturando ticks en tiempo real)\n")
            df_ticks = data_manager.capturar_ticks_tiempo_real(duracion_segundos=3600)
            if df_ticks is not None and len(df_ticks) > 0:
                print(f"\n✅ {len(df_ticks)} ticks capturados")
                df_live = data_manager.obtener_datos_historicos(cantidad=100)

        # =====================================================================
        # PASO 6: MODELO HÍBRIDO (SI HAY DATOS LIVE)
        # =====================================================================
        if df_live is not None and len(df_live) > 0:
            print_header("PASO 6: CREACIÓN DE MODELO HÍBRIDO")

            print("🔀 Combinando modelo histórico con datos live...")

            df_live_features = feature_engineer.generar_todas_features(df_live)
            df_live_features = feature_engineer.crear_target(df_live_features, horizonte=1)

            hybrid_trainer = HybridTrainer()
            modelo_hibrido, metricas_hibrido = hybrid_trainer.crear_modelo_hibrido(
                modelo_historico,
                df_live_features
            )

            if modelo_hibrido is not None:
                print("\n✅ Modelo híbrido creado exitosamente")
                print(f"\n📊 MÉTRICAS DEL MODELO HÍBRIDO:")
                print(f"   • Accuracy: {metricas_hibrido['accuracy']:.2%}")
                print(f"   • Precision: {metricas_hibrido['precision']:.2%}")
                print(f"   • Recall: {metricas_hibrido['recall']:.2%}")
                print(f"   • F1-Score: {metricas_hibrido['f1']:.2%}")

                print("\n💾 Guardando modelo híbrido...")
                modelo_hibrido_path = hybrid_trainer.guardar_modelo(modelo_hibrido)
                print(f"   ✅ Guardado en: {modelo_hibrido_path}")

        # =====================================================================
        # RESUMEN FINAL
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}\n")

        print("📦 Modelos generados:")
        print(f"   • Modelo histórico: ✅")
        if df_live is not None:
            print(f"   • Modelo híbrido: ✅")

        print(f"\n🎯 Próximos pasos:")
        print(f"   1. Ejecuta: python main.py")
        print(f"   2. Selecciona modo de operación")
        print(f"   3. ¡Deja que el bot opere!\n")

        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # --- CORRECCIÓN: método cerrar puede no existir ---
        # Si tienes un método para cerrar la conexión, úsalo aquí. Si no, ignora.
        # try:
        #     mt5.cerrar()
        # except AttributeError:
        #     pass

    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        # --- CORRECCIÓN: método cerrar puede no existir ---
        # try:
        #     mt5.cerrar()
        # except AttributeError:
        #     pass

    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        # --- CORRECCIÓN: método cerrar puede no existir ---
        # try:
        #     mt5.cerrar()
        # except AttributeError:
        #     pass
        print("❌ Proceso terminado con errores\n")

if __name__ == "__main__":
    main()
