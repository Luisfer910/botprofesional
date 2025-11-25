#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOT DE TRADING XM - MAIN
Sistema de trading automatizado con IA
"""

import os
import time
import signal
from datetime import datetime
import json
import joblib
import glob
import sys

# Agregar directorio actual al path para evitar problemas de importación
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from strategy.signal_generator import SignalGenerator
from strategy.risk_manager import RiskManager
from strategy.trade_executor import TradeExecutor as OrderExecutor

class TradingBot:
    """Bot de Trading Principal"""
    
    def __init__(self, config_path='config/xm_config.json'):
        """Inicializa el bot de trading"""
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Componentes
        self.mt5 = None
        self.data_manager = None
        self.feature_engineer = None
        self.modelo_hibrido = None
        self.signal_generator = None
        self.risk_manager = None
        self.order_executor = None
        
        # Estado
        self.running = False
        self.modo = None
        
        # Configurar señales de interrupción
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de interrupción"""
        print("\n\n⚠️  Señal de interrupción recibida...")
        print("🛑 Deteniendo bot de forma segura...")
        self.stop()
        sys.exit(0)
    
    def inicializar(self):
        """Inicializa todos los componentes del bot"""
        try:
            print("\n" + "="*70)
            print("  🤖 BOT DE TRADING XM - INICIALIZANDO")
            print("="*70 + "\n")
            
            print("📋 INICIALIZANDO COMPONENTES...\n")
            
            # 1. Conectar a MT5
            print("1️⃣  Conectando a MT5...")
            self.mt5 = MT5Connector(config_path='config/xm_config.json')
            if not self.mt5.conectar():
                print("❌ Error: No se pudo conectar a MT5")
                return False
            print("   ✅ Conectado\n")
            
            # 2. Data Manager
            print("2️⃣  Inicializando Data Manager...")
            self.data_manager = DataManager(self.mt5)
            print("   ✅ Listo\n")
            
            # 3. Feature Engineer
            print("3️⃣  Inicializando Feature Engineer...")
            self.feature_engineer = FeatureEngineer()
            print("   ✅ Listo\n")
            
            # 4. Cargar Modelo
            print("4️⃣  Cargando modelo de IA...")
            if not self._cargar_modelo():
                print("❌ Error: No se pudo cargar el modelo")
                print("   Ejecuta 'python entrenar_completo.py' primero")
                return False
            
            # 5. Signal Generator
            print("5️⃣  Inicializando Signal Generator...")
            self.signal_generator = SignalGenerator(
                self.modelo_hibrido,
                self.feature_engineer,
                config=self.config
            )
            print("   ✅ Listo\n")
            
            # 6. Risk Manager
            print("6️⃣  Inicializando Risk Manager...")
            self.risk_manager = RiskManager(self.mt5)
            print("   ✅ Listo\n")
            
            # 7. Order Executor
            print("7️⃣  Inicializando Order Executor...")
            self.order_executor = OrderExecutor(self.mt5, self.risk_manager)
            print("   ✅ Listo\n")
            
            print("="*70)
            print("  ✅ TODOS LOS COMPONENTES INICIALIZADOS")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error durante inicialización: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cargar_modelo(self):
        """Carga el modelo híbrido"""
        try:
            # Buscar modelos en la carpeta
            modelos = glob.glob('models/modelo_hibrido_*.pkl')
            
            if not modelos:
                print("   ⚠️  No se encontraron modelos en 'models/'")
                print("   📂 Archivos en models/:")
                
                # Listar todos los archivos en models/
                if os.path.exists('models'):
                    archivos = os.listdir('models')
                    if archivos:
                        for archivo in archivos:
                            print(f"      - {archivo}")
                    else:
                        print("      (carpeta vacía)")
                else:
                    print("      (carpeta no existe)")
                
                return False
            
            # Cargar el más reciente
            modelo_path = max(modelos, key=os.path.getctime)
            print(f"   📂 Cargando: {os.path.basename(modelo_path)}")
            
            self.modelo_hibrido = joblib.load(modelo_path)
            
            print(f"   ✅ Modelo cargado exitosamente")
            print(f"   📊 Tipo: {type(self.modelo_hibrido).__name__}\n")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error al cargar modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def seleccionar_modo(self):
        """Permite al usuario seleccionar el modo de operación"""
        print("\n" + "="*70)
        print("  🎯 SELECCIÓN DE MODO")
        print("="*70 + "\n")
        
        print("Modos disponibles:")
        print("  1. 🤖 Automático (el bot opera solo)")
        print("  2. 📊 Semi-automático (el bot sugiere, tú decides)")
        print("  3. 👁️  Solo observación (sin operar)")
        print("  0. ❌ Salir\n")
        
        while True:
            try:
                opcion = input("Selecciona modo (0-3): ").strip()
                
                if opcion == '0':
                    return None
                elif opcion == '1':
                    self.modo = 'automatico'
                    print("\n✅ Modo automático activado")
                    return 'automatico'
                elif opcion == '2':
                    self.modo = 'semi_automatico'
                    print("\n✅ Modo semi-automático activado")
                    return 'semi_automatico'
                elif opcion == '3':
                    self.modo = 'observacion'
                    print("\n✅ Modo observación activado")
                    return 'observacion'
                else:
                    print("❌ Opción inválida. Intenta de nuevo.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Operación cancelada")
                return None
    
    def ejecutar(self):
        """Ejecuta el loop principal del bot"""
        try:
            self.running = True
            
            print("\n" + "="*70)
            print(f"  🚀 BOT INICIADO - MODO: {self.modo.upper()}")
            print("="*70 + "\n")
            
            print("📊 Información:")
            print(f"   • Par: {self.config['TRADING']['SYMBOL']}")
            print(f"   • Timeframe: {self.config['TRADING']['TIMEFRAME']}")
            print(f"   • Modo: {self.modo}")
            print(f"   • Presiona Ctrl+C para detener\n")
            
            ciclo = 0
            
            while self.running:
                ciclo += 1
                print(f"\n{'─'*70}")
                print(f"🔄 Ciclo #{ciclo} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'─'*70}\n")
                
                # 1. Obtener datos - CORREGIDO: usar cargar_datos_historicos con cantidad
                print("📥 Obteniendo datos del mercado...")
                velas = self.data_manager.cargar_datos_historicos(
                    cantidad=self.config['TRADING']['VELAS_ANALISIS']
                )
                
                if velas is None or len(velas) == 0:
                    print("❌ No se pudieron obtener datos")
                    time.sleep(60)
                    continue
                
                print(f"✅ {len(velas)} velas obtenidas\n")
                
                # 2. Generar señal
                print("🧠 Analizando mercado...")
                senal = self.signal_generator.generar_senal(velas)
                
                if senal is None:
                    print("⚠️  No se pudo generar señal")
                    time.sleep(60)
                    continue
                
                # Mostrar señal
                self._mostrar_senal(senal)
                
                # 3. Evaluar riesgo
                if senal['tipo'] != 'HOLD':
                    print("\n⚖️  Evaluando riesgo...")
                    puede_operar = self.risk_manager.puede_operar()
                    
                    if not puede_operar:
                        print("❌ No se puede operar (límites de riesgo)")
                        time.sleep(60)
                        continue
                    
                    print("✅ Riesgo aceptable\n")
                    
                    # 4. Ejecutar según modo
                    if self.modo == 'automatico':
                        self._ejecutar_automatico(senal)
                    elif self.modo == 'semi_automatico':
                        self._ejecutar_semi_automatico(senal)
                    # En modo observación no hace nada
                
                # 5. Mostrar estadísticas
                self._mostrar_estadisticas()
                
                # 6. Esperar siguiente ciclo
                intervalo = self.config['TRADING'].get('INTERVALO_CICLOS', 60)
                print(f"\n⏳ Esperando {intervalo}s hasta próximo ciclo...")
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada")
            self.stop()
        except Exception as e:
            print(f"\n❌ Error en loop principal: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def _mostrar_senal(self, senal):
        """Muestra la señal generada"""
        tipo = senal['tipo']
        prob = senal.get('probabilidad', 0)
        conf = senal.get('confianza', 0)
        
        if tipo == 'CALL':
            emoji = "📈"
            color = "verde"
        elif tipo == 'PUT':
            emoji = "📉"
            color = "rojo"
        else:
            emoji = "⏸️"
            color = "amarillo"
        
        print(f"\n{emoji} SEÑAL: {tipo}")
        print(f"   • Probabilidad: {prob:.1f}%")
        print(f"   • Confianza: {conf:.1f}%")
        
        if 'motivo' in senal:
            print(f"   • Motivo: {senal['motivo']}")
    
    def _ejecutar_automatico(self, senal):
        """Ejecuta trade automáticamente"""
        print("\n🤖 Ejecutando trade automático...")
        
        resultado = self.order_executor.ejecutar_trade(senal)
        
        if resultado['exito']:
            print(f"✅ Trade ejecutado: {resultado['ticket']}")
        else:
            print(f"❌ Error: {resultado['mensaje']}")
    
    def _ejecutar_semi_automatico(self, senal):
        """Pide confirmación antes de ejecutar"""
        print("\n📊 ¿Ejecutar este trade?")
        print("   1. ✅ Sí")
        print("   2. ❌ No")
        
        try:
            opcion = input("\nSelecciona (1-2): ").strip()
            
            if opcion == '1':
                self._ejecutar_automatico(senal)
            else:
                print("❌ Trade cancelado por el usuario")
                
        except KeyboardInterrupt:
            print("\n❌ Trade cancelado")
    
    def _mostrar_estadisticas(self):
        """Muestra estadísticas del bot"""
        print("\n" + "─"*70)
        print("📊 ESTADÍSTICAS")
        print("─"*70)
        
        # Info de cuenta
        info = self.mt5.obtener_info_cuenta()
        if info:
            print(f"\n💰 Cuenta:")
            print(f"   • Balance: ${info['balance']:.2f}")
            print(f"   • Equity: ${info['equity']:.2f}")
            print(f"   • Profit: ${info['profit']:.2f}")
        
        # Estadísticas de riesgo
        stats_riesgo = self.risk_manager.obtener_estadisticas()
        print(f"\n⚠️  Gestión de Riesgo:")
        print(f"   • Trades hoy: {stats_riesgo.get('trades_hoy', 0)}")
        print(f"   • Pérdida diaria: ${stats_riesgo.get('perdida_diaria', 0):.2f}")
        print(f"   • Drawdown: {stats_riesgo.get('drawdown', 0):.1f}%")
    
    def stop(self):
        """Detiene el bot"""
        print("\n🛑 Deteniendo bot...")
        self.running = False
        
        if self.mt5:
            self.mt5.desconectar()
        
        print("✅ Bot detenido correctamente\n")

def main():
    """Función principal"""
    # Crear bot
    bot = TradingBot()
    
    # Inicializar
    if not bot.inicializar():
        print("\n❌ No se pudo inicializar el bot")
        return
    
    # Seleccionar modo
    modo = bot.seleccionar_modo()
    if modo is None:
        print("\n👋 Saliendo...")
        return
    
    # Ejecutar
    bot.ejecutar()

if __name__ == "__main__":
    main()
