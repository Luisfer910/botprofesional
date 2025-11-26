#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOT DE TRADING XM - VERSIÓN PROFESIONAL
Con análisis tick-by-tick y aprendizaje en vivo
"""

import sys
import os
import time
import signal
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from strategy.signal_generator import SignalGenerator
from strategy.risk_manager import RiskManager
from strategy.trade_executor import TradeExecutor
from training.continuous_learner import ContinuousLearner

class TradingBot:
    def __init__(self):
        with open('config/xm_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.running = False
        self.ciclo = 0
        
        # Componentes
        self.mt5 = None
        self.data_manager = None
        self.feature_engineer = None
        self.signal_generator = None
        self.risk_manager = None
        self.trade_executor = None
        self.continuous_learner = None
        self.modelo = None
        
        print(f"\n{'='*70}")
        print(f"  🤖 BOT DE TRADING XM - INICIALIZANDO")
        print(f"{'='*70}\n")
    
    def inicializar(self):
        """Inicializa todos los componentes"""
        print("📋 INICIALIZANDO COMPONENTES...\n")
        
        # 1. MT5
        print("1️⃣  Conectando a MT5...")
        self.mt5 = MT5Connector()
        if not self.mt5.conectar():
            print("❌ Error de conexión")
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
        
        # 4. Cargar modelo
        print("4️⃣  Cargando modelo de IA...")
        if not self._cargar_modelo():
            print("   ❌ Error al cargar modelo")
            return False
        print("   ✅ Modelo cargado\n")
        
        # 5. Signal Generator
        print("5️⃣  Inicializando Signal Generator...")
        self.signal_generator = SignalGenerator(self.modelo, self.feature_engineer)
        print("   ✅ Listo\n")
        
        # 6. Risk Manager
        print("6️⃣  Inicializando Risk Manager...")
        self.risk_manager = RiskManager(self.mt5)
        print("   ✅ Listo\n")
        
        # 7. Trade Executor
        print("7️⃣  Inicializando Order Executor...")
        self.trade_executor = TradeExecutor(self.mt5, self.risk_manager)
        print("   ✅ Listo\n")
        
        # 8. Continuous Learner
        print("8️⃣  Inicializando Continuous Learner...")
        self.continuous_learner = ContinuousLearner(self.modelo, self.feature_engineer)
        print("   ✅ Listo\n")
        
        print(f"{'='*70}")
        print(f"  ✅ TODOS LOS COMPONENTES INICIALIZADOS")
        print(f"{'='*70}\n")
        
        return True
    
    def _cargar_modelo(self):
        """Carga el modelo más reciente"""
        try:
            import pickle
            import glob
            
            # Buscar modelos
            modelos = glob.glob('models/*.pkl')
            
            if not modelos:
                print("   ⚠️ No hay modelos entrenados")
                print("   💡 Ejecuta 'python entrenar_completo.py'")
                return False
            
            # Ordenar por fecha (más reciente primero)
            modelos.sort(reverse=True)
            
            # Intentar cargar modelos hasta encontrar uno válido
            for modelo_path in modelos:
                try:
                    with open(modelo_path, 'rb') as f:
                        modelo_cargado = pickle.load(f)
                    
                    # Verificar que sea un modelo válido (no un array)
                    if hasattr(modelo_cargado, 'predict'):
                        self.modelo = modelo_cargado
                        print(f"   📂 Cargando: {os.path.basename(modelo_path)}")
                        print(f"   ✅ Modelo cargado exitosamente")
                        print(f"   📊 Tipo: {type(self.modelo).__name__}")
                        return True
                    else:
                        print(f"   ⚠️ {os.path.basename(modelo_path)} no es un modelo válido (es {type(modelo_cargado).__name__})")
                        continue
                        
                except Exception as e:
                    print(f"   ⚠️ Error al cargar {os.path.basename(modelo_path)}: {str(e)}")
                    continue
            
            # Si llegamos aquí, no hay modelos válidos
            print("   ❌ No se encontraron modelos válidos")
            print("   💡 Ejecuta 'python entrenar_completo.py' para entrenar un modelo")
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    def ciclo_principal(self):
        """Ciclo principal del bot"""
        self.ciclo += 1
        
        print(f"\n{'─'*70}")
        print(f"🔄 Ciclo #{self.ciclo} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}\n")
        
        try:
            # 1. Obtener datos con análisis tick-by-tick
            print("📥 Obteniendo datos del mercado...")
            df_velas, features_intravela = self.data_manager.obtener_datos_live_con_ticks()
            
            if df_velas is None:
                print("   ⚠️ No se pudieron obtener datos")
                return
            
            print(f"   ✅ Descargadas {len(df_velas)} velas")
            print(f"   📅 Desde: {df_velas['time'].iloc[0]}")
            print(f"   📅 Hasta: {df_velas['time'].iloc[-1]}")
            
            if features_intravela:
                print(f"   🎯 Features intravela capturadas:")
                print(f"      • Presión neta: {features_intravela.get('presion_neta', 0):.3f}")
                print(f"      • Volatilidad: {features_intravela.get('volatilidad_normalizada', 0):.2f} pips")
                print(f"      • Ticks: {features_intravela.get('num_ticks', 0)}")
            
            print(f"✅ {len(df_velas)} velas obtenidas\n")
            
            # 2. Generar señal
            print("🧠 Analizando mercado...\n")
            senal = self.signal_generator.generar_senal(df_velas, features_intravela)
            
            # Mostrar señal
            self._mostrar_senal(senal)
            
            # 3. Ejecutar trade si es necesario (solo en modo automático)
            if hasattr(self, 'modo') and self.modo == 'automatico':
                if senal['tipo'] in ['CALL', 'PUT']:
                    puede_operar, razon = self.risk_manager.puede_operar(senal)
                    
                    if puede_operar:
                        print("\n🚀 Ejecutando trade...")
                        resultado = self.trade_executor.ejecutar_orden(senal)
                        
                        if resultado:
                            print("   ✅ Trade ejecutado")
                            # Agregar experiencia al learner
                            self.continuous_learner.agregar_experiencia(senal, resultado)
                    else:
                        print(f"\n⚠️  Trade rechazado: {razon}")
            
            # 4. Monitorear trades abiertos (CORREGIDO)
            self.trade_executor.monitorear_trades()
            
            # 5. Aprendizaje continuo
            if self.ciclo % 10 == 0:  # Cada 10 ciclos
                print("\n🧠 Ejecutando aprendizaje continuo...")
                self.continuous_learner.aprender()
            
            # 6. Mostrar estadísticas
            self._mostrar_estadisticas()
            
        except Exception as e:
            print(f"\n❌ Error en ciclo: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _mostrar_senal(self, senal):
        """Muestra la señal generada"""
        print(f"\n{'='*70}")
        print(f"  📊 SEÑAL GENERADA")
        print(f"{'='*70}\n")
        
        tipo = senal['tipo']
        if tipo == 'CALL':
            emoji = '🟢'
        elif tipo == 'PUT':
            emoji = '🔴'
        else:
            emoji = '⚪'
        
        print(f"{emoji} Tipo: {tipo}")
        print(f"💪 Fuerza: {senal['fuerza']:.2f}%")
        print(f"📝 Razón: {senal['razon']}")
        
        print(f"\n{'='*70}\n")
    
    def _mostrar_estadisticas(self):
        """Muestra estadísticas de la cuenta"""
        print(f"\n{'─'*70}")
        print(f"📊 ESTADÍSTICAS")
        print(f"{'─'*70}")
        
        info = self.mt5.obtener_info_cuenta()
        if info:
            print(f"💰 Balance: ${info['balance']:.2f}")
            print(f"📈 Equity: ${info['equity']:.2f}")
            print(f"📊 Margen: ${info['margin']:.2f}")
        
        stats = self.trade_executor.obtener_estadisticas()
        if stats:
            try:
                print(f"📈 Win Rate: {stats['win_rate']:.1f}%")
                print(f"💵 Profit Total: ${stats['profit_total']:.2f}")
                print(f"📊 Trades: {stats['total_trades']}")
            except KeyError as e:
                print(f"⚠️  Estadísticas parciales disponibles")
        
        print(f"{'─'*70}\n")
    
    def ejecutar(self, modo='observacion', intervalo=60):
        """Ejecuta el bot"""
        self.modo = modo
        self.running = True
        
        print(f"\n{'='*70}")
        print(f"  🚀 BOT INICIADO - MODO: {modo.upper()}")
        print(f"{'='*70}\n")
        print(f"📊 Información:")
        print(f"   • Par: {self.config['TRADING']['SYMBOL']}")
        print(f"   • Timeframe: {self.config['TRADING']['TIMEFRAME']}")
        print(f"   • Modo: {modo}")
        print(f"   • Presiona Ctrl+C para detener\n")
        
        try:
            while self.running:
                self.ciclo_principal()
                
                print(f"\n⏳ Esperando {intervalo}s hasta próximo ciclo...\n")
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada")
            self.detener()
    
    def detener(self):
        """Detiene el bot"""
        print(f"\n{'='*70}")
        print(f"  🛑 DETENIENDO BOT...")
        print(f"{'='*70}\n")
        
        self.running = False
        
        # Guardar estado del learner
        if self.continuous_learner:
            print("💾 Guardando aprendizaje...")
            self.continuous_learner.guardar_estado()
        
        # Cerrar conexión
        if self.mt5:
            self.mt5.cerrar()
        
        print("\n✅ Bot detenido correctamente\n")

def main():
    """Función principal"""
    
    def signal_handler(sig, frame):
        if 'bot' in globals():
            bot.detener()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Crear bot
    bot = TradingBot()
    
    # Inicializar
    if not bot.inicializar():
        print("\n❌ No se pudo inicializar el bot")
        return
    
    # Menú
    print(f"\n{'='*70}")
    print(f"  🎯 SELECCIÓN DE MODO")
    print(f"{'='*70}\n")
    print("Modos disponibles:")
    print("  1. 🤖 Automático (el bot opera solo)")
    print("  2. 📊 Semi-automático (el bot sugiere, tú decides)")
    print("  3. 👁️  Solo observación (sin operar)")
    print("  0. ❌ Salir\n")
    
    opcion = input("Selecciona modo (0-3): ")
    
    if opcion == '1':
        print("\n✅ Modo automático activado")
        bot.ejecutar(modo='automatico', intervalo=60)
    elif opcion == '2':
        print("\n✅ Modo semi-automático activado")
        bot.ejecutar(modo='semiautomatico', intervalo=60)
    elif opcion == '3':
        print("\n✅ Modo observación activado")
        bot.ejecutar(modo='observacion', intervalo=60)
    else:
        print("\n👋 Saliendo...")
        bot.mt5.cerrar()

if __name__ == "__main__":
    main()
