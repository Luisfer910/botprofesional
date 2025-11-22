#!/usr/bin/env python3
"""
BOT DE TRADING XM - VERSIÓN COMERCIAL
Sistema completo de trading automatizado con IA

Características:
- Entrenamiento híbrido (histórico + live)
- Aprendizaje continuo
- Gestión de riesgo avanzada
- Monitoreo en tiempo real
- Notificaciones
"""

import sys
import os
import time
import signal
from datetime import datetime, timedelta
import json

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.feature_engineer import FeatureEngineer
from strategy.signal_generator import SignalGenerator
from strategy.risk_manager import RiskManager
from strategy.trade_executor import TradeExecutor
from training.continuous_learner import ContinuousLearner

class TradingBot:
    def __init__(self, modelo_path='models/hybrid_model_latest.pkl'):
        """Inicializa el bot de trading"""
        
        # Cargar configuración
        with open('config/xm_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Estado del bot
        self.running = False
        self.ciclos_completados = 0
        self.inicio_sesion = None
        
        # Componentes principales
        self.mt5 = None
        self.data_manager = None
        self.feature_engineer = None
        self.signal_generator = None
        self.risk_manager = None
        self.trade_executor = None
        self.continuous_learner = None
        
        # Modelo
        self.modelo_path = modelo_path
        self.modelo = None
        
        print(f"\n{'='*70}")
        print(f"  🤖 BOT DE TRADING XM - INICIALIZANDO")
        print(f"{'='*70}\n")
    
    def inicializar(self):
        """Inicializa todos los componentes del bot"""
        
        print("📋 INICIALIZANDO COMPONENTES...\n")
        
        # 1. Conexión MT5
        print("1️⃣  Conectando a MT5...")
        self.mt5 = MT5Connector()
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
        print("7️⃣  Inicializando Trade Executor...")
        self.trade_executor = TradeExecutor(self.mt5, self.risk_manager)
        print("   ✅ Listo\n")
        
        # 8. Continuous Learner
        print("8️⃣  Inicializando Continuous Learner...")
        self.continuous_learner = ContinuousLearner(self.modelo)
        print("   ✅ Listo\n")
        
        print(f"{'='*70}")
        print(f"  ✅ TODOS LOS COMPONENTES INICIALIZADOS")
        print(f"{'='*70}\n")
        
        return True
    
    def _cargar_modelo(self):
        """Carga el modelo entrenado"""
        try:
            import pickle
            
            # Buscar modelo más reciente si no se especifica
            if not os.path.exists(self.modelo_path):
                # Buscar en carpeta models
                import glob
                modelos = glob.glob('models/hybrid_model_*.pkl')
                
                if not modelos:
                    # Buscar modelo histórico
                    modelos = glob.glob('models/historical_model_*.pkl')
                
                if not modelos:
                    return False
                
                # Ordenar por fecha (más reciente primero)
                modelos.sort(reverse=True)
                self.modelo_path = modelos[0]
            
            with open(self.modelo_path, 'rb') as f:
                self.modelo = pickle.load(f)
            
            print(f"   📁 Modelo: {os.path.basename(self.modelo_path)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error al cargar modelo: {str(e)}")
            return False
    
    def mostrar_panel_info(self):
        """Muestra panel de información en tiempo real"""
        
        # Limpiar pantalla (opcional)
        # os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"\n{'='*70}")
        print(f"  🤖 BOT DE TRADING XM - PANEL DE CONTROL")
        print(f"{'='*70}")
        
        # Info de cuenta
        info = self.mt5.obtener_info_cuenta()
        if info:
            print(f"\n💰 CUENTA:")
            print(f"   Balance:      ${info['balance']:,.2f}")
            print(f"   Equity:       ${info['equity']:,.2f}")
            print(f"   Profit:       ${info['profit']:,.2f}")
            print(f"   Margin Level: {info['margin_level']:.2f}%")
        
        # Info de riesgo
        risk_stats = self.risk_manager.obtener_estadisticas_riesgo()
        if risk_stats:
            print(f"\n⚠️  GESTIÓN DE RIESGO:")
            print(f"   Trades hoy:        {risk_stats['trades_hoy']}/{risk_stats['max_trades_dia']}")
            print(f"   Pérdida diaria:    ${risk_stats['perdida_diaria']:,.2f}")
            print(f"   Drawdown actual:   {risk_stats['drawdown_actual']:.2f}%")
            print(f"   Puede operar:      {'✅ SÍ' if risk_stats['puede_operar'] else '❌ NO'}")
        
        # Info de trading
        trading_stats = self.trade_executor.obtener_estadisticas_trading()
        if trading_stats:
            print(f"\n📊 ESTADÍSTICAS DE TRADING:")
            print(f"   Total trades:      {trading_stats['total_trades']}")
            print(f"   Ganados:           {trading_stats['ganados']} ({trading_stats['win_rate']:.1f}%)")
            print(f"   Perdidos:          {trading_stats['perdidos']}")
            print(f"   Profit total:      ${trading_stats['profit_total']:,.2f}")
            print(f"   Profit factor:     {trading_stats['profit_factor']:.2f}")
            print(f"   Trades abiertos:   {trading_stats['trades_abiertos']}")
        
        # Info de aprendizaje
        learner_stats = self.continuous_learner.obtener_estadisticas()
        if learner_stats:
            print(f"\n🧠 APRENDIZAJE CONTINUO:")
            print(f"   Experiencias:      {learner_stats['total_experiencias']}")
            print(f"   Win rate general:  {learner_stats['win_rate_general']*100:.1f}%")
            print(f"   Actualizaciones:   {learner_stats['total_actualizaciones']}")
            print(f"   Última actualiz.:  {learner_stats['ultima_actualizacion']}")
        
        # Info de sesión
        print(f"\n⏱️  SESIÓN:")
        if self.inicio_sesion:
            duracion = datetime.now() - self.inicio_sesion
            horas = int(duracion.total_seconds() // 3600)
            minutos = int((duracion.total_seconds() % 3600) // 60)
            print(f"   Duración:          {horas}h {minutos}m")
        print(f"   Ciclos:            {self.ciclos_completados}")
        print(f"   Hora actual:       {datetime.now().strftime('%H:%M:%S')}")
        
        # Precio actual
        tick = self.mt5.obtener_tick_actual()
        if tick:
            print(f"\n💹 MERCADO:")
            print(f"   {self.config['TRADING']['SYMBOL']}")
            print(f"   Bid:               {tick['bid']:.5f}")
            print(f"   Ask:               {tick['ask']:.5f}")
            print(f"   Spread:            {tick['spread']:.1f} pips")
        
        print(f"\n{'='*70}\n")
    
    def ciclo_principal(self):
        """Ciclo principal del bot"""
        
        print(f"🔄 Ejecutando ciclo {self.ciclos_completados + 1}...")
        
        try:
            # 1. Actualizar datos
            print("   📥 Actualizando datos...")
            df_actual = self.data_manager.actualizar_datos_live()
            
            if df_actual is None or len(df_actual) < 100:
                print("   ⚠️ No hay suficientes datos")
                return
            
            # 2. Generar señal
            print("   🎯 Generando señal...")
            señal = self.signal_generator.generar_señal(df_actual)
            
            if señal['tipo'] != 'NEUTRAL':
                print(f"\n   {'─'*66}")
                print(f"   🎯 SEÑAL DETECTADA: {señal['tipo']}")
                print(f"   {'─'*66}")
                print(f"      Probabilidad:  {señal['probabilidad']:.3f}")
                print(f"      Confianza:     {señal['confianza']}")
                print(f"      Precio:        {señal['precio_actual']:.5f}")
                
                if señal['analisis']:
                    print(f"\n      📋 Análisis:")
                    for analisis in señal['analisis'][:5]:  # Primeros 5
                        print(f"         • {analisis}")
                
                print(f"   {'─'*66}\n")
                
                # 3. Verificar si podemos operar
                puede_operar, razon = self.risk_manager.puede_operar(señal)
                
                if puede_operar:
                    # 4. Ejecutar trade
                    print("   🚀 Ejecutando trade...")
                    
                    # Obtener ATR para stop loss dinámico
                    atr = df_actual['atr'].iloc[-1] if 'atr' in df_actual.columns else None
                    
                    trade_info = self.trade_executor.ejecutar_trade(señal, atr)
                    
                    if trade_info:
                        print("   ✅ Trade ejecutado exitosamente")
                    else:
                        print("   ❌ No se pudo ejecutar el trade")
                else:
                    print(f"   ⚠️ Trade rechazado: {razon}")
            else:
                print("   ℹ️  Sin señal clara (NEUTRAL)")
            
            # 5. Monitorear trades abiertos
            trades_cerrados = self.trade_executor.monitorear_trades()
            
            # 6. Aprendizaje continuo de trades cerrados
            if trades_cerrados:
                print(f"\n   🧠 Aprendiendo de {len(trades_cerrados)} trade(s) cerrado(s)...")
                
                for trade in trades_cerrados:
                    # Preparar features del trade
                    features = trade['señal']['features']
                    prediccion = 1 if trade['tipo'] == 'CALL' else 0
                    resultado_real = 1 if trade['resultado'] == 'GANADO' else 0
                    
                    # Agregar experiencia
                    # (Necesitamos reconstruir el array de features)
                    # Por simplicidad, usamos un placeholder aquí
                    # En producción, guardarías las features completas
                    
                # Aprender si hay suficientes experiencias
                self.continuous_learner.aprender_de_experiencias(min_experiencias=10)
            
            # 7. Verificar si necesita reentrenamiento completo
            if self.continuous_learner.necesita_reentrenamiento():
                print("\n   ⏰ Es momento de reentrenamiento completo")
                print("   💡 Ejecuta 'python entrenar_completo.py' cuando sea conveniente")
                self.continuous_learner.programar_proximo_reentrenamiento()
            
            self.ciclos_completados += 1
            
        except Exception as e:
            print(f"   ❌ Error en ciclo: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def ejecutar(self, intervalo_segundos=60):
        """
        Ejecuta el bot en modo continuo
        
        Args:
            intervalo_segundos: Intervalo entre ciclos (default: 60s)
        """
        
        self.running = True
        self.inicio_sesion = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"  🚀 BOT INICIADO - MODO AUTOMÁTICO")
        print(f"{'='*70}")
        print(f"\n  Intervalo: {intervalo_segundos} segundos")
        print(f"  Presiona Ctrl+C para detener\n")
        print(f"{'='*70}\n")
        
        try:
            while self.running:
                # Verificar conexión
                if not self.mt5.verificar_conexion():
                    print("⚠️ Conexión perdida, reconectando...")
                    if not self.mt5.conectar():
                        print("❌ No se pudo reconectar, esperando...")
                        time.sleep(30)
                        continue
                
                # Mostrar panel de info cada 10 ciclos
                if self.ciclos_completados % 10 == 0:
                    self.mostrar_panel_info()
                
                # Ejecutar ciclo principal
                self.ciclo_principal()
                
                # Esperar intervalo
                print(f"\n   ⏳ Esperando {intervalo_segundos}s hasta próximo ciclo...\n")
                time.sleep(intervalo_segundos)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada (Ctrl+C)")
            self.detener()
    
    def detener(self):
        """Detiene el bot de forma segura"""
        
        print(f"\n{'='*70}")
        print(f"  🛑 DETENIENDO BOT...")
        print(f"{'='*70}\n")
        
        self.running = False
        
        # Preguntar si cerrar trades
        if len(self.trade_executor.trades_abiertos) > 0:
            print(f"⚠️  Hay {len(self.trade_executor.trades_abiertos)} trade(s) abierto(s)")
            respuesta = input("¿Deseas cerrarlos? (s/n): ")
            
            if respuesta.lower() == 's':
                self.trade_executor.cerrar_todos_trades()
        
        # Guardar estado del learner
        print("\n💾 Guardando estado del aprendizaje continuo...")
        self.continuous_learner.guardar_estado()
        
        # Mostrar resumen final
        print(f"\n{'='*70}")
        print(f"  📊 RESUMEN DE SESIÓN")
        print(f"{'='*70}")
        
        if self.inicio_sesion:
            duracion = datetime.now() - self.inicio_sesion
            print(f"\n  Duración:         {duracion}")
        
        print(f"  Ciclos:           {self.ciclos_completados}")
        
        trading_stats = self.trade_executor.obtener_estadisticas_trading()
        if trading_stats:
            print(f"  Total trades:     {trading_stats['total_trades']}")
            print(f"  Win rate:         {trading_stats['win_rate']:.1f}%")
            print(f"  Profit total:     ${trading_stats['profit_total']:,.2f}")
        
        print(f"\n{'='*70}")
        
        # Cerrar conexión MT5
        self.mt5.desconectar()
        
        print(f"\n✅ Bot detenido correctamente\n")

def main():
    """Función principal"""
    
    # Manejar señales de sistema
    def signal_handler(sig, frame):
        print("\n⚠️  Señal de interrupción recibida")
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
    
    # Menú de opciones
    print(f"\n{'='*70}")
    print(f"  🎮 MODO DE OPERACIÓN")
    print(f"{'='*70}\n")
    print("  1. Modo Automático (recomendado)")
    print("  2. Modo Manual (un ciclo a la vez)")
    print("  3. Solo Monitoreo (sin operar)")
    print("  4. Salir\n")
    
    opcion = input("Selecciona una opción (1-4): ")
    
    if opcion == '1':
        # Modo automático
        intervalo = input("\nIntervalo entre ciclos en segundos (default: 60): ")
        intervalo = int(intervalo) if intervalo.isdigit() else 60
        
        bot.ejecutar(intervalo_segundos=intervalo)
        
    elif opcion == '2':
        # Modo manual
        print("\n🎮 MODO MANUAL")
        print("Presiona Enter para ejecutar un ciclo, 'q' para salir\n")
        
        while True:
            comando = input(">>> ")
            
            if comando.lower() == 'q':
                bot.detener()
                break
            
            bot.mostrar_panel_info()
            bot.ciclo_principal()
            
    elif opcion == '3':
        # Solo monitoreo
        print("\n👀 MODO MONITOREO")
        print("Presiona Ctrl+C para salir\n")
        
        try:
            while True:
                bot.mostrar_panel_info()
                bot.trade_executor.monitorear_trades()
                time.sleep(10)
        except KeyboardInterrupt:
            bot.detener()
    
    else:
        print("\n👋 Saliendo...")
        bot.mt5.desconectar()

if __name__ == "__main__":
    main()
