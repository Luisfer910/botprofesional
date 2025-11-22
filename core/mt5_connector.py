"""
Conector MT5 - Versión adaptada del V47 que funciona
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json

class MT5Connector:
    def __init__(self, config_path='config/xm_config.json'):
        """Inicializa el conector MT5"""
        self.config = self._cargar_config(config_path)
        self.connected = False
        
    def _cargar_config(self, config_path):
        """Carga configuración desde JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"❌ Error al cargar configuración: {e}")
            return None
    
    def conectar(self):
        """Conecta a MT5 usando la misma lógica del V47"""
        try:
            # 1. Inicializar MT5
            if not mt5.initialize():
                print("❌ Error al inicializar MT5")
                print(f"   Código de error: {mt5.last_error()}")
                return False
            
            # 2. Login
            login = self.config['MT5']['LOGIN']
            password = self.config['MT5']['PASSWORD']
            server = self.config['MT5']['SERVER']
            
            if not mt5.login(login, password=password, server=server):
                print(f"❌ Error de login: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            # 3. Seleccionar símbolo
            symbol = self.config['TRADING']['SYMBOL']
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Error al seleccionar {symbol}")
                mt5.shutdown()
                return False
            
            self.connected = True
            print(f"✅ Conectado a XM. {symbol} seleccionado.")
            
            # Mostrar info de cuenta
            account_info = mt5.account_info()
            if account_info:
                print(f"   Cuenta: {account_info.login}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Servidor: {account_info.server}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en conexión: {e}")
            return False
    
    def desconectar(self):
        """Desconecta de MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Desconectado de MT5")
    
    def obtener_datos(self, symbol=None, timeframe=None, cantidad=10000):
        """
        Descarga datos históricos (igual que en V47)
        
        Args:
            symbol: Par de divisas (ej: "EURUSD")
            timeframe: Timeframe MT5 (ej: mt5.TIMEFRAME_M5)
            cantidad: Número de velas
        """
        if not self.connected:
            print("❌ No hay conexión a MT5")
            return None
        
        # Usar valores por defecto del config si no se especifican
        if symbol is None:
            symbol = self.config['TRADING']['SYMBOL']
        
        if timeframe is None:
            # Convertir string a constante MT5
            tf_str = self.config['TRADING']['TIMEFRAME']
            timeframe = self._get_timeframe(tf_str)
        
        print(f"📥 Descargando {cantidad:,} velas {symbol}...")
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, cantidad)
            
            if rates is None or len(rates) == 0:
                print(f"❌ Error en descarga: {mt5.last_error()}")
                return None
            
            # Convertir a DataFrame (igual que V47)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            print(f"✅ Descargadas {len(df):,} velas")
            print(f"   Desde: {df.index[0]}")
            print(f"   Hasta: {df.index[-1]}")
            
            return df[['open', 'high', 'low', 'close', 'tick_volume']]
            
        except Exception as e:
            print(f"❌ Error al descargar datos: {e}")
            return None
    
    def obtener_info_cuenta(self):
        """Obtiene información completa de la cuenta"""
        if not self.connected:
            print("❌ No hay conexión a MT5")
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level if account_info.margin > 0 else 0,
                'profit': account_info.profit,
                'server': account_info.server,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
                'name': account_info.name,
                'company': account_info.company
            }
        except Exception as e:
            print(f"❌ Error al obtener info de cuenta: {e}")
            return None
    
    def _get_timeframe(self, tf_string):
        """Convierte string de timeframe a constante MT5"""
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        return timeframes.get(tf_string, mt5.TIMEFRAME_M5)
    
    def obtener_tick_actual(self, symbol=None):
        """Obtiene el último tick del símbolo"""
        if not self.connected:
            return None
        
        if symbol is None:
            symbol = self.config['TRADING']['SYMBOL']
        
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume
            }
        return None
    
    def verificar_conexion(self):
        """Verifica si la conexión sigue activa"""
        if not self.connected:
            return False
        
        # Intentar obtener info de cuenta
        account_info = mt5.account_info()
        return account_info is not None
    
    def get_symbol_info(self, symbol=None):
        """Obtiene información del símbolo"""
        if not self.connected:
            return None
        
        if symbol is None:
            symbol = self.config['TRADING']['SYMBOL']
        
        info = mt5.symbol_info(symbol)
        if info:
            return {
                'name': info.name,
                'digits': info.digits,
                'point': info.point,
                'spread': info.spread,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step
            }
        return None


# --- FUNCIONES DE COMPATIBILIDAD CON EL CÓDIGO ANTIGUO ---

def conectar_xm(config_path='config/xm_config.json'):
    """Función de compatibilidad con el código V47"""
    connector = MT5Connector(config_path)
    return connector.conectar()

def descargar_datos(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, cantidad=10000):
    """Función de compatibilidad con el código V47"""
    connector = MT5Connector()
    if connector.conectar():
        return connector.obtener_datos(symbol, timeframe, cantidad)
    return None
