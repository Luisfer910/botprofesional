"""
Data Manager - Gestión de datos históricos y en tiempo real
Versión: 2.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time

class DataManager:
    """
    Gestiona la obtención y procesamiento de datos desde MT5
    """
    
    def __init__(self, mt5_connector, config_path='config/xm_config.json'):
        """
        Inicializa el Data Manager
        
        Args:
            mt5_connector: Instancia de MT5Connector
            config_path: Ruta al archivo de configuración
        """
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mt5 = mt5_connector
        self.trading_config = self.config['TRADING']
        
        # Configurar logging
        logging.basicConfig(
            filename='logs/data_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Mapeo de timeframes
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        # Cache de datos
        self.ultimo_df = None
        self.ultima_actualizacion = None
    
    def cargar_datos_historicos(self, cantidad=20000):
        """
        Carga datos históricos desde MT5
        
        Args:
            cantidad: Número de velas a descargar
            
        Returns:
            DataFrame con datos históricos
        """
        symbol = self.trading_config['SYMBOL']
        timeframe_str = self.trading_config['TIMEFRAME']
        timeframe = self.timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)
        
        self.logger.info(f"Descargando {cantidad} velas de {symbol} {timeframe_str}")
        
        try:
            # Descargar datos
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, cantidad)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                self.logger.error(f"Error al descargar datos: {error}")
                print(f"   ❌ Error MT5: {error}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            
            # Convertir timestamp a datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Renombrar columnas para consistencia
            df = df.rename(columns={
                'tick_volume': 'volume'
            })
            
            # Ordenar por tiempo
            df = df.sort_values('time').reset_index(drop=True)
            
            self.logger.info(f"Datos históricos cargados: {len(df)} velas")
            print(f"   ✅ Descargadas {len(df)} velas")
            print(f"   📅 Desde: {df['time'].iloc[0]}")
            print(f"   📅 Hasta: {df['time'].iloc[-1]}")
            
            # Guardar en cache
            self.ultimo_df = df.copy()
            self.ultima_actualizacion = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Excepción al cargar datos: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
            return None
    
    def actualizar_datos_live(self, cantidad=500):
        """
        Actualiza datos con las últimas velas
        
        Args:
            cantidad: Número de velas recientes a obtener
            
        Returns:
            DataFrame actualizado
        """
        return self.cargar_datos_historicos(cantidad=cantidad)
    
    def observar_mercado_live(self, duracion_minutos=60, intervalo_segundos=1):
        """
        Observa el mercado en tiempo real tick-by-tick
        
        Args:
            duracion_minutos: Duración de la observación en minutos
            intervalo_segundos: Intervalo entre capturas
            
        Returns:
            DataFrame con observaciones tick-by-tick
        """
        symbol = self.trading_config['SYMBOL']
        
        print(f"\n👁️  Observando mercado en vivo por {duracion_minutos} minutos...")
        print(f"   Capturando ticks cada {intervalo_segundos} segundo(s)")
        print(f"   Presiona Ctrl+C para detener antes\n")
        
        observaciones = []
        inicio = datetime.now()
        fin = inicio + timedelta(minutes=duracion_minutos)
        
        contador = 0
        
        try:
            while datetime.now() < fin:
                # Obtener tick actual
                tick = mt5.symbol_info_tick(symbol)
                
                if tick:
                    obs = {
                        'time': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume,
                        'spread': tick.ask - tick.bid
                    }
                    
                    observaciones.append(obs)
                    contador += 1
                    
                    # Mostrar progreso cada 60 ticks
                    if contador % 60 == 0:
                        tiempo_transcurrido = (datetime.now() - inicio).total_seconds() / 60
                        print(f"   📊 {contador} ticks capturados ({tiempo_transcurrido:.1f} min)")
                
                # Esperar intervalo
                time.sleep(intervalo_segundos)
                
        except KeyboardInterrupt:
            print("\n   ⚠️  Observación interrumpida por el usuario")
        
        if not observaciones:
            self.logger.warning("No se capturaron observaciones")
            return None
        
        # Convertir a DataFrame
        df_live = pd.DataFrame(observaciones)
        
        print(f"\n   ✅ Observación completada")
        print(f"   📊 Total ticks capturados: {len(df_live)}")
        print(f"   ⏱️  Duración real: {(datetime.now() - inicio).total_seconds() / 60:.1f} minutos")
        
        self.logger.info(f"Observación live completada: {len(df_live)} ticks")
        
        return df_live
    
    def agregar_datos_live_a_velas(self, df_velas, df_live):
        """
        Agrega datos de observación live a las velas históricas
        
        Args:
            df_velas: DataFrame con velas históricas
            df_live: DataFrame con observaciones tick-by-tick
            
        Returns:
            DataFrame combinado
        """
        if df_live is None or len(df_live) == 0:
            return df_velas
        
        # Agrupar ticks en velas de 5 minutos
        timeframe_str = self.trading_config['TIMEFRAME']
        
        # Mapeo de timeframe a minutos
        timeframe_minutos = {
            'M1': '1T',
            'M5': '5T',
            'M15': '15T',
            'M30': '30T',
            'H1': '60T',
            'H4': '240T',
            'D1': '1D'
        }
        
        freq = timeframe_minutos.get(timeframe_str, '5T')
        
        # Establecer índice de tiempo
        df_live_copy = df_live.copy()
        df_live_copy.set_index('time', inplace=True)
        
        # Resamplear a velas
        velas_live = df_live_copy.resample(freq).agg({
            'bid': 'ohlc',
            'volume': 'sum'
        })
        
        # Aplanar columnas multi-nivel
        velas_live.columns = ['open', 'high', 'low', 'close', 'volume']
        velas_live = velas_live.dropna()
        velas_live.reset_index(inplace=True)
        
        # Agregar columnas faltantes
        velas_live['tick_volume'] = velas_live['volume']
        velas_live['spread'] = 0
        velas_live['real_volume'] = 0
        
        # Combinar con datos históricos
        df_combinado = pd.concat([df_velas, velas_live], ignore_index=True)
        df_combinado = df_combinado.drop_duplicates(subset=['time'], keep='last')
        df_combinado = df_combinado.sort_values('time').reset_index(drop=True)
        
        print(f"\n   ✅ Datos live agregados")
        print(f"   📊 Velas adicionales: {len(velas_live)}")
        print(f"   📊 Total velas: {len(df_combinado)}")
        
        return df_combinado
    
    def obtener_vela_actual_en_formacion(self):
        """
        Obtiene la vela actual que se está formando
        
        Returns:
            Dict con información de la vela en formación
        """
        symbol = self.trading_config['SYMBOL']
        timeframe_str = self.trading_config['TIMEFRAME']
        timeframe = self.timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M5)
        
        # Obtener última vela cerrada
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)
        
        if rates is None or len(rates) < 2:
            return None
        
        ultima_vela = rates[-1]
        tick_actual = mt5.symbol_info_tick(symbol)
        
        if not tick_actual:
            return None
        
        # Calcular información de vela en formación
        vela_formacion = {
            'time_inicio': datetime.fromtimestamp(ultima_vela['time']),
            'open': ultima_vela['open'],
            'high': max(ultima_vela['high'], tick_actual.bid),
            'low': min(ultima_vela['low'], tick_actual.bid),
            'close': tick_actual.bid,
            'bid': tick_actual.bid,
            'ask': tick_actual.ask,
            'spread': tick_actual.ask - tick_actual.bid,
            'volumen_acumulado': ultima_vela['tick_volume']
        }
        
        # Calcular progreso de la vela
        timeframe_segundos = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400
        }
        
        segundos_tf = timeframe_segundos.get(timeframe_str, 300)
        tiempo_transcurrido = (datetime.now() - vela_formacion['time_inicio']).total_seconds()
        progreso = min(tiempo_transcurrido / segundos_tf * 100, 100)
        
        vela_formacion['progreso_porcentaje'] = progreso
        vela_formacion['segundos_restantes'] = max(0, segundos_tf - tiempo_transcurrido)
        
        return vela_formacion
    
    def verificar_calidad_datos(self, df):
        """
        Verifica la calidad de los datos
        
        Args:
            df: DataFrame a verificar
            
        Returns:
            Dict con métricas de calidad
        """
        if df is None or len(df) == 0:
            return {'valido': False, 'razon': 'DataFrame vacío'}
        
        # Verificar columnas requeridas
        columnas_requeridas = ['time', 'open', 'high', 'low', 'close', 'volume']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            return {
                'valido': False,
                'razon': f'Columnas faltantes: {columnas_faltantes}'
            }
        
        # Verificar valores nulos
        nulos = df[columnas_requeridas].isnull().sum().sum()
        porcentaje_nulos = (nulos / (len(df) * len(columnas_requeridas))) * 100
        
        # Verificar valores negativos en precios
        negativos = (df[['open', 'high', 'low', 'close']] < 0).sum().sum()
        
        # Verificar consistencia OHLC
        inconsistencias = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        calidad = {
            'valido': True,
            'total_registros': len(df),
            'valores_nulos': int(nulos),
            'porcentaje_nulos': round(porcentaje_nulos, 2),
            'valores_negativos': int(negativos),
            'inconsistencias_ohlc': int(inconsistencias),
            'fecha_inicio': df['time'].iloc[0],
            'fecha_fin': df['time'].iloc[-1]
        }
        
        # Determinar si es válido
        if porcentaje_nulos > 5 or negativos > 0 or inconsistencias > len(df) * 0.01:
            calidad['valido'] = False
            calidad['razon'] = 'Calidad de datos insuficiente'
        
        return calidad
    
    def guardar_datos(self, df, nombre='datos_historicos'):
        """
        Guarda datos en archivo CSV
        
        Args:
            df: DataFrame a guardar
            nombre: Nombre base del archivo
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/{nombre}_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        
        self.logger.info(f"Datos guardados: {filename}")
        print(f"   💾 Datos guardados: {filename}")
    
    def cargar_datos_guardados(self, filename):
        """
        Carga datos desde archivo CSV
        
        Args:
            filename: Ruta al archivo
            
        Returns:
            DataFrame con los datos
        """
        try:
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            
            self.logger.info(f"Datos cargados: {filename}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            return None
