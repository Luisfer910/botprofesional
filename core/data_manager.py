"""
Data Manager - Gestión de datos históricos y en tiempo real
Versión: 3.0 - Con análisis tick-by-tick real
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging

class DataManager:
    def __init__(self, mt5_connector, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mt5 = mt5_connector
        self.symbol = self.config['TRADING']['SYMBOL']
        self.timeframe_str = self.config['TRADING']['TIMEFRAME']
        self.timeframe = self._get_timeframe()
        
        # Buffer de ticks para análisis intravela
        self.tick_buffer = []
        self.current_candle_time = None
        self.tick_data_by_candle = {}  # {candle_time: [ticks]}
        
        logging.basicConfig(
            filename='logs/data_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_timeframe(self):
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
        return timeframes.get(self.timeframe_str, mt5.TIMEFRAME_M5)
    
    def obtener_datos_historicos(self, cantidad=20000):
        """
        Obtiene datos históricos de velas
        
        Args:
            cantidad: Número de velas a obtener
            
        Returns:
            DataFrame con datos históricos
        """
        try:
            self.logger.info(f"📥 Obteniendo {cantidad} velas históricas...")
            
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                self.timeframe,
                0,
                cantidad
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error("❌ No se pudieron obtener datos históricos")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            self.logger.info(f"✅ {len(df)} velas históricas obtenidas")
            self.logger.info(f"   Desde: {df['time'].iloc[0]}")
            self.logger.info(f"   Hasta: {df['time'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error al obtener datos históricos: {str(e)}")
            return None
    
    def capturar_ticks_tiempo_real(self, duracion_segundos=300):
        """
        Captura ticks en tiempo real durante la formación de una vela M5
        
        Args:
            duracion_segundos: Duración de captura (300s = 5min para M5)
            
        Returns:
            DataFrame con ticks capturados
        """
        try:
            self.logger.info(f"🎯 Iniciando captura de ticks por {duracion_segundos}s...")
            
            inicio = datetime.now()
            fin = inicio + timedelta(seconds=duracion_segundos)
            
            ticks_capturados = []
            ultimo_tick_time = None
            
            print(f"\n{'─'*60}")
            print(f"🎯 CAPTURA DE TICKS EN TIEMPO REAL")
            print(f"{'─'*60}")
            print(f"   Duración: {duracion_segundos}s ({duracion_segundos/60:.1f} min)")
            print(f"   Inicio: {inicio.strftime('%H:%M:%S')}")
            print(f"   Fin estimado: {fin.strftime('%H:%M:%S')}")
            print(f"{'─'*60}\n")
            
            contador = 0
            
            while datetime.now() < fin:
                # Obtener tick actual
                tick = mt5.symbol_info_tick(self.symbol)
                
                if tick is not None:
                    tick_time = datetime.fromtimestamp(tick.time)
                    
                    # Evitar duplicados
                    if ultimo_tick_time is None or tick_time > ultimo_tick_time:
                        tick_data = {
                            'time': tick_time,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume,
                            'spread': (tick.ask - tick.bid) / tick.bid * 10000  # En pips
                        }
                        
                        ticks_capturados.append(tick_data)
                        ultimo_tick_time = tick_time
                        contador += 1
                        
                        # Mostrar progreso cada 50 ticks
                        if contador % 50 == 0:
                            transcurrido = (datetime.now() - inicio).total_seconds()
                            restante = duracion_segundos - transcurrido
                            print(f"   📊 Ticks: {contador} | Tiempo: {transcurrido:.0f}s / {duracion_segundos}s | Restante: {restante:.0f}s")
                
                # Pequeña pausa para no saturar
                time.sleep(0.1)
            
            print(f"\n{'─'*60}")
            print(f"✅ CAPTURA COMPLETADA")
            print(f"{'─'*60}")
            print(f"   Total ticks: {len(ticks_capturados)}")
            print(f"   Duración real: {(datetime.now() - inicio).total_seconds():.1f}s")
            print(f"{'─'*60}\n")
            
            if len(ticks_capturados) == 0:
                self.logger.warning("⚠️ No se capturaron ticks")
                return None
            
            df_ticks = pd.DataFrame(ticks_capturados)
            
            self.logger.info(f"✅ {len(df_ticks)} ticks capturados")
            
            return df_ticks
            
        except Exception as e:
            self.logger.error(f"❌ Error al capturar ticks: {str(e)}")
            return None
    
    def obtener_ticks_vela_actual(self):
        """
        Obtiene los ticks de la vela M5 que se está formando actualmente
        
        Returns:
            DataFrame con ticks de la vela actual
        """
        try:
            # Obtener tiempo de la vela actual
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
            
            if rates is None or len(rates) == 0:
                return None
            
            candle_time = pd.to_datetime(rates[0]['time'], unit='s')
            
            # Si es una nueva vela, limpiar buffer
            if self.current_candle_time is None or candle_time > self.current_candle_time:
                self.current_candle_time = candle_time
                self.tick_buffer = []
                self.logger.info(f"🕐 Nueva vela M5 iniciada: {candle_time}")
            
            # Capturar tick actual
            tick = mt5.symbol_info_tick(self.symbol)
            
            if tick is not None:
                tick_time = datetime.fromtimestamp(tick.time)
                
                # Solo agregar si es de la vela actual
                if tick_time >= candle_time:
                    tick_data = {
                        'time': tick_time,
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume,
                        'spread': (tick.ask - tick.bid) / tick.bid * 10000,
                        'candle_time': candle_time
                    }
                    
                    self.tick_buffer.append(tick_data)
            
            # Retornar DataFrame con ticks de la vela actual
            if len(self.tick_buffer) > 0:
                return pd.DataFrame(self.tick_buffer)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error al obtener ticks de vela actual: {str(e)}")
            return None
    
    def analizar_ticks_intravela(self, df_ticks):
        """
        Analiza los ticks dentro de una vela y genera features intravela
        
        Args:
            df_ticks: DataFrame con ticks de la vela
            
        Returns:
            dict con features intravela
        """
        if df_ticks is None or len(df_ticks) < 5:
            return None
        
        try:
            features = {}
            
            # Precio
            precio_inicio = df_ticks['bid'].iloc[0]
            precio_fin = df_ticks['bid'].iloc[-1]
            precio_max = df_ticks['bid'].max()
            precio_min = df_ticks['bid'].min()
            
            # Movimiento
            features['precio_cambio'] = precio_fin - precio_inicio
            features['precio_cambio_pct'] = (precio_fin - precio_inicio) / precio_inicio * 100
            features['rango_intravela'] = precio_max - precio_min
            
            # Posición final del precio
            if precio_max != precio_min:
                features['posicion_precio'] = (precio_fin - precio_min) / (precio_max - precio_min)
            else:
                features['posicion_precio'] = 0.5
            
            # Volatilidad intravela
            features['volatilidad_intravela'] = df_ticks['bid'].std()
            features['volatilidad_normalizada'] = features['volatilidad_intravela'] / precio_inicio * 10000  # En pips
            
            # Presión compradora/vendedora
            cambios = df_ticks['bid'].diff()
            cambios_positivos = cambios[cambios > 0].count()
            cambios_negativos = cambios[cambios < 0].count()
            total_cambios = cambios_positivos + cambios_negativos
            
            if total_cambios > 0:
                features['presion_compradora'] = cambios_positivos / total_cambios
                features['presion_vendedora'] = cambios_negativos / total_cambios
                features['presion_neta'] = features['presion_compradora'] - features['presion_vendedora']
            else:
                features['presion_compradora'] = 0.5
                features['presion_vendedora'] = 0.5
                features['presion_neta'] = 0.0
            
            # Velocidad de cambio
            if len(df_ticks) > 1:
                tiempo_total = (df_ticks['time'].iloc[-1] - df_ticks['time'].iloc[0]).total_seconds()
                if tiempo_total > 0:
                    features['velocidad'] = abs(precio_fin - precio_inicio) / tiempo_total
                else:
                    features['velocidad'] = 0.0
            else:
                features['velocidad'] = 0.0
            
            # Momentum intravela
            if len(df_ticks) >= 10:
                mitad = len(df_ticks) // 2
                precio_mitad = df_ticks['bid'].iloc[mitad]
                features['momentum_primera_mitad'] = precio_mitad - precio_inicio
                features['momentum_segunda_mitad'] = precio_fin - precio_mitad
            else:
                features['momentum_primera_mitad'] = 0.0
                features['momentum_segunda_mitad'] = 0.0
            
            # Spread
            features['spread_promedio'] = df_ticks['spread'].mean()
            features['spread_max'] = df_ticks['spread'].max()
            features['spread_min'] = df_ticks['spread'].min()
            
            # Cambios de dirección
            cambios_direccion = 0
            for i in range(1, len(cambios)):
                if cambios.iloc[i-1] * cambios.iloc[i] < 0:  # Cambio de signo
                    cambios_direccion += 1
            features['cambios_direccion'] = cambios_direccion
            
            # Número de ticks
            features['num_ticks'] = len(df_ticks)
            
            # Aceleración
            if len(df_ticks) >= 3:
                velocidades = cambios.abs()
                aceleraciones = velocidades.diff()
                features['aceleracion'] = aceleraciones.mean()
            else:
                features['aceleracion'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error al analizar ticks intravela: {str(e)}")
            return None
    
    def obtener_datos_completos_con_ticks(self, velas_historicas=1000):
        """
        Obtiene datos históricos y los enriquece con features intravela
        simuladas (para entrenamiento histórico)
        
        Args:
            velas_historicas: Número de velas históricas
            
        Returns:
            DataFrame con velas + features intravela simuladas
        """
        try:
            # Obtener velas históricas
            df = self.obtener_datos_historicos(velas_historicas)
            
            if df is None:
                return None
            
            # Simular features intravela basadas en OHLC
            # (En producción, estas se obtienen de ticks reales)
            
            df['rango_intravela'] = df['high'] - df['low']
            df['posicion_precio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            df['volatilidad_intravela'] = df['rango_intravela'] / df['close']
            
            # Presión basada en la posición del cierre
            df['presion_neta'] = (df['posicion_precio'] - 0.5) * 2  # -1 a 1
            df['presion_compradora'] = (df['presion_neta'] + 1) / 2
            df['presion_vendedora'] = 1 - df['presion_compradora']
            
            # Velocidad basada en el cuerpo de la vela
            df['velocidad'] = abs(df['close'] - df['open']) / df['close']
            
            # Momentum
            df['momentum_primera_mitad'] = (df['high'] + df['low']) / 2 - df['open']
            df['momentum_segunda_mitad'] = df['close'] - (df['high'] + df['low']) / 2
            
            # Spread simulado (basado en volatilidad)
            df['spread_promedio'] = df['volatilidad_intravela'] * 10
            df['spread_max'] = df['spread_promedio'] * 1.5
            df['spread_min'] = df['spread_promedio'] * 0.5
            
            # Cambios de dirección (simulado)
            df['cambios_direccion'] = np.random.randint(5, 20, len(df))
            
            # Número de ticks (simulado)
            df['num_ticks'] = np.random.randint(50, 200, len(df))
            
            # Aceleración (simulada)
            df['aceleracion'] = df['velocidad'].diff().fillna(0)
            
            self.logger.info(f"✅ Datos completos con features intravela generados")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error al obtener datos completos: {str(e)}")
            return None
    
    def obtener_datos_live_con_ticks(self):
        """
        Obtiene datos en tiempo real con análisis tick-by-tick real
        
        Returns:
            tuple: (DataFrame velas, dict features_intravela)
        """
        try:
            # Obtener velas recientes
            df_velas = self.obtener_datos_historicos(100)
            
            if df_velas is None:
                return None, None
            
            # Obtener ticks de la vela actual
            df_ticks = self.obtener_ticks_vela_actual()
            
            # Analizar ticks
            features_intravela = None
            if df_ticks is not None and len(df_ticks) >= 5:
                features_intravela = self.analizar_ticks_intravela(df_ticks)
            
            return df_velas, features_intravela
            
        except Exception as e:
            self.logger.error(f"❌ Error al obtener datos live con ticks: {str(e)}")
            return None, None
