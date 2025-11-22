import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, mt5_connector, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mt5 = mt5_connector
        self.risk_config = self.config['RISK']
        self.horarios_config = self.config['HORARIOS']
        
        # Tracking de riesgo
        self.trades_hoy = []
        self.perdida_diaria = 0.0
        self.drawdown_actual = 0.0
        self.equity_maximo = 0.0
        
        logging.basicConfig(
            filename='logs/risk_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def puede_operar(self, señal):
        """
        Verifica si se puede operar según las reglas de riesgo
        
        Returns:
            tuple: (puede_operar: bool, razon: str)
        """
        # 1. Verificar horario
        if not self._verificar_horario():
            return False, "Fuera de horario de trading"
        
        # 2. Verificar límite de trades diarios
        if not self._verificar_limite_trades():
            return False, f"Límite de trades diarios alcanzado ({self.risk_config['MAX_TRADES_DIA']})"
        
        # 3. Verificar pérdida diaria máxima
        if not self._verificar_perdida_diaria():
            return False, f"Pérdida diaria máxima alcanzada ({self.risk_config['MAX_PERDIDA_DIARIA']*100}%)"
        
        # 4. Verificar drawdown máximo
        if not self._verificar_drawdown():
            return False, f"Drawdown máximo alcanzado ({self.risk_config['MAX_DRAWDOWN']*100}%)"
        
        # 5. Verificar spread
        if not self._verificar_spread():
            return False, "Spread demasiado alto"
        
        # 6. Verificar confianza de la señal
        if señal['confianza'] == 'BAJA':
            return False, "Confianza de señal muy baja"
        
        # 7. Verificar que no sea NEUTRAL
        if señal['tipo'] == 'NEUTRAL':
            return False, "Señal neutral"
        
        return True, "OK"
    
    def _verificar_horario(self):
        """Verifica si estamos en horario de trading"""
        return self.mt5.verificar_horario_trading()
    
    def _verificar_limite_trades(self):
        """Verifica límite de trades por día"""
        # Limpiar trades de días anteriores
        self._limpiar_trades_antiguos()
        
        return len(self.trades_hoy) < self.risk_config['MAX_TRADES_DIA']
    
    def _verificar_perdida_diaria(self):
        """Verifica que no se haya alcanzado la pérdida diaria máxima"""
        info = self.mt5.obtener_info_cuenta()
        if info is None:
            return False
        
        capital_inicial = self.risk_config['CAPITAL_INICIAL']
        max_perdida = capital_inicial * self.risk_config['MAX_PERDIDA_DIARIA']
        
        # Calcular pérdida del día
        self.perdida_diaria = sum(t['profit'] for t in self.trades_hoy if t['profit'] < 0)
        
        return abs(self.perdida_diaria) < max_perdida
    
    def _verificar_drawdown(self):
        """Verifica que no se haya alcanzado el drawdown máximo"""
        info = self.mt5.obtener_info_cuenta()
        if info is None:
            return False
        
        equity_actual = info['equity']
        
        # Actualizar equity máximo
        if equity_actual > self.equity_maximo:
            self.equity_maximo = equity_actual
        
        # Calcular drawdown
        if self.equity_maximo > 0:
            self.drawdown_actual = (self.equity_maximo - equity_actual) / self.equity_maximo
        else:
            self.drawdown_actual = 0.0
        
        return self.drawdown_actual < self.risk_config['MAX_DRAWDOWN']
    
    def _verificar_spread(self):
        """Verifica que el spread no sea demasiado alto"""
        spread = self.mt5.obtener_spread_actual()
        
        if spread is None:
            return False
        
        max_spread = self.config['TRADING']['MAX_SPREAD']
        
        return spread <= max_spread
    
    def _limpiar_trades_antiguos(self):
        """Limpia trades de días anteriores"""
        hoy = datetime.now().date()
        self.trades_hoy = [t for t in self.trades_hoy if t['fecha'].date() == hoy]
    
    def calcular_tamaño_posicion(self, señal):
        """
        Calcula el tamaño óptimo de posición usando Kelly Criterion modificado
        
        Returns:
            float: Tamaño de posición en lotes
        """
        info = self.mt5.obtener_info_cuenta()
        if info is None:
            return 0.01  # Mínimo
        
        # Capital disponible
        capital = info['equity']
        
        # Riesgo por trade
        riesgo_por_trade = capital * self.risk_config['RIESGO_POR_TRADE']
        
        # Obtener info del símbolo
        import MetaTrader5 as mt5
        symbol = self.config['TRADING']['SYMBOL']
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return 0.01
        
        # Stop loss en pips
        stop_loss_pips = self.risk_config['STOP_LOSS_PIPS']
        
        # Valor de 1 pip
        pip_value = symbol_info.trade_tick_value
        
        # Calcular tamaño de posición
        # Tamaño = Riesgo / (Stop Loss en pips * Valor de pip)
        tamaño = riesgo_por_trade / (stop_loss_pips * pip_value)
        
        # Ajustar a lote mínimo y máximo
        lote_min = symbol_info.volume_min
        lote_max = symbol_info.volume_max
        lote_step = symbol_info.volume_step
        
        # Redondear al step más cercano
        tamaño = round(tamaño / lote_step) * lote_step
        
        # Limitar
        tamaño = max(lote_min, min(tamaño, lote_max))
        
        # Ajustar según confianza de la señal
        if señal['confianza'] == 'MEDIA':
            tamaño *= 0.7
        elif señal['confianza'] == 'BAJA':
            tamaño *= 0.5
        
        self.logger.info(f"💰 Tamaño de posición calculado: {tamaño:.2f} lotes")
        
        return tamaño
    
    def calcular_stop_loss(self, precio_entrada, tipo_operacion, atr=None):
        """
        Calcula el stop loss dinámico
        
        Args:
            precio_entrada: Precio de entrada
            tipo_operacion: 'CALL' o 'PUT'
            atr: Average True Range (opcional, para stop dinámico)
        
        Returns:
            float: Precio de stop loss
        """
        if atr is not None and atr > 0:
            # Stop loss dinámico basado en ATR
            stop_distance = atr * 1.5
        else:
            # Stop loss fijo en pips
            import MetaTrader5 as mt5
            symbol = self.config['TRADING']['SYMBOL']
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                stop_distance = 0.0015  # Default
            else:
                stop_distance = self.risk_config['STOP_LOSS_PIPS'] * symbol_info.point * 10
        
        if tipo_operacion == 'CALL':
            stop_loss = precio_entrada - stop_distance
        else:  # PUT
            stop_loss = precio_entrada + stop_distance
        
        return stop_loss
    
    def calcular_take_profit(self, precio_entrada, stop_loss, tipo_operacion):
        """
        Calcula el take profit basado en risk-reward ratio
        
        Args:
            precio_entrada: Precio de entrada
            stop_loss: Precio de stop loss
            tipo_operacion: 'CALL' o 'PUT'
        
        Returns:
            float: Precio de take profit
        """
        risk_reward = self.risk_config['RISK_REWARD_MIN']
        
        # Calcular distancia del stop loss
        stop_distance = abs(precio_entrada - stop_loss)
        
        # Take profit a distancia de risk_reward * stop_distance
        if tipo_operacion == 'CALL':
            take_profit = precio_entrada + (stop_distance * risk_reward)
        else:  # PUT
            take_profit = precio_entrada - (stop_distance * risk_reward)
        
        return take_profit
    
    def registrar_trade(self, trade_info):
        """Registra un trade para tracking de riesgo"""
        trade_info['fecha'] = datetime.now()
        self.trades_hoy.append(trade_info)
        
        self.logger.info(
            f"📝 Trade registrado: {trade_info['tipo']} | "
            f"Profit: ${trade_info['profit']:.2f}"
        )
    
    def obtener_estadisticas_riesgo(self):
        """Obtiene estadísticas de riesgo actuales"""
        info = self.mt5.obtener_info_cuenta()
        
        if info is None:
            return None
        
        return {
            'equity': info['equity'],
            'balance': info['balance'],
            'profit': info['profit'],
            'margin_level': info['margin_level'],
            'trades_hoy': len(self.trades_hoy),
            'max_trades_dia': self.risk_config['MAX_TRADES_DIA'],
            'perdida_diaria': self.perdida_diaria,
            'max_perdida_diaria': self.risk_config['CAPITAL_INICIAL'] * self.risk_config['MAX_PERDIDA_DIARIA'],
            'drawdown_actual': self.drawdown_actual * 100,
            'max_drawdown': self.risk_config['MAX_DRAWDOWN'] * 100,
            'puede_operar': self.puede_operar({'tipo': 'CALL', 'confianza': 'ALTA'})[0]
        }
