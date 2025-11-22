import MetaTrader5 as mt5
import json
import logging
from datetime import datetime
import time

class TradeExecutor:
    def __init__(self, mt5_connector, risk_manager, config_path='config/xm_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager
        
        self.trading_config = self.config['TRADING']
        
        # Historial de trades
        self.trades_abiertos = []
        self.trades_cerrados = []
        
        logging.basicConfig(
            filename='logs/trade_executor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def ejecutar_trade(self, señal, atr=None):
        """
        Ejecuta un trade basado en la señal
        
        Returns:
            dict: Información del trade ejecutado o None si falla
        """
        # Verificar si se puede operar
        puede, razon = self.risk_manager.puede_operar(señal)
        
        if not puede:
            self.logger.warning(f"⚠️ No se puede operar: {razon}")
            print(f"⚠️ Trade rechazado: {razon}")
            return None
        
        # Obtener precio actual
        tick = self.mt5.obtener_tick_actual()
        if tick is None:
            self.logger.error("❌ No se pudo obtener precio actual")
            return None
        
        # Determinar tipo de orden
        if señal['tipo'] == 'CALL':
            tipo_orden = mt5.ORDER_TYPE_BUY
            precio_entrada = tick['ask']
        else:  # PUT
            tipo_orden = mt5.ORDER_TYPE_SELL
            precio_entrada = tick['bid']
        
        # Calcular tamaño de posición
        volumen = self.risk_manager.calcular_tamaño_posicion(señal)
        
        # Calcular stop loss y take profit
        stop_loss = self.risk_manager.calcular_stop_loss(precio_entrada, señal['tipo'], atr)
        take_profit = self.risk_manager.calcular_take_profit(precio_entrada, stop_loss, señal['tipo'])
        
        # Preparar request
        symbol = self.trading_config['SYMBOL']
        magic = self.trading_config['MAGIC_NUMBER']
        deviation = self.trading_config['SLIPPAGE']
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volumen,
            "type": tipo_orden,
            "price": precio_entrada,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": magic,
            "comment": f"Bot_{señal['tipo']}_{datetime.now().strftime('%H%M%S')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Ejecutar orden con reintentos
        max_intentos = 3
        for intento in range(max_intentos):
            try:
                print(f"\n{'─'*60}")
                print(f"🚀 EJECUTANDO TRADE (Intento {intento + 1}/{max_intentos})")
                print(f"{'─'*60}")
                print(f"   Tipo:         {señal['tipo']}")
                print(f"   Precio:       {precio_entrada:.5f}")
                print(f"   Volumen:      {volumen:.2f} lotes")
                print(f"   Stop Loss:    {stop_loss:.5f}")
                print(f"   Take Profit:  {take_profit:.5f}")
                print(f"   Confianza:    {señal['confianza']}")
                print(f"   Probabilidad: {señal['probabilidad']:.3f}")
                print(f"{'─'*60}\n")
                
                # Enviar orden
                result = mt5.order_send(request)
                
                if result is None:
                    self.logger.error(f"❌ Intento {intento + 1}: order_send retornó None")
                    time.sleep(1)
                    continue
                
                # Verificar resultado
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(
                        f"❌ Intento {intento + 1}: Error {result.retcode} - {result.comment}"
                    )
                    print(f"❌ Error: {result.comment}")
                    time.sleep(1)
                    continue
                
                # Éxito
                trade_info = {
                    'ticket': result.order,
                    'tipo': señal['tipo'],
                    'volumen': volumen,
                    'precio_entrada': result.price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp_apertura': datetime.now(),
                    'señal': señal,
                    'estado': 'ABIERTO'
                }
                
                self.trades_abiertos.append(trade_info)
                
                print(f"✅ TRADE EJECUTADO EXITOSAMENTE")
                print(f"{'─'*60}")
                print(f"   Ticket:       {result.order}")
                print(f"   Precio real:  {result.price:.5f}")
                print(f"   Slippage:     {abs(result.price - precio_entrada):.5f}")
                print(f"{'='*60}\n")
                
                self.logger.info(
                    f"✅ Trade ejecutado: {señal['tipo']} | "
                    f"Ticket: {result.order} | Precio: {result.price:.5f} | "
                    f"Vol: {volumen:.2f}"
                )
                
                return trade_info
                
            except Exception as e:
                self.logger.error(f"❌ Excepción en intento {intento + 1}: {str(e)}")
                time.sleep(1)
        
        # Si llegamos aquí, todos los intentos fallaron
        print(f"❌ No se pudo ejecutar el trade después de {max_intentos} intentos\n")
        self.logger.error(f"❌ Trade no ejecutado después de {max_intentos} intentos")
        return None
    
    def monitorear_trades(self):
        """
        Monitorea los trades abiertos y actualiza su estado
        
        Returns:
            list: Trades que se cerraron en esta iteración
        """
        if len(self.trades_abiertos) == 0:
            return []
        
        trades_cerrados_ahora = []
        
        # Obtener posiciones abiertas
        positions = mt5.positions_get(symbol=self.trading_config['SYMBOL'])
        
        if positions is None:
            positions = []
        
        tickets_abiertos = [pos.ticket for pos in positions]
        
        # Verificar cada trade
        for trade in self.trades_abiertos[:]:  # Copia para poder modificar
            if trade['ticket'] not in tickets_abiertos:
                # Trade cerrado
                trade['estado'] = 'CERRADO'
                trade['timestamp_cierre'] = datetime.now()
                
                # Obtener info del trade cerrado
                deals = mt5.history_deals_get(ticket=trade['ticket'])
                
                if deals and len(deals) > 0:
                    deal = deals[-1]
                    trade['precio_cierre'] = deal.price
                    trade['profit'] = deal.profit
                    trade['duracion'] = (trade['timestamp_cierre'] - trade['timestamp_apertura']).total_seconds()
                    
                    # Determinar resultado
                    if trade['profit'] > 0:
                        trade['resultado'] = 'GANADO'
                    else:
                        trade['resultado'] = 'PERDIDO'
                    
                    print(f"\n{'='*60}")
                    print(f"🔔 TRADE CERRADO")
                    print(f"{'='*60}")
                    print(f"   Ticket:       {trade['ticket']}")
                    print(f"   Tipo:         {trade['tipo']}")
                    print(f"   Resultado:    {trade['resultado']}")
                    print(f"   Profit:       ${trade['profit']:.2f}")
                    print(f"   Entrada:      {trade['precio_entrada']:.5f}")
                    print(f"   Cierre:       {trade['precio_cierre']:.5f}")
                    print(f"   Duración:     {trade['duracion']:.0f}s")
                    print(f"{'='*60}\n")
                    
                    self.logger.info(
                        f"🔔 Trade cerrado: {trade['tipo']} | "
                        f"Ticket: {trade['ticket']} | "
                        f"Profit: ${trade['profit']:.2f} | "
                        f"Resultado: {trade['resultado']}"
                    )
                    
                    # Registrar en risk manager
                    self.risk_manager.registrar_trade(trade)
                
                # Mover a trades cerrados
                self.trades_cerrados.append(trade)
                self.trades_abiertos.remove(trade)
                trades_cerrados_ahora.append(trade)
        
        return trades_cerrados_ahora
    
    def cerrar_trade(self, ticket):
        """Cierra manualmente un trade específico"""
        # Buscar trade
        trade = next((t for t in self.trades_abiertos if t['ticket'] == ticket), None)
        
        if trade is None:
            self.logger.warning(f"⚠️ Trade {ticket} no encontrado")
            return False
        
        # Obtener posición
        position = mt5.positions_get(ticket=ticket)
        
        if not position or len(position) == 0:
            self.logger.warning(f"⚠️ Posición {ticket} no existe")
            return False
        
        position = position[0]
        
        # Determinar tipo de cierre
        if position.type == mt5.POSITION_TYPE_BUY:
            tipo_orden = mt5.ORDER_TYPE_SELL
            precio = mt5.symbol_info_tick(position.symbol).bid
        else:
            tipo_orden = mt5.ORDER_TYPE_BUY
            precio = mt5.symbol_info_tick(position.symbol).ask
        
        # Preparar request de cierre
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": tipo_orden,
            "position": ticket,
            "price": precio,
            "deviation": self.trading_config['SLIPPAGE'],
            "magic": self.trading_config['MAGIC_NUMBER'],
            "comment": f"Close_{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Ejecutar cierre
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"✅ Trade {ticket} cerrado manualmente")
            print(f"✅ Trade {ticket} cerrado manualmente")
            return True
        else:
            self.logger.error(f"❌ Error al cerrar trade {ticket}")
            return False
    
    def cerrar_todos_trades(self):
        """Cierra todos los trades abiertos"""
        print(f"\n⚠️ Cerrando todos los trades abiertos...")
        
        for trade in self.trades_abiertos[:]:
            self.cerrar_trade(trade['ticket'])
            time.sleep(0.5)
        
        print(f"✅ Todos los trades cerrados\n")
    
    def obtener_estadisticas_trading(self):
        """Obtiene estadísticas de trading"""
        if len(self.trades_cerrados) == 0:
            return None
        
        total = len(self.trades_cerrados)
        ganados = sum(1 for t in self.trades_cerrados if t.get('resultado') == 'GANADO')
        perdidos = sum(1 for t in self.trades_cerrados if t.get('resultado') == 'PERDIDO')
        
        profit_total = sum(t.get('profit', 0) for t in self.trades_cerrados)
        profit_ganados = sum(t.get('profit', 0) for t in self.trades_cerrados if t.get('resultado') == 'GANADO')
        profit_perdidos = sum(t.get('profit', 0) for t in self.trades_cerrados if t.get('resultado') == 'PERDIDO')
        
        win_rate = ganados / total if total > 0 else 0
        
        avg_profit_ganado = profit_ganados / ganados if ganados > 0 else 0
        avg_profit_perdido = profit_perdidos / perdidos if perdidos > 0 else 0
        
        profit_factor = abs(profit_ganados / profit_perdidos) if profit_perdidos != 0 else 0
        
        return {
            'total_trades': total,
            'ganados': ganados,
            'perdidos': perdidos,
            'win_rate': win_rate * 100,
            'profit_total': profit_total,
            'profit_ganados': profit_ganados,
            'profit_perdidos': profit_perdidos,
            'avg_profit_ganado': avg_profit_ganado,
            'avg_profit_perdido': avg_profit_perdido,
            'profit_factor': profit_factor,
            'trades_abiertos': len(self.trades_abiertos)
        }
