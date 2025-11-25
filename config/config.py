"""
Configuración del Bot de Trading XM
Carga configuración desde xm_config.json
"""

import json
import os

# Ruta al archivo de configuración
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'xm_config.json')

def cargar_config():
    """Carga la configuración desde el archivo JSON"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {CONFIG_PATH}")
        print("   Por favor, crea el archivo xm_config.json en la carpeta config/")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error al leer xm_config.json: {e}")
        return None

# Cargar configuración al importar el módulo
CONFIG = cargar_config()

# Valores por defecto si no se puede cargar el archivo
if CONFIG is None:
    print("⚠️  Usando configuración por defecto")
    CONFIG = {
        "MT5": {
            "LOGIN": 0,
            "PASSWORD": "",
            "SERVER": "XMGlobal-MT5 5"
        },
        "TRADING": {
            "SYMBOL": "EURUSD",
            "TIMEFRAME": "M5",
            "LOTE_INICIAL": 0.01,
            "MAX_TRADES_DIA": 10,
            "SPREAD_MAXIMO": 20
        },
        "RISK_MANAGEMENT": {
            "RIESGO_POR_TRADE": 0.01,
            "MAX_PERDIDA_DIARIA": 0.05,
            "MAX_DRAWDOWN": 0.15,
            "STOP_LOSS_ATR_MULTIPLICADOR": 2.0,
            "TAKE_PROFIT_ATR_MULTIPLICADOR": 3.0,
            "USAR_KELLY_CRITERION": True,
            "KELLY_FRACTION": 0.25
        },
        "MODELO": {
            "UMBRAL_CONFIANZA": 0.6,
            "USAR_MODELO_HIBRIDO": True,
            "REENTRENAR_CADA_N_TRADES": 50
        },
        "SISTEMA": {
            "VELAS_HISTORICAS": 20000,
            "INTERVALO_CICLO_SEGUNDOS": 60,
            "GUARDAR_LOGS": True,
            "MODO_DEBUG": False
        }
    }