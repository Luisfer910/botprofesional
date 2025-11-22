# 🤖 Bot de Trading XM - Versión Comercial

Sistema completo de trading automatizado con Inteligencia Artificial para operar en XM (MetaTrader 5).

## 🌟 Características Principales

### 🧠 Inteligencia Artificial Avanzada
- **Entrenamiento Híbrido**: Combina datos históricos (20,000 velas) con observación en vivo (tick-by-tick)
- **Aprendizaje Continuo**: Aprende de cada trade ejecutado y mejora constantemente
- **30+ Features**: Indicadores técnicos, patrones de velas, soportes/resistencias, impulsos, volatilidad
- **Modelo LightGBM**: Alta precisión y velocidad de predicción

### 💰 Gestión de Riesgo Profesional
- **Kelly Criterion**: Cálculo óptimo del tamaño de posición
- **Stop Loss Dinámico**: Basado en ATR (Average True Range)
- **Risk-Reward Ratio**: Mínimo 1:1.5 configurable
- **Límites de Seguridad**:
  - Máximo de trades por día
  - Pérdida diaria máxima
  - Drawdown máximo
  - Control de spread

### 📊 Análisis Técnico Completo
- Indicadores de tendencia (SMA, EMA, MACD, ADX)
- Indicadores de momentum (RSI, Stochastic)
- Bandas de Bollinger
- Detección de soportes y resistencias
- Patrones de velas japonesas
- Análisis de impulsos y retrocesos
- Análisis intravela (formación de vela en tiempo real)

### 🎯 Generación de Señales Inteligente
- Señales CALL/PUT con probabilidad y confianza
- Análisis contextual de cada señal
- Filtros de calidad de señal
- Umbrales configurables

### 📈 Monitoreo y Estadísticas
- Panel de control en tiempo real
- Estadísticas de trading (win rate, profit factor)
- Tracking de riesgo
- Historial completo de trades

## 📋 Requisitos

### Software
- Python 3.8 o superior
- MetaTrader 5
- Cuenta XM (demo o real)

### Dependencias Python
```bash
pip install -r requirements.txt
```

Incluye:
- MetaTrader5
- pandas, numpy
- scikit-learn
- lightgbm
- ta (technical analysis)
- Y más...

## 🚀 Instalación

### 1. Clonar/Descargar el Proyecto

```bash
# Estructura de carpetas
bot_xm_commercial_v1/
├── core/                    # Núcleo del sistema
├── training/                # Entrenamiento de modelos
├── strategy/                # Estrategia y ejecución
├── config/                  # Configuración
├── logs/                    # Logs del sistema
├── models/                  # Modelos entrenados
├── data/                    # Datos históricos
├── main.py                  # Bot principal
├── entrenar_completo.py     # Script de entrenamiento
├── inicio_rapido.py         # Verificación inicial
└── requirements.txt         # Dependencias
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Credenciales

Edita `config/xm_config.json`:

```json
{
  "MT5": {
    "LOGIN": TU_NUMERO_CUENTA,
    "PASSWORD": "TU_PASSWORD",
    "SERVER": "XMGlobal-MT5 3"
  },
  ...
}
```

### 4. Verificar Instalación

```bash
python inicio_rapido.py
```

## 🎓 Entrenamiento del Modelo

### Entrenamiento Completo (Recomendado)

```bash
python entrenar_completo.py
```

Este proceso:
1. ✅ Conecta a MT5
2. ✅ Descarga 20,000 velas históricas
3. ✅ Observa el mercado en vivo por 1 hora (opcional)
4. ✅ Genera 30+ features avanzadas
5. ✅ Entrena modelo histórico con LightGBM
6. ✅ Refina con datos live (si disponibles)
7. ✅ Crea modelo híbrido
8. ✅ Guarda modelos en `models/`

**Tiempo estimado**: 1-2 horas (incluyendo observación live)

**Nota**: Puedes omitir la observación live si tienes prisa. El modelo histórico solo también funciona bien.

## 🎮 Uso del Bot

### Iniciar el Bot

```bash
python main.py
```

### Modos de Operación

#### 1. Modo Automático (Recomendado)
- Opera de forma completamente automática
- Intervalo configurable entre ciclos (default: 60s)
- Monitoreo continuo
- Aprendizaje automático

#### 2. Modo Manual
- Ejecuta un ciclo a la vez
- Control total sobre cada operación
- Ideal para pruebas

#### 3. Modo Monitoreo
- Solo observa (no opera)
- Útil para análisis

### Panel de Control

El bot muestra en tiempo real:

```
💰 CUENTA:
   Balance:      $500.00
   Equity:       $523.45
   Profit:       $23.45

⚠️  GESTIÓN DE RIESGO:
   Trades hoy:        3/10
   Pérdida diaria:    $0.00
   Drawdown actual:   0.00%

📊 ESTADÍSTICAS DE TRADING:
   Total trades:      15
   Ganados:           10 (66.7%)
   Profit total:      $123.45
   Profit factor:     2.34

🧠 APRENDIZAJE CONTINUO:
   Experiencias:      50
   Win rate general:  68.0%
   Actualizaciones:   5
```

## ⚙️ Configuración

### Archivo: `config/xm_config.json`

#### Parámetros de Trading
```json
"TRADING": {
  "SYMBOL": "EURUSD",           // Par a operar
  "TIMEFRAME": "M5",            // Temporalidad
  "MAX_SPREAD": 20              // Spread máximo permitido
}
```

#### Gestión de Riesgo
```json
"RISK": {
  "CAPITAL_INICIAL": 500,       // Capital inicial
  "RIESGO_POR_TRADE": 0.02,     // 2% por trade
  "MAX_TRADES_DIA": 10,          // Máximo 10 trades/día
  "MAX_PERDIDA_DIARIA": 0.05,    // Máximo 5% pérdida/día
  "MAX_DRAWDOWN": 0.15,          // Máximo 15% drawdown
  "STOP_LOSS_PIPS": 15,          // Stop loss en pips
  "TAKE_PROFIT_PIPS": 30,        // Take profit en pips
  "RISK_REWARD_MIN": 1.5         // Ratio mínimo 1:1.5
}
```

#### Horarios de Trading
```json
"HORARIOS": {
  "INICIO": "08:00",
  "FIN": "20:00",
  "EVITAR_HORAS": ["00:00-01:00", "22:00-23:00"],
  "EVITAR_DIAS": []
}
```

#### Modelo de IA
```json
"MODELO": {
  "VELAS_HISTORICAS": 20000,         // Velas para entrenamiento
  "OBSERVACION_LIVE_MINUTOS": 60,    // Minutos de observación live
  "REENTRENAMIENTO_HORAS": 6,        // Reentrenar cada 6 horas
  "UMBRAL_CALL": 0.58,               // Umbral para señal CALL
  "UMBRAL_PUT": 0.42,                // Umbral para señal PUT
  "MIN_PROBABILIDAD": 0.55           // Probabilidad mínima
}
```

## 📊 Estructura del Sistema

### Core (Núcleo)
- `mt5_connector.py`: Conexión robusta con MT5
- `data_manager.py`: Gestión de datos históricos y live
- `feature_engineer.py`: Generación de features

### Training (Entrenamiento)
- `historical_trainer.py`: Entrenamiento con datos históricos
- `hybrid_trainer.py`: Fusión de modelos histórico + live
- `continuous_learner.py`: Aprendizaje continuo

### Strategy (Estrategia)
- `signal_generator.py`: Generación de señales de trading
- `risk_manager.py`: Gestión de riesgo
- `trade_executor.py`: Ejecución y monitoreo de trades

## 🔒 Seguridad y Mejores Prácticas

### ✅ Recomendaciones

1. **Empieza en Demo**
   - Prueba primero en cuenta demo
   - Verifica que todo funcione correctamente
   - Analiza resultados durante al menos 1 semana

2. **Capital Inicial Conservador**
   - Empieza con capital que puedas permitirte perder
   - No uses todo tu capital de trading

3. **Monitoreo Regular**
   - Revisa el bot al menos 2 veces al día
   - Verifica logs en `logs/`
   - Analiza estadísticas

4. **Ajusta Parámetros Gradualmente**
   - No hagas cambios drásticos
   - Prueba un cambio a la vez
   - Documenta los resultados

5. **Reentrenamiento Periódico**
   - Reentrena el modelo cada semana
   - Especialmente después de eventos importantes
   - Mantén backups de modelos anteriores

### ⚠️ Advertencias

- **Trading con riesgo**: El trading de Forex conlleva riesgo de pérdida
- **No garantías**: Ningún sistema garantiza ganancias
- **Supervisión necesaria**: No dejes el bot sin supervisión prolongada
- **Condiciones de mercado**: El rendimiento varía según condiciones
- **Slippage y spreads**: Pueden afectar resultados reales

## 🐛 Solución de Problemas

### Error: "No se pudo conectar a MT5"
- ✅ Verifica que MT5 esté abierto
- ✅ Comprueba credenciales en `config/xm_config.json`
- ✅ Verifica el nombre del servidor
- ✅ Asegúrate de tener conexión a internet

### Error: "Modelo no encontrado"
- ✅ Ejecuta `python entrenar_completo.py`
- ✅ Verifica que existan archivos en `models/`

### Error: "Spread demasiado alto"
- ✅ Espera a que el spread baje
- ✅ Ajusta `MAX_SPREAD` en configuración
- ✅ Evita horarios de baja liquidez

### Trades no se ejecutan
- ✅ Verifica límites de riesgo
- ✅ Comprueba horarios de trading
- ✅ Revisa logs en `logs/trade_executor.log`

## 📈 Optimización y Mejora

### Ajustar Umbrales de Señal
```json
"UMBRAL_CALL": 0.58,    // Más alto = menos señales, más calidad
"UMBRAL_PUT": 0.42,     // Más bajo = menos señales, más calidad
```

### Ajustar Gestión de Riesgo
```json
"RIESGO_POR_TRADE": 0.02,  // Más bajo = más conservador
"RISK_REWARD_MIN": 1.5,    // Más alto = mejor ratio riesgo/beneficio
```

### Reentrenar con Más Datos
```python
# En entrenar_completo.py
df_historico = data_manager.cargar_datos_historicos(cantidad=50000)  # Más velas
```

## 📝 Logs y Debugging

### Ubicación de Logs
```
logs/
├── mt5_connection.log      # Conexión MT5
├── data_manager.log         # Gestión de datos
├── feature_engineer.log     # Generación de features
├── historical_trainer.log   # Entrenamiento
├── signal_generator.log     # Señales
├── risk_manager.log         # Gestión de riesgo
└── trade_executor.log       # Ejecución de trades
```

### Ver Logs en Tiempo Real
```bash
tail -f logs/trade_executor.log
```

## 🤝 Soporte

Para soporte o preguntas:
- 📧 Email: [tu-email]
- 💬 Discord: [tu-discord]
- 📱 Telegram: [tu-telegram]

## 📄 Licencia

Este proyecto es de uso personal/educativo. No me hago responsable de pérdidas financieras.

## 🎯 Roadmap Futuro

- [ ] Interfaz gráfica (GUI)
- [ ] Soporte para más pares de divisas
- [ ] Backtesting avanzado
- [ ] Optimización de hiperparámetros automática
- [ ] Notificaciones por Telegram/Discord
- [ ] Dashboard web en tiempo real
- [ ] Soporte para múltiples cuentas

---

**⚠️ DISCLAIMER**: El trading de Forex conlleva un alto nivel de riesgo y puede no ser adecuado para todos los inversores. El alto grado de apalancamiento puede trabajar en tu contra así como a tu favor. Antes de decidir operar Forex debes considerar cuidadosamente tus objetivos de inversión, nivel de experiencia y apetito de riesgo. Existe la posibilidad de que pierdas parte o toda tu inversión inicial, por lo tanto no debes invertir dinero que no puedas permitirte perder.

---

**Desarrollado con ❤️ para traders que buscan automatizar su estrategia**
"# botprofesional" 
