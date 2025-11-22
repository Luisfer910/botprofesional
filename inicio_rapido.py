#!/usr/bin/env python3
"""
Script de Inicio Rápido
Verifica configuración y guía al usuario
"""

import os
import sys
import json

def verificar_estructura():
    """Verifica que exista la estructura de carpetas"""
    carpetas = [
        'core', 'training', 'strategy', 'analysis', 
        'interface', 'config', 'logs', 'models', 'data'
    ]
    
    print("📁 Verificando estructura de carpetas...")
    
    faltantes = []
    for carpeta in carpetas:
        if not os.path.exists(carpeta):
            faltantes.append(carpeta)
            os.makedirs(carpeta, exist_ok=True)
    
    if faltantes:
        print(f"   ✅ Carpetas creadas: {', '.join(faltantes)}")
    else:
        print("   ✅ Estructura correcta")
    
    return True

def verificar_configuracion():
    """Verifica que exista la configuración"""
    config_path = 'config/xm_config.json'
    
    print("\n⚙️  Verificando configuración...")
    
    if not os.path.exists(config_path):
        print("   ❌ Archivo de configuración no encontrado")
        print("   📝 Crea el archivo 'config/xm_config.json'")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verificar campos críticos
        if config['MT5']['LOGIN'] == 123456789:
            print("   ⚠️  Debes configurar tus credenciales de MT5")
            print("   📝 Edita 'config/xm_config.json'")
            return False
        
        print("   ✅ Configuración válida")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al leer configuración: {str(e)}")
        return False

def verificar_dependencias():
    """Verifica que estén instaladas las dependencias"""
    print("\n📦 Verificando dependencias...")
    
    dependencias = [
        'MetaTrader5',
        'pandas',
        'numpy',
        'sklearn',
        'lightgbm',
        'ta'
    ]
    
    faltantes = []
    
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        print(f"   ❌ Dependencias faltantes: {', '.join(faltantes)}")
        print("\n   Ejecuta: pip install -r requirements.txt")
        return False
    
    print("   ✅ Todas las dependencias instaladas")
    return True

def verificar_modelo():
    """Verifica que exista un modelo entrenado"""
    print("\n🤖 Verificando modelo de IA...")
    
    import glob
    modelos = glob.glob('models/*.pkl')
    
    if not modelos:
        print("   ⚠️  No hay modelos entrenados")
        print("\n   Debes entrenar el modelo primero:")
        print("   python entrenar_completo.py")
        return False
    
    print(f"   ✅ Encontrados {len(modelos)} modelo(s)")
    return True

def main():
    print(f"\n{'='*70}")
    print(f"  🚀 BOT DE TRADING XM - VERIFICACIÓN INICIAL")
    print(f"{'='*70}\n")
    
    # Verificaciones
    checks = [
        verificar_estructura(),
        verificar_configuracion(),
        verificar_dependencias(),
        verificar_modelo()
    ]
    
    print(f"\n{'='*70}")
    
    if all(checks):
        print(f"  ✅ TODO LISTO PARA OPERAR")
        print(f"{'='*70}\n")
        print("  🎯 Próximos pasos:")
        print("     1. python main.py          → Iniciar bot")
        print("     2. Selecciona modo de operación")
        print("     3. ¡Deja que el bot opere!\n")
    else:
        print(f"  ⚠️  HAY PROBLEMAS QUE RESOLVER")
        print(f"{'='*70}\n")
        print("  📋 Checklist:")
        print(f"     {'✅' if checks[0] else '❌'} Estructura de carpetas")
        print(f"     {'✅' if checks[1] else '❌'} Configuración")
        print(f"     {'✅' if checks[2] else '❌'} Dependencias")
        print(f"     {'✅' if checks[3] else '❌'} Modelo entrenado\n")
        
        if not checks[3]:
            print("  💡 Primero entrena el modelo:")
            print("     python entrenar_completo.py\n")

if __name__ == "__main__":
    main()
