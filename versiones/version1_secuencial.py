"""
VERSIÓN 1: Suma Secuencial de Arreglos - Línea Base

Descripción: Implementación básica de suma de arreglos usando solo Python estándar.
             Establece la línea base de rendimiento para comparaciones posteriores.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Propósito: 
- Demostrar la implementación más simple de suma de arreglos
- Establecer tiempo de referencia para métodos paralelos
- Verificar la lógica básica de operaciones con listas
- Preparar el terreno para optimizaciones paralelas

Características:
Solo Python estándar - SIN dependencias externas
Verificación automática de resultados
Medición precisa de tiempo
Interfaz de usuario amigable
Código educativo y documentado
"""

import time
import random

def main():
    # Encabezado de ejecución
    print("="*70)
    print("VERSIÓN 1: SUMA SECUENCIAL BÁSICA")
    print("Autor: Hector Jorge Morales Arch (江久取)")
    print("="*70)
    
    # Configuración
    n = 10  # Tamaño del arreglo
    print(f"\nCONFIGURACIÓN:")
    print(f"   Tamaño de arreglos: {n} elementos")
    print(f"   Total de operaciones: {n} sumas")
    
    # Crear arreglos
    print(f"\nCREANDO ARREGLOS...")
    A = [i + 1 for i in range(n)]           # [1, 2, 3, ..., n]
    B = [(i + 1) * 10 for i in range(n)]    # [10, 20, 30, ..., n*10]
    R = [0] * n  # Arreglo resultado
    
    print(f"   Arreglo A creado: {A[:5]}..." if n > 5 else f"   Arreglo A: {A}")
    print(f"   Arreglo B creado: {B[:5]}..." if n > 5 else f"   Arreglo B: {B}")
    
    # SUMA SECUENCIAL
    print(f"\nEJECUTANDO SUMA SECUENCIAL...")
    
    inicio = time.perf_counter()
    
    # Realizar la suma - Algoritmo secuencial
    for i in range(n):
        R[i] = A[i] + B[i]
    
    fin = time.perf_counter()
    tiempo = fin - inicio
    
    # MOSTRAR RESULTADOS
    print(f"\nSUMA COMPLETADA")
    print(f"   Tiempo de ejecución: {tiempo:.6f} segundos")
    print(f"   Velocidad: {n/tiempo:.0f} operaciones/segundo" if tiempo > 0 else "   Velocidad: Instantánea")
    
    # VERIFICACIÓN
    print(f"\nVERIFICANDO RESULTADOS...")
    errores = 0
    for i in range(n):
        if R[i] != A[i] + B[i]:
            errores += 1
    
    if errores == 0:
        print(f"   Todos los cálculos son correctos")
    else:
        print(f"   Se encontraron {errores} errores")
    
    # TABLA DE RESULTADOS
    print(f"\nTABLA DE RESULTADOS (mostrando primeros {min(5, n)}):")
    print("   " + "="*35)
    print("   Índice\tA\tB\tResultado")
    print("   " + "-"*35)
    
    for i in range(min(5, n)):
        print(f"   {i}\t\t{A[i]}\t{B[i]}\t{R[i]}")
    
    if n > 5:
        print(f"   ... y {n-5} elementos más")
    
    # RESUMEN FINAL
    print(f"\n" + "="*70)
    print("RESUMEN FINAL - VERSIÓN 1:")
    print("="*70)
    print(f"   • Tamaño procesado: {n} elementos")
    print(f"   • Tiempo total: {tiempo:.6f} segundos")
    print(f"   • Precisión: {100 - (errores/n*100 if n>0 else 0):.1f}%")
    print(f"   • Estado: {'ÉXITO' if errores == 0 else 'CON ERRORES'}")
    print(f"   • Método: Algoritmo secuencial tradicional")
    print("="*70)
    
    # CONCLUSIÓN
    print(f"\nCONCLUSIÓN:")
    print(f"   Esta versión establece la línea base de rendimiento.")
    print(f"   Para arreglos pequeños ({n} elementos), el método secuencial")
    print(f"   es adecuado. Para tamaños mayores, ver versiones paralelas.")

if __name__ == "__main__":
    main()