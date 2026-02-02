"""
VERSIÓN 1: Suma Básica de Arreglos (Secuencial)

Esta es la implementación más simple de suma de arreglos.
No utiliza paralelismo, es puramente secuencial.
Autor: Hector Jorge Morales Arch
Alias: 江久取
Propósito: Establecer la línea base de rendimiento.
"""

import time

def main():
    print("="*50)
    print("VERSIÓN 1: SUMA SECUENCIAL BÁSICA")
    print("="*50)
    
    # Tamaño pequeño para demostración
    n = 10
    
    # Crear arreglos simples
    A = list(range(1, n + 1))           # [1, 2, 3, ..., 10]
    B = [i * 10 for i in range(1, n + 1)]  # [10, 20, 30, ..., 100]
    R = [0] * n
    
    print(f"\nArreglo A: {A}")
    print(f"Arreglo B: {B}")
    
    # Suma secuencial
    print("\nRealizando suma secuencial...")
    inicio = time.time()
    
    for i in range(n):
        R[i] = A[i] + B[i]
    
    tiempo = time.time() - inicio
    
    print(f"\nResultado R: {R}")
    print(f"\nResumen:")
    print(f"  Elementos: {n}")
    print(f"  Tiempo: {tiempo:.6f} segundos")
    print(f"  Operaciones: {n} sumas")
    
    # Verificación
    print("\nVerificación:")
    for i in range(n):
        print(f"  R[{i}] = {A[i]} + {B[i]} = {R[i]}")
    
    print("\n✅ Versión 1 completada")

if __name__ == "__main__":
    main()