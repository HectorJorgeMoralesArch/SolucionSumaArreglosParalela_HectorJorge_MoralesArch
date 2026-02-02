"""
VERSIÓN 2: Suma con Threading
Introducción al paralelismo con threading en Python.
Esta versión divide la tarea entre múltiples threads.
Autor: Hector Jorge Morales Arch
Alias: 江久取
Propósito: Establecer la línea base de rendimiento.
"""
import threading
import time
import numpy as np

def worker(A, B, R, start, end, thread_id):
    """Función que ejecuta cada thread"""
    for i in range(start, end):
        R[i] = A[i] + B[i]
    # print(f"  Thread {thread_id} completó elementos {start}-{end-1}")

def main():
    print("="*50)
    print("VERSIÓN 2: SUMA CON THREADING")
    print("="*50)
    
    n = 1000
    num_threads = 4
    
    # Generar arreglos aleatorios
    np.random.seed(42)
    A = np.random.randint(0, 100, n)
    B = np.random.randint(0, 100, n)
    R = np.zeros(n, dtype=int)
    
    print(f"\nConfiguración:")
    print(f"  Elementos: {n}")
    print(f"  Threads: {num_threads}")
    
    # Calcular chunks
    chunk_size = n // num_threads
    threads = []
    
    # Crear y comenzar threads
    inicio = time.time()
    
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        
        thread = threading.Thread(
            target=worker,
            args=(A, B, R, start, end, i)
        )
        threads.append(thread)
        thread.start()
    
    # Esperar a que todos los threads terminen
    for thread in threads:
        thread.join()
    
    tiempo = time.time() - inicio
    
    # Verificación
    R_verificacion = A + B  # Usando NumPy para verificar
    correcto = np.array_equal(R, R_verificacion)
    
    print(f"\nResultados:")
    print(f"  Tiempo total: {tiempo:.6f} segundos")
    print(f"  Correcto: {'✅ Sí' if correcto else '❌ No'}")
    
    # Mostrar primeros 5 resultados
    print(f"\nMuestra de resultados (primeros 5):")
    for i in range(5):
        print(f"  R[{i}] = {A[i]} + {B[i]} = {R[i]}")
    
    print("\n✅ Versión 2 completada")

if __name__ == "__main__":
    main()