"""
VERSIÓN 2: Suma Paralela con Threading - Primera Optimización

Descripción: Implementación de suma de arreglos usando threading de Python.
             Divide el trabajo entre múltiples hilos para acelerar el proceso.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Propósito: 
- Introducir el concepto de paralelismo con threading
- Comparar rendimiento vs versión secuencial
- Demostrar división de trabajo entre hilos
- Mostrar limitaciones del threading en Python (GIL)

Características:
Threading nativo de Python
Distribución automática de trabajo
Comparación con línea base secuencial
Cálculo de speedup y eficiencia
Verificación de resultados cruzada
Experimentación con diferentes configuraciones
"""

import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor

def suma_chunk(args):
    """Suma un chunk del arreglo - Función para threading"""
    A_chunk, B_chunk, start_idx = args
    resultado = []
    for i in range(len(A_chunk)):
        resultado.append(A_chunk[i] + B_chunk[i])
    return start_idx, resultado

def worker_detallado(A, B, R, start, end, thread_id):
    """Worker para threading básico - Muestra información de progreso"""
    for i in range(start, end):
        R[i] = A[i] + B[i]
    print(f"   🧵 Thread {thread_id}: completó elementos {start:,}-{end-1:,}")

def main():
    # Encabezado de ejecución
    print("="*70)
    print("VERSIÓN 2: SUMA CON THREADING")
    print("Autor: Hector Jorge Morales Arch (江久取)")
    print("="*70)
    
    # Configuración
    n = 1000  # Tamaño del arreglo
    num_threads = 4
    
    print(f"\nCONFIGURACIÓN INICIAL:")
    print(f"   • Tamaño de arreglos: {n:,} elementos")
    print(f"   • Número de threads: {num_threads}")
    print(f"   • Total de operaciones: {n:,} sumas")
    
    # Crear arreglos aleatorios
    print(f"\nGENERANDO ARREGLOS ALEATORIOS...")
    random.seed(42)  # Semilla fija para reproducibilidad
    
    A = [random.randint(1, 1000) for _ in range(n)]
    B = [random.randint(1, 1000) for _ in range(n)]
    R_threading = [0] * n
    R_secuencial = [0] * n
    
    print(f"   Arreglos creados:")
    print(f"      A[0:3] = {A[:3]}...")
    print(f"      B[0:3] = {B[:3]}...")
    
    # ========== SUMA SECUENCIAL (LÍNEA BASE) ==========
    print(f"\n" + "="*70)
    print("📏 LÍNEA BASE: SUMA SECUENCIAL")
    print("="*70)
    
    inicio_sec = time.perf_counter()
    for i in range(n):
        R_secuencial[i] = A[i] + B[i]
    tiempo_sec = time.perf_counter() - inicio_sec
    
    print(f"   Tiempo secuencial: {tiempo_sec:.6f} segundos")
    print(f"   Throughput: {n/tiempo_sec:,.0f} operaciones/segundo")
    
    # ========== MÉTODO 1: THREADING BÁSICO ==========
    print(f"\n" + "="*70)
    print("MÉTODO 1: Threading Básico")
    print("="*70)
    
    # Calcular distribución
    chunk_size = n // num_threads
    threads = []
    
    print(f"\n   DISTRIBUCIÓN DEL TRABAJO:")
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        print(f"      Thread {i}: elementos {start:,}-{end-1:,} ({end-start:,} elementos)")
    
    # Crear y ejecutar threads
    inicio_thread = time.perf_counter()
    
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        
        thread = threading.Thread(
            target=worker_detallado,
            args=(A, B, R_threading, start, end, i)
        )
        threads.append(thread)
        thread.start()
    
    # Esperar a que todos terminen
    for thread in threads:
        thread.join()
    
    tiempo_thread = time.perf_counter() - inicio_thread
    
    # ========== MÉTODO 2: THREADPOOLEXECUTOR ==========
    print(f"\n" + "="*70)
    print("MÉTODO 2: ThreadPoolExecutor (API moderna)")
    print("="*70)
    
    R_pool = [0] * n
    
    # Preparar chunks para ThreadPool
    chunks = []
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        A_chunk = A[start:end]
        B_chunk = B[start:end]
        chunks.append((A_chunk, B_chunk, start))
    
    inicio_pool = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        resultados = list(executor.map(suma_chunk, chunks))
    
    # Combinar resultados
    for start_idx, resultado_chunk in resultados:
        for j, valor in enumerate(resultado_chunk):
            R_pool[start_idx + j] = valor
    
    tiempo_pool = time.perf_counter() - inicio_pool
    
    # ========== COMPARACIÓN Y ANÁLISIS ==========
    print(f"\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS Y RENDIMIENTO")
    print("="*70)
    
    # Verificación de exactitud
    correcto_thread = all(R_threading[i] == R_secuencial[i] for i in range(n))
    correcto_pool = all(R_pool[i] == R_secuencial[i] for i in range(n))
    
    print(f"\n   VERIFICACIÓN DE EXACTITUD:")
    print(f"      • Threading básico: {'CORRECTO' if correcto_thread else 'ERROR'}")
    print(f"      • ThreadPoolExecutor: {'CORRECTO' if correcto_pool else 'ERROR'}")
    
    # Métricas de rendimiento
    print(f"\n   TIEMPOS DE EJECUCIÓN:")
    print(f"      • Secuencial:       {tiempo_sec:10.6f} segundos")
    print(f"      • Threading básico: {tiempo_thread:10.6f} segundos")
    print(f"      • ThreadPool:       {tiempo_pool:10.6f} segundos")
    
    # Cálculo de speedup y eficiencia
    if tiempo_thread > 0:
        speedup_thread = tiempo_sec / tiempo_thread
        eficiencia_thread = (speedup_thread / num_threads) * 100
        
        print(f"\n   MÉTRICAS (Threading básico):")
        print(f"      • Speedup:    {speedup_thread:.2f}x más rápido")
        print(f"      • Eficiencia: {eficiencia_thread:.1f}% del ideal")
        print(f"      • Tiempo ahorrado: {tiempo_sec - tiempo_thread:.6f} segundos")
    
    # ========== EXPERIMENTO: ESCALABILIDAD ==========
    print(f"\n" + "="*70)
    print("EXPERIMENTO: Escalabilidad con diferentes threads")
    print("="*70)
    
    print(f"\n   Probando con diferentes configuraciones de threads...")
    
    configs_threads = [1, 2, 4, 8]
    tiempos_experimento = []
    
    for threads in configs_threads:
        R_temp = [0] * n
        chunk_size = n // threads
        
        # Preparar chunks
        chunks_exp = []
        for i in range(threads):
            start = i * chunk_size
            end = n if i == threads - 1 else start + chunk_size
            A_chunk = A[start:end]
            B_chunk = B[start:end]
            chunks_exp.append((A_chunk, B_chunk, start))
        
        # Ejecutar con ThreadPoolExecutor
        inicio = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            resultados_exp = list(executor.map(suma_chunk, chunks_exp))
        
        # Combinar resultados
        for start_idx, resultado_chunk in resultados_exp:
            for j, valor in enumerate(resultado_chunk):
                R_temp[start_idx + j] = valor
        
        tiempo_exp = time.perf_counter() - inicio
        tiempos_experimento.append(tiempo_exp)
        
        speedup_exp = tiempo_sec / tiempo_exp if tiempo_exp > 0 else 0
        print(f"      • {threads:2} threads: {tiempo_exp:8.6f}s (Speedup: {speedup_exp:.2f}x)")
    
    # Encontrar configuración óptima
    mejor_idx = tiempos_experimento.index(min(tiempos_experimento))
    print(f"\n   CONFIGURACIÓN ÓPTIMA: {configs_threads[mejor_idx]} threads")
    print(f"      Tiempo mínimo: {tiempos_experimento[mejor_idx]:.6f}s")
    print(f"      Speedup máximo: {tiempo_sec/tiempos_experimento[mejor_idx]:.2f}x")
    
    # ========== CONCLUSIONES ==========
    print(f"\n" + "="*70)
    print("CONCLUSIONES - VERSIÓN 2")
    print("="*70)
    
    print(f"\n   1. Threading puede acelerar operaciones I/O bound")
    print(f"   2. Para CPU-bound en Python, el GIL limita la ganancia")
    print(f"   3. ThreadPoolExecutor es más moderno y manejable")
    print(f"   4. El overhead de threads afecta para tareas pequeñas")
    print(f"   5. Ideal para: operaciones con espera I/O, no CPU intensivas")
    
    print(f"\nRECOMENDACIÓN PARA n={n:,}:")
    if tiempo_sec / min(tiempo_thread, tiempo_pool) > 1.3:
        print(f"   Threading recomendado (ganancia > 30%)")
    else:
        print(f"   Threading poco beneficioso (considerar multiprocessing)")
    
    print(f"\n" + "="*70)
    print("VERSIÓN 2 COMPLETADA EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()