"""
VERSIÓN 3: Suma Paralela con Multiprocessing - Máxima Aceleración

Descripción: Implementación de suma de arreglos usando multiprocessing de Python.
             Utiliza procesos separados para evitar el GIL y maximizar el uso de CPU.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Propósito: 
- Demostrar paralelismo real utilizando múltiples procesos
- Evitar las limitaciones del GIL en threading
- Maximizar el uso de múltiples núcleos de CPU
- Comparar rendimiento con threading y secuencial
- Analizar overhead de comunicación entre procesos

Características:
Multiprocessing para paralelismo real
Uso óptimo de múltiples núcleos de CPU
Comparación detallada con métodos anteriores
Análisis de overhead y escalabilidad
Distribución inteligente de chunks
Verificación robusta de resultados
Recomendaciones basadas en datos
"""

import multiprocessing as mp
import time
import random
import os
from concurrent.futures import ProcessPoolExecutor
import math

def worker_multiproceso(args):
    """
    Worker para multiprocessing - Ejecuta en proceso separado
    Args:
        args: Tuple con (A_chunk, B_chunk, start_idx)
    Returns:
        Tuple con (start_idx, resultado, pid_del_proceso)
    """
    A_chunk, B_chunk, start_idx = args
    resultado = []
    for i in range(len(A_chunk)):
        resultado.append(A_chunk[i] + B_chunk[i])
    return start_idx, resultado, os.getpid()

def calcular_tamanio_chunk_optimo(n, num_procesos):
    """
    Calcula el tamaño óptimo de chunk para balancear carga y overhead
    Returns:
        Tamaño de chunk optimizado
    """
    chunk_base = n // num_procesos
    # Evitar chunks muy pequeños (alto overhead)
    if chunk_base < 100:
        return max(100, n // max(1, num_procesos // 2))
    # Para chunks muy grandes, limitar tamaño
    if chunk_base > 10000:
        return 10000
    return chunk_base

def main():
    # Encabezado de ejecución
    print("="*70)
    print("VERSIÓN 3: SUMA CON MULTIPROCESSING")
    print("Autor: Hector Jorge Morales Arch (江久取)")
    print("="*70)
    
    # Configuración del sistema
    n = 10000  # Tamaño significativo para ver beneficio
    num_procesos = min(4, mp.cpu_count())  # Usar hasta 4 procesos o CPUs disponibles
    
    print(f"\nCONFIGURACIÓN DEL SISTEMA:")
    print(f"   • CPUs físicos disponibles: {mp.cpu_count()}")
    print(f"   • Procesos a utilizar: {num_procesos}")
    print(f"   • Tamaño de arreglos: {n:,} elementos")
    print(f"   • Memoria estimada: {(n * 3 * 4) / (1024*1024):.2f} MB")
    
    # Crear arreglos de prueba
    print(f"\nCREANDO ARREGLOS DE PRUEBA...")
    random.seed(42)  # Reproducibilidad
    
    A = [random.randint(1, 10000) for _ in range(n)]
    B = [random.randint(1, 10000) for _ in range(n)]
    
    # Arreglos para resultados
    R_multiproc = [0] * n
    R_secuencial = [0] * n
    
    print(f"   Arreglos creados exitosamente")
    print(f"      Muestra A[:3] = {A[:3]}...")
    print(f"      Muestra B[:3] = {B[:3]}...")
    
    # ========== LÍNEA BASE: SUMA SECUENCIAL ==========
    print(f"\n" + "="*70)
    print("LÍNEA BASE: SUMA SECUENCIAL")
    print("="*70)
    
    inicio_sec = time.perf_counter()
    for i in range(n):
        R_secuencial[i] = A[i] + B[i]
    tiempo_sec = time.perf_counter() - inicio_sec
    
    print(f"   Tiempo secuencial: {tiempo_sec:.6f} segundos")
    print(f"   Throughput: {n/tiempo_sec:,.0f} operaciones/segundo")
    
    # ========== MÉTODO 1: PROCESSPOOLEXECUTOR ==========
    print(f"\n" + "="*70)
    print("MÉTODO 1: ProcessPoolExecutor (API moderna)")
    print("="*70)
    
    # Calcular chunks optimizados
    chunk_size = calcular_tamanio_chunk_optimo(n, num_procesos)
    num_chunks = math.ceil(n / chunk_size)
    
    print(f"\n   ESTRATEGIA DE DISTRIBUCIÓN:")
    print(f"      • Tamaño de chunk: {chunk_size:,} elementos")
    print(f"      • Número de chunks: {num_chunks}")
    print(f"      • Chunks por proceso: ~{num_chunks/num_procesos:.1f}")
    
    # Preparar chunks para distribución
    chunks = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        A_chunk = A[i:end]
        B_chunk = B[i:end]
        chunks.append((A_chunk, B_chunk, i))
    
    # Ejecutar con ProcessPoolExecutor
    inicio_pool = time.perf_counter()
    pids_usados = set()  # Para trackear procesos utilizados
    
    with ProcessPoolExecutor(max_workers=num_procesos) as executor:
        resultados = list(executor.map(worker_multiproceso, chunks))
    
    # Combinar resultados y recolectar información
    for start_idx, resultado_chunk, pid in resultados:
        pids_usados.add(pid)
        for j, valor in enumerate(resultado_chunk):
            R_multiproc[start_idx + j] = valor
    
    tiempo_pool = time.perf_counter() - inicio_pool
    
    print(f"\n   PROCESOS UTILIZADOS: {len(pids_usados)}")
    print(f"      IDs de procesos: {sorted(pids_usados)}")
    print(f"    Tiempo ProcessPoolExecutor: {tiempo_pool:.6f} segundos")
    
    # ========== VERIFICACIÓN DE RESULTADOS ==========
    print(f"\n" + "="*70)
    print("VERIFICACIÓN DE EXACTITUD")
    print("="*70)
    
    # Verificación por muestreo (eficiente para grandes arreglos)
    correcto_pool = True
    muestra_verificacion = min(100, n)
    indices_verificar = random.sample(range(n), muestra_verificacion)
    
    for idx in indices_verificar:
        if R_multiproc[idx] != A[idx] + B[idx]:
            correcto_pool = False
            break
    
    print(f"\n   EXACTITUD DE RESULTADOS:")
    print(f"      • ProcessPoolExecutor: {'CORRECTO' if correcto_pool else 'ERROR'}")
    print(f"      • Muestra verificada: {muestra_verificacion} elementos")
    
    # ========== ANÁLISIS DETALLADO DE RENDIMIENTO ==========
    print(f"\n" + "="*70)
    print("ANÁLISIS DE RENDIMIENTO DETALLADO")
    print("="*70)
    
    # Cálculo de métricas
    if tiempo_pool > 0:
        speedup_pool = tiempo_sec / tiempo_pool
        eficiencia_pool = (speedup_pool / num_procesos) * 100
        
        print(f"\n   MÉTRICAS PRINCIPALES:")
        print(f"      • Speedup: {speedup_pool:.2f}x más rápido")
        print(f"      • Eficiencia: {eficiencia_pool:.1f}% del ideal")
        print(f"      • Tiempo ahorrado: {tiempo_sec - tiempo_pool:.6f} segundos")
        print(f"      • Throughput paralelo: {n/tiempo_pool:,.0f} ops/seg")
    
    # Tabla comparativa
    print(f"\n   COMPARATIVA TIEMPOS:")
    print(f"      {'Método':<30} {'Tiempo (s)':<15} {'Speedup':<10}")
    print(f"      {'-'*55}")
    print(f"      {'Secuencial':<30} {tiempo_sec:<15.6f} {'1.00x':<10}")
    print(f"      {'ProcessPoolExecutor':<30} {tiempo_pool:<15.6f} {f'{speedup_pool:.2f}x':<10}")
    
    # ========== EXPERIMENTO: ESCALABILIDAD ==========
    print(f"\n" + "="*70)
    print("EXPERIMENTO: Escalabilidad con diferentes tamaños")
    print("="*70)
    
    tamanios_prueba = [1000, 5000, 10000, 20000]
    resultados_escalabilidad = []
    
    print(f"\n   Probando escalabilidad con diferentes tamaños...")
    
    for tam in tamanios_prueba:
        # Crear arreglos temporales
        A_temp = [random.randint(1, 1000) for _ in range(tam)]
        B_temp = [random.randint(1, 1000) for _ in range(tam)]
        
        # Tiempo secuencial
        inicio = time.perf_counter()
        [A_temp[i] + B_temp[i] for i in range(tam)]
        tiempo_seq = time.perf_counter() - inicio
        
        # Tiempo paralelo
        chunks_temp = []
        chunk_temp_size = calcular_tamanio_chunk_optimo(tam, num_procesos)
        
        for i in range(0, tam, chunk_temp_size):
            end = min(i + chunk_temp_size, tam)
            A_chunk = A_temp[i:end]
            B_chunk = B_temp[i:end]
            chunks_temp.append((A_chunk, B_chunk, i))
        
        inicio = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_procesos) as executor:
            list(executor.map(worker_multiproceso, chunks_temp))
        tiempo_par = time.perf_counter() - inicio
        
        speedup_temp = tiempo_seq / tiempo_par if tiempo_par > 0 else 0
        
        resultados_escalabilidad.append((tam, tiempo_seq, tiempo_par, speedup_temp))
        
        print(f"      • n={tam:6,}: Sec={tiempo_seq:.4f}s, Par={tiempo_par:.4f}s, Speedup={speedup_temp:.2f}x")
    
    # ========== RECOMENDACIONES PRÁCTICAS ==========
    print(f"\n" + "="*70)
    print("RECOMENDACIONES PRÁCTICAS")
    print("="*70)
    
    # Análisis basado en resultados
    mejor_speedup = speedup_pool
    
    print(f"\n   BASADO EN LOS RESULTADOS CON n={n:,}:")
    
    if mejor_speedup > 2.0:
        print(f"   Multiprocessing ALTAMENTE RECOMENDADO")
        print(f"      • Speedup significativo: {mejor_speedup:.2f}x")
        print(f"      • Buen uso de recursos multicore")
        print(f"      • Ideal para procesamiento por lotes")
    elif mejor_speedup > 1.2:
        print(f"   Multiprocessing MODERADAMENTE BENEFICIOSO")
        print(f"      • Speedup moderado: {mejor_speedup:.2f}x")
        print(f"      • Considerar overhead de procesos")
        print(f"      • Útil para operaciones repetitivas")
    else:
        print(f"   Multiprocessing NO RECOMENDADO")
        print(f"      • Overhead mayor que la ganancia")
        print(f"      • Mejor usar threading o secuencial")
        print(f"      • Considerar para n > 50,000")
    
    # Reglas generales de decisión
    print(f"\n   REGLAS GENERALES DE DECISIÓN:")
    print(f"      1. n < 1,000 → Usar secuencial (overhead muy alto)")
    print(f"      2. 1,000 ≤ n < 10,000 → Considerar threading")
    print(f"      3. n ≥ 10,000 → Evaluar multiprocessing")
    print(f"      4. Operaciones CPU-intensivas → Multiprocessing")
    print(f"      5. Operaciones I/O-bound → Threading")
    
    # ========== MUESTRA DE VALIDACIÓN ==========
    print(f"\n" + "="*70)
    print("MUESTRA DE VALIDACIÓN DE RESULTADOS")
    print("="*70)
    
    # Mostrar cálculos verificados
    print(f"\n   Validación de cálculos (índices aleatorios):")
    
    indices_muestra = random.sample(range(n), min(5, n))
    
    print(f"\n   {'Índice':<8} {'A':<8} {'B':<8} {'Resultado':<10} {'Estado':<10}")
    print(f"   {'-'*45}")
    
    for idx in indices_muestra:
        resultado_esperado = A[idx] + B[idx]
        resultado_real = R_multiproc[idx]
        correcto = resultado_real == resultado_esperado
        estado = "OK" if correcto else "ERROR"
        print(f"   {idx:<8} {A[idx]:<8} {B[idx]:<8} {resultado_real:<10} {estado}")
    
    # ========== CONCLUSIÓN FINAL ==========
    print(f"\n" + "="*70)
    print("CONCLUSIÓN FINAL - VERSIÓN 3")
    print("="*70)
    
    print(f"\n   Multiprocessing ofrece paralelismo real en Python al:")
    print(f"   1. Evitar el GIL usando procesos separados")
    print(f"   2. Utilizar múltiples núcleos de CPU eficientemente")
    print(f"   3. Escalar con el número de procesadores disponibles")
    
    print(f"\n   Limitaciones a considerar:")
    print(f"   1. Overhead de creación de procesos")
    print(f"   2. Comunicación entre procesos (serialización)")
    print(f"   3. Mayor uso de memoria")
    print(f"   4. Complejidad de implementación")
    
    print(f"\n" + "="*70)
    print("VERSIÓN 3 COMPLETADA EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    # Importante: main() debe estar protegido para multiprocessing en Windows
    main()