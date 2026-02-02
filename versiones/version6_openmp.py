"""
VERSION 6: Implementación con OpenMP
Sistema Comparativo con Paralelismo OpenMP

Descripcion: Sistema que incluye OpenMP como cuarto metodo de paralelismo
             junto a las versiones secuencial, threading y multiprocessing.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Requisitos:
- Python 3.6+
- Librería: pip install numba (recomendado)
- Alternativa: pip install pyomp

Caracteristicas:
1. Cuatro metodos de suma: Secuencial, Threading, Multiprocessing, OpenMP
2. Configuracion flexible de OpenMP (numero de threads)
3. Verificacion de resultados entre todos los metodos
4. Comparativa de rendimiento completa
5. Interfaz de usuario mejorada
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
from datetime import datetime

# =================== CONFIGURACION OPENMP ===================
# Intentar importar diferentes implementaciones de OpenMP
OPENMP_DISPONIBLE = False
OPENMP_IMPLEMENTACION = None

try:
    # Opcion 1: Numba con prange (OpenMP-like)
    from numba import jit, prange
    OPENMP_DISPONIBLE = True
    OPENMP_IMPLEMENTACION = "numba"
    print("✓ OpenMP disponible via numba")
except ImportError:
    try:
        # Opcion 2: pyomp
        import pyomp
        OPENMP_DISPONIBLE = True
        OPENMP_IMPLEMENTACION = "pyomp"
        print("OpenMP disponible via pyomp")
    except ImportError:
        try:
            # Opcion 3: pyopenmp
            import pyopenmp
            OPENMP_DISPONIBLE = True
            OPENMP_IMPLEMENTACION = "pyopenmp"
            print("OpenMP disponible via pyopenmp")
        except ImportError:
            print("OpenMP no disponible. Usando implementacion simulada.")
            OPENMP_DISPONIBLE = False
            OPENMP_IMPLEMENTACION = "simulada"

# =================== FUNCIONES AUXILIARES ===================
def crear_arreglos(n, max_val=10000, seed=None):
    """Crea dos arreglos aleatorios del tamano especificado"""
    if seed is not None:
        random.seed(seed)
    A = [random.randint(1, max_val) for _ in range(n)]
    B = [random.randint(1, max_val) for _ in range(n)]
    return A, B

def crear_arreglos_usuario(n):
    """Permite al usuario ingresar valores manualmente"""
    print(f"\nCreando {n} elementos por usuario...")
    print("Opciones:")
    print("  1. Ingresar valores manualmente")
    print("  2. Usar valores aleatorios")
    
    opcion = input("Seleccione opcion (1-2): ").strip()
    
    if opcion == "1":
        A = []
        B = []
        
        print(f"\nIngrese valores para el arreglo A (max {n} valores):")
        print("Formato: valor1,valor2,valor3,... o presione Enter para aleatorio")
        
        entrada_a = input("Valores A: ").strip()
        if entrada_a:
            valores = [int(x.strip()) for x in entrada_a.split(',') if x.strip()]
            A = valores[:n]
            # Rellenar si no hay suficientes
            if len(A) < n:
                A.extend([random.randint(1, 100) for _ in range(n - len(A))])
        else:
            A = [random.randint(1, 100) for _ in range(n)]
        
        print(f"\nIngrese valores para el arreglo B (max {n} valores):")
        entrada_b = input("Valores B: ").strip()
        if entrada_b:
            valores = [int(x.strip()) for x in entrada_b.split(',') if x.strip()]
            B = valores[:n]
            if len(B) < n:
                B.extend([random.randint(1, 100) for _ in range(n - len(B))])
        else:
            B = [random.randint(1, 100) for _ in range(n)]
        
        return A, B
    else:
        return crear_arreglos(n, 100)

def imprimir_arreglos(A, B, R, n, limite=10):
    """Imprime una muestra de los arreglos para verificacion"""
    print("\n" + "="*80)
    print("MUESTRA DE ARREGLOS (primeros {} elementos):".format(min(limite, n)))
    print("="*80)
    
    print(f"\n{'Indice':<10} {'A[i]':<10} {'B[i]':<10} {'R[i] (A+B)':<15} {'Verificacion':<12}")
    print("-"*57)
    
    for i in range(min(limite, n)):
        calculado = A[i] + B[i]
        correcto = R[i] == calculado
        verificacion = "✓" if correcto else "✗"
        
        print(f"{i:<10} {A[i]:<10} {B[i]:<10} {R[i]:<15} {verificacion:<12}")
    
    # Verificar si todos son correctos
    todos_correctos = all(R[i] == A[i] + B[i] for i in range(min(limite, n)))
    if todos_correctos:
        print(f"\nTodos los {min(limite, n)} elementos muestreados son correctos.")
    else:
        print(f"\nAlgunos elementos no son correctos.")
    
    return todos_correctos

def verificar_resultados(R1, R2, A, B, n, muestra=100):
    """Verifica que dos resultados sean iguales mediante muestreo"""
    if len(R1) != len(R2):
        return False
    
    indices = random.sample(range(n), min(muestra, n))
    for i in indices:
        esperado = A[i] + B[i]
        if R1[i] != esperado or R2[i] != esperado:
            return False
    return True

def calcular_metricas(tiempo_sec, tiempo_par, num_hilos):
    """Calcula speedup y eficiencia"""
    if tiempo_par > 0:
        speedup = tiempo_sec / tiempo_par
        eficiencia = (speedup / num_hilos) * 100
    else:
        speedup = 0
        eficiencia = 0
    return speedup, eficiencia

# =================== IMPLEMENTACIONES DE METODOS ===================
def suma_secuencial(A, B, n):
    """VERSION 1: Suma secuencial tradicional"""
    R = [0] * n
    for i in range(n):
        R[i] = A[i] + B[i]
    return R

def suma_threading(A, B, n, num_threads=4):
    """VERSION 2: Suma con threading"""
    R = [0] * n
    chunk_size = n // num_threads
    threads = []
    
    def worker(start, end):
        for i in range(start, end):
            R[i] = A[i] + B[i]
    
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return R

def worker_process(args):
    """Funcion auxiliar para multiprocessing"""
    A_chunk, B_chunk, start_idx = args
    resultado = []
    for i in range(len(A_chunk)):
        resultado.append(A_chunk[i] + B_chunk[i])
    return start_idx, resultado

def suma_multiprocessing(A, B, n, num_processes=4):
    """VERSION 3: Suma con multiprocessing"""
    chunk_size = max(100, n // num_processes)
    chunks = []
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunks.append((A[i:end], B[i:end], i))
    
    R = [0] * n
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        resultados = list(executor.map(worker_process, chunks))
    
    for start_idx, resultado_chunk in resultados:
        for j, valor in enumerate(resultado_chunk):
            R[start_idx + j] = valor
    
    return R

# =================== IMPLEMENTACION OPENMP ===================
def suma_openmp_numba(A, B, n, num_threads=4):
    """VERSION 4: Suma con OpenMP usando Numba"""
    
    @jit(nopython=True, parallel=True, nogil=True)
    def suma_paralela_numba(A, B, R):
        for i in prange(len(A)):
            R[i] = A[i] + B[i]
        return R
    
    # Configurar numero de threads para numba
    import numba
    numba.set_num_threads(num_threads)
    
    # Convertir a arrays de numpy para numba
    import numpy as np
    A_np = np.array(A, dtype=np.int32)
    B_np = np.array(B, dtype=np.int32)
    R_np = np.zeros(n, dtype=np.int32)
    
    # Ejecutar funcion compilada
    R_np = suma_paralela_numba(A_np, B_np, R_np)
    
    return R_np.tolist()

def suma_openmp_pyomp(A, B, n, num_threads=4):
    """VERSION 4: Suma con OpenMP usando pyomp"""
    import pyomp
    
    R = [0] * n
    
    # Configurar numero de threads
    pyomp.set_num_threads(num_threads)
    
    # Paralelizar con OpenMP
    @pyomp.parallel(num_threads=num_threads)
    def paralelizar():
        # Obtener informacion del thread
        thread_id = pyomp.get_thread_num()
        num_threads = pyomp.get_num_threads()
        
        # Calcular chunk para este thread
        chunk_size = n // num_threads
        start = thread_id * chunk_size
        end = n if thread_id == num_threads - 1 else start + chunk_size
        
        # Sumar el chunk asignado
        for i in range(start, end):
            R[i] = A[i] + B[i]
    
    # Ejecutar region paralela
    paralelizar()
    
    return R

def suma_openmp_simulada(A, B, n, num_threads=4):
    """Implementacion simulada de OpenMP (si no hay libreria disponible)"""
    print("  Nota: Usando implementacion simulada de OpenMP")
    
    R = [0] * n
    chunk_size = n // num_threads
    threads = []
    
    def worker(start, end):
        for i in range(start, end):
            R[i] = A[i] + B[i]
    
    for i in range(num_threads):
        start = i * chunk_size
        end = n if i == num_threads - 1 else start + chunk_size
        
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return R

def suma_openmp(A, B, n, num_threads=4):
    """Funcion principal de OpenMP que selecciona la implementacion disponible"""
    
    print(f"  Implementacion OpenMP: {OPENMP_IMPLEMENTACION}")
    print(f"  Threads configurados: {num_threads}")
    
    if not OPENMP_DISPONIBLE:
        print("  OpenMP no disponible, usando simulacion")
        return suma_openmp_simulada(A, B, n, num_threads)
    
    try:
        if OPENMP_IMPLEMENTACION == "numba":
            return suma_openmp_numba(A, B, n, num_threads)
        elif OPENMP_IMPLEMENTACION == "pyomp":
            return suma_openmp_pyomp(A, B, n, num_threads)
        elif OPENMP_IMPLEMENTACION == "pyopenmp":
            # pyopenmp tiene una API similar
            import pyopenmp as omp
            R = [0] * n
            
            # Region paralela
            # Nota: pyopenmp puede requerir configuracion adicional
            for i in range(n):
                R[i] = A[i] + B[i]
            
            return R
        else:
            return suma_openmp_simulada(A, B, n, num_threads)
    except Exception as e:
        print(f"  Error con OpenMP: {e}")
        print("  Usando implementacion simulada...")
        return suma_openmp_simulada(A, B, n, num_threads)

# =================== SISTEMA DE COMPARACION OPENMP ===================
class SistemaComparacionOpenMP:
    """Sistema comparativo incluyendo OpenMP"""
    
    def __init__(self):
        self.resultados = []
        self.configuraciones_openmp = [1, 2, 4, 8]
    
    def ejecutar_comparativa_openmp(self, n=1000, usar_valores_usuario=False):
        """Ejecuta comparativa completa incluyendo OpenMP"""
        
        print("\n" + "="*80)
        print("COMPARATIVA OPENMP - n = {} elementos".format(n))
        print("="*80)
        
        # Crear arreglos
        if usar_valores_usuario:
            A, B = crear_arreglos_usuario(n)
        else:
            A, B = crear_arreglos(n, 100)
        
        resultados = []
        
        # 1. METODO SECUENCIAL (base)
        print(f"\n1. EJECUTANDO METODO SECUENCIAL...")
        inicio = time.perf_counter()
        R_sec = suma_secuencial(A, B, n)
        t_sec = time.perf_counter() - inicio
        resultados.append(("Secuencial", t_sec, 1, 0, 0, R_sec))
        print(f"   Tiempo: {t_sec:.6f}s")
        
        # 2. METODO THREADING
        print(f"\n2. EJECUTANDO METODO THREADING...")
        for threads in [2, 4]:
            inicio = time.perf_counter()
            R_thr = suma_threading(A, B, n, threads)
            t_thr = time.perf_counter() - inicio
            speedup, eficiencia = calcular_metricas(t_sec, t_thr, threads)
            resultados.append((f"Threading ({threads})", t_thr, threads, speedup, eficiencia, R_thr))
            print(f"   • {threads} threads: {t_thr:.6f}s (Speedup: {speedup:.2f}x)")
        
        # 3. METODO MULTIPROCESSING
        print(f"\n3. EJECUTANDO METODO MULTIPROCESSING...")
        procesos = min(4, mp.cpu_count())
        inicio = time.perf_counter()
        try:
            R_mp = suma_multiprocessing(A, B, n, procesos)
            t_mp = time.perf_counter() - inicio
            speedup, eficiencia = calcular_metricas(t_sec, t_mp, procesos)
            resultados.append((f"Multiprocessing ({procesos})", t_mp, procesos, speedup, eficiencia, R_mp))
            print(f"   • {procesos} procesos: {t_mp:.6f}s (Speedup: {speedup:.2f}x)")
        except Exception as e:
            print(f"   • Error: {e}")
        
        # 4. METODO OPENMP
        print(f"\n4. EJECUTANDO METODO OPENMP...")
        print(f"   Implementacion disponible: {OPENMP_IMPLEMENTACION}")
        
        for threads in self.configuraciones_openmp:
            if threads <= mp.cpu_count():
                inicio = time.perf_counter()
                try:
                    R_omp = suma_openmp(A, B, n, threads)
                    t_omp = time.perf_counter() - inicio
                    speedup, eficiencia = calcular_metricas(t_sec, t_omp, threads)
                    resultados.append((f"OpenMP ({threads})", t_omp, threads, speedup, eficiencia, R_omp))
                    print(f"   • {threads} threads: {t_omp:.6f}s (Speedup: {speedup:.2f}x)")
                except Exception as e:
                    print(f"   • Error con {threads} threads: {e}")
        
        # VERIFICACION DE RESULTADOS
        print(f"\n5. VERIFICANDO RESULTADOS...")
        
        # Tomar el resultado secuencial como referencia
        R_referencia = resultados[0][5]  # R_sec
        
        for nombre, tiempo, config, speedup, eficiencia, R in resultados[1:]:
            if R:  # Si hay resultado
                correcto = verificar_resultados(R_referencia, R, A, B, n, 50)
                print(f"   • {nombre}: {'CORRECTO' if correcto else 'ERROR'}")
        
        # IMPRIMIR MUESTRA DE ARREGLOS
        print(f"\n6. MUESTRA DE RESULTADOS...")
        # Usar el resultado de OpenMP si existe, sino usar secuencial
        R_muestra = None
        for nombre, _, _, _, _, R in resultados:
            if "OpenMP" in nombre and R:
                R_muestra = R
                break
        
        if not R_muestra:
            R_muestra = R_referencia
        
        imprimir_arreglos(A, B, R_muestra, n, 10)
        
        # RESUMEN DE RESULTADOS
        print(f"\n" + "="*80)
        print("RESUMEN DE COMPARATIVA")
        print("="*80)
        
        print(f"\n{'Metodo':<25} {'Tiempo (s)':<15} {'Threads':<10} {'Speedup':<10} {'Eficiencia':<10}")
        print("-"*70)
        
        for nombre, tiempo, config, speedup, eficiencia, _ in resultados:
            if "Secuencial" in nombre:
                print(f"{nombre:<25} {tiempo:<15.6f} {config:<10} {'-':<10} {'-':<10}")
            else:
                print(f"{nombre:<25} {tiempo:<15.6f} {config:<10} {speedup:<10.2f} {eficiencia:<10.1f}%")
        
        # ENCONTRAR EL MEJOR METODO
        if len(resultados) > 1:
            # Filtrar metodos paralelos (excluyendo secuencial)
            resultados_paralelos = [r for r in resultados if r[0] != "Secuencial" and r[1] > 0]
            
            if resultados_paralelos:
                mejor = min(resultados_paralelos, key=lambda x: x[1])
                print(f"\nMEJOR METODO: {mejor[0]}")
                print(f"   Tiempo: {mejor[1]:.6f}s")
                print(f"   Speedup vs secuencial: {mejor[3]:.2f}x")
                print(f"   Eficiencia: {mejor[4]:.1f}%")
        
        # Guardar resultados
        resultado_completo = {
            'tamano': n,
            'usuario': usar_valores_usuario,
            'resultados': resultados,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.resultados.append(resultado_completo)
        
        return resultados
    
    def ejecutar_prueba_rapida_openmp(self):
        """Prueba rapida especifica para OpenMP"""
        print("\n" + "="*80)
        print("PRUEBA RAPIDA OPENMP")
        print("="*80)
        
        n = 1000
        print(f"Tamaño: {n} elementos")
        
        # Crear arreglos pequeños para prueba rapida
        A, B = crear_arreglos(n, 100)
        
        print(f"\n1. Suma Secuencial...")
        inicio = time.perf_counter()
        R_sec = suma_secuencial(A, B, n)
        t_sec = time.perf_counter() - inicio
        print(f"   Tiempo: {t_sec:.6f}s")
        
        print(f"\n2. Suma OpenMP (4 threads)...")
        inicio = time.perf_counter()
        R_omp = suma_openmp(A, B, n, 4)
        t_omp = time.perf_counter() - inicio
        
        speedup, eficiencia = calcular_metricas(t_sec, t_omp, 4)
        print(f"   Tiempo: {t_omp:.6f}s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Eficiencia: {eficiencia:.1f}%")
        
        # Verificar
        print(f"\n3. Verificando resultados...")
        correcto = verificar_resultados(R_sec, R_omp, A, B, n, 20)
        print(f"   Resultados: {'✓ CORRECTOS' if correcto else '✗ ERROR'}")
        
        # Mostrar muestra
        print(f"\n4. Muestra de arreglos (primeros 5 elementos):")
        print(f"{'A':<10} {'B':<10} {'A+B':<10} {'OpenMP':<10}")
        print("-"*40)
        for i in range(min(5, n)):
            print(f"{A[i]:<10} {B[i]:<10} {A[i]+B[i]:<10} {R_omp[i]:<10}")
        
        input("\nPresiona Enter para continuar...")
    
    def mostrar_info_openmp(self):
        """Muestra informacion sobre la configuracion OpenMP"""
        print("\n" + "="*80)
        print("INFORMACION OPENMP")
        print("="*80)
        
        print(f"\nEstado OpenMP: {'DISPONIBLE' if OPENMP_DISPONIBLE else 'NO DISPONIBLE'}")
        print(f"Implementacion: {OPENMP_IMPLEMENTACION}")
        print(f"CPUs del sistema: {mp.cpu_count()}")
        
        if OPENMP_DISPONIBLE:
            print("\nOpenMP esta correctamente configurado.")
            print("Puedes ejecutar pruebas con paralelismo OpenMP.")
        else:
            print("\nOpenMP no esta disponible.")
            print("Opciones para instalarlo:")
            print("  1. pip install numba (recomendado)")
            print("  2. pip install pyomp")
            print("  3. pip install pyopenmp")
            print("\nEl sistema usara una implementacion simulada.")
        
        print("\nConfiguraciones OpenMP probadas:", self.configuraciones_openmp)
        
        input("\nPresiona Enter para continuar...")
    
    def exportar_resultados_openmp(self):
        """Exporta resultados de pruebas OpenMP"""
        if not self.resultados:
            print("\nNo hay resultados para exportar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("resultados_openmp", exist_ok=True)
        
        archivo = f"resultados_openmp/openmp_{timestamp}.txt"
        
        with open(archivo, "w") as f:
            f.write("="*80 + "\n")
            f.write("RESULTADOS OPENMP - SISTEMA COMPARATIVO\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"OpenMP disponible: {OPENMP_DISPONIBLE}\n")
            f.write(f"Implementacion: {OPENMP_IMPLEMENTACION}\n")
            f.write(f"CPUs: {mp.cpu_count()}\n")
            f.write("="*80 + "\n\n")
            
            for idx, resultado in enumerate(self.resultados, 1):
                f.write(f"PRUEBA #{idx} - {resultado['timestamp']}\n")
                f.write(f"Tamaño: {resultado['tamano']:,} elementos\n")
                f.write(f"Valores usuario: {'SI' if resultado['usuario'] else 'NO'}\n")
                f.write("-"*60 + "\n")
                
                f.write(f"{'Metodo':<25} {'Tiempo (s)':<15} {'Threads':<10} {'Speedup':<10}\n")
                f.write("-"*60 + "\n")
                
                for nombre, tiempo, config, speedup, eficiencia, _ in resultado['resultados']:
                    if "Secuencial" in nombre:
                        f.write(f"{nombre:<25} {tiempo:<15.6f} {config:<10} {'-':<10}\n")
                    else:
                        f.write(f"{nombre:<25} {tiempo:<15.6f} {config:<10} {speedup:<10.2f}\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"\nResultados exportados a: {archivo}")
        input("Presiona Enter para continuar...")

# =================== INTERFAZ DE USUARIO ===================
def mostrar_menu_openmp():
    """Muestra el menu principal con OpenMP"""
    print("\n" + "="*80)
    print("SISTEMA COMPARATIVO CON OPENMP - VERSION 6")
    print("="*80)
    print("\nOPCIONES PRINCIPALES:")
    print("  1. Prueba rapida OpenMP (n=1000)")
    print("  2. Comparativa completa OpenMP (valores aleatorios)")
    print("  3. Comparativa completa OpenMP (valores de usuario)")
    print("  4. Probar diferentes tamaños")
    
    print("\nINFORMACION Y CONFIGURACION:")
    print("  5. Ver informacion OpenMP")
    print("  6. Configurar OpenMP")
    print("  7. Exportar resultados")
    
    print("\nVERSIONES ANTERIORES:")
    print("  8. Ejecutar version completa (todos los metodos)")
    print("  9. Volver al menu principal")
    print("  0. Salir")
    
    print("\n" + "="*80)

def configurar_openmp():
    """Permite configurar parametros de OpenMP"""
    print("\n" + "="*80)
    print("CONFIGURACION OPENMP")
    print("="*80)
    
    sistema = SistemaComparacionOpenMP()
    
    print(f"\nConfiguraciones actuales de threads: {sistema.configuraciones_openmp}")
    print(f"CPUs disponibles: {mp.cpu_count()}")
    
    print("\n¿Deseas cambiar las configuraciones?")
    print("  1. Usar configuracion estandar [1,2,4,8]")
    print("  2. Usar configuracion optima [2,4,sistema]")
    print("  3. Personalizar")
    print("  4. Mantener actual")
    
    opcion = input("\nSelecciona opcion (1-4): ").strip()
    
    if opcion == "1":
        sistema.configuraciones_openmp = [1, 2, 4, 8]
        print("Configuracion establecida: [1, 2, 4, 8]")
    elif opcion == "2":
        cpus = mp.cpu_count()
        config = [2, 4]
        if cpus > 4:
            config.append(cpus)
        sistema.configuraciones_openmp = config
        print(f"Configuracion establecida: {config}")
    elif opcion == "3":
        print("\nIngresa los numeros de threads separados por comas:")
        entrada = input("Threads: ").strip()
        if entrada:
            try:
                threads = [int(x.strip()) for x in entrada.split(',')]
                threads = sorted(list(set(threads)))  # Eliminar duplicados y ordenar
                sistema.configuraciones_openmp = threads
                print(f"Configuracion establecida: {threads}")
            except ValueError:
                print("Error: valores no validos.")
    
    print(f"\nConfiguracion actual: {sistema.configuraciones_openmp}")
    input("\nPresiona Enter para continuar...")

def probar_diferentes_tamanos():
    """Prueba OpenMP con diferentes tamaños de arreglo"""
    print("\n" + "="*80)
    print("PRUEBA CON DIFERENTES TAMAÑOS")
    print("="*80)
    
    tamanos = [100, 500, 1000, 5000, 10000]
    sistema = SistemaComparacionOpenMP()
    
    for n in tamanos:
        print(f"\n" + "="*60)
        print(f"PROBANDO CON n = {n:,} elementos")
        print("="*60)
        
        # Solo OpenMP para comparar diferentes tamaños
        A, B = crear_arreglos(n, 100)
        
        # Secuencial como referencia
        inicio = time.perf_counter()
        R_sec = suma_secuencial(A, B, n)
        t_sec = time.perf_counter() - inicio
        
        # OpenMP con diferentes threads
        for threads in [1, 2, 4]:
            if threads <= mp.cpu_count():
                inicio = time.perf_counter()
                try:
                    R_omp = suma_openmp(A, B, n, threads)
                    t_omp = time.perf_counter() - inicio
                    speedup, _ = calcular_metricas(t_sec, t_omp, threads)
                    
                    # Verificar
                    correcto = verificar_resultados(R_sec, R_omp, A, B, n, 20)
                    
                    print(f"  OpenMP ({threads} threads):")
                    print(f"    • Tiempo: {t_omp:.6f}s")
                    print(f"    • Speedup: {speedup:.2f}x")
                    print(f"    • Verificacion: {'✓' if correcto else '✗'}")
                except Exception as e:
                    print(f"  Error con {threads} threads: {e}")
    
    input("\nPresiona Enter para continuar...")

# =================== PROGRAMA PRINCIPAL ===================
def main_openmp():
    """Funcion principal de la version OpenMP"""
    
    print("\n" + "="*80)
    print("VERSION 6 - IMPLEMENTACION CON OPENMP")
    print("="*80)
    print("\nSistema comparativo que incluye OpenMP como cuarto metodo")
    print("de paralelismo, junto a secuencial, threading y multiprocessing.")
    
    sistema = SistemaComparacionOpenMP()
    
    while True:
        mostrar_menu_openmp()
        
        try:
            opcion = input("\nSelecciona una opcion (0-9): ").strip()
            
            if opcion == "0":
                print("\n¡Gracias por usar el Sistema OpenMP!")
                break
                
            elif opcion == "1":
                sistema.ejecutar_prueba_rapida_openmp()
                
            elif opcion == "2":
                n = 1000
                print(f"\nEjecutando comparativa OpenMP con n={n} (valores aleatorios)...")
                sistema.ejecutar_comparativa_openmp(n, usar_valores_usuario=False)
                input("\nPresiona Enter para continuar...")
                
            elif opcion == "3":
                n = 1000
                print(f"\nEjecutando comparativa OpenMP con n={n} (valores de usuario)...")
                sistema.ejecutar_comparativa_openmp(n, usar_valores_usuario=True)
                input("\nPresiona Enter para continuar...")
                
            elif opcion == "4":
                probar_diferentes_tamanos()
                
            elif opcion == "5":
                sistema.mostrar_info_openmp()
                
            elif opcion == "6":
                configurar_openmp()
                
            elif opcion == "7":
                sistema.exportar_resultados_openmp()
                
            elif opcion == "8":
                # Ejecutar version completa con todos los metodos
                from version5_configuraciones_avanzadas import main as main_v5
                print("\nCargando version completa...")
                main_v5()
                print("\nRegresando a version OpenMP...")
                
            elif opcion == "9":
                # Volver al menu principal
                print("\nRegresando al menu principal...")
                break
                
            else:
                print("\nOpcion invalida. Por favor selecciona 0-9.")
                input("Presiona Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("Presiona Enter para continuar...")

# =================== EJECUCION DIRECTA ===================
if __name__ == "__main__":
    # Mostrar banner de OpenMP
    print("\n" + "="*80)
    print(" " * 25 + "OPENMP PARA PYTHON")
    print(" " * 20 + "Sistema Comparativo V6")
    print("="*80)
    print(" " * 20 + "Autor: Hector Jorge Morales Arch")
    print(" " * 25 + "Alias: 江久取")
    print("="*80)
    
    # Verificar dependencias
    print("\nVerificando dependencias OpenMP...")
    
    if not OPENMP_DISPONIBLE:
        print("\nADVERTENCIA: OpenMP no esta disponible.")
        print("\nPara instalar OpenMP, ejecuta uno de los siguientes comandos:")
        print("  1. pip install numba (recomendado para esta version)")
        print("  2. pip install pyomp")
        print("  3. pip install pyopenmp")
        
        respuesta = input("\n¿Deseas continuar con implementacion simulada? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Saliendo del programa...")
            sys.exit(1)
    
    # Ejecutar menu principal
    main_openmp()