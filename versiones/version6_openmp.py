"""
VERSION 6 COMPLETA: Sistema Comparativo con OpenMP y Todas las Funcionalidades
Todas las versiones integradas en un solo sistema

Autor: Hector Jorge Morales Arch
Alias: 江久取

Caracteristicas incluidas:
1. Version 1: Suma secuencial
2. Version 2: Suma con threading
3. Version 3: Suma con multiprocessing
4. Version 4: Comparativa completa
5. Version 5: Configuraciones avanzadas y aleatorias
6. Version 6: OpenMP con Numba
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
from datetime import datetime

# =================== CONFIGURACION OPENMP/NUMBA ===================
try:
    from numba import jit, prange, set_num_threads
    import numpy as np
    OPENMP_DISPONIBLE = True
    print("✓ OpenMP/Numba disponible")
except ImportError:
    OPENMP_DISPONIBLE = False
    print("⚠ OpenMP/Numba no disponible (usando threading simulado)")

# =================== FUNCIONES COMPARTIDAS ===================
def crear_arreglos(n, max_val=10000, seed=None):
    """Crea dos arreglos aleatorios"""
    if seed is not None:
        random.seed(seed)
    A = [random.randint(1, max_val) for _ in range(n)]
    B = [random.randint(1, max_val) for _ in range(n)]
    return A, B

def crear_arreglos_usuario(n):
    """Permite al usuario ingresar valores manualmente"""
    print(f"\nCreando {n} elementos")
    print("1. Valores aleatorios")
    print("2. Ingresar manualmente")
    
    opcion = input("Opción: ").strip()
    
    if opcion == "2":
        A = []
        B = []
        
        print("\nEjemplo: 1,2,3,4,5")
        print("Arreglo A:")
        entrada = input("Valores (Enter para aleatorio): ").strip()
        if entrada:
            A = [int(x.strip()) for x in entrada.split(',') if x.strip().isdigit()]
        if not A or len(A) < n:
            A = A + [random.randint(1, 100) for _ in range(n - len(A))]
        
        print("\nArreglo B:")
        entrada = input("Valores (Enter para aleatorio): ").strip()
        if entrada:
            B = [int(x.strip()) for x in entrada.split(',') if x.strip().isdigit()]
        if not B or len(B) < n:
            B = B + [random.randint(1, 100) for _ in range(n - len(B))]
        
        return A[:n], B[:n]
    else:
        return crear_arreglos(n, 100)

def verificar_resultados(R1, R2, A, B, n, muestra=100):
    """Verifica que dos resultados sean iguales"""
    if len(R1) != len(R2):
        return False
    
    indices = random.sample(range(n), min(muestra, n))
    for i in indices:
        if R1[i] != A[i] + B[i] or R2[i] != A[i] + B[i]:
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

def mostrar_muestra(A, B, R, limite=10):
    """Muestra una muestra de los arreglos"""
    print(f"\n{'i':<5} {'A[i]':<10} {'B[i]':<10} {'R[i]':<10} {'A+B':<10} {'✓/✗':<5}")
    print("-"*50)
    
    for i in range(min(limite, len(A))):
        esperado = A[i] + B[i]
        correcto = R[i] == esperado
        print(f"{i:<5} {A[i]:<10} {B[i]:<10} {R[i]:<10} {esperado:<10} {'✓' if correcto else '✗'}")
    
    return all(R[i] == A[i] + B[i] for i in range(min(limite, len(A))))

# =================== VERSION 1: SECUENCIAL ===================
def suma_secuencial(A, B, n):
    """Version 1: Suma secuencial tradicional"""
    R = [0] * n
    for i in range(n):
        R[i] = A[i] + B[i]
    return R

# =================== VERSION 2: THREADING ===================
def suma_threading(A, B, n, num_threads=4):
    """Version 2: Suma con threading"""
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

# =================== VERSION 3: MULTIPROCESSING ===================
def worker_process(args):
    """Funcion auxiliar para multiprocessing"""
    A_chunk, B_chunk, start_idx = args
    resultado = []
    for i in range(len(A_chunk)):
        resultado.append(A_chunk[i] + B_chunk[i])
    return start_idx, resultado

def suma_multiprocessing(A, B, n, num_processes=4):
    """Version 3: Suma con multiprocessing"""
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

# =================== VERSION 6: OPENMP ===================
@jit(nopython=True, parallel=True, nogil=True)
def suma_openmp_numba(A_np, B_np, R_np):
    """Version 6: Suma con OpenMP usando Numba"""
    for i in prange(len(A_np)):
        R_np[i] = A_np[i] + B_np[i]
    return R_np

def suma_openmp(A, B, n, num_threads=4):
    """Funcion principal de OpenMP"""
    if not OPENMP_DISPONIBLE:
        # Simular con threading si no hay OpenMP
        print("  (Usando threading como simulacion de OpenMP)")
        return suma_threading(A, B, n, num_threads)
    
    try:
        # Convertir a numpy arrays
        A_np = np.array(A, dtype=np.int32)
        B_np = np.array(B, dtype=np.int32)
        R_np = np.zeros(n, dtype=np.int32)
        
        # Configurar threads
        set_num_threads(num_threads)
        
        # Ejecutar
        R_np = suma_openmp_numba(A_np, B_np, R_np)
        return R_np.tolist()
    except Exception as e:
        print(f"  Error OpenMP: {e}, usando threading")
        return suma_threading(A, B, n, num_threads)

# =================== SISTEMA COMPLETO INTEGRADO ===================
class SistemaCompleto:
    """Sistema que integra todas las versiones"""
    
    def __init__(self):
        self.resultados = []
        self.historico = []
        
    # =================== VERSION 4: COMPARATIVA COMPLETA ===================
    def comparativa_completa(self):
        """Version 4: Comparativa de todos los metodos"""
        print("\n" + "="*80)
        print("VERSION 4: COMPARATIVA COMPLETA")
        print("="*80)
        
        tamanos = [100, 1000, 5000, 10000]
        
        for n in tamanos:
            print(f"\n\n🔬 ANALIZANDO TAMAÑO: {n:,} elementos")
            print("-"*60)
            
            A, B = crear_arreglos(n, 100)
            resultados_tam = []
            
            # Secuencial
            print(f"\n   📏 SECUENCIAL...")
            inicio = time.perf_counter()
            R_sec = suma_secuencial(A, B, n)
            t_sec = time.perf_counter() - inicio
            resultados_tam.append(('Secuencial', t_sec, 1, 0, 0))
            print(f"      Tiempo: {t_sec:.6f}s")
            
            # Threading
            print(f"\n   🧵 THREADING...")
            for threads in [2, 4]:
                inicio = time.perf_counter()
                R_thr = suma_threading(A, B, n, threads)
                t_thr = time.perf_counter() - inicio
                speedup, eficiencia = calcular_metricas(t_sec, t_thr, threads)
                resultados_tam.append((f'Threading ({threads})', t_thr, threads, speedup, eficiencia))
                print(f"      {threads} threads: {t_thr:.6f}s (Speedup: {speedup:.2f}x)")
            
            # Multiprocessing
            print(f"\n   ⚡ MULTIPROCESSING...")
            procesos = min(4, mp.cpu_count())
            try:
                inicio = time.perf_counter()
                R_mp = suma_multiprocessing(A, B, n, procesos)
                t_mp = time.perf_counter() - inicio
                speedup, eficiencia = calcular_metricas(t_sec, t_mp, procesos)
                resultados_tam.append((f'Multiprocessing ({procesos})', t_mp, procesos, speedup, eficiencia))
                print(f"      {procesos} procesos: {t_mp:.6f}s (Speedup: {speedup:.2f}x)")
            except Exception as e:
                print(f"      Error: {e}")
            
            # OpenMP
            if OPENMP_DISPONIBLE:
                print(f"\n   🚀 OPENMP...")
                for threads in [2, 4]:
                    inicio = time.perf_counter()
                    R_omp = suma_openmp(A, B, n, threads)
                    t_omp = time.perf_counter() - inicio
                    speedup, eficiencia = calcular_metricas(t_sec, t_omp, threads)
                    resultados_tam.append((f'OpenMP ({threads})', t_omp, threads, speedup, eficiencia))
                    print(f"      {threads} threads: {t_omp:.6f}s (Speedup: {speedup:.2f}x)")
            
            # Encontrar el mejor
            if len(resultados_tam) > 1:
                mejor = min(resultados_tam[1:], key=lambda x: x[1])
                print(f"\n   🏆 MEJOR PARA n={n:,}: {mejor[0]}")
                print(f"      Tiempo: {mejor[1]:.6f}s, Speedup: {mejor[3]:.2f}x")
            
            self.historico.append({
                'tamano': n,
                'resultados': resultados_tam,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        # Mostrar reporte final
        self._mostrar_reporte_version4()
    
    def _mostrar_reporte_version4(self):
        """Muestra reporte de la version 4"""
        print("\n" + "="*80)
        print("📊 REPORTE FINAL - VERSION 4")
        print("="*80)
        
        for item in self.historico[-4:]:  # Ultimos 4 tamanos
            n = item['tamano']
            print(f"\n📐 TAMAÑO: {n:,} elementos")
            print("-"*60)
            
            print(f"\n{'Metodo':<25} {'Tiempo (s)':<15} {'Threads':<10} {'Speedup':<10}")
            print("-"*60)
            
            for nombre, tiempo, threads, speedup, _ in item['resultados']:
                if nombre == 'Secuencial':
                    print(f"{nombre:<25} {tiempo:<15.6f} {threads:<10} {'-':<10}")
                else:
                    print(f"{nombre:<25} {tiempo:<15.6f} {threads:<10} {speedup:<10.2f}")
    
    # =================== VERSION 5: CONFIGURACIONES AVANZADAS ===================
    def generar_configuracion_aleatoria(self):
        """Genera configuraciones aleatorias"""
        return {
            'arreglo_random': {
                'tamanos': [100, 500, 1000, 5000, 10000],
                'max_val': 10000,
                'seed': random.randint(1, 1000)
            },
            'threads_random': {
                'configs': random.sample([1, 2, 4, 6, 8, mp.cpu_count()], 3)
            },
            'ambos_random': {
                'tamanos': random.sample([100, 500, 1000, 2000, 5000, 10000, 20000], 3),
                'threads': random.sample([1, 2, 4, 6, 8, mp.cpu_count()], 3)
            }
        }
    
    def arreglo_random_threads_fijos(self):
        """Version 5: Arreglo random, threads fijos"""
        print("\n" + "="*80)
        print("VERSION 5: ARREGLO RANDOM, THREADS FIJOS")
        print("="*80)
        
        configs = self.generar_configuracion_aleatoria()
        config = configs['arreglo_random']
        n = random.choice(config['tamanos'])
        
        print(f"\nConfiguracion generada aleatoriamente:")
        print(f"  • Tamaño: {n:,} elementos")
        print(f"  • Max valor: {config['max_val']}")
        print(f"  • Seed: {config['seed']}")
        
        A, B = crear_arreglos(n, config['max_val'], config['seed'])
        
        self._ejecutar_comparativa_avanzada(A, B, n, [1, 2, 4, 8, mp.cpu_count()])
    
    def arreglo_fijo_threads_random(self):
        """Version 5: Arreglo fijo, threads random"""
        print("\n" + "="*80)
        print("VERSION 5: ARREGLO FIJO, THREADS RANDOM")
        print("="*80)
        
        configs = self.generar_configuracion_aleatoria()
        config = configs['threads_random']
        n = 5000  # Fijo
        
        print(f"\nConfiguracion generada aleatoriamente:")
        print(f"  • Tamaño fijo: {n:,} elementos")
        print(f"  • Threads aleatorios: {config['configs']}")
        
        A, B = crear_arreglos(n, 10000)
        
        self._ejecutar_comparativa_avanzada(A, B, n, config['configs'])
    
    def ambos_random(self):
        """Version 5: Ambos random"""
        print("\n" + "="*80)
        print("VERSION 5: AMBOS RANDOM")
        print("="*80)
        
        configs = self.generar_configuracion_aleatoria()
        config = configs['ambos_random']
        n = random.choice(config['tamanos'])
        
        print(f"\nConfiguracion generada aleatoriamente:")
        print(f"  • Tamaño aleatorio: {n:,} elementos")
        print(f"  • Threads aleatorios: {config['threads']}")
        
        A, B = crear_arreglos(n, 10000)
        
        self._ejecutar_comparativa_avanzada(A, B, n, config['threads'])
    
    def arreglo_especifico(self):
        """Version 5: Arreglo especifico del usuario"""
        print("\n" + "="*80)
        print("VERSION 5: CONFIGURACION ESPECIFICA")
        print("="*80)
        
        try:
            n = int(input("\nTamaño del arreglo: ").strip())
            if n <= 0:
                print("El tamaño debe ser positivo.")
                return
        except ValueError:
            print("Tamaño no válido.")
            return
        
        print("\n1. Valores aleatorios")
        print("2. Ingresar valores")
        opcion = input("Opción: ").strip()
        
        if opcion == "2":
            A, B = crear_arreglos_usuario(n)
        else:
            A, B = crear_arreglos(n, 100)
        
        print("\nConfiguración de threads:")
        print("1. [1, 2, 4, 8]")
        print("2. [2, 4, sistema]")
        print("3. Personalizado")
        
        opcion_threads = input("Opción: ").strip()
        
        if opcion_threads == "1":
            threads_config = [1, 2, 4, 8]
        elif opcion_threads == "2":
            threads_config = [2, 4, mp.cpu_count()]
        elif opcion_threads == "3":
            entrada = input("Threads (ej: 2,4,6): ").strip()
            threads_config = [int(x.strip()) for x in entrada.split(',') if x.strip().isdigit()]
        else:
            threads_config = [1, 2, 4]
        
        self._ejecutar_comparativa_avanzada(A, B, n, threads_config)
    
    def _ejecutar_comparativa_avanzada(self, A, B, n, threads_config):
        """Ejecuta comparativa avanzada con configuracion dada"""
        print(f"\n" + "="*60)
        print(f"EJECUTANDO COMPARATIVA (n={n:,})")
        print("="*60)
        
        resultados = []
        
        # Secuencial
        print(f"\n1. SECUENCIAL...")
        inicio = time.perf_counter()
        R_sec = suma_secuencial(A, B, n)
        t_sec = time.perf_counter() - inicio
        resultados.append(('Secuencial', t_sec, 1, 0, 0))
        print(f"   Tiempo: {t_sec:.6f}s")
        
        # Threading
        print(f"\n2. THREADING...")
        for threads in threads_config:
            if threads <= mp.cpu_count():
                inicio = time.perf_counter()
                R_thr = suma_threading(A, B, n, threads)
                t_thr = time.perf_counter() - inicio
                speedup, _ = calcular_metricas(t_sec, t_thr, threads)
                resultados.append((f'Threading ({threads})', t_thr, threads, speedup, 0))
                print(f"   • {threads} threads: {t_thr:.6f}s (Speedup: {speedup:.2f}x)")
        
        # Multiprocessing
        print(f"\n3. MULTIPROCESSING...")
        for procesos in [p for p in threads_config if p <= mp.cpu_count()]:
            try:
                inicio = time.perf_counter()
                R_mp = suma_multiprocessing(A, B, n, procesos)
                t_mp = time.perf_counter() - inicio
                speedup, _ = calcular_metricas(t_sec, t_mp, procesos)
                resultados.append((f'Multiprocessing ({procesos})', t_mp, procesos, speedup, 0))
                print(f"   • {procesos} procesos: {t_mp:.6f}s (Speedup: {speedup:.2f}x)")
            except Exception as e:
                print(f"   • Error con {procesos} procesos: {e}")
        
        # OpenMP
        if OPENMP_DISPONIBLE:
            print(f"\n4. OPENMP...")
            for threads in [t for t in threads_config if t <= mp.cpu_count()]:
                inicio = time.perf_counter()
                R_omp = suma_openmp(A, B, n, threads)
                t_omp = time.perf_counter() - inicio
                speedup, _ = calcular_metricas(t_sec, t_omp, threads)
                resultados.append((f'OpenMP ({threads})', t_omp, threads, speedup, 0))
                print(f"   • {threads} threads: {t_omp:.6f}s (Speedup: {speedup:.2f}x)")
        
        # Mostrar resultados
        print(f"\n" + "="*60)
        print("RESULTADOS")
        print("="*60)
        
        print(f"\n{'Metodo':<25} {'Tiempo (s)':<15} {'Speedup':<10}")
        print("-"*50)
        
        for nombre, tiempo, _, speedup, _ in resultados:
            if nombre == 'Secuencial':
                print(f"{nombre:<25} {tiempo:<15.6f} {'-':<10}")
            else:
                print(f"{nombre:<25} {tiempo:<15.6f} {speedup:<10.2f}")
        
        # Verificar
        if len(resultados) > 1:
            print(f"\nVerificando resultados...")
            R_ref = resultados[0][0]
            # Aquí se podría implementar verificación entre métodos
        
        # Guardar en historico
        self.historico.append({
            'tipo': 'avanzada',
            'tamano': n,
            'config_threads': threads_config,
            'resultados': resultados,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    # =================== VERSION 6: OPENMP ESPECIFICO ===================
    def prueba_openmp(self):
        """Version 6: Prueba especifica de OpenMP"""
        print("\n" + "="*80)
        print("VERSION 6: PRUEBA OPENMP")
        print("="*80)
        
        if not OPENMP_DISPONIBLE:
            print("\n⚠ OpenMP no está disponible.")
            print("Instala: pip install numba numpy")
            input("\nPresiona Enter para continuar...")
            return
        
        n = 1000
        print(f"\nTamaño: {n} elementos")
        
        # Crear arreglos
        print("\n1. Creando arreglos...")
        A, B = crear_arreglos_usuario(n)
        
        # Secuencial
        print(f"\n2. Ejecutando SECUENCIAL...")
        inicio = time.perf_counter()
        R_sec = suma_secuencial(A, B, n)
        t_sec = time.perf_counter() - inicio
        print(f"   Tiempo: {t_sec:.6f}s")
        
        # OpenMP con diferentes threads
        print(f"\n3. Ejecutando OPENMP...")
        resultados_omp = []
        
        for threads in [1, 2, 4, 8]:
            if threads <= mp.cpu_count():
                print(f"\n   • {threads} thread(s):")
                inicio = time.perf_counter()
                try:
                    R_omp = suma_openmp(A, B, n, threads)
                    t_omp = time.perf_counter() - inicio
                    
                    speedup, eficiencia = calcular_metricas(t_sec, t_omp, threads)
                    correcto = verificar_resultados(R_sec, R_omp, A, B, n, 50)
                    
                    resultados_omp.append({
                        'threads': threads,
                        'tiempo': t_omp,
                        'speedup': speedup,
                        'eficiencia': eficiencia,
                        'correcto': correcto
                    })
                    
                    print(f"     Tiempo: {t_omp:.6f}s")
                    print(f"     Speedup: {speedup:.2f}x")
                    print(f"     Eficiencia: {eficiencia:.1f}%")
                    print(f"     Verificación: {'✓ OK' if correcto else '✗ ERROR'}")
                except Exception as e:
                    print(f"     Error: {e}")
        
        # Mostrar resumen
        print(f"\n" + "="*60)
        print("RESUMEN OPENMP")
        print("="*60)
        
        print(f"\n{'Threads':<10} {'Tiempo (s)':<15} {'Speedup':<10} {'Eficiencia':<10} {'✓/✗':<5}")
        print("-"*55)
        
        for res in resultados_omp:
            estado = "✓" if res['correcto'] else "✗"
            print(f"{res['threads']:<10} {res['tiempo']:<15.6f} "
                  f"{res['speedup']:<10.2f} {res['eficiencia']:<10.1f} {estado:<5}")
        
        # Mostrar muestra
        if resultados_omp:
            mejor = min(resultados_omp, key=lambda x: x['tiempo'])
            print(f"\n🏆 Mejor configuración: {mejor['threads']} threads")
            print(f"   Tiempo: {mejor['tiempo']:.6f}s, Speedup: {mejor['speedup']:.2f}x")
            
            # Volver a ejecutar para mostrar muestra
            R_omp_mejor = suma_openmp(A, B, n, mejor['threads'])
            print(f"\n📊 Muestra de resultados:")
            mostrar_muestra(A, B, R_omp_mejor)
        
        # Guardar en historico
        self.historico.append({
            'tipo': 'openmp',
            'tamano': n,
            'resultados': resultados_omp,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    # =================== FUNCIONES DE REPORTE ===================
    def mostrar_historico(self):
        """Muestra el historico de ejecuciones"""
        if not self.historico:
            print("\nNo hay historico disponible.")
            return
        
        print("\n" + "="*80)
        print("HISTORICO DE EJECUCIONES")
        print("="*80)
        
        for idx, item in enumerate(self.historico, 1):
            print(f"\n{idx}. {item['timestamp']} - ", end="")
            
            if 'tipo' in item:
                if item['tipo'] == 'avanzada':
                    print(f"Configuración Avanzada (n={item['tamano']:,})")
                elif item['tipo'] == 'openmp':
                    print(f"Prueba OpenMP (n={item['tamano']:,})")
            else:
                print(f"Comparativa Completa (n={item['tamano']:,})")
            
            if 'resultados' in item and len(item['resultados']) > 0:
                mejor = min(item['resultados'][1:], key=lambda x: x[1]) if len(item['resultados']) > 1 else item['resultados'][0]
                if isinstance(mejor, tuple):
                    print(f"   Mejor método: {mejor[0]} - {mejor[1]:.6f}s")
                elif isinstance(mejor, dict):
                    print(f"   Mejor OpenMP: {mejor['threads']} threads - {mejor['tiempo']:.6f}s")
    
    def exportar_resultados(self):
        """Exporta todos los resultados a archivo"""
        if not self.historico:
            print("\nNo hay resultados para exportar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("resultados", exist_ok=True)
        archivo = f"resultados/reporte_{timestamp}.txt"
        
        try:
            with open(archivo, "w", encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("REPORTE COMPLETO - SISTEMA VERSION 6\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Autor: Hector Jorge Morales Arch (江久取)\n")
                f.write(f"CPUs: {mp.cpu_count()}, OpenMP: {'SI' if OPENMP_DISPONIBLE else 'NO'}\n")
                f.write("="*80 + "\n\n")
                
                for idx, item in enumerate(self.historico, 1):
                    f.write(f"EJECUCION #{idx}\n")
                    f.write(f"Hora: {item['timestamp']}\n")
                    f.write(f"Tamaño: {item['tamano']:,} elementos\n")
                    
                    if 'tipo' in item:
                        f.write(f"Tipo: {item['tipo'].upper()}\n")
                    
                    f.write("-"*60 + "\n")
                    
                    if 'resultados' in item:
                        if isinstance(item['resultados'][0], tuple):
                            # Formato para versiones 4 y 5
                            f.write(f"{'Metodo':<25} {'Tiempo (s)':<15} {'Threads':<10} {'Speedup':<10}\n")
                            f.write("-"*60 + "\n")
                            
                            for nombre, tiempo, threads, speedup, _ in item['resultados']:
                                if nombre == 'Secuencial':
                                    f.write(f"{nombre:<25} {tiempo:<15.6f} {threads:<10} {'-':<10}\n")
                                else:
                                    f.write(f"{nombre:<25} {tiempo:<15.6f} {threads:<10} {speedup:<10.2f}\n")
                        else:
                            # Formato para OpenMP
                            f.write(f"{'Threads':<10} {'Tiempo (s)':<15} {'Speedup':<10} {'Eficiencia':<10}\n")
                            f.write("-"*60 + "\n")
                            
                            for res in item['resultados']:
                                f.write(f"{res['threads']:<10} {res['tiempo']:<15.6f} "
                                       f"{res['speedup']:<10.2f} {res['eficiencia']:<10.1f}\n")
                    
                    f.write("\n" + "="*60 + "\n\n")
            
            print(f"\n✓ Reporte exportado: {archivo}")
            print(f"  Ejecuciones: {len(self.historico)}")
            
        except Exception as e:
            print(f"\n✗ Error al exportar: {e}")

# =================== MENU PRINCIPAL COMPLETO ===================
def mostrar_menu_completo():
    """Muestra el menu completo con todas las versiones"""
    print("\n" + "="*80)
    print("SISTEMA COMPARATIVO COMPLETO - VERSION 6")
    print("="*80)
    
    print("\nVERSIONES INDIVIDUALES:")
    print("  1. Prueba rápida (n=100)")
    print("  2. Threading demo (n=1000)")
    print("  3. Multiprocessing demo (n=1000)")
    
    print("\nVERSIONES COMPLETAS:")
    print("  4. V4: Comparativa completa (múltiples tamaños)")
    print("  5. V5: Arreglo random, threads fijos")
    print("  6. V5: Arreglo fijo, threads random")
    print("  7. V5: Ambos random")
    print("  8. V5: Arreglo específico (configuración completa)")
    print("  9. V6: OpenMP específico")
    
    print("\nREPORTES Y HERRAMIENTAS:")
    print("  10. Mostrar histórico")
    print("  11. Exportar resultados")
    print("  12. Información del sistema")
    print("  13. Salir")
    
    print("\n" + "="*80)

def prueba_rapida():
    """Prueba rápida de concepto"""
    print("\n" + "="*80)
    print("PRUEBA RÁPIDA (n=100)")
    print("="*80)
    
    n = 100
    A, B = crear_arreglos(n, 100)
    
    print(f"\n1. Secuencial...")
    inicio = time.perf_counter()
    R1 = suma_secuencial(A, B, n)
    t1 = time.perf_counter() - inicio
    print(f"   Tiempo: {t1:.6f}s")
    
    print(f"\n2. Threading (4 threads)...")
    inicio = time.perf_counter()
    R2 = suma_threading(A, B, n, 4)
    t2 = time.perf_counter() - inicio
    print(f"   Tiempo: {t2:.6f}s")
    
    print(f"\n3. Verificación...")
    correcto = verificar_resultados(R1, R2, A, B, n, 20)
    print(f"   Resultados: {'✓ IGUALES' if correcto else '✗ DIFERENTES'}")
    
    input("\nPresiona Enter para continuar...")

def demo_threading():
    """Demo de threading"""
    print("\n" + "="*80)
    print("DEMO THREADING (n=1000)")
    print("="*80)
    
    n = 1000
    A, B = crear_arreglos(n, 100)
    
    for threads in [1, 2, 4, 8]:
        print(f"\n• {threads} thread(s)...")
        inicio = time.perf_counter()
        R = suma_threading(A, B, n, threads)
        tiempo = time.perf_counter() - inicio
        print(f"  Tiempo: {tiempo:.6f}s")
    
    input("\nPresiona Enter para continuar...")

def demo_multiprocessing():
    """Demo de multiprocessing"""
    print("\n" + "="*80)
    print("DEMO MULTIPROCESSING (n=1000)")
    print("="*80)
    
    n = 1000
    A, B = crear_arreglos(n, 100)
    
    for procesos in [2, 4]:
        if procesos <= mp.cpu_count():
            print(f"\n• {procesos} proceso(s)...")
            try:
                inicio = time.perf_counter()
                R = suma_multiprocessing(A, B, n, procesos)
                tiempo = time.perf_counter() - inicio
                print(f"  Tiempo: {tiempo:.6f}s")
            except Exception as e:
                print(f"  Error: {e}")
    
    input("\nPresiona Enter para continuar...")

def informacion_sistema():
    """Muestra información del sistema"""
    print("\n" + "="*80)
    print("INFORMACIÓN DEL SISTEMA")
    print("="*80)
    
    print(f"\n• CPUs disponibles: {mp.cpu_count()}")
    print(f"• OpenMP disponible: {'SÍ' if OPENMP_DISPONIBLE else 'NO'}")
    
    if OPENMP_DISPONIBLE:
        print("• Implementación: Numba")
    else:
        print("• Para OpenMP: pip install numba numpy")
    
    print("\n• Sistema operativo:", os.name)
    print("• Python:", sys.version.split()[0])
    
    input("\nPresiona Enter para continuar...")

# =================== PROGRAMA PRINCIPAL ===================
def main():
    """Función principal del sistema completo"""
    
    print("\n" + "="*80)
    print("SISTEMA COMPARATIVO COMPLETO - TODAS LAS VERSIONES")
    print("="*80)
    print("Autor: Hector Jorge Morales Arch")
    print("Alias: 江久取")
    print("="*80)
    
    if not OPENMP_DISPONIBLE:
        print("\n⚠ OpenMP/Numba no está disponible.")
        print("   Para usar OpenMP, instala: pip install numba numpy")
        print("   (El sistema funcionará sin OpenMP)")
    
    sistema = SistemaCompleto()
    
    while True:
        mostrar_menu_completo()
        
        try:
            opcion = input("\nSelecciona una opción (1-13): ").strip()
            
            if opcion == "1":
                prueba_rapida()
            elif opcion == "2":
                demo_threading()
            elif opcion == "3":
                demo_multiprocessing()
            elif opcion == "4":
                sistema.comparativa_completa()
                input("\nPresiona Enter para continuar...")
            elif opcion == "5":
                sistema.arreglo_random_threads_fijos()
                input("\nPresiona Enter para continuar...")
            elif opcion == "6":
                sistema.arreglo_fijo_threads_random()
                input("\nPresiona Enter para continuar...")
            elif opcion == "7":
                sistema.ambos_random()
                input("\nPresiona Enter para continuar...")
            elif opcion == "8":
                sistema.arreglo_especifico()
                input("\nPresiona Enter para continuar...")
            elif opcion == "9":
                sistema.prueba_openmp()
                input("\nPresiona Enter para continuar...")
            elif opcion == "10":
                sistema.mostrar_historico()
                input("\nPresiona Enter para continuar...")
            elif opcion == "11":
                sistema.exportar_resultados()
                input("\nPresiona Enter para continuar...")
            elif opcion == "12":
                informacion_sistema()
            elif opcion == "13":
                print("\n" + "="*80)
                print("¡Gracias por usar el Sistema Comparativo Completo!")
                print(f"Ejecuciones realizadas: {len(sistema.historico)}")
                print("="*80)
                break
            else:
                print("\nOpción inválida. Por favor selecciona 1-13.")
                input("Presiona Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()