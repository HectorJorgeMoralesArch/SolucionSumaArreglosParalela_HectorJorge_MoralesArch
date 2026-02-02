"""
VERSION 4: Comparativa Completa de Metodos - Analisis Exhaustivo

Descripcion: Sistema completo de comparacion que evalua todos los metodos
             (secuencial, threading, multiprocessing) con diferentes configuraciones
             y tamanos de datos. Incluye analisis estadistico y recomendaciones.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Proposito: 
- Proporcionar analisis comparativo exhaustivo de todos los metodos
- Ayudar en la toma de decisiones para seleccion de estrategia optima
- Demostrar evolucion del rendimiento con diferentes tamanos de datos
- Generar recomendaciones basadas en datos empiricos
- Servir como herramienta educativa para entender paralelismo

Caracteristicas:
- Comparacion de 3 metodos principales
- Analisis con multiples tamanos de datos
- Evaluacion estadistica de resultados
- Recomendaciones automatizadas
- Interfaz de usuario interactiva
- Exportacion de resultados
- Codigo educativo y documentado
- Solo Python estandar - SIN dependencias
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os
import math
from datetime import datetime

# =================== FUNCIONES AUXILIARES ===================
def crear_arreglos(n, max_val=10000, seed=42):
    """Crea dos arreglos aleatorios del tamano especificado"""
    random.seed(seed)
    A = [random.randint(1, max_val) for _ in range(n)]
    B = [random.randint(1, max_val) for _ in range(n)]
    return A, B

def verificar_resultados(R1, R2, A, B, n, muestra=100):
    """Verifica que dos resultados sean iguales mediante muestreo"""
    if len(R1) != len(R2):
        return False
    
    # Verificar solo una muestra para eficiencia
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

# =================== FUNCION PARA MULTIPROCESSING ===================
def worker_process(args):
    """Funcion auxiliar para multiprocessing (debe estar en nivel global)"""
    A_chunk, B_chunk, start_idx = args
    resultado = []
    for i in range(len(A_chunk)):
        resultado.append(A_chunk[i] + B_chunk[i])
    return start_idx, resultado

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

# =================== SISTEMA DE COMPARACION ===================
class SistemaComparacion:
    """Sistema completo para comparar metodos de suma"""
    
    def __init__(self):
        self.resultados = []
        self.configuraciones = {
            'tamanios': [100, 1000, 5000, 10000, 50000],
            'threads_config': [1, 2, 4, 8],
            'procesos_config': [2, 4],
            'repeticiones': 3  # Para promediar resultados
        }
    
    def ejecutar_prueba(self, n, metodo, config):
        """Ejecuta una prueba individual"""
        A, B = crear_arreglos(n)
        
        if metodo == 'secuencial':
            inicio = time.perf_counter()
            R = suma_secuencial(A, B, n)
            tiempo = time.perf_counter() - inicio
            return {'metodo': 'Secuencial', 'tiempo': tiempo, 'config': 1}
        
        elif metodo == 'threading':
            threads = config
            inicio = time.perf_counter()
            R = suma_threading(A, B, n, threads)
            tiempo = time.perf_counter() - inicio
            return {'metodo': f'Threading ({threads})', 'tiempo': tiempo, 'config': threads}
        
        elif metodo == 'multiprocessing':
            procesos = config
            inicio = time.perf_counter()
            R = suma_multiprocessing(A, B, n, procesos)
            tiempo = time.perf_counter() - inicio
            return {'metodo': f'Multiprocessing ({procesos})', 'tiempo': tiempo, 'config': procesos}
        
        return None
    
    def comparativa_completa(self):
        """Ejecuta comparativa completa con todos los metodos y configuraciones"""
        print("\n" + "="*80)
        print("INICIANDO COMPARATIVA COMPLETA")
        print("="*80)
        
        for n in self.configuraciones['tamanios']:
            print(f"\n\nANALIZANDO TAMANIO: {n:,} elementos")
            print("-"*60)
            
            resultados_tam = []
            
            # Secuencial (linea base)
            print(f"\n   Ejecutando SECUENCIAL...")
            res_sec = self.ejecutar_prueba(n, 'secuencial', 1)
            resultados_tam.append(res_sec)
            print(f"      Tiempo: {res_sec['tiempo']:.6f}s")
            
            # Threading con diferentes configs
            print(f"\n   Ejecutando THREADING...")
            for threads in self.configuraciones['threads_config']:
                res_thr = self.ejecutar_prueba(n, 'threading', threads)
                if res_thr:
                    speedup, eficiencia = calcular_metricas(
                        res_sec['tiempo'], res_thr['tiempo'], threads
                    )
                    res_thr.update({'speedup': speedup, 'eficiencia': eficiencia})
                    resultados_tam.append(res_thr)
                    print(f"      {threads} threads: {res_thr['tiempo']:.6f}s "
                          f"(Speedup: {speedup:.2f}x, Efic: {eficiencia:.1f}%)")
            
            # Multiprocessing
            print(f"\n   Ejecutando MULTIPROCESSING...")
            for procesos in self.configuraciones['procesos_config']:
                if procesos <= mp.cpu_count():
                    try:
                        res_mp = self.ejecutar_prueba(n, 'multiprocessing', procesos)
                        if res_mp:
                            speedup, eficiencia = calcular_metricas(
                                res_sec['tiempo'], res_mp['tiempo'], procesos
                            )
                            res_mp.update({'speedup': speedup, 'eficiencia': eficiencia})
                            resultados_tam.append(res_mp)
                            print(f"      {procesos} procesos: {res_mp['tiempo']:.6f}s "
                                  f"(Speedup: {speedup:.2f}x, Efic: {eficiencia:.1f}%)")
                    except Exception as e:
                        print(f"      Error con {procesos} procesos: {str(e)[:50]}...")
            
            # Encontrar el mejor metodo para este tamano
            if len([r for r in resultados_tam if r['metodo'] != 'Secuencial']) > 0:
                mejor_resultado = min(
                    [r for r in resultados_tam if r['metodo'] != 'Secuencial'],
                    key=lambda x: x['tiempo']
                )
                
                print(f"\n   MEJOR METODO PARA n={n:,}:")
                print(f"      {mejor_resultado['metodo']}")
                print(f"      Tiempo: {mejor_resultado['tiempo']:.6f}s")
                print(f"      Speedup vs secuencial: {mejor_resultado.get('speedup', 0):.2f}x")
                
                self.resultados.append({
                    'tamanio': n,
                    'resultados': resultados_tam,
                    'mejor_metodo': mejor_resultado['metodo'],
                    'mejor_speedup': mejor_resultado.get('speedup', 0)
                })
            else:
                print(f"\n   No se pudo determinar el mejor metodo para n={n:,}")
    
    def generar_reporte(self):
        """Genera un reporte detallado de los resultados"""
        if not self.resultados:
            print("\nNo hay resultados para generar reporte. Ejecuta primero la opcion 4.")
            return
        
        print("\n" + "="*80)
        print("REPORTE COMPARATIVO DETALLADO")
        print("="*80)
        
        for resultado in self.resultados:
            n = resultado['tamanio']
            print(f"\nTAMANIO: {n:,} elementos")
            print("-"*60)
            
            print(f"\n{'Metodo':<25} {'Tiempo (s)':<12} {'Speedup':<10} {'Eficiencia':<12}")
            print("-"*65)
            
            for res in resultado['resultados']:
                if res['metodo'] == 'Secuencial':
                    print(f"{res['metodo']:<25} {res['tiempo']:<12.6f} {'-':<10} {'-':<12}")
                else:
                    print(f"{res['metodo']:<25} {res['tiempo']:<12.6f} "
                          f"{res.get('speedup', 0):<10.2f} {res.get('eficiencia', 0):<10.1f}%")
            
            print(f"\nRECOMENDACION: {resultado['mejor_metodo']}")
            print(f"   Speedup alcanzado: {resultado['mejor_speedup']:.2f}x")
    
    def generar_recomendaciones_generales(self):
        """Genera recomendaciones basadas en todos los resultados"""
        if not self.resultados:
            print("\nNo hay resultados para generar recomendaciones. Ejecuta primero la opcion 4.")
            return
        
        print("\n" + "="*80)
        print("RECOMENDACIONES GENERALES BASADAS EN ANALISIS")
        print("="*80)
        
        print("\nGUIA DE SELECCION DE METODO:")
        
        # Analizar patrones
        patrones = []
        for resultado in self.resultados:
            n = resultado['tamanio']
            mejor = resultado['mejor_metodo']
            speedup = resultado['mejor_speedup']
            
            if 'Secuencial' in mejor or speedup < 1.1:
                patrones.append(f"n={n:,}: Secuencial (speedup: {speedup:.2f}x)")
            elif 'Threading' in mejor:
                patrones.append(f"n={n:,}: Threading (speedup: {speedup:.2f}x)")
            elif 'Multiprocessing' in mejor:
                patrones.append(f"n={n:,}: Multiprocessing (speedup: {speedup:.2f}x)")
        
        print("\n   Patrones observados:")
        for patron in patrones:
            print(f"      • {patron}")
        
        print("\nREGLAS PRACTICAS:")
        print("""
    1. n < 1,000 -> SECUENCIAL
       • Overhead de paralelismo > ganancia
       • Simple y efectivo
    
    2. 1,000 ≤ n < 10,000 -> THREADING
       • Ganancia moderada
       • Bajo overhead de memoria
       • Bueno para I/O mixto
    
    3. n ≥ 10,000 -> MULTIPROCESSING
       • Maxima ganancia de rendimiento
       • Usa multiples nucleos eficientemente
       • Ideal para CPU-bound
        """)
        
        print("\nCONSIDERACIONES IMPORTANTES:")
        print("""
    • Threading limitado por GIL en Python
    • Multiprocessing tiene overhead de comunicacion
    • Tamano de chunk afecta el rendimiento
    • Hardware disponible (CPUs) es crucial
    • Tipo de operacion (CPU vs I/O bound)
        """)

# =================== IMPLEMENTACION DE OPCIONES 2 Y 3 ===================
def ejecutar_comparativa_basica():
    """Opcion 2: Comparativa basica con n=1,000"""
    print("\n" + "="*80)
    print("COMPARATIVA BASICA (n=1,000)")
    print("="*80)
    
    n = 1000
    print(f"\nTamano de prueba: {n:,} elementos")
    
    A, B = crear_arreglos(n, max_val=10000)
    
    print(f"\n1. Ejecutando SECUENCIAL...")
    inicio = time.perf_counter()
    R1 = suma_secuencial(A, B, n)
    t1 = time.perf_counter() - inicio
    print(f"   Completado en {t1:.6f}s")
    
    print(f"\n2. Ejecutando THREADING (4 threads)...")
    inicio = time.perf_counter()
    R2 = suma_threading(A, B, n, 4)
    t2 = time.perf_counter() - inicio
    speedup2, eficiencia2 = calcular_metricas(t1, t2, 4)
    print(f"   Completado en {t2:.6f}s")
    print(f"   Speedup: {speedup2:.2f}x, Eficiencia: {eficiencia2:.1f}%")
    
    print(f"\n3. Ejecutando MULTIPROCESSING (2 procesos)...")
    try:
        inicio = time.perf_counter()
        R3 = suma_multiprocessing(A, B, n, 2)
        t3 = time.perf_counter() - inicio
        speedup3, eficiencia3 = calcular_metricas(t1, t3, 2)
        print(f"   Completado en {t3:.6f}s")
        print(f"   Speedup: {speedup3:.2f}x, Eficiencia: {eficiencia3:.1f}%")
    except Exception as e:
        print(f"   Error: {e}")
        t3 = 0
        R3 = None
    
    # Verificacion
    print(f"\nVERIFICACION:")
    v12 = verificar_resultados(R1, R2, A, B, n)
    print(f"   • Secuencial vs Threading: {'OK' if v12 else 'ERROR'}")
    
    if R3:
        v13 = verificar_resultados(R1, R3, A, B, n)
        print(f"   • Secuencial vs Multiprocessing: {'OK' if v13 else 'ERROR'}")
    
    # Determinar el mejor metodo
    tiempos = [t1, t2, t3]
    metodos = ["Secuencial", "Threading (4)", "Multiprocessing (2)"]
    
    # Filtrar tiempos validos (mayores que 0)
    tiempos_validos = [(t, m) for t, m in zip(tiempos, metodos) if t > 0]
    
    if tiempos_validos:
        mejor_tiempo, mejor_metodo = min(tiempos_validos, key=lambda x: x[0])
        print(f"\nMEJOR METODO: {mejor_metodo} con {mejor_tiempo:.6f}s")
    
    input("\nPresiona Enter para continuar...")

def ejecutar_comparativa_avanzada():
    """Opcion 3: Comparativa avanzada con n=10,000"""
    print("\n" + "="*80)
    print("COMPARATIVA AVANZADA (n=10,000)")
    print("="*80)
    
    n = 10000
    print(f"\nTamano de prueba: {n:,} elementos")
    print("(Esto puede tomar unos segundos...)")
    
    A, B = crear_arreglos(n, max_val=10000)
    
    resultados = []
    
    print(f"\n1. Ejecutando SECUENCIAL...")
    inicio = time.perf_counter()
    R1 = suma_secuencial(A, B, n)
    t1 = time.perf_counter() - inicio
    print(f"   Completado en {t1:.6f}s")
    resultados.append(("Secuencial", t1, None, None))
    
    # Probar diferentes configuraciones de threading
    configs_threading = [1, 2, 4, 8]
    print(f"\n2. Ejecutando THREADING...")
    for threads in configs_threading:
        print(f"   • Con {threads} threads...", end=" ", flush=True)
        inicio = time.perf_counter()
        R2 = suma_threading(A, B, n, threads)
        t2 = time.perf_counter() - inicio
        speedup, eficiencia = calcular_metricas(t1, t2, threads)
        print(f"Tiempo: {t2:.6f}s, Speedup: {speedup:.2f}x")
        resultados.append((f"Threading ({threads})", t2, speedup, eficiencia))
    
    # Probar diferentes configuraciones de multiprocessing
    configs_multiprocessing = [2, 4]
    print(f"\n3. Ejecutando MULTIPROCESSING...")
    for procesos in configs_multiprocessing:
        if procesos <= mp.cpu_count():
            print(f"   • Con {procesos} procesos...", end=" ", flush=True)
            try:
                inicio = time.perf_counter()
                R3 = suma_multiprocessing(A, B, n, procesos)
                t3 = time.perf_counter() - inicio
                speedup, eficiencia = calcular_metricas(t1, t3, procesos)
                print(f"Tiempo: {t3:.6f}s, Speedup: {speedup:.2f}x")
                resultados.append((f"Multiprocessing ({procesos})", t3, speedup, eficiencia))
            except Exception as e:
                print(f"Error: {str(e)[:40]}...")
        else:
            print(f"   • Saltando {procesos} procesos (solo {mp.cpu_count()} CPUs disponibles)")
    
    # Mostrar resumen
    print(f"\n" + "-"*80)
    print("RESUMEN DE RESULTADOS:")
    print("-"*80)
    print(f"\n{'Metodo':<25} {'Tiempo (s)':<15} {'Speedup':<12} {'Eficiencia':<12}")
    print("-"*70)
    
    for metodo, tiempo, speedup, eficiencia in resultados:
        if metodo == "Secuencial":
            print(f"{metodo:<25} {tiempo:<15.6f} {'-':<12} {'-':<12}")
        else:
            print(f"{metodo:<25} {tiempo:<15.6f} {speedup:<12.2f} {eficiencia:<12.1f}%")
    
    # Encontrar el mejor metodo (excluyendo secuencial)
    if len(resultados) > 1:
        resultados_validos = [r for r in resultados if r[0] != "Secuencial" and r[1] > 0]
        if resultados_validos:
            mejor_resultado = min(resultados_validos, key=lambda x: x[1])
            print(f"\nMEJOR METODO: {mejor_resultado[0]}")
            print(f"Tiempo: {mejor_resultado[1]:.6f}s")
            if mejor_resultado[2]:
                print(f"Speedup vs secuencial: {mejor_resultado[2]:.2f}x")
    
    input("\nPresiona Enter para continuar...")

# =================== INTERFAZ DE USUARIO ===================
def mostrar_encabezado():
    """Muestra el encabezado del programa"""
    print("\n" + "="*80)
    print(" " * 25 + "PARALELISMO EN PYTHON")
    print(" " * 15 + "Sistema Comparativo de Metodos de Suma")
    print("="*80)
    print(" " * 20 + "Autor: Hector Jorge Morales Arch")
    print(" " * 25 + "Alias: 江久取")
    print("="*80)

def mostrar_menu():
    """Muestra el menu principal"""
    print("\nMENU PRINCIPAL:")
    print("   1. Prueba rapida (n=100)")
    print("   2. Comparativa basica (n=1,000)")
    print("   3. Comparativa avanzada (n=10,000)")
    print("   4. Comparativa completa (multiples tamanos)")
    print("   5. Generar reporte detallado")
    print("   6. Exportar resultados")
    print("   7. Ayuda y documentacion")
    print("   8. Salir")
    print("\n" + "="*80)

def ejecutar_prueba_rapida():
    """Opcion 1: Prueba rapida"""
    print("\n" + "="*80)
    print("PRUEBA RAPIDA DEMOSTRATIVA")
    print("="*80)
    
    n = 100
    print(f"\nTamano de prueba: {n} elementos")
    
    A, B = crear_arreglos(n, max_val=100)
    
    # Ejecutar metodos
    print(f"\n1. Ejecutando SECUENCIAL...")
    inicio = time.perf_counter()
    R1 = suma_secuencial(A, B, n)
    t1 = time.perf_counter() - inicio
    print(f"   Completado en {t1:.6f}s")
    
    print(f"\n2. Ejecutando THREADING (4 threads)...")
    inicio = time.perf_counter()
    R2 = suma_threading(A, B, n, 4)
    t2 = time.perf_counter() - inicio
    print(f"   Completado en {t2:.6f}s")
    
    print(f"\n3. Ejecutando MULTIPROCESSING (2 procesos)...")
    try:
        inicio = time.perf_counter()
        R3 = suma_multiprocessing(A, B, n, 2)
        t3 = time.perf_counter() - inicio
        print(f"   Completado en {t3:.6f}s")
    except Exception as e:
        print(f"   Error: {e}")
        R3 = None
    
    # Verificacion
    v12 = verificar_resultados(R1, R2, A, B, n)
    
    print(f"\nVERIFICACION:")
    print(f"   • Secuencial vs Threading: {'OK' if v12 else 'ERROR'}")
    
    if R3:
        v13 = verificar_resultados(R1, R3, A, B, n)
        print(f"   • Secuencial vs Multiprocessing: {'OK' if v13 else 'ERROR'}")
    
    input("\nPresiona Enter para continuar...")

def mostrar_ayuda():
    """Muestra la ayuda del programa"""
    print("\n" + "="*80)
    print("AYUDA Y DOCUMENTACION")
    print("="*80)
    
    print("""
DESCRIPCION:
Este programa compara diferentes metodos para sumar arreglos en Python,
desde el enfoque secuencial tradicional hasta metodos paralelos avanzados.

PROPOSITO:
- Demostrar la evolucion de tecnicas de procesamiento
- Ayudar a seleccionar la estrategia optima segun el caso
- Educar sobre paralelismo en Python
- Proporcionar metricas reales de rendimiento

METODOS IMPLEMENTADOS:
1. SECUENCIAL: Metodo tradicional, linea base de rendimiento
2. THREADING: Paralelismo con hilos (limitado por GIL)
3. MULTIPROCESSING: Paralelismo real con procesos separados

METRICAS CALCULADAS:
• Tiempo de ejecucion
• Speedup (aceleracion vs secuencial)
• Eficiencia (% del ideal)

RECOMENDACIONES DE USO:
• Empiece con pruebas pequenas (opcion 1)
• Use comparativa basica para entender conceptos (opcion 2)
• Use comparativa completa para analisis detallado (opcion 4)
• Consulte las recomendaciones generadas
• Experimente con diferentes tamanos de datos

NOTAS IMPORTANTES:
• Todos los metodos usan solo Python estandar
• Los resultados pueden variar segun hardware
• El overhead afecta mas en tamanos pequenos
• Multiprocessing es mejor para CPU-intensive
    """)
    
    input("\nPresiona Enter para continuar...")

# =================== PROGRAMA PRINCIPAL ===================
def main():
    """Funcion principal del programa"""
    
    mostrar_encabezado()
    
    print("\n" + "="*80)
    print("¡Bienvenido al Sistema Comparativo de Metodos de Suma!")
    print("\nEste programa te permitira comparar diferentes enfoques para")
    print("sumar arreglos en Python, desde el metodo secuencial basico")
    print("hasta tecnicas paralelas avanzadas.")
    print("="*80)
    
    sistema = SistemaComparacion()
    
    while True:
        mostrar_menu()
        
        try:
            opcion = input("\nSelecciona una opcion (1-8): ").strip()
            
            if opcion == "1":
                ejecutar_prueba_rapida()
            elif opcion == "2":
                ejecutar_comparativa_basica()
            elif opcion == "3":
                ejecutar_comparativa_avanzada()
            elif opcion == "4":
                print("\nEJECUTANDO COMPARATIVA COMPLETA...")
                print("   Esto puede tomar varios segundos...")
                sistema.comparativa_completa()
                sistema.generar_reporte()
                sistema.generar_recomendaciones_generales()
                input("\nPresiona Enter para continuar...")
            elif opcion == "5":
                if sistema.resultados:
                    sistema.generar_reporte()
                else:
                    print("\nPrimero ejecuta la comparativa completa (opcion 4)")
                input("Presiona Enter para continuar...")
            elif opcion == "6":
                if sistema.resultados:
                    print("\nExportando resultados...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("resultados", exist_ok=True)
                    
                    with open(f"resultados/reporte_{timestamp}.txt", "w") as f:
                        f.write("="*80 + "\n")
                        f.write("REPORTE DE COMPARACION - Sistema de Suma de Arreglos\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Autor: Hector Jorge Morales Arch (江久取)\n")
                        f.write("="*80 + "\n\n")
                        
                        for resultado in sistema.resultados:
                            f.write(f"TAMANIO: {resultado['tamanio']:,} elementos\n")
                            f.write("-"*60 + "\n")
                            
                            for res in resultado['resultados']:
                                if res['metodo'] == 'Secuencial':
                                    f.write(f"{res['metodo']:<25} {res['tiempo']:.6f}s\n")
                                else:
                                    f.write(f"{res['metodo']:<25} {res['tiempo']:.6f}s "
                                           f"(Speedup: {res.get('speedup', 0):.2f}x, "
                                           f"Efic: {res.get('eficiencia', 0):.1f}%)\n")
                            
                            f.write(f"\nMEJOR METODO: {resultado['mejor_metodo']}\n")
                            f.write(f"MEJOR SPEEDUP: {resultado['mejor_speedup']:.2f}x\n\n")
                    
                    print(f"   Reporte guardado en: resultados/reporte_{timestamp}.txt")
                else:
                    print("\nNo hay resultados para exportar. Ejecuta primero la opcion 4.")
                input("Presiona Enter para continuar...")
            elif opcion == "7":
                mostrar_ayuda()
            elif opcion == "8":
                print("\n" + "="*80)
                print("¡Gracias por usar el Sistema Comparativo!")
                print("="*80)
                break
            else:
                print("\nOpcion invalida. Por favor selecciona 1-8.")
                input("Presiona Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()