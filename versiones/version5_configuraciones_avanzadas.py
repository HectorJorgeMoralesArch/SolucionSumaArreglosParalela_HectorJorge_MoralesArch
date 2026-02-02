"""
VERSION 5: Sistema Comparativo Mejorado con Opciones Aleatorias y Especificas

Descripcion: Sistema completo de comparacion que evalua todos los metodos
             con configuraciones aleatorias, especificas y personalizadas.

Autor: Hector Jorge Morales Arch
Alias: 江久取

Caracteristicas:
- Comparacion de 3 metodos principales
- Opciones aleatorias y especificas
- Configuracion personalizada por usuario
- Analisis estadistico completo
- Recomendaciones automatizadas
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import os
import math
from datetime import datetime
import sys

# =================== FUNCIONES AUXILIARES ===================
def crear_arreglos(n, max_val=10000, seed=None):
    """Crea dos arreglos aleatorios del tamano especificado"""
    if seed is not None:
        random.seed(seed)
    A = [random.randint(1, max_val) for _ in range(n)]
    B = [random.randint(1, max_val) for _ in range(n)]
    return A, B

def crear_arreglo_personalizado(n, valores_personalizados=None):
    """Crea arreglos con valores personalizados o aleatorios"""
    if valores_personalizados:
        # Si se proporcionan valores personalizados
        if len(valores_personalizados) >= n*2:
            A = valores_personalizados[:n]
            B = valores_personalizados[n:n*2]
        else:
            # Rellenar con valores aleatorios si no hay suficientes
            A = valores_personalizados + [random.randint(1, 100) for _ in range(n - len(valores_personalizados))]
            B = [random.randint(1, 100) for _ in range(n)]
    else:
        # Valores aleatorios
        A = [random.randint(1, 100) for _ in range(n)]
        B = [random.randint(1, 100) for _ in range(n)]
    return A, B

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

def obtener_configuracion_threads():
    """Obtiene configuracion de threads basada en el sistema"""
    cpus = mp.cpu_count()
    configs = [1, 2, 4]
    if cpus >= 6:
        configs.append(6)
    if cpus >= 8:
        configs.append(8)
    configs.append(cpus)  # Maximo del sistema
    return configs

# =================== FUNCION PARA MULTIPROCESSING ===================
def worker_process(args):
    """Funcion auxiliar para multiprocessing"""
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

# =================== NUEVAS OPCIONES DE CONFIGURACION ===================
def generar_configuracion_aleatoria():
    """Genera una configuracion aleatoria para pruebas"""
    configs = {
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
    return configs

def obtener_input_usuario():
    """Obtiene configuracion especifica del usuario"""
    print("\n" + "="*80)
    print("CONFIGURACION ESPECIFICA DEL USUARIO")
    print("="*80)
    
    # Tamano del arreglo
    while True:
        try:
            n = int(input("\nTamano del arreglo (ej: 1000): ").strip())
            if n > 0:
                break
            else:
                print("Por favor ingresa un numero positivo.")
        except ValueError:
            print("Por favor ingresa un numero valido.")
    
    # Valores personalizados
    valores_personalizados = None
    opcion_valores = input("\n¿Deseas ingresar valores personalizados? (s/n): ").strip().lower()
    if opcion_valores == 's':
        print("\nIngresa los valores separados por comas (ej: 1,2,3,4,5,6):")
        entrada = input("Valores: ").strip()
        if entrada:
            try:
                valores = [int(x.strip()) for x in entrada.split(',')]
                if len(valores) >= 2:
                    valores_personalizados = valores
                    print(f"Se usaran {len(valores)} valores personalizados")
                else:
                    print("Se necesitan al menos 2 valores. Se usaran valores aleatorios.")
            except ValueError:
                print("Valores no validos. Se usaran valores aleatorios.")
    
    # Configuracion de threads
    print(f"\nCPUs disponibles en el sistema: {mp.cpu_count()}")
    print("\nSelecciona configuracion de threads:")
    print("  1. Todos los disponibles [1,2,4,6,8,sistema]")
    print("  2. Solo pares [2,4,6,8]")
    print("  3. Solo optimos [2,4,sistema]")
    print("  4. Personalizado")
    
    while True:
        opcion_threads = input("\nOpcion (1-4): ").strip()
        if opcion_threads in ['1', '2', '3', '4']:
            break
        print("Opcion invalida. Selecciona 1-4.")
    
    if opcion_threads == '1':
        threads_config = [1, 2, 4, 6, 8, mp.cpu_count()]
    elif opcion_threads == '2':
        threads_config = [2, 4, 6, 8]
    elif opcion_threads == '3':
        threads_config = [2, 4, mp.cpu_count()]
    else:  # opcion 4
        print("\nIngresa los numeros de threads separados por comas (ej: 2,4,8):")
        entrada = input("Threads: ").strip()
        threads_config = []
        if entrada:
            try:
                threads_config = [int(x.strip()) for x in entrada.split(',')]
                threads_config = list(set(threads_config))  # Eliminar duplicados
                threads_config.sort()
            except ValueError:
                print("Valores no validos. Se usara configuracion por defecto.")
                threads_config = [1, 2, 4, mp.cpu_count()]
    
    # Metodos a comparar
    print("\nSelecciona metodos a comparar:")
    print("  1. Todos (Secuencial, Threading, Multiprocessing)")
    print("  2. Solo Secuencial vs Threading")
    print("  3. Solo Secuencial vs Multiprocessing")
    print("  4. Solo Threading vs Multiprocessing")
    
    while True:
        opcion_metodos = input("\nOpcion (1-4): ").strip()
        if opcion_metodos in ['1', '2', '3', '4']:
            break
        print("Opcion invalida. Selecciona 1-4.")
    
    if opcion_metodos == '1':
        metodos = ['secuencial', 'threading', 'multiprocessing']
    elif opcion_metodos == '2':
        metodos = ['secuencial', 'threading']
    elif opcion_metodos == '3':
        metodos = ['secuencial', 'multiprocessing']
    else:  # opcion 4
        metodos = ['threading', 'multiprocessing']
    
    return {
        'tamano': n,
        'valores_personalizados': valores_personalizados,
        'threads_config': threads_config,
        'metodos': metodos,
        'repeticiones': 2  # Numero de repeticiones para promediar
    }

# =================== SISTEMA DE COMPARACION MEJORADO ===================
class SistemaComparacionMejorado:
    """Sistema mejorado para comparar metodos de suma"""
    
    def __init__(self):
        self.resultados = []
        self.config_actual = {}
        self.resultados_acumulados = []  # Para acumular resultados de todas las ejecuciones
    
    def ejecutar_prueba_completa(self, config, guardar_resultados=True):
        """Ejecuta una prueba completa con configuracion especifica"""
        print(f"\n" + "="*80)
        print(f"EJECUTANDO PRUEBA COMPLETA")
        print("="*80)
        
        n = config['tamano']
        
        # Crear arreglos
        if config.get('valores_personalizados'):
            print(f"\nCreando arreglos con valores personalizados...")
            A, B = crear_arreglo_personalizado(n, config['valores_personalizados'])
        else:
            print(f"\nCreando arreglos aleatorios (n={n:,})...")
            A, B = crear_arreglos(n)
        
        resultados_prueba = []
        
        # Ejecutar metodos segun configuracion
        if 'secuencial' in config['metodos']:
            print(f"\n1. EJECUTANDO METODO SECUENCIAL...")
            tiempos = []
            R_sec = None
            for i in range(config.get('repeticiones', 1)):
                inicio = time.perf_counter()
                R_sec = suma_secuencial(A, B, n)
                tiempo = time.perf_counter() - inicio
                tiempos.append(tiempo)
            
            tiempo_promedio = sum(tiempos) / len(tiempos)
            resultados_prueba.append({
                'metodo': 'Secuencial',
                'tiempo': tiempo_promedio,
                'config': 1,
                'resultado': R_sec
            })
            print(f"   Tiempo promedio: {tiempo_promedio:.6f}s")
        
        # Threading con diferentes configuraciones
        if 'threading' in config['metodos']:
            print(f"\n2. EJECUTANDO METODO THREADING...")
            for threads in config['threads_config']:
                tiempos = []
                R_thr = None
                for i in range(config.get('repeticiones', 1)):
                    inicio = time.perf_counter()
                    R_thr = suma_threading(A, B, n, threads)
                    tiempo = time.perf_counter() - inicio
                    tiempos.append(tiempo)
                
                tiempo_promedio = sum(tiempos) / len(tiempos)
                if 'secuencial' in config['metodos'] and resultados_prueba:
                    speedup, eficiencia = calcular_metricas(
                        resultados_prueba[0]['tiempo'], tiempo_promedio, threads
                    )
                else:
                    speedup, eficiencia = 0, 0
                
                resultados_prueba.append({
                    'metodo': f'Threading ({threads})',
                    'tiempo': tiempo_promedio,
                    'config': threads,
                    'speedup': speedup,
                    'eficiencia': eficiencia,
                    'resultado': R_thr
                })
                print(f"   • {threads} threads: {tiempo_promedio:.6f}s "
                      f"(Speedup: {speedup:.2f}x, Efic: {eficiencia:.1f}%)")
        
        # Multiprocessing
        if 'multiprocessing' in config['metodos']:
            print(f"\n3. EJECUTANDO METODO MULTIPROCESSING...")
            cpus_disponibles = mp.cpu_count()
            procesos_config = [p for p in config['threads_config'] if p <= cpus_disponibles]
            
            for procesos in procesos_config:
                try:
                    tiempos = []
                    R_mp = None
                    for i in range(config.get('repeticiones', 1)):
                        inicio = time.perf_counter()
                        R_mp = suma_multiprocessing(A, B, n, procesos)
                        tiempo = time.perf_counter() - inicio
                        tiempos.append(tiempo)
                    
                    tiempo_promedio = sum(tiempos) / len(tiempos)
                    if 'secuencial' in config['metodos'] and resultados_prueba:
                        speedup, eficiencia = calcular_metricas(
                            resultados_prueba[0]['tiempo'], tiempo_promedio, procesos
                        )
                    else:
                        speedup, eficiencia = 0, 0
                    
                    resultados_prueba.append({
                        'metodo': f'Multiprocessing ({procesos})',
                        'tiempo': tiempo_promedio,
                        'config': procesos,
                        'speedup': speedup,
                        'eficiencia': eficiencia,
                        'resultado': R_mp
                    })
                    print(f"   • {procesos} procesos: {tiempo_promedio:.6f}s "
                          f"(Speedup: {speedup:.2f}x, Efic: {eficiencia:.1f}%)")
                except Exception as e:
                    print(f"   • Error con {procesos} procesos: {str(e)[:50]}")
        
        # Verificar resultados si hay multiples metodos
        if len([r for r in resultados_prueba if 'resultado' in r]) >= 2:
            print(f"\n4. VERIFICANDO RESULTADOS...")
            resultados_con_data = [r for r in resultados_prueba if 'resultado' in r]
            
            for i in range(len(resultados_con_data)-1):
                for j in range(i+1, len(resultados_con_data)):
                    r1 = resultados_con_data[i]
                    r2 = resultados_con_data[j]
                    verificacion = verificar_resultados(r1['resultado'], r2['resultado'], A, B, n, 50)
                    print(f"   • {r1['metodo']} vs {r2['metodo']}: {'OK' if verificacion else 'ERROR'}")
        
        # Encontrar el mejor metodo (excluyendo secuencial si existe)
        resultados_no_secuencial = [r for r in resultados_prueba if r.get('metodo') != 'Secuencial' and r.get('tiempo', 0) > 0]
        if resultados_no_secuencial:
            mejor_resultado = min(resultados_no_secuencial, key=lambda x: x['tiempo'])
            print(f"\n" + "-"*80)
            print(f"MEJOR METODO: {mejor_resultado['metodo']}")
            print(f"TIEMPO: {mejor_resultado['tiempo']:.6f}s")
            if 'speedup' in mejor_resultado and mejor_resultado['speedup'] > 0:
                print(f"SPEEDUP: {mejor_resultado['speedup']:.2f}x")
                print(f"EFICIENCIA: {mejor_resultado.get('eficiencia', 0):.1f}%")
        else:
            print(f"\nNo se pudo determinar el mejor metodo.")
            mejor_resultado = None
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            resultado_completo = {
                'configuracion': config,
                'tamano': n,
                'resultados': resultados_prueba,
                'mejor_metodo': mejor_resultado['metodo'] if mejor_resultado else None,
                'mejor_tiempo': mejor_resultado['tiempo'] if mejor_resultado else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.resultados.append(resultado_completo)
            self.resultados_acumulados.append(resultado_completo)
            self.config_actual = config
        
        return resultados_prueba
    
    def generar_reporte_detallado(self):
        """Genera un reporte detallado de todos los resultados acumulados"""
        if not self.resultados_acumulados:
            print("\nNo hay resultados para generar reporte. Ejecuta primero alguna prueba.")
            return
        
        print("\n" + "="*80)
        print("REPORTE DETALLADO - TODAS LAS PRUEBAS")
        print("="*80)
        
        total_pruebas = len(self.resultados_acumulados)
        print(f"\nTotal de pruebas ejecutadas: {total_pruebas}")
        
        for idx, resultado in enumerate(self.resultados_acumulados, 1):
            print(f"\n" + "="*60)
            print(f"PRUEBA #{idx} - {resultado['timestamp']}")
            print("="*60)
            
            n = resultado['tamano']
            print(f"\nConfiguracion:")
            print(f"  • Tamano del arreglo: {n:,} elementos")
            print(f"  • Metodos probados: {len(resultado['resultados'])}")
            
            if resultado.get('mejor_metodo'):
                print(f"\nResultados:")
                print(f"{'Metodo':<25} {'Tiempo (s)':<15} {'Speedup':<10} {'Eficiencia':<10}")
                print("-"*60)
                
                for res in resultado['resultados']:
                    if res['metodo'] == 'Secuencial':
                        print(f"{res['metodo']:<25} {res['tiempo']:<15.6f} {'-':<10} {'-':<10}")
                    else:
                        print(f"{res['metodo']:<25} {res['tiempo']:<15.6f} "
                              f"{res.get('speedup', 0):<10.2f} {res.get('eficiencia', 0):<10.1f}%")
                
                print(f"\nMEJOR METODO: {resultado['mejor_metodo']}")
                print(f"TIEMPO: {resultado['mejor_tiempo']:.6f}s")
            else:
                print("\nNo se pudo determinar el mejor metodo para esta prueba.")
        
        # Resumen estadistico
        print(f"\n" + "="*80)
        print("RESUMEN ESTADISTICO")
        print("="*80)
        
        mejores_tiempos = []
        metodos_ganadores = {}
        
        for resultado in self.resultados_acumulados:
            if resultado.get('mejor_metodo'):
                mejores_tiempos.append(resultado['mejor_tiempo'])
                metodo = resultado['mejor_metodo']
                metodos_ganadores[metodo] = metodos_ganadores.get(metodo, 0) + 1
        
        if mejores_tiempos:
            print(f"\nEstadisticas de tiempo del mejor metodo:")
            print(f"  • Mejor tiempo: {min(mejores_tiempos):.6f}s")
            print(f"  • Peor tiempo: {max(mejores_tiempos):.6f}s")
            print(f"  • Tiempo promedio: {sum(mejores_tiempos)/len(mejores_tiempos):.6f}s")
        
        if metodos_ganadores:
            print(f"\nMetodos ganadores por frecuencia:")
            for metodo, frecuencia in sorted(metodos_ganadores.items(), key=lambda x: x[1], reverse=True):
                porcentaje = (frecuencia / total_pruebas) * 100
                print(f"  • {metodo}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    def exportar_resultados(self):
        """Exporta todos los resultados a un archivo"""
        if not self.resultados_acumulados:
            print("\nNo hay resultados para exportar. Ejecuta primero alguna prueba.")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("resultados", exist_ok=True)
        nombre_archivo = f"resultados/reporte_completo_{timestamp}.txt"
        
        try:
            with open(nombre_archivo, "w", encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("REPORTE COMPLETO - SISTEMA COMPARATIVO AVANZADO\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Autor: Hector Jorge Morales Arch (江久取)\n")
                f.write(f"Sistema: {os.name}, CPUs: {mp.cpu_count()}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"TOTAL DE PRUEBAS: {len(self.resultados_acumulados)}\n\n")
                
                for idx, resultado in enumerate(self.resultados_acumulados, 1):
                    f.write(f"PRUEBA #{idx} - {resultado['timestamp']}\n")
                    f.write("-"*60 + "\n")
                    f.write(f"Tamano del arreglo: {resultado['tamano']:,} elementos\n")
                    
                    if resultado.get('configuracion', {}).get('valores_personalizados'):
                        f.write("Valores: Personalizados\n")
                    else:
                        f.write("Valores: Aleatorios\n")
                    
                    f.write("\nResultados detallados:\n")
                    f.write(f"{'Metodo':<25} {'Tiempo (s)':<15} {'Speedup':<10} {'Eficiencia':<10}\n")
                    f.write("-"*60 + "\n")
                    
                    for res in resultado['resultados']:
                        if res['metodo'] == 'Secuencial':
                            f.write(f"{res['metodo']:<25} {res['tiempo']:<15.6f} {'-':<10} {'-':<10}\n")
                        else:
                            f.write(f"{res['metodo']:<25} {res['tiempo']:<15.6f} "
                                   f"{res.get('speedup', 0):<10.2f} {res.get('eficiencia', 0):<10.1f}%\n")
                    
                    if resultado.get('mejor_metodo'):
                        f.write(f"\nMEJOR METODO: {resultado['mejor_metodo']}\n")
                        f.write(f"TIEMPO: {resultado['mejor_tiempo']:.6f}s\n")
                    
                    f.write("\n" + "="*60 + "\n\n")
                
                # Resumen estadistico
                f.write("\n" + "="*80 + "\n")
                f.write("RESUMEN ESTADISTICO\n")
                f.write("="*80 + "\n\n")
                
                mejores_tiempos = []
                metodos_ganadores = {}
                
                for resultado in self.resultados_acumulados:
                    if resultado.get('mejor_metodo'):
                        mejores_tiempos.append(resultado['mejor_tiempo'])
                        metodo = resultado['mejor_metodo']
                        metodos_ganadores[metodo] = metodos_ganadores.get(metodo, 0) + 1
                
                if mejores_tiempos:
                    f.write("Estadisticas de tiempo del mejor metodo:\n")
                    f.write(f"  • Mejor tiempo: {min(mejores_tiempos):.6f}s\n")
                    f.write(f"  • Peor tiempo: {max(mejores_tiempos):.6f}s\n")
                    f.write(f"  • Tiempo promedio: {sum(mejores_tiempos)/len(mejores_tiempos):.6f}s\n")
                    f.write(f"  • Rango: {max(mejores_tiempos) - min(mejores_tiempos):.6f}s\n\n")
                
                if metodos_ganadores:
                    f.write("Metodos ganadores por frecuencia:\n")
                    for metodo, frecuencia in sorted(metodos_ganadores.items(), key=lambda x: x[1], reverse=True):
                        porcentaje = (frecuencia / len(self.resultados_acumulados)) * 100
                        f.write(f"  • {metodo}: {frecuencia} veces ({porcentaje:.1f}%)\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("FIN DEL REPORTE\n")
                f.write("="*80 + "\n")
            
            print(f"\nReporte exportado exitosamente:")
            print(f"  • Archivo: {nombre_archivo}")
            print(f"  • Pruebas incluidas: {len(self.resultados_acumulados)}")
            print(f"  • Tamanio del archivo: {os.path.getsize(nombre_archivo)} bytes")
            
            return True
            
        except Exception as e:
            print(f"\nError al exportar resultados: {e}")
            return False

# =================== FUNCIONES GLOBALES ===================
# Variable global para mantener el sistema principal
sistema_principal = None

def inicializar_sistema():
    """Inicializa el sistema principal global"""
    global sistema_principal
    if sistema_principal is None:
        sistema_principal = SistemaComparacionMejorado()
    return sistema_principal

# =================== NUEVAS FUNCIONES DE MENU ===================
def ejecutar_arreglo_random():
    """Opcion: Arreglo random, threads fijos"""
    print("\n" + "="*80)
    print("ARREGLO RANDOM - THREADS FIJOS")
    print("="*80)
    
    configs = generar_configuracion_aleatoria()
    config_arreglo = configs['arreglo_random']
    
    # Seleccionar tamano aleatorio
    n = random.choice(config_arreglo['tamanos'])
    
    config = {
        'tamano': n,
        'valores_personalizados': None,
        'threads_config': [1, 2, 4, 8, mp.cpu_count()],
        'metodos': ['secuencial', 'threading', 'multiprocessing'],
        'repeticiones': 2
    }
    
    print(f"\nConfiguracion generada aleatoriamente:")
    print(f"  • Tamano del arreglo: {n:,} elementos")
    print(f"  • Maximo valor: {config_arreglo['max_val']}")
    print(f"  • Seed aleatoria: {config_arreglo['seed']}")
    print(f"  • Threads a probar: {config['threads_config']}")
    
    sistema = inicializar_sistema()
    sistema.ejecutar_prueba_completa(config)
    
    input("\nPresiona Enter para continuar...")

def ejecutar_threads_random():
    """Opcion: Arreglo fijo, threads random"""
    print("\n" + "="*80)
    print("ARREGLO FIJO - THREADS RANDOM")
    print("="*80)
    
    configs = generar_configuracion_aleatoria()
    config_threads = configs['threads_random']
    
    # Tamano fijo
    n = 5000
    
    config = {
        'tamano': n,
        'valores_personalizados': None,
        'threads_config': config_threads['configs'],
        'metodos': ['secuencial', 'threading', 'multiprocessing'],
        'repeticiones': 2
    }
    
    print(f"\nConfiguracion generada aleatoriamente:")
    print(f"  • Tamano del arreglo: {n:,} elementos (fijo)")
    print(f"  • Configuraciones de threads: {config_threads['configs']}")
    print(f"  • CPUs disponibles: {mp.cpu_count()}")
    
    sistema = inicializar_sistema()
    sistema.ejecutar_prueba_completa(config)
    
    input("\nPresiona Enter para continuar...")

def ejecutar_ambos_random():
    """Opcion: Ambos random"""
    print("\n" + "="*80)
    print("AMBOS RANDOM - ARREGLO Y THREADS ALEATORIOS")
    print("="*80)
    
    configs = generar_configuracion_aleatoria()
    config_ambos = configs['ambos_random']
    
    # Seleccionar tamano aleatorio
    n = random.choice(config_ambos['tamanos'])
    
    config = {
        'tamano': n,
        'valores_personalizados': None,
        'threads_config': config_ambos['threads'],
        'metodos': ['secuencial', 'threading', 'multiprocessing'],
        'repeticiones': 2
    }
    
    print(f"\nConfiguracion generada aleatoriamente:")
    print(f"  • Tamano del arreglo: {n:,} elementos")
    print(f"  • Configuraciones de threads: {config_ambos['threads']}")
    print(f"  • Total de combinaciones: {len(config_ambos['threads'])}")
    
    sistema = inicializar_sistema()
    sistema.ejecutar_prueba_completa(config)
    
    input("\nPresiona Enter para continuar...")

def ejecutar_arreglo_especifico():
    """Opcion: Arreglo especifico dado por el usuario"""
    print("\n" + "="*80)
    print("CONFIGURACION ESPECIFICA DEL USUARIO")
    print("="*80)
    
    config = obtener_input_usuario()
    
    print(f"\n" + "="*80)
    print("RESUMEN DE CONFIGURACION:")
    print("="*80)
    print(f"  • Tamano del arreglo: {config['tamano']:,} elementos")
    print(f"  • Valores personalizados: {'SI' if config['valores_personalizados'] else 'NO'}")
    print(f"  • Configuracion de threads: {config['threads_config']}")
    print(f"  • Metodos a comparar: {config['metodos']}")
    print(f"  • Repeticiones: {config['repeticiones']}")
    
    confirmacion = input("\n¿Ejecutar con esta configuracion? (s/n): ").strip().lower()
    
    if confirmacion == 's':
        sistema = inicializar_sistema()
        sistema.ejecutar_prueba_completa(config)
    else:
        print("\nEjecucion cancelada.")
    
    input("\nPresiona Enter para continuar...")

def mostrar_menu_completo():
    """Muestra el menu completo con todas las opciones"""
    print("\n" + "="*80)
    print("MENU COMPLETO - SISTEMA COMPARATIVO AVANZADO")
    print("="*80)
    print("\nOPCIONES PRINCIPALES:")
    print("  1. Prueba rapida (n=100)")
    print("  2. Comparativa basica (n=1,000)")
    print("  3. Comparativa avanzada (n=10,000)")
    print("  4. Comparativa completa (multiples tamanos)")
    
    print("\nOPCIONES AVANZADAS:")
    print("  5. ARREGLO RANDOM, THREADS FIJOS")
    print("  6. ARREGLO FIJO, THREADS RANDOM")
    print("  7. AMBOS RANDOM")
    print("  8. ARREGLO ESPECIFICO (configuracion completa)")
    
    print("\nOPCIONES DE REPORTE:")
    print("  9. Generar reporte detallado")
    print(" 10. Exportar resultados")
    print(" 11. Ayuda y documentacion")
    print(" 12. Salir")
    print("\n" + "="*80)

# =================== FUNCIONES EXISTENTES (actualizadas) ===================
def mostrar_encabezado():
    """Muestra el encabezado del programa"""
    print("\n" + "="*80)
    print(" " * 25 + "PARALELISMO EN PYTHON - SISTEMA AVANZADO")
    print(" " * 10 + "Sistema Comparativo de Metodos de Suma con Configuraciones Aleatorias")
    print("="*80)
    print(" " * 20 + "Autor: Hector Jorge Morales Arch")
    print(" " * 25 + "Alias: 江久取")
    print("="*80)

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

def ejecutar_comparativa_basica():
    """Opcion 2: Comparativa basica con n=1,000"""
    print("\n" + "="*80)
    print("COMPARATIVA BASICA (n=1,000)")
    print("="*80)
    
    n = 1000
    config = {
        'tamano': n,
        'valores_personalizados': None,
        'threads_config': [1, 2, 4, 8],
        'metodos': ['secuencial', 'threading', 'multiprocessing'],
        'repeticiones': 2
    }
    
    sistema = inicializar_sistema()
    sistema.ejecutar_prueba_completa(config)
    
    input("\nPresiona Enter para continuar...")

def ejecutar_comparativa_avanzada():
    """Opcion 3: Comparativa avanzada con n=10,000"""
    print("\n" + "="*80)
    print("COMPARATIVA AVANZADA (n=10,000)")
    print("="*80)
    
    n = 10000
    config = {
        'tamano': n,
        'valores_personalizados': None,
        'threads_config': [1, 2, 4, 8, mp.cpu_count()],
        'metodos': ['secuencial', 'threading', 'multiprocessing'],
        'repeticiones': 2
    }
    
    sistema = inicializar_sistema()
    sistema.ejecutar_prueba_completa(config)
    
    input("\nPresiona Enter para continuar...")

def ejecutar_comparativa_completa():
    """Opcion 4: Comparativa completa con multiples tamanos"""
    print("\n" + "="*80)
    print("COMPARATIVA COMPLETA - MULTIPLES TAMANOS")
    print("="*80)
    
    tamanos = [100, 1000, 5000, 10000, 50000]
    sistema = inicializar_sistema()
    
    for n in tamanos:
        print(f"\n" + "="*60)
        print(f"PROCESANDO TAMANO: {n:,} elementos")
        print("="*60)
        
        config = {
            'tamano': n,
            'valores_personalizados': None,
            'threads_config': [1, 2, 4, 8, mp.cpu_count()],
            'metodos': ['secuencial', 'threading', 'multiprocessing'],
            'repeticiones': 2
        }
        
        sistema.ejecutar_prueba_completa(config, guardar_resultados=True)
    
    # Mostrar resumen final
    print(f"\n" + "="*80)
    print("RESUMEN FINAL - TODOS LOS TAMANOS")
    print("="*80)
    
    if sistema.resultados_acumulados:
        for resultado in sistema.resultados_acumulados[-len(tamanos):]:  # Mostrar solo los ultimos
            n = resultado['tamano']
            mejor = resultado['mejor_metodo']
            tiempo = resultado['mejor_tiempo']
            
            if mejor and tiempo:
                print(f"  • n={n:,}: {mejor} - {tiempo:.6f}s")
    else:
        print("No se generaron resultados.")
    
    input("\nPresiona Enter para continuar...")

def generar_reporte_detallado():
    """Opcion 9: Generar reporte detallado"""
    sistema = inicializar_sistema()
    sistema.generar_reporte_detallado()
    input("\nPresiona Enter para continuar...")

def exportar_resultados():
    """Opcion 10: Exportar resultados"""
    sistema = inicializar_sistema()
    sistema.exportar_resultados()
    input("\nPresiona Enter para continuar...")

def mostrar_ayuda():
    """Opcion 11: Mostrar ayuda"""
    print("\n" + "="*80)
    print("AYUDA DEL SISTEMA AVANZADO")
    print("="*80)
    print("""
OPCIONES DISPONIBLES:

1-4: Pruebas basicas y comparativas estandar
5-8: Configuraciones avanzadas con aleatoriedad

5. ARREGLO RANDOM, THREADS FIJOS:
   • Tamano del arreglo aleatorio
   • Threads predefinidos [1,2,4,8,sistema]

6. ARREGLO FIJO, THREADS RANDOM:
   • Tamano fijo (5,000 elementos)
   • Configuracion de threads aleatoria

7. AMBOS RANDOM:
   • Tamano del arreglo aleatorio
   • Configuracion de threads aleatoria

8. ARREGLO ESPECIFICO:
   • Configuracion completa personalizada
   • Valores personalizados opcionales
   • Seleccion de metodos a comparar

9. GENERAR REPORTE DETALLADO:
   • Muestra todos los resultados acumulados
   • Incluye estadisticas y resumen

10. EXPORTAR RESULTADOS:
   • Guarda todos los resultados en archivo
   • Formato legible y completo

NOTAS:
• Los resultados se acumulan entre ejecuciones
• Puedes ejecutar multiples pruebas y luego generar reporte
• El sistema mantiene un historico de todas las pruebas
    """)
    input("\nPresiona Enter para continuar...")

# =================== PROGRAMA PRINCIPAL ===================
def main():
    """Funcion principal del programa"""
    
    mostrar_encabezado()
    
    print("\n" + "="*80)
    print("¡Bienvenido al Sistema Comparativo Avanzado!")
    print("\nEste sistema te permite comparar diferentes enfoques para")
    print("sumar arreglos con configuraciones aleatorias y especificas.")
    print("NOTA: Los resultados se acumulan entre ejecuciones.")
    print("="*80)
    
    # Inicializar sistema global
    inicializar_sistema()
    
    while True:
        mostrar_menu_completo()
        
        try:
            opcion = input("\nSelecciona una opcion (1-12): ").strip()
            
            if opcion == "1":
                ejecutar_prueba_rapida()
            elif opcion == "2":
                ejecutar_comparativa_basica()
            elif opcion == "3":
                ejecutar_comparativa_avanzada()
            elif opcion == "4":
                ejecutar_comparativa_completa()
            elif opcion == "5":
                ejecutar_arreglo_random()
            elif opcion == "6":
                ejecutar_threads_random()
            elif opcion == "7":
                ejecutar_ambos_random()
            elif opcion == "8":
                ejecutar_arreglo_especifico()
            elif opcion == "9":
                generar_reporte_detallado()
            elif opcion == "10":
                exportar_resultados()
            elif opcion == "11":
                mostrar_ayuda()
            elif opcion == "12":
                print("\n" + "="*80)
                print("¡Gracias por usar el Sistema Comparativo Avanzado!")
                
                # Mostrar estadisticas finales
                sistema = inicializar_sistema()
                if sistema.resultados_acumulados:
                    print(f"\nResumen de sesion:")
                    print(f"  • Pruebas ejecutadas: {len(sistema.resultados_acumulados)}")
                    
                    # Contar metodos ganadores
                    metodos_ganadores = {}
                    for resultado in sistema.resultados_acumulados:
                        if resultado.get('mejor_metodo'):
                            metodo = resultado['mejor_metodo']
                            metodos_ganadores[metodo] = metodos_ganadores.get(metodo, 0) + 1
                    
                    if metodos_ganadores:
                        print(f"  • Metodos ganadores:")
                        for metodo, frecuencia in sorted(metodos_ganadores.items(), key=lambda x: x[1], reverse=True):
                            print(f"      • {metodo}: {frecuencia} veces")
                
                print("="*80)
                break
            else:
                print("\nOpcion invalida. Por favor selecciona 1-12.")
                input("Presiona Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()