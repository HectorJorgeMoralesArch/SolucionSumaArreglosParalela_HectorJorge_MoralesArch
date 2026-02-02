/**
 * VERSION 7 COMPLETA: Sistema Comparativo con OpenMP y Todas las Funcionalidades
 * Todas las versiones integradas en un solo sistema
 * 
 * Autor: Hector Jorge Morales Arch
 * 
 * Caracteristicas incluidas:
 * 1. Version 1: Suma secuencial
 * 2. Version 2: Suma con threading
 * 3. Version 3: Suma con multiprocessing (simulado)
 * 4. Version 4: Comparativa completa
 * 5. Version 5: Configuraciones avanzadas y aleatorias
 * 6. Version 7: OpenMP nativo
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <math.h>

// =================== CONFIGURACION ===================
#define MAX_VAL 10000
#define MAX_HISTORICO 100
#define MAX_NOMBRE 100
#define MAX_RESULTADOS 20

#ifdef _OPENMP
    #define OPENMP_DISPONIBLE 1
#else
    #define OPENMP_DISPONIBLE 0
#endif

// =================== ESTRUCTURAS DE DATOS ===================
typedef struct {
    char nombre[MAX_NOMBRE];
    double tiempo;
    int threads;
    double speedup;
    double eficiencia;
} Resultado;

typedef struct {
    int tamano;
    Resultado resultados[MAX_RESULTADOS];
    int num_resultados;
    char timestamp[20];
} HistoricoItem;

typedef struct {
    HistoricoItem historico[MAX_HISTORICO];
    int num_ejecuciones;
} SistemaCompleto;

typedef struct {
    int* A;
    int* B;
    int* R;
    int start;
    int end;
} ThreadArgs;

typedef struct {
    int tipo;  // 1: avanzada, 2: openmp, 3: completa
    int tamano;
    int config_threads[10];
    int num_threads;
    Resultado resultados[MAX_RESULTADOS];
    int num_resultados;
} ConfiguracionAvanzada;

// =================== FUNCIONES AUXILIARES ===================
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void crear_arreglos(int n, int max_val, int** A, int** B, int seed) {
    *A = (int*)malloc(n * sizeof(int));
    *B = (int*)malloc(n * sizeof(int));
    
    if (seed != -1) {
        srand(seed);
    } else {
        srand(time(NULL));
    }
    
    for (int i = 0; i < n; i++) {
        (*A)[i] = rand() % max_val + 1;
        (*B)[i] = rand() % max_val + 1;
    }
}

void crear_arreglos_usuario(int n, int** A, int** B) {
    *A = (int*)malloc(n * sizeof(int));
    *B = (int*)malloc(n * sizeof(int));
    
    printf("\nCreando %d elementos\n", n);
    printf("1. Valores aleatorios\n");
    printf("2. Ingresar manualmente\n");
    
    char opcion[10];
    fgets(opcion, 10, stdin);
    
    if (opcion[0] == '2') {
        printf("\nEjemplo: 1,2,3,4,5\n");
        
        // Arreglo A
        printf("Arreglo A:\n");
        printf("Valores (Enter para aleatorio): ");
        
        char entrada[1000];
        fgets(entrada, 1000, stdin);
        
        if (strlen(entrada) > 1) {
            char* token = strtok(entrada, ",");
            int i = 0;
            while (token != NULL && i < n) {
                (*A)[i++] = atoi(token);
                token = strtok(NULL, ",");
            }
            for (; i < n; i++) {
                (*A)[i] = rand() % 100 + 1;
            }
        } else {
            for (int i = 0; i < n; i++) {
                (*A)[i] = rand() % 100 + 1;
            }
        }
        
        // Arreglo B
        printf("\nArreglo B:\n");
        printf("Valores (Enter para aleatorio): ");
        
        fgets(entrada, 1000, stdin);
        
        if (strlen(entrada) > 1) {
            char* token = strtok(entrada, ",");
            int i = 0;
            while (token != NULL && i < n) {
                (*B)[i++] = atoi(token);
                token = strtok(NULL, ",");
            }
            for (; i < n; i++) {
                (*B)[i] = rand() % 100 + 1;
            }
        } else {
            for (int i = 0; i < n; i++) {
                (*B)[i] = rand() % 100 + 1;
            }
        }
    } else {
        crear_arreglos(n, 100, A, B, -1);
    }
}

int verificar_resultados(int* R1, int* R2, int* A, int* B, int n, int muestra) {
    if (muestra > n) muestra = n;
    
    for (int i = 0; i < muestra; i++) {
        int idx = rand() % n;
        int esperado = A[idx] + B[idx];
        if (R1[idx] != esperado || R2[idx] != esperado) {
            return 0;
        }
    }
    return 1;
}

void calcular_metricas(double tiempo_sec, double tiempo_par, int num_hilos, 
                      double* speedup, double* eficiencia) {
    if (tiempo_par > 0) {
        *speedup = tiempo_sec / tiempo_par;
        *eficiencia = (*speedup / num_hilos) * 100;
    } else {
        *speedup = 0;
        *eficiencia = 0;
    }
}

void mostrar_muestra(int* A, int* B, int* R, int limite, int n) {
    printf("\n%-5s %-10s %-10s %-10s %-10s %-5s\n", 
           "i", "A[i]", "B[i]", "R[i]", "A+B", "OK");
    printf("--------------------------------------------------\n");
    
    int max = (limite < n) ? limite : n;
    int correcto = 1;
    
    for (int i = 0; i < max; i++) {
        int esperado = A[i] + B[i];
        int es_correcto = R[i] == esperado;
        if (!es_correcto) correcto = 0;
        printf("%-5d %-10d %-10d %-10d %-10d %s\n", 
               i, A[i], B[i], R[i], esperado, 
               es_correcto ? "OK" : "ERROR");
    }
    
    if (correcto) {
        printf("\nTodos los elementos mostrados son correctos.\n");
    } else {
        printf("\nHay errores en los resultados.\n");
    }
}

// =================== VERSION 1: SECUENCIAL ===================
void suma_secuencial(int* A, int* B, int* R, int n) {
    for (int i = 0; i < n; i++) {
        R[i] = A[i] + B[i];
    }
}

// =================== VERSION 2: THREADING ===================
void* worker_thread(void* args) {
    ThreadArgs* targs = (ThreadArgs*)args;
    for (int i = targs->start; i < targs->end; i++) {
        targs->R[i] = targs->A[i] + targs->B[i];
    }
    return NULL;
}

void suma_threading(int* A, int* B, int* R, int n, int num_threads) {
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];
    
    int chunk_size = n / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        args[i].A = A;
        args[i].B = B;
        args[i].R = R;
        args[i].start = i * chunk_size;
        args[i].end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        
        pthread_create(&threads[i], NULL, worker_thread, &args[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// =================== VERSION 3: MULTIPROCESSING (SIMULADO) ===================
void suma_multiprocessing(int* A, int* B, int* R, int n, int num_processes) {
    // En C puro, usamos threads para simular multiprocessing
    // En un sistema real se usarían fork() y procesos
    suma_threading(A, B, R, n, num_processes);
}

// =================== VERSION 7: OPENMP ===================
void suma_openmp(int* A, int* B, int* R, int n, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        R[i] = A[i] + B[i];
    }
}

// =================== SISTEMA COMPLETO ===================
void inicializar_sistema(SistemaCompleto* sistema) {
    sistema->num_ejecuciones = 0;
}

void comparativa_completa(SistemaCompleto* sistema) {
    printf("\n================================================================================\n");
    printf("VERSION 4: COMPARATIVA COMPLETA\n");
    printf("================================================================================\n");
    
    int tamanos[] = {100, 1000, 5000, 10000};
    int num_tamanos = 4;
    
    for (int t = 0; t < num_tamanos; t++) {
        int n = tamanos[t];
        printf("\n\nANALIZANDO TAMAÑO: %d elementos\n", n);
        printf("------------------------------------------------------------\n");
        
        int *A, *B;
        crear_arreglos(n, 100, &A, &B, -1);
        
        // Secuencial
        printf("\n   SECUENCIAL...\n");
        int* R_sec = (int*)malloc(n * sizeof(int));
        double inicio = get_time();
        suma_secuencial(A, B, R_sec, n);
        double t_sec = get_time() - inicio;
        printf("      Tiempo: %.6fs\n", t_sec);
        
        // Threading
        printf("\n   THREADING...\n");
        int thread_configs[] = {2, 4};
        for (int i = 0; i < 2; i++) {
            int threads = thread_configs[i];
            int* R_thr = (int*)malloc(n * sizeof(int));
            inicio = get_time();
            suma_threading(A, B, R_thr, n, threads);
            double t_thr = get_time() - inicio;
            
            double speedup, eficiencia;
            calcular_metricas(t_sec, t_thr, threads, &speedup, &eficiencia);
            printf("      %d threads: %.6fs (Speedup: %.2fx)\n", threads, t_thr, speedup);
            
            free(R_thr);
        }
        
        // Multiprocessing
        printf("\n   MULTIPROCESSING...\n");
        int procesos = 4;  // Valor fijo para ejemplo
        int* R_mp = (int*)malloc(n * sizeof(int));
        inicio = get_time();
        suma_multiprocessing(A, B, R_mp, n, procesos);
        double t_mp = get_time() - inicio;
        
        double speedup_mp, eficiencia_mp;
        calcular_metricas(t_sec, t_mp, procesos, &speedup_mp, &eficiencia_mp);
        printf("      %d procesos: %.6fs (Speedup: %.2fx)\n", procesos, t_mp, speedup_mp);
        
        // OpenMP
        if (OPENMP_DISPONIBLE) {
            printf("\n   OPENMP...\n");
            for (int i = 0; i < 2; i++) {
                int threads = thread_configs[i];
                int* R_omp = (int*)malloc(n * sizeof(int));
                inicio = get_time();
                suma_openmp(A, B, R_omp, n, threads);
                double t_omp = get_time() - inicio;
                
                double speedup_omp, eficiencia_omp;
                calcular_metricas(t_sec, t_omp, threads, &speedup_omp, &eficiencia_omp);
                printf("      %d threads: %.6fs (Speedup: %.2fx)\n", threads, t_omp, speedup_omp);
                
                free(R_omp);
            }
        }
        
        // Limpieza
        free(A);
        free(B);
        free(R_sec);
        free(R_mp);
        
        printf("\nPresiona Enter para continuar...\n");
        getchar();
    }
}

void ejecutar_comparativa_avanzada(int* A, int* B, int n, int* threads_config, int num_threads) {
    printf("\n================================================================================\n");
    printf("EJECUTANDO COMPARATIVA (n=%d)\n", n);
    printf("================================================================================\n");
    
    int* R_sec = (int*)malloc(n * sizeof(int));
    
    // Secuencial
    printf("\n1. SECUENCIAL...\n");
    double inicio = get_time();
    suma_secuencial(A, B, R_sec, n);
    double t_sec = get_time() - inicio;
    printf("   Tiempo: %.6fs\n", t_sec);
    
    // Threading
    printf("\n2. THREADING...\n");
    for (int i = 0; i < num_threads; i++) {
        int threads = threads_config[i];
        int* R_thr = (int*)malloc(n * sizeof(int));
        inicio = get_time();
        suma_threading(A, B, R_thr, n, threads);
        double t_thr = get_time() - inicio;
        
        double speedup, eficiencia;
        calcular_metricas(t_sec, t_thr, threads, &speedup, &eficiencia);
        printf("   • %d threads: %.6fs (Speedup: %.2fx)\n", threads, t_thr, speedup);
        
        free(R_thr);
    }
    
    // OpenMP
    if (OPENMP_DISPONIBLE) {
        printf("\n3. OPENMP...\n");
        for (int i = 0; i < num_threads; i++) {
            int threads = threads_config[i];
            int* R_omp = (int*)malloc(n * sizeof(int));
            inicio = get_time();
            suma_openmp(A, B, R_omp, n, threads);
            double t_omp = get_time() - inicio;
            
            double speedup, eficiencia;
            calcular_metricas(t_sec, t_omp, threads, &speedup, &eficiencia);
            printf("   • %d threads: %.6fs (Speedup: %.2fx)\n", threads, t_omp, speedup);
            
            free(R_omp);
        }
    }
    
    free(R_sec);
}

void prueba_openmp_especifica() {
    printf("\n================================================================================\n");
    printf("VERSION 7: PRUEBA OPENMP\n");
    printf("================================================================================\n");
    
    if (!OPENMP_DISPONIBLE) {
        printf("\nOpenMP no está disponible.\n");
        printf("Compila con: gcc -fopenmp -o version7 version7.c\n");
        printf("\nPresiona Enter para continuar...\n");
        getchar();
        return;
    }
    
    int n = 1000;
    printf("\nTamaño: %d elementos\n", n);
    
    // Crear arreglos
    printf("\n1. Creando arreglos...\n");
    int *A, *B;
    crear_arreglos_usuario(n, &A, &B);
    
    // Secuencial
    printf("\n2. Ejecutando SECUENCIAL...\n");
    int* R_sec = (int*)malloc(n * sizeof(int));
    double inicio = get_time();
    suma_secuencial(A, B, R_sec, n);
    double t_sec = get_time() - inicio;
    printf("   Tiempo: %.6fs\n", t_sec);
    
    // OpenMP con diferentes threads
    printf("\n3. Ejecutando OPENMP...\n");
    int threads_config[] = {1, 2, 4, 8};
    int num_configs = 4;
    
    for (int i = 0; i < num_configs; i++) {
        int threads = threads_config[i];
        printf("\n   • %d thread(s):\n", threads);
        
        int* R_omp = (int*)malloc(n * sizeof(int));
        inicio = get_time();
        suma_openmp(A, B, R_omp, n, threads);
        double t_omp = get_time() - inicio;
        
        double speedup, eficiencia;
        calcular_metricas(t_sec, t_omp, threads, &speedup, &eficiencia);
        
        int correcto = verificar_resultados(R_sec, R_omp, A, B, n, 50);
        
        printf("     Tiempo: %.6fs\n", t_omp);
        printf("     Speedup: %.2fx\n", speedup);
        printf("     Eficiencia: %.1f%%\n", eficiencia);
        printf("     Verificación: %s\n", correcto ? "OK" : "ERROR");
        
        free(R_omp);
    }
    
    // Mostrar muestra
    printf("\n================================================================================\n");
    printf("MUESTRA DE RESULTADOS\n");
    printf("================================================================================\n");
    
    mostrar_muestra(A, B, R_sec, 10, n);
    
    // Limpieza
    free(A);
    free(B);
    free(R_sec);
}

// =================== MENU PRINCIPAL ===================
void mostrar_menu_completo() {
    printf("\n================================================================================\n");
    printf("SISTEMA COMPARATIVO COMPLETO - VERSION 7\n");
    printf("================================================================================\n");
    
    printf("\nVERSIONES INDIVIDUALES:\n");
    printf("  1. Prueba rápida (n=100)\n");
    printf("  2. Threading demo (n=1000)\n");
    printf("  3. Multiprocessing demo (n=1000)\n");
    
    printf("\nVERSIONES COMPLETAS:\n");
    printf("  4. V4: Comparativa completa (múltiples tamaños)\n");
    printf("  5. V7: Prueba OpenMP específica\n");
    
    printf("\nREPORTES Y HERRAMIENTAS:\n");
    printf("  6. Información del sistema\n");
    printf("  7. Salir\n");
    
    printf("\n================================================================================\n");
}

void prueba_rapida() {
    printf("\n================================================================================\n");
    printf("PRUEBA RÁPIDA (n=100)\n");
    printf("================================================================================\n");
    
    int n = 100;
    int *A, *B;
    crear_arreglos(n, 100, &A, &B, -1);
    
    printf("\n1. Secuencial...\n");
    int* R1 = (int*)malloc(n * sizeof(int));
    double inicio = get_time();
    suma_secuencial(A, B, R1, n);
    double t1 = get_time() - inicio;
    printf("   Tiempo: %.6fs\n", t1);
    
    printf("\n2. Threading (4 threads)...\n");
    int* R2 = (int*)malloc(n * sizeof(int));
    inicio = get_time();
    suma_threading(A, B, R2, n, 4);
    double t2 = get_time() - inicio;
    printf("   Tiempo: %.6fs\n", t2);
    
    printf("\n3. Verificación...\n");
    int correcto = verificar_resultados(R1, R2, A, B, n, 20);
    printf("   Resultados: %s\n", correcto ? "IGUALES" : "DIFERENTES");
    
    free(A);
    free(B);
    free(R1);
    free(R2);
    
    printf("\nPresiona Enter para continuar...\n");
    getchar();
    getchar();
}

void demo_threading() {
    printf("\n================================================================================\n");
    printf("DEMO THREADING (n=1000)\n");
    printf("================================================================================\n");
    
    int n = 1000;
    int *A, *B;
    crear_arreglos(n, 100, &A, &B, -1);
    
    int threads_config[] = {1, 2, 4, 8};
    int num_configs = 4;
    
    for (int i = 0; i < num_configs; i++) {
        int threads = threads_config[i];
        printf("\n• %d thread(s)...\n", threads);
        
        int* R = (int*)malloc(n * sizeof(int));
        double inicio = get_time();
        suma_threading(A, B, R, n, threads);
        double tiempo = get_time() - inicio;
        printf("  Tiempo: %.6fs\n", tiempo);
        
        free(R);
    }
    
    free(A);
    free(B);
    
    printf("\nPresiona Enter para continuar...\n");
    getchar();
    getchar();
}

void demo_multiprocessing() {
    printf("\n================================================================================\n");
    printf("DEMO MULTIPROCESSING (n=1000)\n");
    printf("================================================================================\n");
    
    int n = 1000;
    int *A, *B;
    crear_arreglos(n, 100, &A, &B, -1);
    
    int procesos_config[] = {2, 4};
    int num_configs = 2;
    
    for (int i = 0; i < num_configs; i++) {
        int procesos = procesos_config[i];
        printf("\n• %d proceso(s)...\n", procesos);
        
        int* R = (int*)malloc(n * sizeof(int));
        double inicio = get_time();
        suma_multiprocessing(A, B, R, n, procesos);
        double tiempo = get_time() - inicio;
        printf("  Tiempo: %.6fs\n", tiempo);
        
        free(R);
    }
    
    free(A);
    free(B);
    
    printf("\nPresiona Enter para continuar...\n");
    getchar();
    getchar();
}

void informacion_sistema() {
    printf("\n================================================================================\n");
    printf("INFORMACIÓN DEL SISTEMA\n");
    printf("================================================================================\n");
    
    printf("\n• OpenMP disponible: %s\n", OPENMP_DISPONIBLE ? "SI" : "NO");
    
    if (OPENMP_DISPONIBLE) {
        printf("• Máximo threads OpenMP: %d\n", omp_get_max_threads());
    }
    
    printf("• Tamaño de int: %lu bytes\n", sizeof(int));
    printf("• Tiempo actual: %.0f\n", get_time());
    
    printf("\nPresiona Enter para continuar...\n");
    getchar();
    getchar();
}

int main() {
    printf("\n================================================================================\n");
    printf("SISTEMA COMPARATIVO COMPLETO - VERSION 7\n");
    printf("================================================================================\n");
    printf("Autor: Hector Jorge Morales Arch\n");
    printf("================================================================================\n");
    
    if (!OPENMP_DISPONIBLE) {
        printf("\nOpenMP no está disponible.\n");
        printf("Para usar OpenMP, compila con: gcc -fopenmp -o version7 version7.c\n");
    }
    
    SistemaCompleto sistema;
    inicializar_sistema(&sistema);
    
    while (1) {
        mostrar_menu_completo();
        
        char opcion[10];
        printf("\nSelecciona una opción (1-7): ");
        fgets(opcion, 10, stdin);
        
        if (opcion[0] == '1') {
            prueba_rapida();
        } else if (opcion[0] == '2') {
            demo_threading();
        } else if (opcion[0] == '3') {
            demo_multiprocessing();
        } else if (opcion[0] == '4') {
            comparativa_completa(&sistema);
        } else if (opcion[0] == '5') {
            prueba_openmp_especifica();
            printf("\nPresiona Enter para continuar...\n");
            getchar();
        } else if (opcion[0] == '6') {
            informacion_sistema();
        } else if (opcion[0] == '7') {
            printf("\n================================================================================\n");
            printf("¡Gracias por usar el Sistema Comparativo Completo!\n");
            printf("================================================================================\n");
            break;
        } else {
            printf("\nOpción inválida. Por favor selecciona 1-7.\n");
            printf("Presiona Enter para continuar...\n");
            getchar();
        }
    }
    
    return 0;
}