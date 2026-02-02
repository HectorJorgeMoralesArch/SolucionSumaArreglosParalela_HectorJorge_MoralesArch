/**
 * VERSION 8 COMPLETA: Sistema Comparativo con CUDA
 * Implementación con GPU usando CUDA
 * 
 * Autor: Hector Jorge Morales Arch
 * 
 * Caracteristicas incluidas:
 * 1. Version 1: Suma secuencial en CPU
 * 2. Version 2: Suma con OpenMP
 * 3. Version 8: Suma con CUDA en GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// =================== CONFIGURACION CUDA ===================
#define BLOCK_SIZE 256
#define MAX_VAL 10000

// =================== KERNEL CUDA ===================
__global__ void suma_cuda_kernel(int* A, int* B, int* R, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        R[idx] = A[idx] + B[idx];
    }
}

// =================== FUNCIONES AUXILIARES ===================
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

void calcular_metricas(double tiempo_sec, double tiempo_par, double* speedup) {
    if (tiempo_par > 0) {
        *speedup = tiempo_sec / tiempo_par;
    } else {
        *speedup = 0;
    }
}

void mostrar_muestra(int* A, int* B, int* R, int limite, int n) {
    printf("\n%-5s %-10s %-10s %-10s %-10s %-5s\n", 
           "i", "A[i]", "B[i]", "R[i]", "A+B", "OK");
    printf("--------------------------------------------------\n");
    
    int max = (limite < n) ? limite : n;
    
    for (int i = 0; i < max; i++) {
        int esperado = A[i] + B[i];
        int es_correcto = R[i] == esperado;
        printf("%-5d %-10d %-10d %-10d %-10d %s\n", 
               i, A[i], B[i], R[i], esperado, 
               es_correcto ? "OK" : "ERROR");
    }
}

// =================== VERSION 1: SECUENCIAL CPU ===================
void suma_secuencial(int* A, int* B, int* R, int n) {
    for (int i = 0; i < n; i++) {
        R[i] = A[i] + B[i];
    }
}

// =================== VERSION 2: OPENMP ===================
#ifdef _OPENMP
#include <omp.h>

void suma_openmp(int* A, int* B, int* R, int n, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        R[i] = A[i] + B[i];
    }
}
#else
void suma_openmp(int* A, int* B, int* R, int n, int num_threads) {
    // Si OpenMP no está disponible, usar secuencial
    suma_secuencial(A, B, R, n);
}
#endif

// =================== VERSION 8: CUDA GPU ===================
void suma_cuda(int* A, int* B, int* R, int n) {
    int *d_A, *d_B, *d_R;
    
    // 1. Reservar memoria en GPU
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    cudaMalloc((void**)&d_R, n * sizeof(int));
    
    // 2. Copiar datos de CPU a GPU
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // 3. Configurar y ejecutar kernel
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    suma_cuda_kernel<<<grid_size, BLOCK_SIZE>>>(d_A, d_B, d_R, n);
    
    // 4. Sincronizar y copiar resultados de GPU a CPU
    cudaDeviceSynchronize();
    cudaMemcpy(R, d_R, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 5. Liberar memoria de GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
}

// =================== SISTEMA COMPARATIVO CUDA ===================
void comparativa_cuda_completa() {
    printf("\n================================================================================\n");
    printf("VERSION 8: COMPARATIVA CUDA\n");
    printf("================================================================================\n");
    
    int tamanos[] = {1000, 10000, 100000, 1000000, 10000000};
    int num_tamanos = 5;
    
    for (int t = 0; t < num_tamanos; t++) {
        int n = tamanos[t];
        printf("\n\nANALIZANDO TAMAÑO: %d elementos\n", n);
        printf("------------------------------------------------------------\n");
        
        int *A, *B;
        crear_arreglos(n, 100, &A, &B, -1);
        
        // Secuencial CPU
        printf("\n   1. CPU SECUENCIAL...\n");
        int* R_cpu = (int*)malloc(n * sizeof(int));
        double inicio = get_time();
        suma_secuencial(A, B, R_cpu, n);
        double t_cpu = get_time() - inicio;
        printf("      Tiempo: %.6fs\n", t_cpu);
        
        // OpenMP
        #ifdef _OPENMP
        printf("\n   2. CPU OPENMP (4 threads)...\n");
        int* R_omp = (int*)malloc(n * sizeof(int));
        inicio = get_time();
        suma_openmp(A, B, R_omp, n, 4);
        double t_omp = get_time() - inicio;
        double speedup_omp;
        calcular_metricas(t_cpu, t_omp, &speedup_omp);
        printf("      Tiempo: %.6fs (Speedup: %.2fx)\n", t_omp, speedup_omp);
        #endif
        
        // CUDA GPU
        printf("\n   3. GPU CUDA...\n");
        int* R_gpu = (int*)malloc(n * sizeof(int));
        
        // Tiempo total incluyendo transferencias
        inicio = get_time();
        suma_cuda(A, B, R_gpu, n);
        double t_gpu_total = get_time() - inicio;
        
        double speedup_gpu_total;
        calcular_metricas(t_cpu, t_gpu_total, &speedup_gpu_total);
        printf("      Tiempo total (con transferencias): %.6fs (Speedup: %.2fx)\n", 
               t_gpu_total, speedup_gpu_total);
        
        // Verificar resultados
        printf("\n   4. VERIFICACION...\n");
        int correcto = verificar_resultados(R_cpu, R_gpu, A, B, n, 100);
        printf("      GPU vs CPU: %s\n", correcto ? "OK" : "ERROR");
        
        // Limpieza
        free(A);
        free(B);
        free(R_cpu);
        #ifdef _OPENMP
        free(R_omp);
        #endif
        free(R_gpu);
        
        if (t < num_tamanos - 1) {
            printf("\nPresiona Enter para continuar con el siguiente tamaño...\n");
            getchar();
        }
    }
}

void prueba_cuda_especifica() {
    printf("\n================================================================================\n");
    printf("PRUEBA ESPECIFICA CUDA\n");
    printf("================================================================================\n");
    
    // Verificar disponibilidad de CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("ERROR: No se encontraron dispositivos CUDA\n");
        return;
    }
    
    printf("Dispositivos CUDA disponibles: %d\n", deviceCount);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Usando dispositivo: %s\n", prop.name);
    printf("Capacidad de computo: %d.%d\n", prop.major, prop.minor);
    printf("Memoria global: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("MPs: %d\n", prop.multiProcessorCount);
    
    int n;
    printf("\nIngresa el tamaño del arreglo (ej: 1000000): ");
    scanf("%d", &n);
    getchar();  // Consumir newline
    
    printf("\nTamaño seleccionado: %d elementos\n", n);
    printf("Memoria requerida: %.2f MB\n", 
           (n * sizeof(int) * 3) / 1024.0 / 1024.0);
    
    // Crear arreglos
    printf("\n1. Creando arreglos...\n");
    int *A, *B;
    crear_arreglos(n, 100, &A, &B, -1);
    
    // CPU Secuencial
    printf("\n2. Ejecutando en CPU (Secuencial)...\n");
    int* R_cpu = (int*)malloc(n * sizeof(int));
    double inicio = get_time();
    suma_secuencial(A, B, R_cpu, n);
    double t_cpu = get_time() - inicio;
    printf("   Tiempo CPU: %.6fs\n", t_cpu);
    printf("   Operaciones/segundo: %.2f Mops\n", 
           (n / t_cpu) / 1000000.0);
    
    // GPU CUDA
    printf("\n3. Ejecutando en GPU (CUDA)...\n");
    int* R_gpu = (int*)malloc(n * sizeof(int));
    inicio = get_time();
    suma_cuda(A, B, R_gpu, n);
    double t_gpu = get_time() - inicio;
    
    double speedup;
    calcular_metricas(t_cpu, t_gpu, &speedup);
    
    printf("   Tiempo GPU: %.6fs\n", t_gpu);
    printf("   Speedup: %.2fx\n", speedup);
    printf("   Operaciones/segundo: %.2f Mops\n", 
           (n / t_gpu) / 1000000.0);
    
    // Verificación detallada
    printf("\n4. Verificación detallada...\n");
    int errores = 0;
    int muestras = (n > 100) ? 100 : n;
    
    for (int i = 0; i < muestras; i++) {
        int idx = rand() % n;
        if (R_cpu[idx] != R_gpu[idx]) {
            errores++;
            if (errores <= 5) {  // Mostrar solo primeros 5 errores
                printf("   Error en índice %d: CPU=%d, GPU=%d\n", 
                       idx, R_cpu[idx], R_gpu[idx]);
            }
        }
    }
    
    if (errores == 0) {
        printf("   Todos los resultados son correctos\n");
    } else {
        printf("   Se encontraron %d errores en %d muestras\n", errores, muestras);
    }
    
    // Mostrar muestra
    printf("\n5. Muestra de resultados (primeros 10 elementos):\n");
    mostrar_muestra(A, B, R_gpu, 10, n);
    
    // Liberar memoria
    free(A);
    free(B);
    free(R_cpu);
    free(R_gpu);
}

void benchmark_cuda() {
    printf("\n================================================================================\n");
    printf("BENCHMARK CUDA - DIFERENTES TAMAÑOS DE BLOQUE\n");
    printf("================================================================================\n");
    
    int n = 1000000;
    printf("Tamaño fijo: %d elementos\n\n", n);
    
    int *A, *B;
    crear_arreglos(n, 100, &A, &B, -1);
    
    int* R_cpu = (int*)malloc(n * sizeof(int));
    double inicio_cpu = get_time();
    suma_secuencial(A, B, R_cpu, n);
    double t_cpu = get_time() - inicio_cpu;
    
    printf("Tiempo CPU de referencia: %.6fs\n\n", t_cpu);
    printf("%-15s %-15s %-15s %-15s\n", 
           "Block Size", "Tiempo GPU", "Speedup", "GB/s");
    printf("----------------------------------------------------------------\n");
    
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = 6;
    
    for (int i = 0; i < num_sizes; i++) {
        int block_size = block_sizes[i];
        
        // Modificar kernel para tamaño de bloque específico
        int *d_A, *d_B, *d_R;
        cudaMalloc((void**)&d_A, n * sizeof(int));
        cudaMalloc((void**)&d_B, n * sizeof(int));
        cudaMalloc((void**)&d_R, n * sizeof(int));
        
        cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);
        
        int grid_size = (n + block_size - 1) / block_size;
        
        int* R_gpu = (int*)malloc(n * sizeof(int));
        
        // Calcular tiempo
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        suma_cuda_kernel<<<grid_size, block_size>>>(d_A, d_B, d_R, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(R_gpu, d_R, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        double t_gpu = milliseconds / 1000.0;
        double speedup = t_cpu / t_gpu;
        double bandwidth = (n * sizeof(int) * 3) / (t_gpu * 1024 * 1024 * 1024);
        
        printf("%-15d %-15.6f %-15.2f %-15.2f\n", 
               block_size, t_gpu, speedup, bandwidth);
        
        // Verificar
        int correcto = 1;
        for (int j = 0; j < 100; j++) {
            int idx = rand() % n;
            if (R_cpu[idx] != R_gpu[idx]) {
                correcto = 0;
                break;
            }
        }
        
        if (!correcto) {
            printf("   (Resultados incorrectos)\n");
        }
        
        // Liberar
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_R);
        free(R_gpu);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    free(A);
    free(B);
    free(R_cpu);
}

// =================== MENU PRINCIPAL CUDA ===================
void mostrar_menu_cuda() {
    printf("\n================================================================================\n");
    printf("SISTEMA COMPARATIVO CUDA - VERSION 8\n");
    printf("================================================================================\n");
    
    printf("\nOPCIONES:\n");
    printf("  1. Comparativa completa CPU vs GPU\n");
    printf("  2. Prueba específica CUDA\n");
    printf("  3. Benchmark diferentes tamaños de bloque\n");
    printf("  4. Información del dispositivo CUDA\n");
    printf("  5. Salir\n");
    
    printf("\n================================================================================\n");
}

void informacion_dispositivo_cuda() {
    printf("\n================================================================================\n");
    printf("INFORMACIÓN DEL DISPOSITIVO CUDA\n");
    printf("================================================================================\n");
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No se encontraron dispositivos CUDA\n");
        return;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDispositivo %d: %s\n", i, prop.name);
        printf("  Capacidad de computo: %d.%d\n", prop.major, prop.minor);
        printf("  MPs: %d\n", prop.multiProcessorCount);
        printf("  Memoria global: %.2f GB\n", 
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Memoria compartida por bloque: %lu KB\n", 
               prop.sharedMemPerBlock / 1024);
        printf("  Registers por bloque: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Máx threads por bloque: %d\n", prop.maxThreadsPerBlock);
        printf("  Máx threads por MP: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Máx bloques por MP: %d\n", prop.maxBlocksPerMultiProcessor);
        
        printf("  Máx dimensiones de bloque: %d x %d x %d\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Máx dimensiones de grid: %d x %d x %d\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        printf("  Clock rate: %.2f GHz\n", prop.clockRate * 1e-6);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate * 1e-6);
        printf("  Bus width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache size: %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    }
}

int main() {
    printf("\n================================================================================\n");
    printf("SISTEMA COMPARATIVO CUDA - VERSION 8\n");
    printf("================================================================================\n");
    printf("Autor: Hector Jorge Morales Arch\n");
    printf("================================================================================\n");
    
    // Verificar CUDA
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        printf("ADVERTENCIA: No se encontraron dispositivos CUDA\n");
        printf("El programa funcionará pero las opciones CUDA no estarán disponibles\n");
    }
    
    while (1) {
        mostrar_menu_cuda();
        
        char opcion[10];
        printf("\nSelecciona una opción (1-5): ");
        fgets(opcion, 10, stdin);
        
        if (opcion[0] == '1') {
            if (deviceCount > 0) {
                comparativa_cuda_completa();
            } else {
                printf("CUDA no está disponible en este sistema\n");
            }
            printf("\nPresiona Enter para continuar...\n");
            getchar();
        } else if (opcion[0] == '2') {
            if (deviceCount > 0) {
                prueba_cuda_especifica();
            } else {
                printf("CUDA no está disponible en este sistema\n");
            }
            printf("\nPresiona Enter para continuar...\n");
            getchar();
        } else if (opcion[0] == '3') {
            if (deviceCount > 0) {
                benchmark_cuda();
            } else {
                printf("CUDA no está disponible en este sistema\n");
            }
            printf("\nPresiona Enter para continuar...\n");
            getchar();
        } else if (opcion[0] == '4') {
            informacion_dispositivo_cuda();
            printf("\nPresiona Enter para continuar...\n");
            getchar();
        } else if (opcion[0] == '5') {
            printf("\n================================================================================\n");
            printf("¡Gracias por usar el Sistema Comparativo CUDA!\n");
            printf("================================================================================\n");
            break;
        } else {
            printf("\nOpción inválida. Por favor selecciona 1-5.\n");
            printf("Presiona Enter para continuar...\n");
            getchar();
        }
    }
    
    return 0;
}