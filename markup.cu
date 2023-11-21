#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

long POLYGON_DIM; //размер полигона = POLYGON_DIM * POLYGON_DIM

__global__ void markup(int* data, int* res, long* size) {
    int h = 3;
    if(gridDim.x * blockDim.x < (*size) * (*size)) { //если общее количество нитей меньше количества элементов полигона
        int tid = threadIdx.x + blockIdx.x * blockDim.x; //номер нити
        int number = ((*size) * (*size)) / (gridDim.x * blockDim.x); //количество элементов полигона, просматриваемых данной нитью
        int start_idx = tid * number; //начальный индекс элемента
        int finish_idx = start_idx + number; //конечный индекс элемента
        while (start_idx < finish_idx) {
            int curr = data[start_idx];
            if( start_idx / (*size) != 0     && curr - data[start_idx - (*size)] > h ||
                start_idx / (*size) != (*size) - 1 && curr - data[start_idx + (*size)] > h ||
                start_idx % (*size) != 0     && curr - data[start_idx - 1] > h ||
                start_idx % (*size) != (*size) - 1 && curr - data[start_idx + 1] > h ) 
                    res[start_idx] = -1;
            start_idx++;
        }
    } else { //если общее количество нитей больше количества элементов полигона
        int tid = threadIdx.x + blockIdx.x * blockDim.x; //номер нити
        if(tid < (*size) * (*size)) { //обязательно проверить, что не происходит выход за границы полигона
            int curr = data[tid];
            if( tid / (*size) != 0     && curr - data[tid - (*size)] > h ||
                tid / (*size) != (*size) - 1 && curr - data[tid + (*size)] > h ||
                tid % (*size) != 0     && curr - data[tid - 1] > h ||
                tid % (*size) != (*size) - 1 && curr - data[tid + 1] > h ) 
                    res[tid] = -1;
        }
    }
}

__host__ void fill(int* P) {
    for(int i = 0; i < POLYGON_DIM * POLYGON_DIM; i++) {
        P[i] = rand() % 10 + 1;
    }
}

//первый параметр - размер полигона
//второй параметр - количество блоков
//третий параметр - количество нитей
int main(int argc, const char* argv[]) {

    if (argc != 4) {
		fprintf(stderr, "Incorrect number of arguments\n");
		return -1;
	}

    POLYGON_DIM = strtol(argv[1], NULL, 10);
    long BLOCK_NUMBER = strtol(argv[2], NULL, 10); //количество блоков
    long THREAD_NUMBER = strtol(argv[3], NULL, 10); //количество нитей в блоке

    int* P;
    int* original_P;
    int* res_P;
    long* size;

    P = (int*) malloc( POLYGON_DIM * POLYGON_DIM * sizeof(int) );
    cudaMalloc( (void**)&original_P, POLYGON_DIM * POLYGON_DIM * sizeof(int) );
    cudaMalloc( (void**)&res_P, POLYGON_DIM * POLYGON_DIM * sizeof(int) );
    cudaMalloc( (void**)&size, sizeof(long) );

    srand( time( 0 ) );
    fill(P);
    cudaMemcpy( original_P, P, POLYGON_DIM * POLYGON_DIM * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( res_P, original_P, POLYGON_DIM * POLYGON_DIM * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy( size, &POLYGON_DIM, sizeof(long), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    markup<<< BLOCK_NUMBER, THREAD_NUMBER >>>(original_P, res_P, size);

    cudaEventSynchronize(stop);
    cudaEventRecord(stop, 0);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed Time: %3.1f ms\n", elapsedTime);

    for(int i = 0; i < POLYGON_DIM; i++) {
        for(int j = 0; j < POLYGON_DIM; j++) {
            printf("%d ", P[POLYGON_DIM*i + j]);
        }
        printf("\n");
    }
    printf("\n");

    cudaMemcpy( P, res_P, POLYGON_DIM * POLYGON_DIM * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < POLYGON_DIM; i++) {
        for(int j = 0; j < POLYGON_DIM; j++) {
            printf("%d ", P[POLYGON_DIM*i + j]);
        }
        printf("\n");
    }

    free(P);
    cudaFree(original_P);
    cudaFree(res_P);
    cudaFree(size);

    return 0;
}