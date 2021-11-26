#include <cuda.h>
#include <malloc.h>
#include <time.h>

__global__
void vecAddKernel(float* d_A, float* d_B, float* d_C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, size);
    vecAddKernel<<<ceil(n/256.0), 256>>> (d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    int n = 1000000000;
    float *h_A = (float*)malloc(n*sizeof(float));
    float *h_B = (float*)malloc(n*sizeof(float));
    float *h_C = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }
    clock_t start, end;
    start = clock();
    vecAdd(h_A, h_B, h_C, n);
    end = clock();
    printf("duration %f\n", (double)(end-start)/CLOCKS_PER_SEC);
    free(h_A);
    h_A = NULL;
    free(h_B);
    h_B = NULL;
    free(h_C);
    h_C = NULL;
    return 0;
}