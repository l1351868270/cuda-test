#include <cuda.h>
#include <malloc.h>
#include <time.h>
#include <cuda_runtime.h>

__global__
void matrixAddKernel(float* d_A, float* d_B, float* d_C, int m, int n) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy*m;
    if (idx < 100) {
        printf("threadIdx.x=%d, threadIdx.y=%d, blockDim.x=%d, blockDim.y=%d, blockIdx.x=%d, blockIdx.y=%d, idx=%d\n",
        threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, idx);
    }

    if (ix < m && iy < n) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

void matrixAdd(float* h_A, float* h_B, float* h_C, int m, int n) {
    int size = m*n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, size);
    dim3 threads(16, 32);
    dim3 blocks(ceil(m/16.0),ceil(n/32.0));
    matrixAddKernel<<<blocks, threads>>> (d_A, d_B, d_C, m, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m*n; i++) {
        if (h_A[i]+h_B[i] != h_C[i]) {
            printf("h[%d]=%f, d[%d]=%f\n", i, h_A[i]+h_B[i], i, h_C[i]);
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    int m = 96, n = 32;
    float *h_A = (float*)malloc(m*n*sizeof(float));
    float *h_B = (float*)malloc(m*n*sizeof(float));
    float *h_C = (float*)malloc(m*n*sizeof(float));
    for (int i = 0; i < m*n; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }
    clock_t start, end;
    start = clock();
    matrixAdd(h_A, h_B, h_C, m, n);
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