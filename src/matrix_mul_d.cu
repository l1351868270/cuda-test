#include <cuda.h>
#include <malloc.h>
#include <time.h>
#include <cuda_runtime.h>

__global__
void matrixMulKernel(float* d_A, float* d_B, float* d_C, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                d_C[idx*n+j] += d_A[idx*n+k]*d_B[k*n+j];
            }
            // printf("%f ", d_C[idx*n+j]);
        }
        // printf("\n");
    }

}

void matrixMul(float* h_A, float* h_B, float* h_C, int n) {
    int size = n*n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, size);
    // dim3 threads(16, 32);
    // dim3 blocks(ceil(m/16.0),ceil(n/32.0));
    dim3 threads(16);
    dim3 blocks(ceil(n/16.0));
    matrixMulKernel<<<blocks, threads>>> (d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    // int h_tmp = 
    // for (int i = 0; i < n; i++) {
    //     for(int j = 0; j < n; j++) {
    //         for (int k = 0; k < n; k++) {
    //             h_C[i*n+j] += h_A[i*n+k] * h_B[k*n+j];
    //         }
    //         if printf("%f ", h_C[i*n+j]);
    //     }
    // }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    int n = 2000; // 1.4s 比cpu版快20倍
    float *h_A = (float*)malloc(n*n*sizeof(float));
    float *h_B = (float*)malloc(n*n*sizeof(float));
    float *h_C = (float*)malloc(n*n*sizeof(float));
    for (int i = 0; i < n*n; i++) {
        h_A[i] = i;
        h_B[i] = i;
        h_C[i] = 0;
    }
    clock_t start, end;
    start = clock();
    matrixMul(h_A, h_B, h_C, n);
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