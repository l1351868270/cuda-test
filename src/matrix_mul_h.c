#include <malloc.h>
#include <time.h>

void matrixMul(float* h_A, float* h_B, float* h_C, int n) {
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                h_C[i*n+j] += h_A[i*n+k] * h_B[k*n+j];
            }
            // printf("%f ", h_C[i*n+j]);
        }
        // printf("\n");
    }
}

int main(int argc, char** argv) {
    int n = 2000; // 47s
    // int n = 3;
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