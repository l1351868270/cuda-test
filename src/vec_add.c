#include <malloc.h>
#include <time.h>

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
    for (int i = 0; i < n; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
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