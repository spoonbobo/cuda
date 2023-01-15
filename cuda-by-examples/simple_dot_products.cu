#include <stdio.h>
#include <cuda_runtime.h>

void handleError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s", cudaGetErrorString(err));
        exit(err);
    }
}