#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaError_t err = cudaSuccess;
    int count;
    err = cudaGetDeviceCount (& count) ;
    printf("%d", count);
    return 0;
}
