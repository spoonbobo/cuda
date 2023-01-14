#include <stdio.h>
#include <cuda_runtime.h>

void handleError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s", cudaGetErrorString(err));
        exit(err);
    }
}

int main(void) {
    cudaError_t err = cudaSuccess;
    int count;
    handleError( cudaGetDeviceCount (& count) );

    cudaDeviceProp prop;
    for (int i=0; i<count; i++) {
        handleError( cudaGetDeviceProperties(&prop, i) );
        printf("   --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    }
    return 0;
}