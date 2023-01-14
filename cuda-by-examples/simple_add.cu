#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    int c;
    int *dev_c;
    cout << "Pointer value (before allocating memory on GPU): " << dev_c << endl;
    cout << "Pointer address: " <<  &dev_c << endl;

    err = cudaMalloc( (void**) &dev_c, sizeof(int) );

    cout << "Pointer value (after allocating memory on GPU): " << dev_c << endl;
    cout << "Pointer address: " << &dev_c << endl;
    
    if (err != cudaSuccess) {
        printf("Memorry alloc error!");
    }

    add <<<1,1>>>( 2, 7, dev_c );

    err = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        printf("cuda Memcpy error!");
    }

    printf("%d", c);

    cudaFree(dev_c);

    return 0;
}
