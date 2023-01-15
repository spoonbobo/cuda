#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

void handleError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s", cudaGetErrorString(err));
        exit(err);
    }
}

void cpu_vector_add(int *a, int *b, int *c) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid++;
    }
}

__global__ void gpu_vector_add(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a[N], b[N], c[N], c_gpu[N];
    int *dev_a, *dev_b, *dev_c;

    // define threads
    dim3 threads(N);

    // allocate memory
    handleError(cudaMalloc( (void**)&dev_a, N * sizeof(int) ));
    handleError(cudaMalloc( (void**)&dev_b, N * sizeof(int) ));
    handleError(cudaMalloc( (void**)&dev_c, N * sizeof(int) ));
    
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i*i;
    }

    // copy host to device
    handleError(cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice ));
    handleError(cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice ));

    gpu_vector_add<<<1, threads>>>(dev_a, dev_b, dev_c);

    // copy device to host
    handleError(cudaMemcpy( c_gpu, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost ));

    cpu_vector_add(a, b, c);

    printf("CPU simple vector add version: \n");

    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    printf("GPU simple vector add version: \n");

    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c_gpu[i]);
    }

    return 0;
}