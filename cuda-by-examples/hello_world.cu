#include <stdio.h>

__global__ void hello_world (void) {
}

int main(void) {
    
    printf("Hello World!");
    hello_world<<<1, 1>>>();

    return 0;
}
