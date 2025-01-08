#include<stdio.h>

__global__ void hello()
{
    printf("Hello from block: %u, thread: %u \n", blockIdx.x, threadIdx.x);
}

int main(int argc, char const *argv[])
{
    hello<<<2, 2>>>();
    // In CUDA, after a kernel is launched,
    // the host program doesn't wait for it to stop,
    // and just move to the next statement.
    // To make sure the content of the GPU buffer is sent back to the host,
    // we have to synchronize between the device and the host.
    cudaDeviceSynchronize();
    return 0;
}
