#include<stdio.h>
#include<math.h>

const int DATA_DIM = 2048*2048;
const int DATA_SIZE = DATA_DIM * sizeof(float);
const int THREADS_PER_BLOCK = 512;

__global__ void add(float *d_A, float *d_B, float *d_C)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_C[index] = d_A[index] + d_B[index];
}

int main(int argc, char const *argv[])
{
    // Arrays on the host side
    float *h_A, *h_B, *h_C;
    // Arrays on the device side
    float *d_A, *d_B, *d_C;
     
    // Allocation on host side
    h_A = (float *)malloc(DATA_SIZE);
    h_B = (float *)malloc(DATA_SIZE);
    h_C = (float *)malloc(DATA_SIZE);
    
    // Allocation on device side
    // Note that here we're directly chancing the addresses:
    // that's why we pass &d_A to cudaMalloc, i.e. "the pointer to the pointer we want to change"
    cudaMalloc((void **)&d_A, DATA_SIZE);
    cudaMalloc((void **)&d_B, DATA_SIZE);
    cudaMalloc((void **)&d_C, DATA_SIZE);
    
#ifndef NOINIT    
    // Initialization
    // In a real program real data is loaded into h_A and h_B
    for (size_t i = 0; i < DATA_DIM; i++)
    {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
#endif
    
    // Transfer data to GPU
    cudaMemcpy(d_A, h_A, DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DATA_SIZE, cudaMemcpyHostToDevice);
    
    // Do vector addition
    add<<<DATA_DIM/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);

    // Transfer data back to host;
    // note that cudaMemcpy is blocking and it waits
    // until the kernels launched above have finished.
    cudaMemcpy(h_C, d_C, DATA_SIZE, cudaMemcpyDeviceToHost);

#ifdef COMPARE
    // Test results
    printf("Doing comparison. \n");
    bool consistency_exist = false;
    for (size_t i = 0; i < DATA_DIM; i++)
    {
        float diff = abs(h_C[i] - (h_A[i] + h_B[i]));
        if (diff > 1e-8)
        {
            printf("Inconsistency at %u = %6f \n", i, diff);
            consistency_exist = true;
        }
    }
    
    if (!consistency_exist)
    {
        printf("No inconsistency is found. \n");
    }
    
#endif

    return 0;
}
