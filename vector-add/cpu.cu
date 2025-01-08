#include<stdio.h>
#include<math.h>

const int DATA_DIM = 1024;
const int DATA_SIZE = DATA_DIM * sizeof(float);

int main(int argc, char const *argv[])
{
    // Arrays on the host side
    float *h_A, *h_B, *h_C;

    // Allocation on host side
    h_A = (float *)malloc(DATA_SIZE);
    h_B = (float *)malloc(DATA_SIZE);
    h_C = (float *)malloc(DATA_SIZE);

    for (size_t i = 0; i < DATA_DIM; i++)
    {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    for (size_t i = 0; i < DATA_DIM; i++)
    {
        h_C[i] = h_A[i] + h_B[i];
    }

    return 0;
}
