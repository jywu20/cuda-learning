
const int DATA_SIZE = 4096;

int main(int argc, char const *argv[])
{
    // Arrays on the host side
    float *h_A, *h_B, *h_C;
    // Arrays on the device side
    float *d_A, *d_B, *d_C;
    
    h_A = new float[DATA_SIZE];
    h_B = new float[DATA_SIZE];    
    h_C = new float[DATA_SIZE];    
    
    

    return 0;
}
