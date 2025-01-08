module load gpu
nvcc -o hello.x hello.cu
./hello.x 