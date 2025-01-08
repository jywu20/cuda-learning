module load gpu
nvcc -o simple.x -DCOMPARE simple.cu
./simple.x