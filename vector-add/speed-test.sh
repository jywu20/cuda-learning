module load gpu
nvcc -o simple.x simple.cu
nvcc -o cpu.x cpu.cu

echo Simple GPU implementation
time ./simple.x
echo

echo CPU implementation
time ./cpu.x
echo 
