nvcc -o simple-no-init.x -DNOINIT simple.cu
nsys profile -o simple-no-init-report simple-no-init.x
