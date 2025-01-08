using CUDA
using BenchmarkTools
using Test

const DATA_DIM = 2048^2
const THREADS_PER_BLOCK = 512

function simple_add!(d_A, d_B, d_C)
    index = threadIdx().x + blockDim().x * (blockIdx().x - 1) 
    d_C[index] = d_A[index] + d_B[index]
    return
end

function simple_add_run!(d_A, d_B, d_C)
    CUDA.@sync begin
        # The number of threads is hard-coded here.
        @cuda blocks=Int(DATA_DIM / THREADS_PER_BLOCK) threads=THREADS_PER_BLOCK simple_add!(d_A, d_B, d_C)
    end
end

let 
    d_A = CUDA.fill(1.0f0, DATA_DIM) 
    d_B = CUDA.fill(1.0f0, DATA_DIM)
    d_C = CUDA.fill(0.0f0, DATA_DIM)
    
    println("Correctness test")

    simple_add_run!(d_A, d_B, d_C)
    
    @test all(Array(d_C) .== 2.0)
end

let 
    d_A = CUDA.fill(1.0f0, DATA_DIM) 
    d_B = CUDA.fill(1.0f0, DATA_DIM)
    d_C = CUDA.fill(0.0f0, DATA_DIM)

    println("Time cost measured by @btime:")
    @btime simple_add_run!($d_A, $d_B, $d_C)
end

let 
    d_A = CUDA.fill(1.0f0, DATA_DIM) 
    d_B = CUDA.fill(1.0f0, DATA_DIM)
    d_C = CUDA.fill(0.0f0, DATA_DIM)

    println("Time cost measured by @time:")
    @time simple_add_run!(d_A, d_B, d_C)
end