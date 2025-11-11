# Matrix Addition with Numba CUDA (GPU) and Numba (CPU)
# Import and Initialize numpy, time, and numba
import numpy as np
import time
from numba import cuda, njit, prange

#################### Array size
N = 4000

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N, N)).astype(np.uint32)
b_cpu = np.random.uniform(1.0, 100.0, size=(N, N)).astype(np.uint32)
c_cpu = np.zeros((N, N), np.uint32)

#################### Numba CUDA kernel
@cuda.jit
def addition(a_gpu, b_gpu, c_gpu, N):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if row < N and col < N:
        idx = row * N + col
        c_gpu[idx] = a_gpu[idx] + b_gpu[idx]

#################### Allocate and transfer data to GPU
a_gpu = cuda.to_device(a_cpu.reshape(-1))
b_gpu = cuda.to_device(b_cpu.reshape(-1))
c_gpu = cuda.device_array(a_cpu.size, dtype=np.uint32)

#################### Start GPU timing
start_gpu = time.time()

#################### Launch Numba CUDA kernel
block_size = 32
grid_size = ( (N + block_size - 1) // block_size,
              (N + block_size - 1) // block_size )

addition[grid_size, (block_size, block_size)](a_gpu, b_gpu, c_gpu, N)

cuda.synchronize()
gpu_time = time.time() - start_gpu

#################### Copy back and reshape
c_gpu_res = c_gpu.copy_to_host().reshape(N, N)

print("Elapsed time using GPU (sec): ", gpu_time)
print("---------------------")

#################### Numba parallel CPU addition
@njit(parallel=True)
def add_matrices_parallel(a, b, c):
    for i in prange(N):
        for j in prange(N):
            c[i, j] = a[i, j] + b[i, j]
    return c

#################### Starting array c with zeros
c_numba = np.zeros((N, N), np.uint32)

#################### Launch Numba CPU version
start_cpu = time.time()
add_matrices_parallel(a_cpu, b_cpu, c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

#################### Print result
print("Elapsed time using CPU parallel numba for-loop (sec): ", cpu_time)
print("---------------------")

#################### Implement and print validation
dif = np.sum(c_gpu_res != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")