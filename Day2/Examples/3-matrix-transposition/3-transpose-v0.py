# Matrix transpose with numba(GPU) and numba(CPU)
# Import and initialize numpy, numba, and time
import numpy as np
import time
from numba import cuda, njit, prange

#################### Array size
N = 4000

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1, 100, size=(N, N)).astype(np.uint32)
c_cpu = np.zeros((N, N), np.uint32)

#################### Numba CUDA kernel
@cuda.jit
def transpose(a_gpu, c_gpu, N):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    id = row * N + col
    id_transposed = col * N + row
    if row < N and col < N:
        c_gpu[id_transposed] = a_gpu[id]

#################### Allocate and transfer data to GPU
a_gpu = cuda.to_device(a_cpu.reshape(-1))
c_gpu = cuda.device_array(a_cpu.size, dtype=np.uint32)

#################### Grid and Block size
#################### Launch Numba CUDA kernel
block_size = 32
grid_size = ( (N + block_size - 1) // block_size,
              (N + block_size - 1) // block_size )

#################### Start GPU timing
start_gpu = time.time()

#################### Launch GPU kernel and time it
transpose[grid_size, (block_size, block_size)](a_gpu, c_gpu, N)

cuda.synchronize()
#################### End GPU timer
gpu_time = time.time() - start_gpu

#################### Copy back and reshape
c_gpu_res = c_gpu.copy_to_host().reshape(N, N)

print("Elapsed time using GPU Numba (sec): ", gpu_time)
print("---------------------")

#################### Numba CPU parallel transposition
@njit(parallel=True)
def transpose_numba(a, c):
    for i in prange(N):
        for j in prange(N):
            c[i, j] = a[j, i]
    return c

#################### Starting array c with zeros
c_numba = np.zeros((N, N), np.uint32)

#################### Launch Numba CPU version
start_cpu = time.time()
transpose_numba(a_cpu, c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

#################### Print result
print("Elapsed time using CPU parallel numba for-loop (sec): ", cpu_time)
print("---------------------")

#################### Implement and print validation
dif = np.sum(c_gpu_res != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")