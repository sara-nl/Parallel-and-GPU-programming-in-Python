#Vector Addition with Numba(GPU) and Numba(CPU)
# Import and Initialize numpy, numba, and time
import numpy as np
import time
from numba import njit, prange, cuda

#################### Array size
N = 10000000

# Create arrays on CPU
a_cpu = np.random.uniform(1.0, 100.0, size=N).astype(np.uint32)
b_cpu = np.random.uniform(1.0, 100.0, size=N).astype(np.uint32)
c_cpu = np.zeros(N, np.uint32)

#################### Numba GPU kernel
@cuda.jit
def gpu_addition(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

#################### Allocate GPU arrays and transfer data
a_gpu = cuda.to_device(a_cpu)
b_gpu = cuda.to_device(b_cpu)
c_gpu = cuda.device_array_like(c_cpu)

#################### Grid and block size
threads_per_block = 512
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

#################### Launch GPU kernel and time it
start_gpu = time.time()
gpu_addition[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
cuda.synchronize()  # Wait for GPU to finish
end_gpu = time.time()
gpu_time = end_gpu - start_gpu
print("Elapsed on GPU with Numba (sec):", gpu_time)
print("---------------------")

#################### Copy result back to CPU
c_numba_gpu = c_gpu.copy_to_host()

#################### Numba CPU parallel addition
@njit(parallel=True)
def cpu_addition(a, b, c):
    for i in prange(len(a)):
        c[i] = a[i] + b[i]

c_numba_cpu = np.zeros(N, np.uint32)
start_cpu = time.time()
cpu_addition(a_cpu, b_cpu, c_numba_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time on CPU using Numba (sec):", cpu_time)
print("---------------------")

#################### Validation
dif = np.sum(c_numba_gpu != c_numba_cpu)
print(f"Validation: there are {dif} different element(s)!")
print("---------------------")