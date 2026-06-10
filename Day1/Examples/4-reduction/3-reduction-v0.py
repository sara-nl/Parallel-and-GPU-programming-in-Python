# Vector addition with numba(GPU) and numba(CPU)
# Import and initialize numpy, numba, and time
import numpy as np
import time 
from functools import reduce
from operator import add
from numba import njit, prange, cuda

#################### Array size
N = 1024

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = a_cpu.copy()

@cuda.jit
def reduction(a):
	tid = cuda.threadIdx.x
	l = 1
	while l < cuda.blockDim.x:
		if tid % (2 * l) == 0:
			a[tid] += a[tid + l]
		cuda.syncthreads()
		l *= 2

#################### Allocate and transfer data to GPU
a_gpu = cuda.to_device(a_cpu)

#################### Allocate GPU arrays and transfer data
block_size = 1024
grid_size = int(np.ceil(N/block_size))

#################### Launch GPU kernel and time it
start_gpu = time.time()
reduction[grid_size, block_size](a_gpu)
cuda.synchronize()  # Wait for GPU to finish

#################### Copy result back to CPU
a_cpu = a_gpu.copy_to_host()

end_gpu = time.time()
gpu_time = end_gpu - start_gpu
print("Elapsed on GPU with Numba (sec):", gpu_time)
print("total: ", a_cpu[0])
print("---------------------")


#################### Numba CPU parallel reduction

@njit(parallel=True)
def reduce_parallel(a):
    t = 0
    # prange automatically distributes iterations among threads
    for i in prange(N):
        t += a[i]
    return t

#################### Launch Numba CPU version
start_cpu = time.time()
total = reduce_parallel(b_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time using numba CPU(sec): ", cpu_time)
print("total: ", total)
print("---------------------")

#################### Reduce operator 
start_cpu_op = time.time()
total_op = reduce(add, b_cpu)
end_cpu_op = time.time()
cpu_time_op = end_cpu_op - start_cpu_op
print("Elapsed time using sequential reduce function (sec): ", cpu_time_op)
print("total: ", total_op)
print("---------------------")

