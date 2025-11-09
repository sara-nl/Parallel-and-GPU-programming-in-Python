#Matrix Addition with Pycuda(GPU) and Numba(CPU)
# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from numba import njit, prange

import time 

#################### Array size
N = 4000

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32)
c_cpu = np.zeros((N,N), np.uint32)

#################### Satrt GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Write a GPU kernel
module = SourceModule(""" 
	__global__ void addition(int* a_gpu, int* b_gpu, int* c_gpu, int N){
		// Global thread indices
		int col = threadIdx.x + blockIdx.x * blockDim.x;
		int row = threadIdx.y + blockIdx.y * blockDim.y;
		int id = col + row * N;
		if(row < N && col < N){
			c_gpu[id] = a_gpu[id] + b_gpu[id];
		}
	}

""")

#################### Launch the GPU kernel
func = module.get_function("addition")
block_size = 32
grid_size = int(np.ceil(N/block_size))
func(cuda.In(a_cpu), cuda.In(b_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size, grid_size, 1), block=(block_size, block_size, 1))

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU (sec): ", gpu_time)
print("---------------------")

#################### Numba parallel addition
#################### Function definition
@njit(parallel=True)
def add_matrices_parallel(a, b, c):
	for i in prange(N):
		for j in prange(N):
			c[i,j]=a[i,j]+b[i,j]
	return c

#################### Starting array c with zeros
c_numba=np.zeros((N,N), np.uint32)

#################### Launch Numba parallel function and time it
start_cpu = time.time()
add_matrices_parallel(a_cpu,b_cpu,c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
#################### Print result
print("Elapsed time using CPU parallel numba for-loop (sec): ", cpu_time)
print("---------------------")

#################### Implement and print validation
dif = np.sum(c_cpu != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")

