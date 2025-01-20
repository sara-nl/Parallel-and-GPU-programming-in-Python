# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
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

#################### Sequesntial addition
c_seq = np.zeros((N,N), np.uint32)
start_cpu = time.time()
for i in range(N):
	for j in range(N):
		c_seq[i][j] = a_cpu[i][j] + b_cpu[i][j]
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time using sequential for-loop (sec): ", cpu_time)
print("---------------------")

#################### Validation
dif = 0
for i in range(N):
	for j in range(N):
		if(c_cpu[i][j] != c_seq[i][j]):
			dif += 1
print ("Validation: there are %d different element(s)!" %dif)
print("---------------------")

