# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 
from functools import reduce
from operator import add

#################### Array size
N = 33554432

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = a_cpu.copy()

#################### Create an array of one element and initialize to zero
total_gpu = np.array([0])
total_gpu[0] = 0

#################### Satrt GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Write a GPU kernel
module = SourceModule(""" 
	__global__ void reduction(int* a_gpu, int* total){
		// Thread global indices
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
		for(int l=1; l<blockDim.x; l*=2){
		
			if(threadIdx.x%(2*l) == 0){
				a_gpu[tid] += a_gpu[tid+l];
			}
			
			__syncthreads();
		}
		
		if (threadIdx.x == 0){
			atomicAdd(&total[0], a_gpu[tid]);
		}
		
	}

""")

#################### Launch the GPU kernel
func = module.get_function("reduction")
block_size = 1024
grid_size = int(np.ceil(N/block_size))
func(cuda.InOut(a_cpu), cuda.InOut(total_gpu), grid=(grid_size, 1, 1), block=(block_size, 1, 1))

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU (sec): ", gpu_time)
print("total: ", total_gpu[0])
print("---------------------")

#################### Sequesntial addition
total_seq = 0
start_cpu_seq = time.time()
for num in b_cpu:
	total_seq += num
end_cpu_seq = time.time()
cpu_time_seq = end_cpu_seq - start_cpu_seq
print("Elapsed time using sequential for-loop (sec): ", cpu_time_seq)
print("total: ", total_seq)
print("---------------------")

#################### Reduce operator 
start_cpu_op = time.time()
total_op = reduce(add, b_cpu)
end_cpu_op = time.time()
cpu_time_op = end_cpu_op - start_cpu_op
print("Elapsed time using sequential reduce function (sec): ", cpu_time_op)
print("total: ", total_op)
print("---------------------")

