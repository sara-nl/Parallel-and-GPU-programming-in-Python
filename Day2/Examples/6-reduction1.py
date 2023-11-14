# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 
from functools import reduce
from operator import add

# Array size
N = 1024

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = a_cpu.copy()

###########################################################
# Satrt GPU timing
start0 = cuda.Event()
end0 = cuda.Event()
start0.record()

# Write a GPU kernel
module = SourceModule(""" 
	__global__ void reduction(int* a_gpu){
		// Thread indices
		int tid = threadIdx.x;
		
		for(int l=1; l<blockDim.x; l*=2){
		
			if(tid%(2*l) == 0){
				a_gpu[tid] += a_gpu[tid+l];
			}
			
			__syncthreads();
		}
		
		
	}

""")


# Launch the GPU kernel
func = module.get_function("reduction")
block_size = 1024
grid_size = int(np.ceil(N/block_size))
func(cuda.InOut(a_cpu), grid=(grid_size, 1, 1), block=(block_size, 1, 1))

# End GPU timing
end0.record()
cuda.Context.synchronize()
sec = start0.time_till(end0)*1e-3
print("Elapsed time using GPU (sec): ", sec)
print("total: ", a_cpu[0])
print("---------------------")

######################################################
# Sequesntial reduction
total_seq = 0
start1 = time.time()
for num in b_cpu:
	total_seq += num
end1 = time.time()
print("Elapsed time using sequential for-loop (sec): ", end1-start1)
print("total: ", total_seq)
print("---------------------")

#####################################################
# Reduce operator 
start3 = time.time()
total_op = reduce(add, b_cpu)
end3 = time.time()
print("Elapsed time using sequential reduce function (sec): ", end3-start3)
print("total: ", total_op)
print("---------------------")


