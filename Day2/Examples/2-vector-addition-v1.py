# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 

# Array size
N = 10000000

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32)
c_cpu = np.zeros(N, np.uint32)
 
########################################################
# Write GPU kernel
module = SourceModule(""" 
	__global__ void addition(int* a_gpu, int* b_gpu, int* c_gpu, int N){
		// Global thread indices
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		if(id < N){
			c_gpu[id] = a_gpu[id] + b_gpu[id];
		}
	}

""")

# Grid and Block size
block_size = 512
grid_size = int(np.ceil(N/block_size))

# Satrt GPU timing
start0 = cuda.Event()
end0 = cuda.Event()
start0.record()

# Launch the GPU kernel
func = module.get_function("addition")
func(cuda.In(a_cpu), cuda.In(b_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size, 1, 1), block=(block_size, 1, 1))

# End GPU timing
end0.record()
cuda.Context.synchronize()
sec = start0.time_till(end0)*1e-3
print("Elapsed time using GPU: ", sec)
print("---------------------")

######################################################
# Sequesntial addition
c_seq = np.zeros(N, np.uint32)
start1 = time.time()
for i in range(N):
	c_seq[i] = a_cpu[i] + b_cpu[i]
end1 = time.time()
print("Elapsed time using sequential for-loop: ", end1-start1)
print("---------------------")

######################################################
# Validation
dif = 0
for i in range(N):
	if (c_cpu[i] != c_seq[i]):
		dif += 1
print ("Validation: there are %d different element(s)! " %dif)
print("---------------------")

