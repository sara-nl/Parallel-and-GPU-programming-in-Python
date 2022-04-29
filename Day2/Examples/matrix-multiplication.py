# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 

# Array size
N = 2000

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32)
c_cpu = np.zeros((N,N), np.uint32)

# Satrt GPU timing
start0 = cuda.Event()
end0 = cuda.Event()
start0.record()

# Write a GPU kernel
module = SourceModule(""" 
	__global__ void multiplication(int* a_gpu, int* b_gpu, int* c_gpu, int N){
		// Global thread indices
		int col = threadIdx.x + blockIdx.x * blockDim.x;
		int row = threadIdx.y + blockIdx.y * blockDim.y;
		int id = col + row * N;
		int tmp = 0;
		if(row < N && col < N){
			for(int k=0; k<N; k++){
				tmp += a_gpu[row*N+k] * b_gpu[k*N+col];
			}
			c_gpu[id] = tmp;
		}
	}

""")


# Launch the GPU kernel
func = module.get_function("multiplication")
block_size = 32
grid_size = int(np.ceil(N/block_size))
func(cuda.In(a_cpu), cuda.In(b_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size, grid_size, 1), block=(block_size, block_size, 1))

# End GPU timing
end0.record()
sec = start0.time_till(end0)*1e-3
print("Elapsed time using GPU: ", sec)

'''
# Sequesntial addition
c_seq = np.zeros((N,N), np.uint32)
start1 = time.time()
for i in range(N):
	for j in range(N):
		for k in range(N):
			c_seq[i][j] += a_cpu[i][k] * b_cpu[k][j]
end1 = time.time()
print("Elapsed time using sequential for-loop: ", end1-start1)
'''

# Numpy addition
start2 = time.time()
c_np = np.matmul(a_cpu, b_cpu)
end2 = time.time()
print("Elapsed time using sequential numpy add func: ", end2-start2)

# Simple addition
start3 = time.time()
c_simp = a_cpu @ b_cpu
end3 = time.time()
print("Elapsed time using sequential simple addition: ", end3-start3)

# Validation
dif = 0
for i in range(N):
	for j in range(N):
		if(c_cpu[i][j] != c_np[i][j]):
			dif += 1
print ("Validation: there are %d different element(s)!" %dif)

