# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 

#################### Array size
N = 400

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

#################### Launch the GPU kernel
func = module.get_function("multiplication")
block_size = 32
grid_size = int(np.ceil(N/block_size))
func(cuda.In(a_cpu), cuda.In(b_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size, grid_size, 1), block=(block_size, block_size, 1))

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU (sec): ", gpu_time)
print("---------------------")

#################### Sequesntial multiplication
c_seq = np.zeros((N,N), np.uint32)
start_cpu_seq = time.time()
for i in range(N):
	for j in range(N):
		for k in range(N):
			c_seq[i][j] += a_cpu[i][k] * b_cpu[k][j]
end_cpu_seq = time.time()
cpu_time_seq = end_cpu_seq - start_cpu_seq
print("Elapsed time using sequential for-loop (sec): ", cpu_time_seq)
print("---------------------")

#################### Numpy multiplication
'''
start_cpu_np = time.time()
c_np = np.matmul(a_cpu, b_cpu)
end_cpu_np = time.time()
cpu_time_np = end_cpu_np - start_cpu_np
print("Elapsed time using sequential numpy func (sec): ", cpu_time_np)
print("---------------------")
'''
#################### The @ operator multiplication
'''
start_cpu_op = time.time()
c_op = a_cpu @ b_cpu
end_cpu_op = time.time()
cpu_time_op = end_cpu_op - start_cpu_op
print("Elapsed time using sequential @ operator (sec): ", cpu_time_op)
print("---------------------")
'''
#################### Validation
dif = 0
for i in range(N):
	for j in range(N):
		if(c_cpu[i][j] != c_seq[i][j]):
			dif += 1
print ("Validation: there are %d different element(s)!" %dif)
print("---------------------")

