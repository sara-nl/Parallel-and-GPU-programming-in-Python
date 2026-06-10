# Matrix multiplication with numba(GPU) and numba(CPU)
# Import and initialize numpy, numba and time
import numpy as np
import time 
from numba import njit, prange, cuda

#################### Array size
N = 400

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N,N)).astype(np.uint32)
c_cpu = np.zeros((N,N), np.uint32)

@cuda.jit
def multiplication(a_gpu, b_gpu, c_gpu, N):
	col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
	if row < N and col < N:
		tmp = 0
		for k in range(N):
			tmp += a_gpu[row * N + k] * b_gpu[k * N + col]
		c_gpu[row * N + col] = tmp

#################### Allocate and transfer data to GPU
a_gpu = cuda.to_device(a_cpu.reshape(-1))
b_gpu = cuda.to_device(b_cpu.reshape(-1))
c_gpu = cuda.device_array(a_cpu.size, dtype=np.uint32)

#################### Define Numba CUDA kernel
block_size = 32
grid_size = ( (N + block_size - 1) // block_size,
              (N + block_size - 1) // block_size )

#################### Launch GPU kernel and time it
start_gpu = time.time()
multiplication[grid_size, (block_size, block_size)](a_gpu, b_gpu, c_gpu, N)
cuda.synchronize()
gpu_time = time.time() - start_gpu

#################### Copy back and reshape
c_gpu_res = c_gpu.copy_to_host().reshape(N, N)

print("Elapsed time using GPU Numba (sec): ", gpu_time)
print("---------------------")

@njit(parallel=True)
def multiplication_numba(a, b, c):
	for i in prange(N): 		# parallelized outer loop
		for j in range(N):
			tmp = 0
			for k in range(N):
				tmp += a[i, k] * b[k, j]
			c[i, j] = tmp
	return c

c_numba=np.zeros((N,N), np.uint32)

#################### Launch Numba parallel function and time it
start_cpu = time.time()
multiplication_numba(a_cpu,b_cpu,c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

#################### Print result
print("Elapsed time using CPU parallel numba for-loop (sec): ", cpu_time)
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
		if(c_gpu_res[i][j] != c_numba[i][j]):
			dif += 1
print ("Validation: there are %d different element(s)!" %dif)
print("---------------------")