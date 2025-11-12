# Vector addition with pycuda(GPU) and numba(CPU)
# Import and initialize pycuda, numpy, numba, and time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 
from numba import njit, prange

#################### Array size
N = 10000000

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32)
c_cpu = np.zeros(N, np.uint32)

#################### Allocate some space on GPU/DEVICE 
a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
c_gpu = cuda.mem_alloc(c_cpu.nbytes)

#################### Write GPU kernel
module = SourceModule(""" 
    __global__ void addition(int* a_gpu, int* b_gpu, int* c_gpu, int N){
        // Global thread indices
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        if(id < N){
            c_gpu[id] = a_gpu[id] + b_gpu[id];
        }
    }
    
""")

#################### Start GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Transfer data from CPU to GPU
cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)
cuda.memcpy_htod(c_gpu, c_cpu)

#################### Grid and Block size
block_size = 512
grid_size = int(np.ceil(N/block_size))

#################### Launch the GPU kernel
func = module.get_function("addition")
func(a_gpu, b_gpu, c_gpu, np.uint32(N), grid=(grid_size, 1, 1), block=(block_size, 1, 1))

#################### Transfer data from GPU to CPU
cuda.memcpy_dtoh(c_cpu, c_gpu)

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed on GPU with PyCuda (sec): ", gpu_time)
print("---------------------")

#################### Numba CPU parallel addition
@njit(parallel=True)
def cpu_addition(a, b, c):
    for i in prange(len(a)):
        c[i] = a[i] + b[i]

c_numba = np.zeros(N, np.uint32)
start_cpu = time.time()
cpu_addition(a_cpu, b_cpu, c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time on CPU using Numba (sec): ", cpu_time)
print("---------------------")

#################### Validation
dif = np.sum(c_cpu != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")