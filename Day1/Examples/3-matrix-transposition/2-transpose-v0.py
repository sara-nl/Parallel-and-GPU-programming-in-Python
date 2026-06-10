# Matrix transpose with pycuda(GPU) and numba (CPU)
# Import and initialize pycuda, numpy, numba, and time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
from numba import njit, prange

#################### Array size
N = 4000

# Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1, 100, size=(N, N)).astype(np.uint32)
c_cpu = np.zeros((N, N), np.uint32)

#################### Allocate some space on GPU/DEVICE 
a_gpu = cuda.mem_alloc(a_cpu.nbytes)
c_gpu = cuda.mem_alloc(c_cpu.nbytes)

#################### Write a kernel
mod = SourceModule("""
    
        __global__ void transpose(int* a_gpu, int* c_gpu, int N){
        
            // Define global thread IDs
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int id = row * N + col;
            int id_transposed = col * N + row;
            if(row < N && col < N){
                c_gpu[id_transposed] = a_gpu[id];
                
            }
            
        }

""")

#################### Start GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Transfer data from CPU to GPU
cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(c_gpu, c_cpu)

#################### Grid and Block size
block_size = 32
grid_size = int(np.ceil(N/block_size))

#################### Launch the GPU kernel
func = mod.get_function("transpose")
func(a_gpu, c_gpu, np.uint32(N), grid=(grid_size , grid_size, 1), block=(block_size , block_size, 1))

#################### Transfer data from GPU to CPU
cuda.memcpy_dtoh(c_cpu, c_gpu)

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed on GPU with PyCuda (sec): ", gpu_time)
print("---------------------")

#################### Numba CPU parallel transposition
@njit(parallel=True)
def transpose_numba(a, c):
    for i in prange(N):
        for j in prange(N):
            c[i, j] = a[j, i]
    return c

#################### Starting array c with zeros
c_numba = np.zeros((N, N), np.uint32)

#################### Launch Numba CPU version
start_cpu = time.time()
transpose_numba(a_cpu, c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

#################### Print result
print("Elapsed time using CPU parallel numba for-loop (sec): ", cpu_time)
print("---------------------")

#################### Implement and print validation
dif = np.sum(c_cpu != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")