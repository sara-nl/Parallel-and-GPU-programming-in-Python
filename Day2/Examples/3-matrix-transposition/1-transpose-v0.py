#Vector Addition with Pycuda(GPU) and Numpy(CPU)
# Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

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

#################### Sequential version
c_seq = np.zeros((N, N), np.uint32)
start_cpu = time.time()
for i in range(N):
    for j in range(N):
        c_seq[i][j] = a_cpu[j][i]
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time using CPU (sec): ", cpu_time)
print("---------------------")

#################### Validation
dif = 0
for s in range(N):
    for t in range(N):
        if(c_cpu[s][t] != c_seq[s][t]):
            dif += 1

print("Validation: there are %d different element(s)!" %dif)
print("---------------------")