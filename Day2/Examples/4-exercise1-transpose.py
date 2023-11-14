# Import modules
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Array size
N = 4000

# CPU allocation and iitialization
a_cpu = np.random.uniform(1, 100, size=(N, N)).astype(np.uint32)
#b_cpu = np.random.uniform(1, 100, size=(N, N)).astype(np.uint32)
c_cpu = np.zeros((N, N), np.uint32)

###############################################################
# Satrt GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()


# Write a kernel
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


# Launch the GPU kernel

func = mod.get_function("transpose")

block_size = 32
grid_size = int(np.ceil(N/block_size))

func(cuda.In(a_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size , grid_size, 1), block=(block_size , block_size, 1))


end_gpu.record()
cuda.Context.synchronize()
sec_gpu = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU: ", sec_gpu)
print("---------------------")

#########################################################
# Sequential version
c_seq = np.zeros((N, N), np.uint32)
start = time.time()
for i in range(N):
    for j in range(N):
        c_seq[i][j] = a_cpu[j][i]
end = time.time()
sec = end - start
print("Elapsed time using CPU: ", sec)
print("---------------------")

#########################################################
# Validation
dif = 0
for s in range(N):
    for t in range(N):
        if(c_cpu[s][t] != c_seq[s][t]):
            dif += 1

print("Validation: there are %d different element(s)!" %dif)
print("---------------------")

