# Import modules
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Array size
N = 512
W = 32 

# CPU allocation and iitialization
a_cpu = np.random.uniform(1, 100, size=(N, W)).astype(np.uint32)
b_cpu = np.random.uniform(1, 100, size=(W, N)).astype(np.uint32)
c_cpu = np.zeros((N, N), np.uint32)



start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

# Write GPU kernel
mod = SourceModule(""" 
    __global__ void multiplication(int* a_gpu, int* b_gpu, int* c_gpu, int N){
        //global thread indices
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        int id = col + row * N;
        int sum = 0;
        __shared__ int a_shared[32][32];
        __shared__ int b_shared[32][32];
        a_shared[threadIdx.y][threadIdx.x] = a_gpu[row * blockDim.x + threadIdx.x];
        b_shared[threadIdx.y][threadIdx.x] = b_gpu[threadIdx.y * N + col];
        __syncthreads();
        if(row < N && col < N){
            for(int k=0; k<32; k++){
                sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
            }
            c_gpu[id] = sum;
        }
    }

""")


# Launch the kernel
func = mod.get_function("multiplication")

block_size = 32
grid_size = int(np.ceil(N/block_size))

func(cuda.In(a_cpu), cuda.In(b_cpu), cuda.Out(c_cpu), np.uint32(N), grid=(grid_size , grid_size, 1), block=(block_size , block_size, 1))


end_gpu.record()
sec_gpu = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU: ", sec_gpu)

# Sequential version
c_seq = np.zeros((N, N), np.uint32)
start = time.time()
for i in range(N):
    for j in range(N):
        for k in range(W):
            c_seq[i][j] += a_cpu[i][k] * b_cpu[k][j]
end = time.time()
sec = end - start
print("Elapsed time using CPU: ", sec)

# Validation
dif = 0
for s in range(N):
    for t in range(N):
        if(c_cpu[s][t] != c_seq[s][t]):
            dif += 1

print("Validation: there are %d different element(s)!" %dif)

