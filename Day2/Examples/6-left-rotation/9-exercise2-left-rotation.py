####################  Import and Initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time 

#################### Array size
N = 1024

#################### Create some space on CPU/HOST (random 32-bit ints)
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = a_cpu.copy()

#################### Start GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Write a GPU kernel
module = SourceModule(""" 
    __global__ void left_rotation(int* a_gpu, int N){
        // Thread indices
        int id = threadIdx.x;
        int temp = a_gpu[id];
        __syncthreads();
        if(id != 0){
            a_gpu[id-1] = temp;
        }else{
            a_gpu[N-1] = temp;
        }

    }

""")

#################### Launch the GPU kernel
func = module.get_function("left_rotation")
block_size = 1024
grid_size = int(np.ceil(N/block_size))
func(cuda.InOut(a_cpu), np.uint32(N), grid=(grid_size, 1, 1), block=(block_size, 1, 1))

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu)*1e-3
print("Elapsed time using GPU (sec): ", gpu_time)
print("---------------------")

#################### Sequesntial version
start_cpu_seq = time.time()
temp = b_cpu[0]
for i in range(N):
    if (i != N-1):
        b_cpu[i] = b_cpu[i+1]
    else:
        b_cpu[i] = temp
end_cpu_seq = time.time()
cpu_time_seq = end_cpu_seq - start_cpu_seq
print("Elapsed time using sequential for-loop (sec): ", cpu_time_seq)
print("---------------------")

#################### Validation
dif = 0
for j in range(N):
    if (a_cpu[j] != b_cpu[j]):
        dif += 1
print ("Validation: there are %d different element(s)! " %dif)
print("---------------------")
