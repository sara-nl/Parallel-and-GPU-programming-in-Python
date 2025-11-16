import numpy as np
import time
from numba import njit, prange

#################### Array size
N = 10000000

# Allocate and initialise arrays on the CPU
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32) 
b_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32)
c_seq = np.zeros(N, np.uint32)
c_numba = np.zeros(N, np.uint32)

#################### Sequential addition
start_cpu = time.time()
for i in range(N):
	c_seq[i] = a_cpu[i] + b_cpu[i]
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time using CPU sequential for-loop (sec): ", cpu_time)
print("---------------------")

#################### Numba CPU parallel addition
@njit(parallel=True)
def cpu_addition(a, b, c):
    for i in prange(len(a)):
        c[i] = a[i] + b_cpu[i]

start_cpu = time.time()
cpu_addition(a_cpu, b_cpu, c_numba)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time on CPU using Numba (sec): ", cpu_time)
print("---------------------")

#################### Validation
dif = np.sum(c_seq != c_numba)
print("Validation: there are %d different element(s)!" % dif)
print("---------------------")