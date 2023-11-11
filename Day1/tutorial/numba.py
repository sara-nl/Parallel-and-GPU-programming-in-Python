{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "95c4a2b8",
   "metadata": {},
   "source": [
    "# Numba\n",
    "\n",
    "[Numba](https://numba.pydata.org/numba-doc/dev/user/overview.html) is a compiler for Python array and numerical functions that gives you the power to speed up your applications with high performance functions written directly in Python.\n",
    "\n",
    "Numba generates optimized machine code from pure Python code using the LLVM compiler infrastructure. With a few simple annotations, array-oriented and math-heavy Python code can be just-in-time optimized to performance similar as C, C++ and Fortran, without having to switch languages or Python interpreters.\n",
    "\n",
    "Numba’s main features are:\n",
    "\n",
    "* on-the-fly code generation (at import time or runtime, at the user’s preference)\n",
    "\n",
    "* native code generation for the CPU (default) and GPU hardware\n",
    "\n",
    "* integration with the Python scientific software stack (thanks to Numpy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bed8843",
   "metadata": {},
   "source": [
    "## Compiling Python code with `@jit`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94dd0e10",
   "metadata": {},
   "source": [
    "### Lazy compilation\n",
    "The recommended way to use the `@jit` decorator is to let Numba decide when and how to optimize:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "25db579a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/scratch-local/benjamic.4363307/ipykernel_279463/1011572751.py:4: NumbaDeprecationWarning: \u001b[1mThe 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.\u001b[0m\n",
      "  def sum(x, y):\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The slowest run took 14.33 times longer than the fastest. This could mean that an intermediate result is being cached.\n",
      "1.44 µs ± 2.23 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "from numba import jit\n",
    "\n",
    "@jit\n",
    "def sum(x, y):\n",
    "    return x + y\n",
    "\n",
    "%timeit sum(2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ae48a41",
   "metadata": {},
   "source": [
    "In this mode, compilation will be deferred until the first function execution. Numba will infer the argument types at call time, and generate optimized code based on this information. Numba will also be able to compile separate specializations depending on the input types. For example, calling the `f()` function above with integer or complex numbers will generate different code paths:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e11cbcbd",
   "metadata": {},
   "source": [
    "### Eager compilation\n",
    "\n",
    "You can also tell Numba the function signature you are expecting. The function f() would now look like:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4830603b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/scratch-local/benjamic.4363307/ipykernel_279463/2588414568.py:1: NumbaDeprecationWarning: \u001b[1mThe 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.\u001b[0m\n",
      "  @jit('int8(int8,int8)')\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "278 ns ± 1.11 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)\n"
     ]
    }
   ],
   "source": [
    "@jit('int8(int8,int8)')\n",
    "def sum(x, y):\n",
    "    return x + y\n",
    "\n",
    "%timeit sum(2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "662319f3",
   "metadata": {},
   "source": [
    "In this case, the corresponding specialization will be compiled by the `@jit` decorator, and no other specialization will be allowed. This is useful if you want fine-grained control over types chosen by the compiler (for example, to use single-precision floats)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f61a80f",
   "metadata": {},
   "source": [
    "### Signature specifications\n",
    "\n",
    "Explicit `@jit` signatures can use a number of types. Here are some common ones:\n",
    "\n",
    "- `void` is the return type of functions returning nothing (which actually return None when called from Python)\n",
    "- `intp` and `uintp` are pointer-sized integers (signed and unsigned, respectively)\n",
    "- `intc` and `uintc` are equivalent to C int and unsigned int integer types\n",
    "- `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64` are fixed-width integers of the corresponding bit width (signed and unsigned)\n",
    "- `float32` and `float64` are single- and double-precision floating-point numbers, respectively\n",
    "- `complex64` and `complex128` are single- and double-precision complex numbers, respectively\n",
    "- array types can be specified by indexing any numeric type, e.g. `float32[:]` for a one-dimensional single-precision array or `int8[:,:]` for a two-dimensional array of 8-bit integers."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a13dff4b",
   "metadata": {},
   "source": [
    "### Compilation options\n",
    "\n",
    "There are a number of keyword-only arguments can be passed to the `@jit` decorator."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7bf8aa8",
   "metadata": {},
   "source": [
    "#### nopython\n",
    "\n",
    "Numba has two compilation modes: `nopython` mode and `object` mode. The former produces much faster code, but has limitations that can force Numba to fall back to the latter. To prevent Numba from falling back, and instead raise an error, pass `nopython=True`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9402b8c0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "268 ns ± 1.49 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)\n"
     ]
    }
   ],
   "source": [
    "@jit(nopython=True)\n",
    "def sum(x, y):\n",
    "    return x + y\n",
    "\n",
    "%timeit sum(2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c60b3473",
   "metadata": {},
   "source": [
    "#### nogil\n",
    "\n",
    "Whenever Numba optimizes Python code to native code that only works on native types and variables (rather than Python objects), it is not necessary anymore to hold Python’s global interpreter lock (GIL). Numba will release the GIL when entering such a compiled function if you passed `nogil=True`.\n",
    "\n",
    " This will not be possible if the function is compiled in `object` mode."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1b943c41",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "249 ns ± 0.259 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)\n"
     ]
    }
   ],
   "source": [
    "@jit(nogil=True)\n",
    "def sum(x, y):\n",
    "    return x + y\n",
    "\n",
    "%timeit sum(2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad364af5",
   "metadata": {},
   "source": [
    "#### cache\n",
    "To avoid compilation times each time you invoke a Python program, you can instruct Numba to write the result of function compilation into a file-based cache. This is done by passing cache=True:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "88055614",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "191 ns ± 0.563 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)\n"
     ]
    }
   ],
   "source": [
    "@jit(cache=True)\n",
    "def sum(x, y):\n",
    "    return x + y\n",
    "\n",
    "%timeit sum(2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "762086dd",
   "metadata": {},
   "source": [
    "#### parallel\n",
    "\n",
    "Enables automatic parallelization (and related optimizations) for those operations in the function known to have parallel semantics. For a list of supported operations, see Automatic parallelization with `@jit`. This feature is enabled by passing `parallel=True` and must be used in conjunction with `nopython=True`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8a67551d",
   "metadata": {},
   "outputs": [],
   "source": [
    "@jit(nopython=True, parallel=True)\n",
    "def sum(x, y):\n",
    "    return x + y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2580f3dc-e0f2-4e61-8331-13525af5abc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Matrix addition\n",
    "![matrix addition](https://media.geeksforgeeks.org/wp-content/uploads/20230608165718/Matrix-Addition.png)\n",
    "*Image from geeksforgeeks.org*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "738f6cd3-d1df-4fca-9a59-b9d5e12eefb6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the input matricies\n",
    "A = [[1, 2], [3, 4]]\n",
    "B = [[4, 5], [6, 7]]\n",
    " \n",
    "# Initialize the result matrix\n",
    "C = [[0, 0], [0, 0]]\n",
    "\n",
    "# just loop through each dimension \n",
    "for i in range(len(A)):\n",
    "    for j in range(len(A[0])):\n",
    "        C[i][j] = A[i][j] + B[i][j]\n",
    "    \n",
    "print(C)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45882985-7a92-4e0b-864d-b9976f18092e",
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile ben.py\n",
    "import random\n",
    "import time\n",
    "from concurrent import futures  \n",
    "from functools import partial\n",
    "\n",
    "def naive_matrix_addition(A,B,C):\n",
    "    \n",
    "    # just loop through each dimension \n",
    "    for i in range(len(A)):\n",
    "        for j in range(len(A[0])):\n",
    "            C[i][j] = A[i][j] + B[i][j]\n",
    "\n",
    "    return(C)\n",
    "\n",
    "def naive_matrix_addition_parallel(A,B,C,N_workers,worker_id):\n",
    "    \n",
    "    block = int(len(A[0])/N_workers)\n",
    "    # just loop through each dimension \n",
    "    for i in range(len(A)):\n",
    "        for j in range(block*worker_id,block*(worker_id+1)):\n",
    "            \n",
    "            C[i][j] = A[i][j] + B[i][j]\n",
    "\n",
    "    return(C)\n",
    "\n",
    "number_cols = 1000 \n",
    "number_rows = 10000 \n",
    "A = [[random.randrange(1, 50, 1)] * number_cols for i in range(number_rows)]\n",
    "B = [[random.randrange(1, 50, 1)] * number_cols for i in range(number_rows)]\n",
    "C = [[0] * number_cols for i in range(number_rows)]\n",
    "\n",
    "#A = [[random.randrange(1, 50, 1) for i in range(size)], [random.randrange(1, 50, 1) for i in range(size)]]\n",
    "#B = [[random.randrange(1, 50, 1) for i in range(size)], [random.randrange(1, 50, 1) for i in range(size)]]\n",
    "#C = [[0 for i in range(size)], [0 for i in range(size)]]\n",
    "\n",
    "\n",
    "start = time.time()\n",
    "result_serial = naive_matrix_addition(A,B,C)\n",
    "end = time.time()\n",
    "print(\"Time serial:\", end-start)\n",
    "\n",
    "C = [[0] * number_cols for i in range(number_rows)]\n",
    "\n",
    "#C = [[0 for i in range(size)], [0 for i in range(size)]]\n",
    "\n",
    "N_workers = 10\n",
    "executor = futures.ProcessPoolExecutor(max_workers=N_workers)\n",
    "\n",
    "new_function = partial(naive_matrix_addition_parallel,A,B,C,N_workers)  \n",
    "\n",
    "start = time.time()\n",
    "future = executor.map(new_function,range(N_workers))\n",
    "\n",
    "result_parallel = list(future)\n",
    "end = time.time()\n",
    "print(\"Time parallel:\", end-start)\n",
    "#print(result_serial - result_parallel)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29d0f753-c447-4579-9d42-86eefcaae1b1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "425c1628-91fd-4ba7-89ea-13a8eb979073",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "16a7f2e9",
   "metadata": {},
   "source": [
    "## The explicit Matrix mulitplication example now with Numba!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "349639fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Multiplying 2 matricies of shape (5,5)\n",
      "122 µs ± 95.8 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)\n",
      "1.21 µs ± 12.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)\n"
     ]
    }
   ],
   "source": [
    "from numba import jit\n",
    "from random import random\n",
    "import numpy as np\n",
    "\n",
    "#what kind of decorator will you put here?\n",
    "def explicit_matmul(A,B):\n",
    "    #A[m][n]\n",
    "    #B[n][p]\n",
    "    #C[m][p]    \n",
    "    C_temp = np.zeros((np.shape(A)[0],np.shape(A)[1]))  \n",
    "    for i in range(np.shape(A)[0]): #(i=1...m) Rows in A\n",
    "        for j in range(np.shape(B)[1]): # (j=1...p) Columns in B\n",
    "            for k in range(np.shape(A)[1]): # (k=1...n) Columns in A\n",
    "                C_temp[i][j] += A[i][k] * B[k][j]\n",
    "    return(C_temp)\n",
    "\n",
    "AX=AY=BX=BY=500\n",
    "print(\"Multiplying 2 matricies of shape (\" +str(AX)+\",\"+str(AY)+\")\")\n",
    "\n",
    "A = np.random.rand(AX,AY)\n",
    "B = np.random.rand(BX,BY)  \n",
    "\n",
    "%timeit explicit_matmul(A,B)\n",
    "\n",
    "%timeit np.matmul(A,B)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6e0654f-4d5f-4b05-a901-5c08d91dca73",
   "metadata": {},
   "source": [
    "## Examples for lightweight profiling your code\n",
    "\n",
    " -  **%timeit** A very usefull magic function (especially for this course!)\n",
    " -  **time** (module) This module provides various time-related functions.\n",
    " -  **cProfile** (module) This module is recommended for most users; it’s a C extension with reasonable overhead that makes it suitable for profiling long-running programs. Based on lsprof, contributed by Brett Rosen and Ted Czotter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d22abf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time \n",
    "\n",
    "start = time.perf_counter_ns()\n",
    "explicit_matmul(A,B)\n",
    "end = time.perf_counter_ns()\n",
    "\n",
    "print(\"Time of function execution is \" +str(round(end-start)) + \" ns\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "692ca06d-2ac3-443f-964f-ebbdd19c11bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cProfile\n",
    "\n",
    "cProfile.run('explicit_matmul(A,B)') #By default the run method prints to the std out\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "716a00d7-136e-4ee3-98ec-1fee2e4a3095",
   "metadata": {},
   "outputs": [],
   "source": [
    "cProfile.run('explicit_matmul(A,B)',\"my_perf_file.out\") #By default the run method prints to the std out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "868fcb41-fdc5-4cf5-9173-4934cdccc4cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pstats\n",
    "from pstats import SortKey\n",
    "\n",
    "p = pstats.Stats('my_perf_file.out')  #read in the profile data\n",
    "\n",
    "#you can sort by the internal time\n",
    "p.sort_stats('time')\n",
    "p.print_stats()\n",
    "\n",
    "#you can sort by the number of calls\n",
    "p.sort_stats('calls')\n",
    "p.print_stats()\n",
    "\n",
    "#you can reverse the order\n",
    "p.reverse_order()\n",
    "p.print_stats()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c753d03-887d-4ecb-976c-f9ff3bf27566",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cProfile\n",
    "\n",
    "def do_profile(func):\n",
    "    def profiled_func(*args, **kwargs):\n",
    "        profile = cProfile.Profile()\n",
    "        try:\n",
    "            profile.enable()\n",
    "            result = func(*args, **kwargs)\n",
    "            profile.disable()\n",
    "            return result\n",
    "        finally:\n",
    "            profile.print_stats()\n",
    "    return profiled_func"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6e3e31d-2b8e-4cdf-b18d-9ee86a00701e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple Matrix multiplication algorithm\n",
    "@do_profile\n",
    "def numpy_matmul(A,B):\n",
    "    npA = np.array(A)\n",
    "    npB = np.array(B)\n",
    "    C = np.matmul(A,B)\n",
    "    return C\n",
    "\n",
    "@do_profile\n",
    "def explicit_matmul(A,B):\n",
    "    C = [[0 for x in range(len(A))] for y in range(len(B[0]))]\n",
    "    for i in range(len(A)):\n",
    "        for j in range(len(B[0])):\n",
    "            for k in range(len(B)):\n",
    "                C[i][j] += A[i][k] * B[k][j]\n",
    "    return C\n",
    "\n",
    "#Set matrix dimension\n",
    "AX=AY=BX=BY=100\n",
    "\n",
    "#Define Matrix A\n",
    "A = [[random() for x in range(AX)] for y in range(AY)]\n",
    "\n",
    "#Define Matrix B\n",
    "B = [[random() for x in range(BX)] for y in range(BY)]\n",
    "\n",
    "res = numpy_matmul(A,B)\n",
    "\n",
    "res = explicit_matmul(A,B)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
