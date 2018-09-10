# CUDA
# Projects on parallel processing using CUDA.
I this project I have tried to inplement parallel processing framework in cuda for breath first search tee transversal.
Content

1.Introduction to CUDA and GPU

2.Grids, Blocks, and Threads

3.Processing flow on CUDA and compiler used

4.Installation

5.Basic syntax for CUDA programs

6.Memory in GPU

7.Vector addition in GPU using CUDA

8.Matrix Multiplication in GPU using CUDA

9.Breath First Search in GPU using CUDA
10.

1.INTRODUCTION TO CUDA AND GPU

CUDA stands for Compute Unified Device Architecture. It is an extension of C programming, an API model for parallel computing created by Nvidia. Programs written using CUDA harness the power of GPU. Thus, increasing the computing performance. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach termed GPGPU (General-Purpose computing on Graphics Processing Units). CUDA can only be used with Nvidia graphics card which supports CUDA, all the newer versions of graphics card made by Nvidia support CUDA.

GPU stands for Graphics Processing Unit. The GPU’s advanced capabilities were originally used primarily for 3D game rendering. But now those capabilities are being harnessed more broadly to accelerate computational workloads in areas such as financial modeling, cutting-edge scientific research and oil and gas exploration.

Architecturally, the CPU is composed of just few cores with lots of cache memory that can handle a few software threads at a time. In contrast, a GPU is composed of hundreds of cores that can handle thousands of threads simultaneously. The ability of a GPU with 100+ cores to process thousands of threads can accelerate some software by 100x over a CPU alone. The GPU achieves this acceleration while being more power- and cost-efficient than a CPU. As many threads can run in many different cores of GPU simultaneously, so it eventually increases the throughput and also increases Floating Point Operations per Second (FLOPS) for a parallel process.



2.GRIDS, BLOCKS, AND THREADS

Threads can be defined a single independent processes, all running in parallel. For example let us assume a for loop in c which runs in CPU for say 10 times, in this case CPU does a same work but with similar or different inputs 10 times one after other, but when it come to CUDA, GPU have many cores, so we can divide the work to different threads all doing similar work but with different inputs, say for previous example we use for loop to run a task 10 time can be did in CUDA with say 10 threads holding each iteration and running parallel to each other can process it in one cycle. So works can done faster using threads. CUDA gives each thread a unique ThreadID to distinguish between each other even though the kernel instructions are the same.

Blocks are groups of threads and grids are blocks. So grid consist of blocks and blocks consist of threads. There is a limit to how many threads can a block hold and also on how many blocks a grid can hold, and it varies for different graphics cards. 

Execution resources are assigned to threads per block. Resources are organized into Streaming Multiprocessors (SM). Multiple blocks of threads can be assigned to a single SM. The number varies with CUDA device. For example, a CUDA device may allow up to 8 thread blocks to be assigned to an SM. This is the upper limit, and it is not necessary that for any configuration of threads, a SM will run 8 blocks. 

The CUDA API has a method, __syncthreads() to synchronize threads. When the method is encountered in the kernel, all threads in a block will be blocked at the calling location until each of them reaches the location.

3.PROCESSING FLOW ON CUDA AND COMPILER USED

A typical CUDA program has code intended both for the GPU and the CPU. By default, a traditional C program is a CUDA program with only the host code. The CPU is referred to as the host, and the GPU is referred to as the device. Whereas the host code can be compiled by a traditional C compiler as the GCC, the device code needs a special compiler to understand the API functions that are used. For Nvidia GPUs, the compiler is called the NVCC (Nvidia C Compiler).

The device code runs on the GPU, and the host code runs on the CPU. The NVCC processes a CUDA program, and separates the host code from the device code. To accomplish this, special CUDA keywords are looked for. The code intended to run of the GPU (device code) is marked with special CUDA keywords for labeling data-parallel functions, called ‘Kernels’. The device code is further compiled by the NVCC and executed on the GPU.

4.INSTALLATION

CUDA can be used in windows, linux and macOs. I am using it in windows 10 in visual studio 2015 community with Nsight plugin, this plugin is installed when you download the CUDA toolkit on a system which has a CUDA enabled graphics card, and a version of visual studio which is compatible with the version of CUDA that is to be installed. I my case I am using visual studio 2015 community with CUDA 9.0 and it works fine, it would not work with visual studio 2017. And also one thing should be taken care about is that visual studio should be installed before CUDA, if trying to install visual studio after CUDA then Nsight plugin would not be available and CUDA project can’t be created in visual studio.
CUDA toolkit can be downloaded from https://developer.nvidia.com/cuda-downloads

5.BASIC SYNTAX FOR CUDA PROGRAM

CUDA program has a same syntax as c++ and c for the host code, the part which is designed to run in GPU is written inside a function with no return type and starting from “__global__” which instructs a compiler to run that function in GPU. 
To call this function we have to specify the number of blocks and number of threads in each block in side”<<<,>>>”(three angular brackets).
As this function has no return type so all the data is passes as argument as pointer to let the function access the memory where it is stored.
For example
__global__ void kernel()
{
}
int main()
{
cout<<”Hello World”;
kernel<<<1,1>>>();
return 0;
}
In the above example we have used 1 block consisting of only one thread. The first parameter inside the angular brackets define the number of blocks and the second parameter defines the number of threads in each block.

Apart from this there are few function in CUDA for memory allocation and assigning a device variable a constant value and for other functions too. Some of them are :-
   A. cudaMalloc(varriable_name,size_of_memory) :-
     It is used to allocate memory in host and is accessed through a pointer.
   B. cudaMemset(varriable_name,constant_value_to_assign,size_of_memory)
     It is used to assign a constant value to a memory in device.
   C. cudaMemcpy(to_mem,from_mem,size_of_memory,cudaMemcpyDeviceToHost) 
     It is used to copy data from device memory to host memory.
   D. cudaMemcpy(to_mem,from_mem,size_of_memory,cudaMemcpyHostToDevice) 
     It is used to copy data from host memory to device memory.
   E. cudaFree(variable_name)
     It is used to deallocate a device memory one allocated.
   F. cudaSuccess
     It is used to check if a CUDA function is executed properly. It is used in our further programs and can be better understood there. 

6.MEMORY IN GPU

CUDA devices have several different memory spaces: Global, local, texture, constant, shared and register memory. Each type of memory on the device has its advantages and disadvantages. Incorrectly making use of the available memory in applications can can rob the desired performance.

In terms of speed, if all the various types of device memory is compared then the ranking would be:
1: Register file
2: Shared Memory
3: Constant Memory
4: Texture Memory
5: Local Memory and Global Memory

The only two types of memory that actually reside on the GPU chip are register and shared memory. Local, Global, Constant, and Texture memory all reside off chip. Local, Constant, and Texture are all cached.

While it would seem that the fastest memory is the best, the other two characteristics of the memory that dictate how that type of memory should be utilized are the scope and lifetime of the memory:
Data stored in register memory is visible only to the thread that wrote it and lasts only for the lifetime of that thread.
Local memory has the same scope rules as register memory, but performs slower.
Data stored in shared memory is visible to all threads within that block and lasts for the duration of the block. This is invaluable because this type of memory allows for threads to communicate and share data between one another.
Data stored in global memory is visible to all threads within the application (including the host), and lasts for the duration of the host allocation.
Constant and texture memory won’t be used here because they are beneficial for only very specific types of applications. Constant memory is used for data that will not change over the course of a kernel execution and is read only. Using constant rather than global memory can reduce the required memory bandwidth, however, this performance gain can only be realized when a warp of threads read the same location.Similar to constant memory, texture memory is another variety of read-only memory on the device. When all reads in a warp are physically adjacent, using texture memory can reduce memory traffic and increase performance compared to global memory.
R/W per-thread registers
R/W per-thread local memory
R/W per-block shared memory
R/W per-grid global memory
Read only per-grid constant memory
The following table summarizes the lifetime, scope and memory of different types of CUDA variables −
Variable declaration	Memory	Scope	Lifetime
Automatic variables other than arrays	Register	Thread	Kernel
Automatic array variables	Local	Thread	Kernel
__device__ __shared__ int sharedVar	Shared	Block	Kernel
__device__ int globalVar	Global	Grid	Application
__device__ __constant__ int constVar	Constant	Grid	Application
