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

Example of a shared memory program is as follows:-
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

__global__ void staticReverse(int *d, int n){
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];}

__global__ void dynamicReverse(int *d, int n){
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];}
int main(void){
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 

  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
if (d[i] != r[i]) 
printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
if (d[i] != r[i]) 
printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
return 0;
}

In the above program a array of numbers is reversed in device using shared memory, first the the array is stored in shared memory just as it is, then all the elements are then store the value of it’s position in reverse from shared memory. In the program two types of shared memory are shown that are: 
1: Static shared memory-
This memory size is defined inside the kernel call function and the size is defined by a constant number variable can be passed to define the size. This memory is allocated by using “__shared__” before the declaration of the variable.   
2: Dynamic shared memory-
This memory size is defined with the kernel call with the third parameter inside the three angular brackets which defines the size of shared memory and it can be determined by a variable so it’s size can differ depending on the input. This memory is allocated by using “extern __shared__” in place of only “__shared__”, before the declaration of the variable.



7.VECTOR ADDITION IN GPU USING CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>

using namespace std;

__global__ void add(int *a, int *b, int size)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < size)
  {  
  a[id] = a[id] + b[id];
  }
}

int main()
{
int size=100;
int *h_a=new int[size];
int *h_b=new int[size];

int i;

for(i=0;i<size;i++)
  {
  h_a[i]= rand()%1000;
  h_b[i]= rand()%1000;
  }

for(i=0;i<5;i++)
  {
  cout<<h_a[i]<<"  "<<h_b[i]<<"\n";
  }

int*d_a,*d_b;

if(cudaMalloc(&d_a, sizeof(int)*size) != cudaSuccess)
  {
  cout<<"Memory allocation failed\n";
  cin>>i;
  return 0;
  }

if(cudaMalloc(&d_b, sizeof(int)*size) != cudaSuccess)
  {
  cout<<"Memory allocation failed\n";
  cudaFree(d_a);
  cin>>i;
  return 0;
  }

if(cudaMemcpy(d_a, h_a, sizeof(int)*size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
  cout<<"Failed to copy data from host to device";
  cudaFree(d_a);
  cudaFree(d_b);
  cin>>i;
  return 0;
  }

if(cudaMemcpy(d_b, h_b, sizeof(int)*size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
  cout<<"Failed to copy data from host to device";
  cudaFree(d_a);
  cudaFree(d_b);
  cin>>i;
  return 0;
  }

add<<<10, 10>>>(d_a, d_b, size);


if(cudaMemcpy(h_a, d_a, size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
  cout<<"Failed to copy data from device to host";
  delete[] h_a;
  delete[] h_b;
  cudaFree(d_a);
  cudaFree(d_b);
  cin>>i;
  return 0;
  }

for(i=0;i<5;i++)
{
  cout<<h_a[i]<<"\n";
}

delete[] h_a;
delete[] h_b;
cudaFree(d_a);
cudaFree(d_b);

cin>>i;
return 0;

}



8.MATRIX MULTIPLICATION IN GPU USING CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>

using namespace std;

__global__ void multiply(int *a, int *b, int *c, int m, int n, int q)
{
int id_x = threadIdx.x;
int id_y = threadIdx.y;

int i,d=0;
for(i=0;i<n;i++)
  {
  d = d + (a[(id_y * n) + i] * b[(i * q) + id_x]);
  }
  c[(id_y * q) + id_x] = d;
}

int main()
{
int i;
int m,n,x,y;

cout<<"Enter the size of first matrix:\n";
cin>>m>>n;
cout<<"Enter the size of second matrix:\n";
cin>>x>>y;

if(n != x)
  {
  cout<<"matrix size is not valid for multiplication";
  cin>>i;
  return 0;
  }

int size_a = m*n;
int size_b = x*y;
int size_c = m*y;

int *h_a=new int[size_a];
int *h_b=new int[size_b];
int *h_c=new int[size_c];

cout<<"Enter the first matrix\n";

for(i=0;i<size_a;i++)
  {
  cin>>h_a[i];
  }

cout<<"Enter the second array:\n";
for(i=0;i<size_b;i++)
  {
  cin>>h_b[i];
  }

int*d_a,*d_b,*d_c;

if(cudaMalloc(&d_a, sizeof(int)*size_a) != cudaSuccess)
  {
  cout<<"Memory allocation failed\n";
  cin>>i;
  return 0;
  }

if(cudaMalloc(&d_b, sizeof(int)*size_b) != cudaSuccess)
  {
  cout<<"Memory allocation failed\n";
  cudaFree(d_a);
  cin>>i;
  return 0;
  }

if(cudaMalloc(&d_c, sizeof(int)*size_c) != cudaSuccess)
  {
  cout<<"Memory allocation failed\n";
  cudaFree(d_a);
  cudaFree(d_b);
  cin>>i;
  return 0;
  }

if(cudaMemcpy(d_a, h_a, sizeof(int)*size_a, cudaMemcpyHostToDevice) != cudaSuccess)
  {
  cout<<"Failed to copy data from host to device";
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cin>>i;
  return 0;
  }

if(cudaMemcpy(d_b, h_b, sizeof(int)*size_b, cudaMemcpyHostToDevice) != cudaSuccess)
  {
  cout<<"Failed to copy data from host to device";
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cin>>i;
  return 0;
  }

dim3 block(1,1,1);
dim3 thread(y,m,1);

multiply<<<block, thread>>>(d_a, d_b, d_c, m, n, y);

if(cudaMemcpy(h_c, d_c, size_c * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
  cout<<"Failed to copy data from device to host";
  delete[] h_a;
  delete[] h_b;
  cudaFree(d_a);
  cudaFree(d_b);
  cin>>i;
  return 0;
  }

cout<<"Multiplied Matrix is:\n";
for(i=0;i<size_c;i++)
{
  cout<<h_c[i]<<"  ";
  if((i+1)%y == 0) cout<<endl;
}

delete[] h_a;
delete[] h_b;
delete[] h_c;
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

cin>>i;
return 0;

}



9.BREATH FIRST SEARCH IN GPU USING CUDA


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;
__global__ void depth(int *mat, int *stack, int *index, int start, int n, int *result, int *depth)
{
int id_x = threadIdx.x;
int id_y = blockIdx.x;
depth[start] = 0;
index[start] = start;
stack[start] = 1;
while(index[id_x]<0)
{
if(mat[id_y*n+id_x]==1)
  {
      if(index[id_x]==-1)
	  {
	    if(index[id_y]!=-1) 
		  {
		    index[id_x] = index[id_y]*10  + id_x;
			depth[id_x] = depth[id_y] +1;
			stack[id_x] = 1;
		  }
	  }
   }
   __syncthreads();
if(mat[id_y*n+id_x]==1)
 {
	  if(index[id_y]==-1)
	  {
	    if(index[id_x]!=-1)
		  {
		    index[id_y] = index[id_x]*10 + id_y;
			depth[id_y] = depth[id_x] +1;
	        stack[id_y] = 1;
		  }
	  }
  }
__syncthreads();
}

__syncthreads();
for(int i=0;i<4;i++)
if(mat[id_y*n+id_x]==1)
{
  if(depth[id_x]!=depth[id_y])
  {
    if(depth[id_x] < depth[id_y] && index[id_x] < index[id_y]/10)
	{
	  if(id_y>99 && id_y<1000) index[id_y]=index[id_x]*1000+id_y;
	  else if(id_y>9 && id_y<100) index[id_y]=index[id_x]*100+id_y;
	  else
	  index[id_y]=index[id_x]*10+id_y;
	}
	__syncthreads();
    if(depth[id_y] < depth[id_x] && index[id_y] < index[id_x]/10)
	{
	  if(id_x>99 && id_x<1000) index[id_x]=index[id_y]*1000+id_x;
	  else if(id_x>9 && id_x<100) index[id_x]=index[id_y]*100+id_x;
	  else
	  index[id_x]=index[id_y]*10+id_x;
	}
	__syncthreads();
  }
}

result[id_x] = 0;
__syncthreads();
for(int i=0;i<n;i++)
{
  if(index[id_x] > index[i])
  {
    result[id_x]++;
  }
}
__syncthreads();
}


int main()
{
int n;
int i,j,k,l;

cout<<"Enter the number of vertices: ";
cin>>n;

int *h_mat = new int[n*n*sizeof(int)];
int *d_mat;
int *d_stack;

if(cudaMalloc(&d_mat,n*n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	return 0;
  }

if(cudaMalloc(&d_stack,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	return 0;
  }

cudaMemset(d_stack, -1, n*sizeof(int));
cudaMemset(d_stack, -1, n*sizeof(int));

for(i=0;i<(n*n);i++)
h_mat[i]=0;

for(i=0;i<n;i++)
  {
    cout<<"\nEnter the number of connections of "<<i<<" vertex: ";
	cin>>j;

	cout<<"\nEnter the vertices which are connected to "<<i<<" :\n";
	for(k=0;k<j;k++)
	  {
	    cin>>l;
	    h_mat[(i * n) + l] = 1;
      }
  }

for(i=0;i<n;i++)
  {
    for(j=0;j<n;j++)
	  cout<<h_mat[i * n + j]<<"  ";
	cout<<endl;
  }

int *h_output = new int[sizeof(int)*n];
int *d_output;
if(cudaMalloc(&d_output,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	return 0;
  }

int *d_depth;
if(cudaMalloc(&d_depth,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	return 0;
  }

int *d_index;
if(cudaMalloc(&d_index,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	return 0;
  }

cudaMemset(d_output, -1, n*sizeof(int));
cudaMemset(d_depth, -1, n*sizeof(int));
cudaMemset(d_index, -1, n*sizeof(int));

int h_vertex;
cout << "Enter the starting vertex: ";
cin >> h_vertex;


//h_output[0] = h_vertex;
for(i=0;i<n*sizeof(int);i++)
h_output[i] = -1;
 
if(cudaMemcpy(d_mat, h_mat, n*n*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cout<<"Memory copy from host to device failed.88";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	cudaFree(d_depth);
	return 0;
  }

if(cudaMemcpy(d_output,h_output,n*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cout<<"Memory copy from host to device failed.97";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	cudaFree(d_depth);
	return 0;
  }


 
depth<<<n,n>>>(d_mat, d_stack, d_index,h_vertex,n,d_output, d_depth);


int *h_depth = new int[n*sizeof(int)];
if(cudaMemcpy(h_depth,d_index,n*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    cout<<"Memory copy from device to host failed.126";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	cudaFree(d_depth);
	return 0;
  } 

cout<<"depth is:\n";
for(i=0;i<n;i++)
{
cout<<h_depth[i]<< " for " << i << endl;
}

 
if(cudaMemcpy(h_output,d_output,n*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    cout<<"Memory copy from device to host failed.126";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_output);
	cudaFree(d_depth);
	return 0;
  } 
 

 cout<<"Output bfs is: ";
 for(i=0;i<n;i++)
 cout<<h_output[i]<<"  ";

cin >> i;
cudaFree(d_depth);
cudaFree(d_mat);
cudaFree(d_stack);
cudaFree(d_output);
return 0;
}

In the above program all the connecting points are stored in an matrix, as 1 given to a point if it’s x and y coordinates are connected and all the rest not connected vertex are given 0, from this matrix we get which pint is connected to which other points. Apart from this we also have a index array of size of total number of vertex.This BFS program  uses a kernel with n blocks with each having n threads(say n be the total number of vertices) to calculate a index to each vertex which holds a number that stores the connecting points to it in sequence. For example if a vertex numbered as 5 is connected to let it be 3 and 3 is connected to 1 which is also our starting vertex then vertex 5 will be given a index of 135. A benefit of this is that if it also provides us the depth of the vertex as the number of digits in index, but in the above program the depth is separately calculated in an array along with the calculation of index, though not used. But the only problem is if a vertex is connected to more than one vertices from the above depth then it always stores the value of the vertex with highest numerical value in the index, it is a problem because if it would have stored the least numerical value vertex in it’s index if we only arrange the vertices according to its index number then it would give us the BFS of the given input. This problem was later solved in program inside kernel after calculation of initial index, and after rearrangement finally index stores the value if index according to its least numeral connected vertex. So now all need to be done is to arrange the vertices in ascending order according to it’s index number, and that will give us the BFS of the input. 
(I have used cin>>i before every return because getch() was not working even after including stdio.h so to keep the output screen I have used cin>>i insted of getch().)
