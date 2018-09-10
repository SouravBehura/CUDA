
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

__global__ void depth(int *mat, int *stack, int *index, int start, int n, int *depth)
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

int *d_depth;
if(cudaMalloc(&d_depth,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	return 0;
  }

int *d_index;
if(cudaMalloc(&d_index,n*sizeof(int)) != cudaSuccess)
  {
    cout<<"Memory allocation failed.";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	return 0;
  }

cudaMemset(d_depth, -1, n*sizeof(int));
cudaMemset(d_index, -1, n*sizeof(int));

int h_vertex;
cout << "Enter the starting vertex: ";
cin >> h_vertex;

 
if(cudaMemcpy(d_mat, h_mat, n*n*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cout<<"Memory copy from host to device failed.88";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_depth);
	return 0;
  }

 
depth<<<n,n>>>(d_mat, d_stack, d_index,h_vertex,n, d_depth);


int *h_depth = new int[n*sizeof(int)];
if(cudaMemcpy(h_depth,d_index,n*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    cout<<"Memory copy from device to host failed.126";
	cin>>i;
	cudaFree(d_mat);
	cudaFree(d_stack);
	cudaFree(d_depth);
	return 0;
  } 

cout<<"depth is:\n";
for(i=0;i<n;i++)
{
cout<<h_depth[i]<< " for " << i << endl;
}

 for(i=0;i<n;i++)
 {
   h_output[i]=0;
   for(j=0;j<n;j++)
   {
     if(h_depth[i]>h_depth[j]) h_output[i]++;
   }
 }
 

 cout<<"Output bfs is: ";
 for(i=0;i<n;i++)
 cout<<h_output[i]<<"  ";

cin >> i;
cudaFree(d_depth);
cudaFree(d_mat);
cudaFree(d_stack);
return 0;
}