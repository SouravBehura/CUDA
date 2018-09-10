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