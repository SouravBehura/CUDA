
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

const double learning_rate = 2;

__global__ void assign_weights(double *d, int size)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * size) + i;
	d[index] = 0;
}

__global__ void mat_mull(double* weights, double* prev_activation, int size)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * size) + i;
	weights[index] = weights[index] * prev_activation[i];
	//weights[index] = weights[index] / size;
}

__global__ void reduce_mat(double* weights, int size_block, int size)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * size_block) + i;
	weights[index] = weights[index] + weights[size + index];
}

__global__ void rectify(double* weights, int size_block, int i)
{
	int j = threadIdx.x;
	int index = (j * size_block) + i;
	weights[(j * size_block)] = weights[index] + weights[(j * size_block)];
}

__global__ void add_bias(double* activation, double* weights, double* bias, int prev_neurons)
{
	int i = threadIdx.x;
	int index = (i * prev_neurons);
	activation[i] = weights[index] + bias[i];
	//activation[i] = activation[i] / prev_neurons;
}

__global__ void sigmoid(double* activation)
{
	int i = threadIdx.x;
	activation[i] = pow(2.72, activation[i]);
	activation[i] = activation[i] / (1 + activation[i]);
}

__global__ void update_weights(double* target, double* next_neurons, double* weights, double* bias, double* cost, int learning_rate, int no_neurons, int no_next_neuron)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * no_neurons) + i;
	cost[j] = target[j] - next_neurons[j];
	double B = cost[j] * (1 - next_neurons[j]) * next_neurons[j];
	double C = B / cost[j];
	weights[index] = weights[index] - (learning_rate * cost[j] * B);
	bias[i] = bias[i] - (learning_rate * C);
}

__global__ void update_weights_output (double* target, double* activation, double* weights, double* bias, double* cost, int learning_rate, int neurons, int prev_no_neurons)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * prev_no_neurons) + i;
	cost[j] = target[j] - activation[j];
	double B = cost[j] * (1 - activation[j]) * activation[j];
	double C = B / cost[j];
	weights[index] = weights[index] - (learning_rate * cost[j] * B);
	bias[i] = bias[i] - (learning_rate * C);
}

__global__ void calc_target (double* next_neurons, double* target, double* next_weights, int no_neurons)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int index = (j * no_neurons) + i;
	target[i] = next_weights[index] * next_neurons[i];
}

__global__ void average(double* target, int next_no_neurons)
{
	int i = threadIdx.x;
	target[i] = target[i] / next_no_neurons;
}

class neuron_layer
{
private:
	int neurons;
	int no_weights;
	double* weights;
	double* activation;
	double* bias;
public:
	neuron_layer(int no_neuron)
	{
		neurons = no_neuron;
		activation = new double[neurons];
		bias = new double[neurons];
		for (int i = 0; i < neurons; i++)
		{
			bias[i] = 0;
		}
	}

	neuron_layer(int no_neuron,neuron_layer prev_layer)
	{
		int prev_neurons = prev_layer.get_number_neurons();
		neurons = no_neuron;
		activation = new double[neurons];
		bias = new double[neurons];
		no_weights = neurons * prev_neurons;
		weights = new double[neurons * prev_neurons];
		for (int i = 0; i < neurons; i++)
		{
			bias[i] = 0;
		}
		double* dev_weights = 0;
		if (cudaMalloc((void**)&dev_weights, neurons * prev_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed.\n";
			cudaFree(dev_weights);
		}

		assign_weights<<<neurons,prev_neurons>>>(dev_weights,prev_neurons);

		if (cudaMemcpy(weights, dev_weights, neurons*prev_neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host weight assignment failed.\n";
			cudaFree(dev_weights);
		}

		//cout << "\nWeights = ";
		/*for (int i = 0; i < neurons*prev_neurons; i++)
		{
			cout << weights[i] << ",";
		}
		cout << endl;*/

		cudaFree(dev_weights);
	}

	void print_layer()
	{
		/*cout << "Activation values = ";
		for (int i = 0; i < neurons; i++)
		{
			cout<< activation[i] << ",";
		}*/
		cout << "\nBias values = ";
		for (int i = 0; i < neurons; i++)
		{
			cout << bias[i] << ",";
		}
		cout << "\nWeight values = ";
		for (int i = 0; i < 10; i++)
			cout << weights[i] << ",";
	}

	void input(double* image)
	{
		/*double* dev_activation;
		if (cudaMalloc((void**)& dev_activation, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda image assignment failed during activation.\n";
			cudaFree(dev_activation);
		}
		if (cudaMemcpy(dev_activation, image, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device image assignment failed during activation.\n";
			cudaFree(dev_activation);
		}
		if (cudaMemcpy(activation, dev_activation, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from yDevice To Host image assignment failed during activation.\n";
			cudaFree(dev_activation);
		}
		cudaFree(dev_activation);*/
		activation = image;
	}

	int get_number_neurons()
	{
		return neurons;
	}

	int get_no_weights()
	{
		return no_weights;
	}

	(double*)get_activation()
	{
		return activation;
	}

	(double*)get_weights()
	{
		return weights;
	}

	void activate_layer(neuron_layer prev_layer)
	{
		int prev_neurons = prev_layer.get_number_neurons();
		double* dev_weights = 0;
		double* dev_activation = 0;
		double* dev_bias = 0;
		double* dev_prev_activation = 0;
		if(cudaMalloc((void**)& dev_weights, neurons * prev_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMalloc((void**)& dev_activation, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda activation assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMalloc((void**)& dev_bias, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda bias assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMalloc((void**)& dev_prev_activation, prev_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda bias assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMemcpy(dev_weights, weights, neurons * prev_neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device weight assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMemcpy(dev_bias, bias, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device bias assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		if (cudaMemcpy(dev_prev_activation, prev_layer.get_activation(), prev_neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device previous layer activation assignment failed during activation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}

		mat_mull<<<neurons, prev_neurons>>>(dev_weights, dev_prev_activation, prev_neurons);

		/*if (cudaMemcpy(weights, dev_weights, prev_neurons * neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host activation assignment failed.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}

		cout << "\nWeights = ";
		for (int i = 0; i < neurons * prev_neurons; i++)
		{
			cout << weights[i] << ",";
		}
		cout << endl;*/

		if (prev_neurons % 2 == 1)
		{
			rectify <<<1, neurons >>> (dev_weights, prev_neurons, (prev_neurons-1));
		}
		
		/*if (cudaMemcpy(weights, dev_weights, prev_neurons * neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host activation assignment failed.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}

		cout << "\nWeights = ";
		for (int i = 0; i < neurons * prev_neurons; i++)
		{
			cout << weights[i] << ",";
		}
		cout << endl;*/

		int i = prev_neurons / 2;
		while (i >= 1)
		{
			reduce_mat <<<neurons, i >>> (dev_weights, prev_neurons, i);
			if (((i % 2) == 1) && i != 1)
			{
				rectify <<<1, neurons >>> (dev_weights, prev_neurons, i);
			}
			i = i / 2;
		}

		/*if (cudaMemcpy(weights, dev_weights, prev_neurons * neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host activation assignment failed.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}

		cout << "\nWeights = ";
		for (int i = 0; i < neurons * prev_neurons; i++)
		{
			cout << weights[i] << ",";
		}
		cout << endl;*/

		add_bias <<<1, neurons >>> (dev_activation, dev_weights, dev_bias, prev_neurons);

		sigmoid <<<1, neurons >>> (dev_activation);

		if (cudaMemcpy(activation, dev_activation, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host activation assignment failed.\n";
			cudaFree(dev_weights);
			cudaFree(dev_activation);
			cudaFree(dev_bias);
			cudaFree(dev_prev_activation);
		}
		cudaFree(dev_weights);
		cudaFree(dev_activation);
		cudaFree(dev_bias);
		cudaFree(dev_prev_activation);
	}

	void activate_input_layer()
	{
		double* dev_activation;
		if (cudaMalloc((void**)& dev_activation, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda activation assignment failed during first layer activation.\n";
			cudaFree(dev_activation);
		}
		if (cudaMemcpy(dev_activation, activation, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device previous layer activation assignment failed during first layer activation.\n";
			cudaFree(dev_activation);
		}

		sigmoid <<<1, neurons >>> (dev_activation);

		if (cudaMemcpy(activation, dev_activation, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from cuda to host activation assignment failed during first layer activation.\n";
			cudaFree(dev_activation);
		}
		cudaFree(dev_activation);
	}

	void back_propagate_output(double* target, neuron_layer prev_layer)
	{
		int prev_no_neurons = prev_layer.get_number_neurons();
		//double* next_neurons = new double[next_no_neurons];
		//next_neurons = next_layer.get_activation();
		double* cost = new double[10];
		double* dev_weights;
		double* dev_target;
		double* dev_activation;
		double* dev_cost;
		double*dev_bias;
		if (cudaMalloc((void**)& dev_weights, neurons * prev_no_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMalloc((void**)& dev_bias, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMalloc((void**)& dev_target, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda target assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMalloc((void**)& dev_activation, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda neurons assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMalloc((void**)& dev_cost, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(dev_weights, weights, neurons * prev_no_neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(dev_target, target, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device target assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(dev_activation, activation, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device next neuron assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(dev_bias, bias,neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}

		update_weights_output <<<neurons, prev_no_neurons>>> (dev_target, dev_activation, dev_weights, dev_bias, dev_cost, learning_rate, neurons, prev_no_neurons);
		
		if (cudaMemcpy(weights, dev_weights, neurons * prev_no_neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from devide to host weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(cost, dev_cost, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from devide to host weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		if (cudaMemcpy(bias, dev_bias, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from device to host bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_activation);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
		}
		cudaFree(dev_weights);
		cudaFree(dev_target);
		cudaFree(dev_activation);
		cudaFree(dev_cost);
		cudaFree(dev_bias);
	}

	void back_propagate(neuron_layer next_layer)
	{
		int next_no_neurons = next_layer.get_number_neurons();
		int next_no_weights = next_layer.get_no_weights();
		int next_next_neurons = next_no_weights / next_no_neurons;
		double* next_neurons = new double[next_no_neurons];
		next_neurons = next_layer.get_activation();
		double* cost = new double[10];
		double* dev_weights;
		double* dev_target;
		double* dev_next_neurons;
		double* dev_cost;
		double* dev_bias;
		double* dev_next_weights;
		if (cudaMalloc((void**)& dev_weights, neurons * next_no_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMalloc((void**)& dev_next_weights, next_no_weights * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda next weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMalloc((void**)& dev_bias, neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMalloc((void**)& dev_target, next_no_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda target assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMalloc((void**)& dev_next_neurons, next_no_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda neurons assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMalloc((void**)& dev_cost, next_no_neurons * sizeof(double)) != cudaSuccess)
		{
			cout << "Memory allocation for cuda weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(dev_weights, weights, neurons * next_no_neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(dev_next_weights, next_layer.get_weights(), next_no_weights * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device next layer weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(dev_next_neurons, next_neurons, next_no_neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device next neuron assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(dev_bias, bias, neurons * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
			cout << "Memory copy from host to device bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}

		calc_target <<<next_no_neurons , next_next_neurons>>> (dev_next_neurons, dev_target, dev_next_weights, neurons);

		average <<<1, next_no_neurons >>> (dev_target, next_no_neurons);

		update_weights <<<next_no_neurons, neurons >>> (dev_target, dev_next_neurons, dev_weights, dev_bias, dev_cost, learning_rate, neurons, next_no_neurons);

		if (cudaMemcpy(weights, dev_weights, neurons * next_no_neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from devide to host weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(cost, dev_cost, next_no_neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from devide to host weight assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		if (cudaMemcpy(bias, dev_bias, neurons * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "Memory copy from device to host bias assignment failed during back propagation.\n";
			cudaFree(dev_weights);
			cudaFree(dev_target);
			cudaFree(dev_next_neurons);
			cudaFree(dev_cost);
			cudaFree(dev_bias);
			cudaFree(dev_next_weights);
		}
		/*cudaFree(dev_weights);
		cudaFree(dev_target);
		cudaFree(dev_next_neurons);
		cudaFree(dev_cost);
		cudaFree(dev_bias);
		cudaFree(dev_next_weights);*/
	}
};

class training_data
{
private:
	double* image = new double[784 * sizeof(double)];
	double* label = new double[10 * sizeof(double)];
public:
	training_data()
	{
		for (int i = 0; i < 10; i++)
			label[i] = 0;
	}
	void get_image(int i, double data)
	{
		image[i] = data;
	}
	void get_label(int data)
	{
		label[data] = 1;
	}
	(double*)give_image()
	{
		return image;
	}
	(double*)give_label()
	{
		return label;
	}
	void print_img()
	{
		for (int i = 0; i < 784; i++)
			cout << image[i] << " ";
		cout << endl;
		for (int i = 0; i < 10; i++)
			cout << label[i] << " ";
		cout << endl;
	}
}training_set[60000];

int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images()
{
	ifstream file("train-images.idx3-ubyte", ios::in | ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					double data = temp;
					training_set[i].get_image(((r * 28) + c), data);
				}
			}
		}
	}
}

void read_mnist_labels()
{
	ifstream file("train-labels.idx1-ubyte", ios::in | ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)& temp, sizeof(temp));
			int data = temp;
			training_set[i].get_label(data);
		}
	}
}

int main()
{
	cout << "Do you want to train the model or load the weights(0 for 1st and 1 for 2nd option):";
	int dec;
	cin >> dec;
	neuron_layer input_layer(784);
	neuron_layer hidden_layer_1(16, input_layer);
	neuron_layer hidden_layer_2(16, hidden_layer_1);
	neuron_layer output_layer(10, hidden_layer_1);
	output_layer.print_layer();
	cout << endl << "--------------------------------------------------------";
	if (dec == 0)
	{
		read_mnist_images();
		read_mnist_labels();

		double* image = new double[784];
		double* target = new double[10];

		for (int x = 0; x <= 0; x++)
		{
			image = training_set[x].give_image();
			target = training_set[x].give_label();
			input_layer.input(image);
			input_layer.activate_input_layer();
			hidden_layer_1.activate_layer(input_layer);
			hidden_layer_2.activate_layer(hidden_layer_1);
			output_layer.activate_layer(hidden_layer_2);
			output_layer.back_propagate_output(target, hidden_layer_2);
			cout << "0";
			hidden_layer_2.back_propagate_output(target, output_layer);
			cout << "1";
		    hidden_layer_1.back_propagate(hidden_layer_2);
			cout << "2";
		}
	}
	else if (dec == 1)
	{
		cout << "Still in progress.\nYou can try later when it is developed.";
		return 0;
	}
	
	/*input_layer.print_layer();
	cout << "\nEND\n";
	hidden_layer_1.print_layer();
	cout << "\nEND\n";
	hidden_layer_2.print_layer();
	cout << "\nEND\n";*/
	output_layer.print_layer();
	cout << "\nEND\n";
	

	return 0;
}
