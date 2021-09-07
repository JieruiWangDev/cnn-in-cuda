#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

// includes, system
#include <string>


#ifndef LAYER_H
#define LAYER_H



const static float lr = 0.1f;
const static float threshold = 0.01f;

class Layer{
    public:
        int M;
        int N;
        int bytes;

        float *output;
        float *preact;

        float *bias;
        float *weight;

        float *d_output;
        float *d_preact;
        float *d_weight;

        Layer(int in_width, int in_height, int bytes);
        ~Layer(); // free the memory allocation

        void setInput(float *data); // set data
        void clear();     //reset the GPU memory
        void bp_clear();  
};

// Layer::Layer(int in_width, int in_height, int in_depth): width(in_width), height(in_height), size(in_size){

// }

__device__ float sigmoid(float v);
__global__ void apply_sigmoid(float *input, float *output, const int N);
__global__ void backward_sigmoid(float* X, int size_in);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);

__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M);
__global__ void ConvLayerBackward_Kernel(
	float input[28][28], 
	float d_output[6][24][24], 
	float preact[6][24][24], 
	float d_preact[6][24][24], 
	float d_weight[6][5][5], 
	int C, int H_in, int W_in, int W_out, int K, int M);

//pooling
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ,int H_in, int W_in, int M, int pool_size);

// FullyConnect

__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out);


__global__ void bp_f(
	float l_f_d_weight[10][6][6][6],
	float l_f_d_preact[10],
	float l_f_bias[10],
	float l_f_weight[10][6][6][6],
	float l_s1_output[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6]);

__global__ void bp_s1(
	float l_s1_preact[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6],
	float l_s1_d_weight[1][4][4],
	float l_s1_weight[1][4][4],
	float l_c1_output[6][24][24],
	float l_c1_d_output[6][24][24],
	float l_s1_bias[6]);

__global__ void bp_c1(
	float l_c1_preact[6][24][24],
	float l_c1_d_preact[6][24][24],
	float l_c1_d_output[6][24][24],
	float l_c1_d_weight[6][5][5],
	float l_c1_weight[6][5][5],
	float l_input_output[28][28],
	float l_c1_bias[6]);



#endif

