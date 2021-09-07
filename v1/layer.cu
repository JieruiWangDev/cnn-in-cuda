#include "layer.h"


#define TILE_WIDTH 16


// Layer constructor:
Layer::Layer(int in_width, int in_height, int in_size): M(in_width), N(in_height), bytes(in_size){

    float h_bias[N];
    float h_weight[N][M];


    output = NULL;
    preact = NULL;
    bias = NULL;
    weight = NULL;

    for (int i = 0; i < N; i++){
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);  // initial bias
        for (int j = 0; j < M; j++){
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);  // initial weight
        }
    }

    cudaMalloc(&output, sizeof(float) * bytes);
	cudaMalloc(&preact, sizeof(float) * bytes);

	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * bytes);
	cudaMalloc(&d_preact, sizeof(float) * bytes);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// de-constructor
Layer::~Layer(){

    // TODO: free cuda memory
    cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);

}

void Layer:: setInput(float *data){
    cudaMemcpy(output, data, sizeof(float)*bytes, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * bytes);
	cudaMemset(preact, 0x00, sizeof(float) * bytes);
}


void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * bytes);
	cudaMemset(d_preact, 0x00, sizeof(float) * bytes);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float sigmoid(float s){
    return 1/(1 + exp(-s));
}

__global__ void apply_sigmoid(float *input, float *output, const int N){
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    // TODO:
    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = sigmoid(input[idx]);
	}
}

// __global__ void backward_sigmoid(float* X, int size_in)
// {
// 	int t = blockIdx.x * 1024 + threadIdx.x;

// 	if(t < size_in)
// 	{
// 		double tmp = 1 / (1 + exp(-X[t]));
// 		tmp = (1-tmp)*tmp;
// 		X[t] = X[t]*tmp;
// 	}
// }


__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

#define TILE_WIDTH 16

//input_pointer,  Output_pointer, W_pointer, Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M){
    int H_out = H_in - K + 1;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	// int l = blockIdx.x;
	int m = blockIdx.y;
	int x = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	int y = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

	float acc = 0;
	int c, p, q;
	for (c = 0; c < C; c++) { // sum over all input channels
		for (p = 0; p < K; p++) // loop over KxK filter
			for (q = 0; q < K; q++)
				if(x < H_out && y < W_out)
                    acc += input[x+p][y+q] * weight[m][p][q];
					//acc = acc + X[n*(C*H_in*W_in) + c*(H_in*W_in) + (hx+p)*(W_in) + (y+q)] * W[m*(C*K*K) + c*(K*K) + p*(K) + q];
	}
	__syncthreads();
	if(x < H_out && y < W_out)
	{
        output[m][x][y] = acc + bias[m];
    }
}


// input_pointer, output_pointer, inputimage_height, inputimage_width, outputimage_channel, pool_size 
__global__ void MaxPool2dForward_Kernel_1(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ,int H_in, int W_in, int M, int pool_size){
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0){
        W_grid = 1;
    }
		
	// int l = blockIdx.x;
	int m = blockIdx.y;
	int x = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	int y = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	
	float acc = 0;
	int p, q;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(x < H_out && y < W_out)
				// acc = acc + input[l*(M*H_in*W_in)+ m*(H_in*W_in) +
				//               (pool_size * x + p)*(W_in) + (pool_size * y + q)] / (pool_size * pool_size);
                acc = acc + input[m][pool_size * x+p][pool_size * y+q] * weight[0][p][q];
	}
	__syncthreads();
	if(x < H_out && y < W_out)
	{
		// Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = acc;
		output[m][x][y] = acc + bias[0];
	}
}


__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out) {
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;

	// int n = blockIdx.x;
	int m = blockIdx.y;  // 10
	int h = threadIdx.x;  // 6
	int w = threadIdx.y;  // 6
	int y = threadIdx.z;  // 6

	float Pvalue = 0;
	int o, p, q;
	for (o = 0; o < 6; o++) {
		for (p = 0; p < 6; p++) {
			for (q = 0; q < 6; q++){
				if(h < 6 && w < 6 && y < 6)
				// Pvalue += input[y][h+p][w+q] * weight[m][y][h+p][w+q];
				// Pvalue += input[h][w][y] * weight[m][h+o][w+p][y+q];
				Pvalue+= input[o][p][q] * weight[m][o][p][q];
			}
		}
	}
	__syncthreads();

    if(m < W_out && h < 6 && w < 6 && y < 6)
		output[m] = Pvalue + bias[m]; // Output
}

// input_height, input_width, weight_width, output_height, output_width
//      1             6          10          1              10
// __global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out) {
// 	int W_grid = ceilf((float)W_out/TILE_WIDTH);
// 	if(W_grid==0)
// 		W_grid = 1;

// 	// int l = blockIdx.x;
// 	int m = blockIdx.y;  // 10
// 	// int x = threadIdx.x;
// 	// int y = threadIdx.y;
// 	// int z = threadIdx.z;

// 	float Pvalue = 0;
// 	int o, p, q;
// 	for (o = 0; o < 6; o++) {
// 		for (p = 0; p < 6; p++) {
// 			for (q = 0; q < 6; q++){
// 				Pvalue += input[o][p][q] * weight[m][o][p][q];
// 			}
// 		}
// 	}
// 	__syncthreads();

// 	if(m < W_out)
// 		output[m] = Pvalue + bias[m]; // Output

// 	// float Pvalue = 0;
// 	// for (int i = 0; i < 6; i++){
// 	// 	if(x < 6 && y < 6)
// 	// 	Pvalue += input[x][y][i] * weight[m][x][y][i];
// 	// }
// 	// // __syncthreads();

//     // if(m < 10 && x < 6 && y < 6)
// 	// 	atomicAdd(&output[m], Pvalue);
// 	// if(x==0 && y==0)
// 	// 	output[m] += bias[m];
// }


// __global__ void FullyConLayerBackward_kernel(
// 	float lf_output[10],
// 	float l_f_d_preact[10],
// 	float ls1_preact[6][6][6],
// 	float lf_weight[10][6][6][6],
// 	float lf_d_weight[10][6][6][6],
// 	float lf_bias[10]
// ) {
// 	// int l = blockIdx.x;
// 	int m = blockIdx.y;  // 10
// 	int x = threadIdx.x;  // 6
// 	int y = threadIdx.y;  // 6
// 	int z = threadIdx.z;  // 6

// 	l_f_d_preact[m] *= lf_output[m] * (1- lf_output[m]);
// 	__syncthreads();
// 	// ls1_d_preact[m] = l_f_d_preact[m] * lf_output[m] * (1- lf_output[m]);

// 	lf_bias[m] += lr + l_f_d_preact[m];
	
// 	lf_d_weight[m][x][y][z] = l_f_d_preact[m] * ls1_preact[x][y][z] ;
// 	lf_d_weight[m][x][y][z] += lf_weight[m][x][y][z];
// }


// //input_pointer, Inputimage_height, Inputimage_width, output_pointer, Outputimage_channel, pool_size
// __global__ void poolingLayer_backward_GPU(float input[6][24][24], int H_in, int W_in, float output[6][6][6], int M, int pool_size)

// {
// 	int H_out = H_in/pool_size;
// 	int W_out = W_in/pool_size;
// 	int W_grid = ceilf((float)W_out/TILE_WIDTH);
// 	if(W_grid==0)
// 		W_grid = 1;
// 	// int l = blockIdx.x;
// 	int m = blockIdx.y;
// 	int x = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
// 	int y = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

// 	//h and w is not center point of calculating, it's upper left corner point of Input image
// 	float acc = 0;
// 	for (int p = 0; p < pool_size; p++) { // loop over KxK input samples
// 		for (int q = 0; q < pool_size; q++)
// 			if(x < H_out && y < W_out)
// 			input[m][h+p][w+q] = output[m][x][y] / (pool_size * pool_size);
// 	}
// 	__syncthreads();

// }



// __global__ void ConvLayerBackward_Kernel(
// 	float input[28][28], 
// 	float d_output[6][24][24], 
// 	float preact[6][24][24], 
// 	float d_preact[6][24][24], 
// 	float d_weight[6][5][5], 
// 	int C, int H_in, int W_in, int W_out, int K, int M) {

//     int H_out = H_in - K + 1;
// 	int c, p, q;
// 	int W_grid = ceilf((float)W_out/TILE_WIDTH);
// 	if(W_grid==0)
// 		W_grid = 1;
// 	int l = blockIdx.x;
// 	int m = blockIdx.y;
// 	int x = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
// 	int y = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

// 	float d = 24.0f * 24.0f;

// 	float o = sigmoid(preact[m][x][y]);
	
// 	// float dv = d_output[m][x][y] * o * (1 - o);
// 	d_preact[m][x][y] = d_output[m][x][y] * o * (1 - o);
// 	__syncthreads();

// 	for (c = 0; c < C; c++) {
// 		for (p = 0; p < K; p++) {
// 			for (q = 0; q < K; q++) {
// 				if(x < H_out && y < W_out) {
// 					d_weight[m][p][q] = d_preact[m][x][y] * input[28][28]/d;
// 				}
// 			}
// 		}
// 	}
// }


__global__ void bp_f(
	float l_f_d_weight[10][6][6][6],
	float l_f_d_preact[10],
	float l_f_bias[10],
	float l_f_weight[10][6][6][6],
	float l_s1_output[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6]
){
	// int l = blockIdx.x;
	int m = blockIdx.y;  // 10
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6


	l_f_d_weight[m][x][y][z] = l_f_d_preact[m] * l_s1_output[x][y][z];
	// l_s1_d_output[x][y][z] += l_f_weight[m][x][y][z] * l_f_d_preact[m];

	atomicAdd(&l_s1_d_output[x][y][z], l_f_weight[m][x][y][z] * l_f_d_preact[m]);
	if(x==0 && y==0 && z==0 )
		l_f_bias[m] += lr * l_f_d_preact[m];

	l_f_weight[m][x][y][z] += lr * l_f_d_weight[m][x][y][z];
}

__global__ void bp_s1(
	float l_s1_preact[6][6][6],
	float l_s1_d_output[6][6][6],
	float l_s1_d_preact[6][6][6],
	float l_s1_d_weight[1][4][4],
	float l_s1_weight[1][4][4],
	float l_c1_output[6][24][24],
	float l_c1_d_output[6][24][24],
	float l_s1_bias[6]
){
	// int l = blockIdx.x;
	int m = blockIdx.y;  // 6
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	// int z = threadIdx.z;

	float o = sigmoid(l_s1_preact[m][x][y]);
	l_s1_d_preact[m][x][y] = l_s1_d_output[m][x][y] * o * (1 - o);

	// l_s1_d_preact[m][x][y] = l_s1_d_output[m][x][y] * l_s1_output[m][x][y] * (1 - l_s1_output[m][x][y]);
	__syncthreads();

	l_s1_bias[0] += lr * l_s1_d_preact[m][x][y]/(6*6*6);

	int i,j;
	for(i=0; i<4; i++) {
		for(j=0; j<4; j++) {
			// l_s1_d_weight[0][i][j] += l_s1_d_preact[m][x][y] * l_c1_output[m][h*4+i][w*4+j];
			// l_c1_d_output[m][h*4+i][w*4+j] += l_s1_weight[0][i][j] * l_s1_d_preact[m][x][y];

			atomicAdd(&l_s1_d_weight[0][i][j], l_s1_d_preact[m][x][y] * l_c1_output[m][x*4+i][y*4+j]);
			atomicAdd(&l_c1_d_output[m][x*4+i][y*4+j], l_s1_weight[0][i][j] * l_s1_d_preact[m][x][y]);
		}
	}

	if(m==0 && x<4 && y<4)
		l_s1_weight[0][x][y] += lr * l_s1_d_weight[0][x][y];
}


__global__ void bp_c1(
	float l_c1_preact[6][24][24],
	float l_c1_d_preact[6][24][24],
	float l_c1_d_output[6][24][24],
	float l_c1_d_weight[6][5][5],
	float l_c1_weight[6][5][5],
	float l_input_output[28][28],
	float l_c1_bias[6]
){
	// int l = blockIdx.x;
	int m = blockIdx.y;  // 6
	int x = threadIdx.x;  // 24
	int y = threadIdx.y;  // 24
	// int z = threadIdx.z;


	float o = sigmoid(l_c1_preact[m][x][y]);
	l_c1_d_preact[m][x][y] = l_c1_d_output[m][x][y] * o * (1 - o);

	int i, j;
	for(i=0; i<5; i++){
		for(j=0; j<5; j++){
			l_c1_d_weight[m][i][j] += l_c1_d_preact[m][x][y] * l_input_output[x + i][y + j] / (24*24);
		}
	}

	l_c1_bias[m] += lr * l_c1_d_preact[m][x][y] / (6*24*24);

	if(m==6 && x<5 && y<5)
		l_c1_weight[m][x][y] += lr * l_c1_d_weight[m][x][y];
}