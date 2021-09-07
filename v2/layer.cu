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

	cudaMalloc(&b_output, sizeof(float) * bytes);
	cudaMalloc(&b_preact, sizeof(float) * bytes);
	cudaMalloc(&b_weight, sizeof(float) * M * N);

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

	cudaFree(b_output);
	cudaFree(b_preact);
	cudaFree(b_weight);

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
	cudaMemset(b_output, 0x00, sizeof(float) * bytes);
	cudaMemset(b_preact, 0x00, sizeof(float) * bytes);
	cudaMemset(b_weight, 0x00, sizeof(float) * M * N);
}

__device__ float ReLU(float r){
	if(r<=0) return 0;
	else return r;
}

__device__ float dReLU(float r){
	if(r<=0) return 0;
	else return 1;
}

__device__ float sigmoid(float s){
    return 1/(1 + exp(-s));
}

__device__ float sigmoidPrime(float s){
	float o = sigmoid(s);
	return o * (1 - o);
}

__global__ void apply_sigmoid(float *input, float *output, const int N){
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    // TODO:
    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = sigmoid(input[idx]);
	}
}

__global__ void loss_func(float *err, float *output, const int Y, const int N)
{
	// int l = blockIdx.x;  // 1
	int x = threadIdx.x; // 10

	if(x == Y) err[x] = 1.0f - output[x];
	else err[x] = 0.0f - output[x];

	// err[x] = ((x==Y ? 1.0f : 0.0f) - output[x]);
}

#define TILE_WIDTH 16

//__constant__ float conv_input[128 * 128];
//input_pointer,  Output_pointer, W_pointer, Inputimage_channel, Inputimage_height, Inputimage_width , Outputimage_width, W_width_height, Outputimage_channel
__global__ void ConvLayerForward_Kernel_1(float input[28][28], float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M){
// __global__ void ConvLayerForward_Kernel(float output[6][24][24], float weight[6][5][5], float bias[6], int C, int H_in, int W_in, int W_out, int K, int M){

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
                    // acc += conv_input[(x+p)*W_in + (y+q)] * weight[m][p][q];
					//acc = acc + X[n*(C*H_in*W_in) + c*(H_in*W_in) + (h+p)*(W_in) + (w+q)] * W[m*(C*K*K) + c*(K*K) + p*(K) + q];
	}
	__syncthreads();
	if(x < H_out && y < W_out)
	{
        output[m][x][y] = acc + bias[m];
    }
}

__global__ void ConvLayerBackward_Kernel(
	float l_c1_preact[6][24][24],
	float l_c1_b_output[6][24][24],
	float l_c1_weight[6][5][5],
	float l_input_output[28][28],
	float l_c1_bias[6]
){
	__shared__ float l_c1_b_preact[24][24];
	__shared__ float l_c1_b_weight[5][5];

	// int l = blockIdx.x;
	int m = blockIdx.y;  // 6
	int x = threadIdx.x;  // 24
	int y = threadIdx.y;  // 24
	// int z = threadIdx.z;

	if(x<5 && y <5)
		l_c1_b_weight[x][y] = 0;
	__syncthreads();

	// float o = sigmoid(l_c1_preact[m][x][y]);
	l_c1_b_preact[x][y] = l_c1_b_output[m][x][y] * sigmoidPrime(l_c1_preact[m][x][y]);

	int i, j;
	for(i=0; i<5; i++){
		for(j=0; j<5; j++){
			l_c1_b_weight[i][j] += l_c1_b_preact[x][y] * l_input_output[x + i][y + j] / (24*24);
		}
	}

	l_c1_bias[m] += lr * l_c1_b_preact[x][y] / (6*24*24);

	if(m==0 && x<5 && y<5)
		l_c1_weight[m][x][y] += lr * l_c1_b_weight[x][y];
}

// input_pointer, output_pointer, inputimage_height, inputimage_width, outputimage_channel, pool_size 
__global__ void PoolLayerForward_Kernel(float input[6][24][24], float output[6][6][6], float weight[1][4][4], float bias[1] ,int H_in, int W_in, int M, int pool_size){
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	// int W_grid = ceilf((float)W_out/TILE_WIDTH);
	// if(W_grid==0){
    //     W_grid = 1;
    // }
		
	// // int l = blockIdx.x;
	// int m = blockIdx.y;
	// int x = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	// int y = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	

	// int l = blockIdx.x;
	// int m = blockIdx.y;
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6

	
	float acc = 0;
	int p, q;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(y < H_out && z < W_out)
				// acc = acc + input[l*(M*H_in*W_in)+ x*(H_in*W_in) +
				//               (pool_size * y + p)*(W_in) + (pool_size * z + q)] / (pool_size * pool_size);
                acc += input[x][pool_size * y+p][pool_size * z+q] * weight[0][p][q];
	}
	__syncthreads();
	if(y < H_out && z < W_out) {
		// Y[n*(M*H_out*W_out)+ x*(H_out*W_out) + h*(W_out) + w] = acc;
		output[x][y][z] = acc + bias[0];
	}
}

__global__ void PoolLayerBackward_Kernel(
	float l_p_preact[6][6][6],
	float l_p_b_output[6][6][6],
	float l_p_b_weight[1][4][4],
	float l_p_weight[1][4][4],
	float l_c1_output[6][24][24],
	float l_c1_b_output[6][24][24],
	float l_p_bias[1]
){
	__shared__ float l_p_b_preact[6][6][6];
	// __shared__ float l_p_b_weight[4][4];
	// int l = blockIdx.x;
	// int m = blockIdx.y;
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6
	
	// if(x<1 && y<4 && z <4)
	// 	l_p_b_weight[y][z] = 0;
	// __syncthreads();

	// float o = sigmoid(l_p_preact[x][y][z]);
	l_p_b_preact[x][y][z] = l_p_b_output[x][y][z] * sigmoidPrime(l_p_preact[x][y][z]);

	l_p_bias[0] += lr * l_p_b_preact[x][y][z]/(6*6*6);

	int i,j;
	for(i=0; i<4; i++) {
		for(j=0; j<4; j++) {
			// l_p_b_weight[0][i][j] += l_p_b_preact[x][y][z] * l_c1_output[x][y*4+i][z*4+j];
			// l_c1_b_output[x][y*4+i][z*4+j] = l_p_weight[0][i][j] * l_p_b_preact[x][y][z];

			atomicAdd(&l_p_b_weight[0][i][j], l_p_b_preact[x][y][z] * l_c1_output[x][y*4+i][z*4+j]);
			atomicAdd(&l_c1_b_output[x][y*4+i][z*4+j], l_p_weight[0][i][j] * l_p_b_preact[x][y][z]);
		}
	}
	__syncthreads();

	if(x==0 && y<4 && z<4)
		l_p_weight[0][y][z] += lr * l_p_b_weight[0][y][z];
}

__global__ void AvgPoolLayerForward_Kernel(float input[6][24][24], float output[6][6][6], int H_in, int W_in, int M, int pool_size){
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6

	
	float acc = 0;
	int p, q;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(y < H_out && z < W_out)
                acc += input[x][y*pool_size + p][z*pool_size + q];
	}
	// __syncthreads();
	if(y < H_out && z < W_out) {
		output[x][y][z] = acc / (pool_size*pool_size);
	}
}


__global__ void AvgPoolLayerBackward_Kernel(
	float input[6][6][6],
	float output[6][24][24]
){
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6

	int i,j;
	for(i=0; i<4; i++) {
		for(j=0; j<4; j++) {
			output[x][y*4+i][z*4+j] = input[x][y][z] / (4*4);
		}
	}
}

__global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out) {
	// int n = blockIdx.x;
	int m = blockIdx.y;  // 10
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6

	float Pvalue = 0;
	int o, p, q;
	for (o = 0; o < 6; o++) {
		for (p = 0; p < 6; p++) {
			for (q = 0; q < 6; q++){
				if(x < 6 && y < 6 && z < 6)
				// Pvalue += input[x][y][z] * weight[m][x+o][y+p][z+q];
				Pvalue += input[o][p][q] * weight[m][o][p][q];
			}
		}
	}
	__syncthreads();

    if(m < 10 && x < 1 && y < 1 && z < 1)
		output[m] = Pvalue + bias[m]; // Output
}

// __global__ void FullyConLayerForward_kernel(float input[6][6][6], float weight[10][6][6][6], float output[10], float bias[10], int H_in, int W_in, int W_we , int H_out, int W_out) {
// 	// int n = blockIdx.x;
// 	// int m = blockIdx.y;  // -
// 	int x = threadIdx.x;  // 6
// 	int y = threadIdx.y;  // 6
// 	int z = threadIdx.z;  // 6

// 	__shared__ float Pvalue[10];
// 	if((x+y)<=10 && x<6 && y<6 && z<0)
// 		Pvalue[x+y] = 0;
// 	__syncthreads();

// 	for(int i=0; i<10; i++){
// 		Pvalue[i] += input[x][y][z] * weight[i][x][y][z];
// 	}

// 	if((x+y)<=10 && x<6 && y<6 && z<0)
// 		output[x+y] = Pvalue[x+y] + bias[x+y];
// }

__global__ void FullyConLayerBackward_kernel(
	float l_f_b_preact[10],
	float l_f_bias[10],
	float l_f_weight[10][6][6][6],
	float l_p_output[6][6][6],
	float l_p_b_output[6][6][6]
){
	__shared__ float l_f_b_weight[6][6][6];
	
	// int l = blockIdx.x;
	int m = blockIdx.y;  // 10
	int x = threadIdx.x;  // 6
	int y = threadIdx.y;  // 6
	int z = threadIdx.z;  // 6

	l_f_b_weight[x][y][z] = l_f_b_preact[m] * l_p_output[x][y][z];
	// l_p_b_output[x][y][z] += l_f_weight[m][x][y][z] * l_f_b_preact[m];

	atomicAdd(&l_p_b_output[x][y][z], l_f_weight[m][x][y][z] * l_f_b_preact[m]);
	if(x==0 && y==0 && z==0 )
		l_f_bias[m] += lr * l_f_b_preact[m];

	l_f_weight[m][x][y][z] += lr * l_f_b_weight[x][y][z];
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
// 	float l_f_b_preact[10],
// 	float ls1_preact[6][6][6],
// 	float lf_weight[10][6][6][6],
// 	float lf_b_weight[10][6][6][6],
// 	float lf_bias[10]
// ) {
// 	// int l = blockIdx.x;
// 	int m = blockIdx.y;  // 10
// 	int x = threadIdx.x;  // 6
// 	int y = threadIdx.y;  // 6
// 	int z = threadIdx.z;  // 6

// 	l_f_b_preact[m] *= lf_output[m] * (1- lf_output[m]);
// 	__syncthreads();
// 	// ls1_b_preact[m] = l_f_b_preact[m] * lf_output[m] * (1- lf_output[m]);

// 	lf_bias[m] += lr + l_f_b_preact[m];
	
// 	lf_b_weight[m][x][y][z] = l_f_b_preact[m] * ls1_preact[x][y][z] ;
// 	lf_b_weight[m][x][y][z] += lf_weight[m][x][y][z];
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
// 	float b_output[6][24][24], 
// 	float preact[6][24][24], 
// 	float b_preact[6][24][24], 
// 	float b_weight[6][5][5], 
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
	
// 	// float dv = b_output[m][x][y] * o * (1 - o);
// 	b_preact[m][x][y] = b_output[m][x][y] * o * (1 - o);
// 	__syncthreads();

// 	for (c = 0; c < C; c++) {
// 		for (p = 0; p < K; p++) {
// 			for (q = 0; q < K; q++) {
// 				if(x < H_out && y < W_out) {
// 					b_weight[m][p][q] = b_preact[m][x][y] * input[28][28]/d;
// 				}
// 			}
// 		}
// 	}
// }




