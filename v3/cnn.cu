#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE


// includes, system
#include <string>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <time.h>

#include "layer.h"
#include "layer.cu"

struct mnist_data {
	double data[28][28];
	int label;  //0-9
};

// set Layer
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_p = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

// static Layer l_f1 = Layer(6*6*6, 10, 36);
// static Layer l_f2 = Layer(36, 1, 10);

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

unsigned int dataToInt(char* c) {
	unsigned int d = 0;
	for (int i = 0; i < 4; i++) {
		d <<= 8;
		d |= (unsigned char)c[i];
	}
	return d;
}

int mnist_load(
    const char *image_filename,
	const char *label_filename,
	mnist_data **data,
	unsigned int *count) 
{
    char tmp[4];
    unsigned char read_data[28*28];
    unsigned int im, l, i, j, k, ic1, ic2, image_cnt, label_cnt;

    FILE *ifp = fopen(image_filename, "rb");
	FILE *lfp = fopen(label_filename, "rb");

    if (!ifp || !lfp) {
        printf("file not open");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
        return -1;
    }

    fread(tmp, 1, 4, ifp);
	im = dataToInt(tmp);
	fread(tmp, 1, 4, lfp);
	l = dataToInt(tmp);
    fread(tmp, 1, 4, ifp);
	image_cnt = dataToInt(tmp);
	fread(tmp, 1, 4, lfp);
	label_cnt = dataToInt(tmp);

    fread(tmp, 1, 4, ifp);
	ic1 = dataToInt(tmp);
    fread(tmp, 1, 4, ifp);
	ic2 = dataToInt(tmp);

    // printf("im, l, image_cnt, label_cnt, ic1, ic2 \n");
    // printf("%d, %d, %d, %d, %d, %d \n", im, l, image_cnt, label_cnt, ic1, ic2);

    if(im != 2051 || l != 2049 || image_cnt != label_cnt || ic1 != 28 || ic2 != 28){
        printf("get wrong file");
        fclose(ifp);
        fclose(lfp);
        return -2;
    }

    *count = image_cnt;
	*data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

    for (i = 0; i < image_cnt; i++) {
        mnist_data *d = &(*data)[i];

        fread(read_data, 1, 28*28, ifp);
        for(j=0; j<28; j++){
            for(k=0; k<28; k++)
                d->data[j][k] = read_data[j*28+k]/255.0;
        }

        fread(tmp, 1, 1, lfp);
		d->label = tmp[0]%10;
    }
    fclose(ifp);
    fclose(lfp);
    return 0;
}

static inline void loadData(){
    clock_t t;
	t = clock();

    mnist_load("MNIST_data/train-images.idx3-ubyte", "MNIST_data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("MNIST_data/t10k-images.idx3-ubyte", "MNIST_data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);

    t = clock() - t;
	float load_time = (float)t/CLOCKS_PER_SEC;
    printf("loadData spend %.2f seconds \n", load_time);
}

static float forward(const double data[28][28]){

    // printf("run forward\n");

    
    float input[28][28];

    for (int i = 0; i<28; i++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
            // printf("%.2f ", data[i][j]);
        }
        // printf("\n");
    }

    l_input.clear();
	l_c1.clear();
	l_p.clear();
	l_f.clear();

    // printf("**************************************\n");


    //example for convLayer 1:

    l_input.setInput((float *)input);
    // cudaMemcpyToSymbol(conv_input, input, sizeof(float) * 28 * 28);

    //printf("input image: %f\n", &l_input.output[0][0]);


    //timer
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    

    int bz;
    bz = ceil((float)24/TILE_WIDTH)*ceil((float)24/TILE_WIDTH);
    dim3 gridDim(1, 6, bz);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //constant memory test
    // ConvLayerForward_Kernel<<<gridDim,blockDim>>>((float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, l_c1.bias, 1, 28, 28, 24, 5, 6);
    ConvLayerForward_Kernel_1<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, l_c1.bias, 1, 28, 28, 24, 5, 6);

    apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.bytes);

    // for pooling layer example:
    dim3 gridDimPool(1, 1, 1);
    dim3 blockDimPool(6, 6, 6);
    PoolLayerForward_Kernel<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_p.preact, (float (*)[4][4])l_p.weight, l_p.bias, 24, 24, 6, 4);
    // AvgPoolLayerForward_Kernel<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_p.preact, 24, 24, 6, 4);
    apply_sigmoid <<<64,64>>>(l_p.preact, l_p.output, l_p.bytes);

    // for fully connected layer
    dim3 gridDimfc(1, 1, 1);
    dim3 blockDimfc(10, 1, 1);
    FullyConLayerForward_kernel<<<gridDimfc,blockDimfc>>>((float (*)[6][6])l_p.output, (float (*)[6][6][6])l_f.weight, l_f.preact, l_f.bias, 1, 6, 10, 1, 10);
	apply_sigmoid<<<64, 64>>>(l_f.preact, l_f.output, l_f.bytes);


    //end timer:
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); // after cudaEventRecord
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return time;
}

static float backward(){
    //timer
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    
    dim3 gridDimfc(1, 10, 1);
    dim3 blockDimfc(6, 6, 6);
    FullyConLayerBackward_kernel<<<gridDimfc, blockDimfc>>>(
        l_f.b_preact,
        l_f.bias,
        (float (*)[6][6][6]) l_f.weight,
        (float (*)[6][6])l_p.output,
        (float (*)[6][6])l_p.b_output);

    
    dim3 gridDims(1, 1, 1);
    dim3 blockDims(6, 6, 6);
    PoolLayerBackward_Kernel<<<gridDims, blockDims>>>(
        (float (*)[6][6])l_p.preact,
        (float (*)[6][6])l_p.b_output,
        (float (*)[4][4])l_p.b_weight,
        (float (*)[4][4])l_p.weight,
        (float (*)[24][24])l_c1.output,
        (float (*)[24][24])l_c1.b_output,
        l_p.bias);
    // AvgPoolLayerBackward_Kernel<<<gridDims, blockDims>>>(
    //     (float (*)[6][6])l_p.preact,
    //     (float (*)[24][24])l_c1.b_output,
    //     4 );

    
    dim3 gridDimc(1, 6, 1);
    dim3 blockDimc(24, 24, 1);
    ConvLayerBackward_Kernel<<<gridDimc, blockDimc>>>(
        (float (*)[24][24])l_c1.preact,
        (float (*)[24][24])l_c1.b_output,
        (float (*)[5][5])l_c1.weight,
        (float (*)[28])l_input.output,
        l_c1.bias);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // after cudaEventRecord
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

static void learn(){

    float time_taken = 0.0;

    clock_t t;
	t = clock();

    for(int i=0; i< train_cnt; i++){
    //for(int i=0; i<10; i++){
    //     printf("label: %d \n", train_set[i].label);

        l_f.bp_clear();
		l_p.bp_clear();
		l_c1.bp_clear();
        
        time_taken += forward(train_set[i].data);
        loss_func<<<1, 10>>>(l_f.b_preact, l_f.output, train_set[i].label, 10);
        time_taken += backward();

    }

    printf("time on GPU: %.5f seconds\n", time_taken /  1000);

    t = clock() - t;
	float cpu_time = (float)t/CLOCKS_PER_SEC;
    printf("Total spend %.2f seconds \n", cpu_time);
}


static unsigned int classify(double data[28][28])
{
	float res[10];

	forward(data);

	unsigned int max = 0;

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	// cudaMemcpy(res, l_f.b_preact, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	printf("Test Accuracy:: %.2f%%\n", 100 - ( double(error) / double(test_cnt) * 100.0));
}


int main(){
    int epoch = 5;
    printf("CNN CUDA version result: \n");
    printf("Number of epoch: %d  \n\n", epoch);
    loadData();
    
    for (int i = 0; i < epoch; i++){
        printf("epoch: %d  \n", i + 1);
        learn();
        test();
    }
    
    
    printf("finish\n");

    return 0;
}