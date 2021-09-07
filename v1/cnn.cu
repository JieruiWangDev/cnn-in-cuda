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

//define the kernel size
#define TILE_WIDTH 16  //for small example

// set Layer
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

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
    mnist_load("MNIST_data/train-images.idx3-ubyte", "MNIST_data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("MNIST_data/t10k-images.idx3-ubyte", "MNIST_data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

static float forward(const double data[28][28]){

    // printf("run forward\n");

    
    float input[28][28];

    for (int i = 0; i<28; i++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
        }
    }

    

    l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

    // printf("**************************************\n");


    //example for convLayer 1:

    l_input.setInput((float *)input);

    //printf("input image: %f\n", &l_input.output[0][0]);


    //timer
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    

    int bz;
    bz = ceil((float)28/TILE_WIDTH)*ceil((float)28/TILE_WIDTH);
    dim3 gridDim(1, 6, bz);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    ConvLayerForward_Kernel_1<<<gridDim,blockDim>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, l_c1.bias, 1, 28, 28, 24, 5, 6);
    apply_sigmoid <<<64,64>>>(l_c1.preact, l_c1.output, l_c1.bytes);

    // for pooling layer example:
    bz = ceil((float)6/TILE_WIDTH)*ceil((float)6/TILE_WIDTH);
    dim3 gridDimPool(1, 6, bz);
    dim3 blockDimPool(TILE_WIDTH, TILE_WIDTH, 1);
    MaxPool2dForward_Kernel_1<<<gridDimPool,blockDimPool>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight, l_s1.bias ,24, 24, 6, 4);
    apply_sigmoid <<<64,64>>>(l_s1.preact, l_s1.output, l_s1.bytes);

    // for fully connected layer
    bz = ceil((float)10/TILE_WIDTH);
    dim3 gridDimfc(1, 10, 1);
    dim3 blockDimfc(6, 6, 6);
    FullyConLayerForward_kernel<<<gridDimfc,blockDimfc>>>((float (*)[6][6])l_s1.output, (float (*)[6][6][6])l_f.weight, l_f.preact, l_f.bias, 1, 6, 10, 1, 10);
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
	cudaEvent_t start2, stop2;
	float time;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
    
    dim3 gridDimfc(1, 10, 1);
    dim3 blockDimfc(6, 6, 6);
    bp_f<<<gridDimfc, blockDimfc>>>(
        (float (*)[6][6][6])l_f.d_weight, 
        l_f.d_preact,
        l_f.bias,
        (float (*)[6][6][6]) l_f.weight,
        (float (*)[6][6])l_s1.output,
        (float (*)[6][6])l_s1.d_output,
        (float (*)[6][6])l_s1.d_preact);

    
    dim3 gridDims(1, 6, 1);
    dim3 blockDims(6, 6, 1);
    bp_s1<<<gridDims, blockDims>>>(
        (float (*)[6][6])l_s1.preact,
        (float (*)[6][6])l_s1.d_output,
        (float (*)[6][6])l_s1.d_preact,
        (float (*)[4][4])l_s1.d_weight,
        (float (*)[4][4])l_s1.weight,
        (float (*)[24][24])l_c1.output,
        (float (*)[24][24])l_c1.d_output,
        l_s1.bias);

    
    dim3 gridDimc(1, 6, 1);
    dim3 blockDimc(24, 24, 1);
    bp_c1<<<gridDimc, blockDimc>>>(
        (float (*)[24][24])l_c1.preact,
        (float (*)[24][24])l_c1.d_preact,
        (float (*)[24][24])l_c1.d_output,
        (float (*)[5][5])l_c1.d_weight,
        (float (*)[5][5])l_c1.weight,
        (float (*)[28])l_input.output,
        l_c1.bias);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2); // after cudaEventRecord
    cudaEventElapsedTime(&time, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

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
		l_s1.bp_clear();
		l_c1.bp_clear();
        
        time_taken += forward(train_set[i].data);
        makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
        time_taken += backward();

     }

     printf("time on GPU: %.5f seconds\n", time_taken /  1000);

     t = clock() - t;
     float cpu_time = (float)t/CLOCKS_PER_SEC;
     printf("Total spend %.2f s.\n", cpu_time);
}


static unsigned int classify(double data[28][28])
{
	float res[10];

	forward(data);

	unsigned int max = 0;

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	// cudaMemcpy(res, l_f.d_preact, sizeof(float) * 10, cudaMemcpyDeviceToHost);

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

	printf("Test Accuracy:: %.2lf%%\n", 100 - ( double(error) / double(test_cnt) * 100.0));
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