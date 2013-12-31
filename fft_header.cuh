#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h> 
#include <iomanip> 
#include <iostream>
#include <vector>
#include <ctime>

using namespace std;

#define NRANK 2
#define BATCH 10



const size_t NX = 4;
const size_t NY = 6;

void fft_R2C_pitched(float2 *h_out_data,float *h_in_data,int I,int J);
void fft_R2C(float2 *h_out_data,float *h_in_data,int I,int J);
void forward_tf();
void backward_tf();