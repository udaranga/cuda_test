#define NRANK 2
#define BATCH 10

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h> 
#include <iomanip> 
#include <iostream>
#include <vector>

using namespace std;

const size_t NX = 4;
const size_t NY = 6;

int main()
	{ 
	// Input array (static) - host side 
	float h_in_data_static[NX][NY] ={ 
		{0.7943 ,   0.6020 ,   0.7482  ,  0.9133  ,  0.9961 , 0.9261},
		{0.3112 ,   0.2630 ,   0.4505  ,  0.1524  ,  0.0782 ,  0.1782},
		{0.5285 ,   0.6541 ,   0.0838  ,  0.8258  ,  0.4427,  0.3842},
		{0.1656 ,   0.6892 ,   0.2290  ,  0.5383  ,  0.1067,  0.1712}
		};

	// --------------------------------
	// Input array (dynamic) - host side 
	float *h_in_data_dynamic = new float[NX*NY];  

	// Set the values
	size_t h_ipitch;
	for (int r = 0; r < NX; ++r)  // this can be also done on GPU
		{  	 
		for (int c = 0; c < NY; ++c)
			{	h_in_data_dynamic[NY*r + c] = h_in_data_static[r][c];	}
		}
	// --------------------------------

	// Output array - host side
	float2 *h_out_data_temp = new float2[NX*(NY/2+1)] ; 


	// Input and Output array - device side	
	cufftHandle plan;
	cufftReal *d_in_data;      
	cufftComplex * d_out_data;
	int n[NRANK] = {NX, NY};

	//  Copy input array from Host to Device
	size_t ipitch;
	cudaError  cudaStat1 = 	cudaMallocPitch((void**)&d_in_data,&ipitch,NY*sizeof(cufftReal),NX);	
	cout << cudaGetErrorString(cudaStat1) << endl;
	cudaError  cudaStat2 = 	cudaMemcpy2D(d_in_data,ipitch,h_in_data_dynamic,NY*sizeof(float),NY*sizeof(float),NX,cudaMemcpyHostToDevice);   
	cout << cudaGetErrorString(cudaStat2) << endl;

	//  Allocate memory for output array - device side
	size_t opitch;
	cudaError  cudaStat3 = 	cudaMallocPitch((void**)&d_out_data,&opitch,(NY/2+1)*sizeof(cufftComplex),NX);	
	cout << cudaGetErrorString(cudaStat3) << endl;
	
	//  Performe the fft
	int rank = 2; // 2D fft     
	int istride = 1, ostride = 1; // Stride lengths
	int idist = 1, odist = 1;     // Distance between batches
	int inembed[] = {ipitch, NX}; // Input size with pitch
	int onembed[] = {opitch, NX}; // Output size with pitch
	int batch = 1;
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	//cufftPlan2d(&plan, NX, NY , CUFFT_R2C);
	cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
	cufftExecR2C(plan, d_in_data, d_out_data);
	cudaThreadSynchronize();

	// Copy d_in_data back from device to host
	cudaError  cudaStat4 = cudaMemcpy2D(h_out_data_temp,(NY/2+1)*sizeof(float2), d_out_data, opitch, (NY/2+1)*sizeof(cufftComplex), NX, cudaMemcpyDeviceToHost); 
	cout << cudaGetErrorString(cudaStat4) << endl;
	
	// Print the results
	for (int i = 0; i < NX; i++)	
		{
		for (int j =0 ; j< NY/2 + 1; j++)		
			printf(" %f + %fi",h_out_data_temp[i*(NY/2+1) + j].x ,h_out_data_temp[i*(NY/2+1) + j].y);
		printf("\n");	 
		}
	cudaFree(d_in_data);

	return 0;
	}

