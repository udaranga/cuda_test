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
	float h_in_data_temp[NX][NY] ={ 
 		 {0.7943 ,   0.6020 ,   0.7482  ,  0.9133  ,  0.9961 , 0.9261},
 	     {0.3112 ,   0.2630 ,   0.4505  ,  0.1524  ,  0.0782 ,  0.1782},
         {0.5285 ,   0.6541 ,   0.0838  ,  0.8258  ,  0.4427,  0.3842},
         {0.1656 ,   0.6892 ,   0.2290  ,  0.5383  ,  0.1067,  0.1712}
  		};

	// --------------------------------
	// Input array (dynamic) - host side 
	// Allocated Memory
// 	float **a = new float*[NX];  
// 	for (int r = 0; r < NX; ++r)  // this can be also done on GPU
//   		a[r] = new float[NY]; 
	float *a = new float[NX*NY];  
  	
	// Set the values
	size_t h_ipitch;
	for (int r = 0; r < NX; ++r)  // this can be also done on GPU
		{  	 
		for (int c = 0; c < NY; ++c)
			{	a[NY*r + c] = h_in_data_temp[r][c];	}
		}
	// --------------------------------

	// Output array - host side
	float2 *h_out_data_temp = new float2[NX*(NY/2+1)] ;
	//float2 h_out_data_temp[NX][NY/2+1] ;
    
	// Input and Output array - device side	
	cufftHandle plan;
	cufftReal *d_in_data;      
	cufftComplex * d_out_data;
	int n[NRANK] = {NX, NY};

  	cudaMalloc((void**)&d_in_data, sizeof(cufftReal)*NX*NY);    
    cudaMemcpy(d_in_data, a, sizeof(cufftReal)*NY*NX, cudaMemcpyHostToDevice);
    	 

//  Copy input array from Host to Device
	//size_t ipitch;
	//cudaError  cudaStat1 = 	cudaMallocPitch((void**)&d_in_data,&ipitch,NY*sizeof(cufftReal),NX);	
	//cout << cudaGetErrorString(cudaStat1) << endl;
	//cudaError  cudaStat2 = 	cudaMemcpy2D(d_in_data,ipitch,a,sizeof(float),sizeof(float),NY*NX,cudaMemcpyHostToDevice);  //<------THIS DOESN"T WORK (DYNAMIC ARRAY) 
	////  //cudaError  cudaStat2 = 	cudaMemcpy2D(d_in_data,ipitch,h_in_data_temp,sizeof(float)*NY,sizeof(float)*NY,NX,cudaMemcpyHostToDevice);  //<------THIS WORKS (STATIC ARRAY)
	//cout << cudaGetErrorString(cudaStat2) << endl;

	cudaMalloc((void**)&d_out_data, sizeof(cufftComplex)*NX*(NY/2 + 1));
  

    //Performe the fft
    //cufftPlanMany(&plan, NRANK, n,NULL, 1, 0,NULL, 1, 0,CUFFT_R2C,BATCH);
	cufftPlan2d(&plan, NX, NY , CUFFT_R2C);
    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
    cufftExecR2C(plan, d_in_data, d_out_data);
    cudaThreadSynchronize();

	// Copy d_in_data back to host
	//cudaError  cudaStat4 = cudaMemcpy2D(h_out_data_temp,(NY)*sizeof(float),d_in_data,ipitch,NY*sizeof(cufftReal),NX,cudaMemcpyDeviceToHost); 
	//cudaMemcpy(h_out_data_temp,d_in_data ,  sizeof(cufftReal)*NY*NX, cudaMemcpyDeviceToHost);  // --- working direct in - out copy
	cudaMemcpy(h_out_data_temp,d_out_data,  sizeof(cufftComplex)*NX*(NY/2 + 1), cudaMemcpyDeviceToHost);
	//cout << cudaGetErrorString(cudaStat4) << endl;


	// Print the results
	for (int i = 0; i < NX; i++)	
		{
		for (int j =0 ; j< NY/2+1; j++)			 
			 printf(" %f + %fi",h_out_data_temp[i*(NY/2+1) + j].x ,h_out_data_temp[i*(NY/2+1) + j].y);
		printf("\n");	 
		}
	cudaFree(d_in_data);

	return 0;
	}

 