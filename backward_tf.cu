#include "fft_header.cuh"

void backward_tf()
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
	float2 *h_out_data = new float2[NX*NY] ; 
	int N = 1;
	double elapsed_secs = 0;
	int mode = 0;
	switch (mode)
		{
	case 0 :
		// ***************************
		// FFT - with pitched arrays
		elapsed_secs = 0;
	
		for (int i = 0 ; i < N ; i++)
			{
			clock_t begin = clock();
			fft_R2C_pitched(h_out_data,h_in_data_dynamic,NX,NY);
			clock_t end = clock();
			elapsed_secs = elapsed_secs  +  double(end - begin) / CLOCKS_PER_SEC;	
			}

		cout << "Pitched " <<  elapsed_secs/N << endl;

		// ***************************
		break;
	case 1 : 
		// ***************************
		// FFT - normal arrays 
		elapsed_secs = 0;
	 
		for (int i = 0 ; i < N ; i++)
			{
			clock_t begin = clock();
			fft_R2C(h_out_data,h_in_data_dynamic,NX,NY);
			clock_t end = clock();
			elapsed_secs = elapsed_secs  +  double(end - begin) / CLOCKS_PER_SEC;	
			}

		cout << "Normal " << elapsed_secs/N << endl;
		break;
		}
	// ***************************

	// Print the results
	for (int i = 0; i < NX; i++)	
		{
		for (int j =0 ; j< NY ; j++)		
			printf(" %f + %fi",h_out_data[i*NY + j].x ,h_out_data[i*NY + j].y);
		printf("\n");	 
		}
}

void fft_R2C_pitched(float2 *h_out_data,float *h_in_data,int I,int J){

	// Output array - host side
	float2 *h_out_data_temp = new float2[I*(J/2+1)] ; 

	// Input and Output array - device side	
	cufftHandle plan;
	cufftReal *d_in_data;      
	cufftComplex * d_out_data;
	int n[NRANK] = {I, J};

	//  Copy input array from Host to Device
	size_t ipitch;
	cudaError  cudaStat1 = 	cudaMallocPitch((void**)&d_in_data,&ipitch,J*sizeof(cufftReal),I);		 
	cudaError  cudaStat2 = 	cudaMemcpy2D(d_in_data,ipitch,h_in_data,J*sizeof(float),J*sizeof(float),I,cudaMemcpyHostToDevice);   
	 
	//  Allocate memory for output array - device side
	size_t opitch;
	cudaError  cudaStat3 = 	cudaMallocPitch((void**)&d_out_data,&opitch,(J/2+1)*sizeof(cufftComplex),I);	
	
	//  Performe the fft
	int rank = 2; // 2D fft     
	int istride = 1, ostride = 1; // Stride lengths
	int idist = 1, odist = 1;     // Distance between batches
	int inembed[] = {I, ipitch/sizeof(cufftReal)}; // Input size with pitch
    int onembed[] = {I, opitch/sizeof(cufftComplex)}; // Output size with pitch
	int batch = 1;
	cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
	//cufftPlan2d(&plan, I, J , CUFFT_R2C);
	cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
	cufftExecR2C(plan, d_in_data, d_out_data);
	cudaThreadSynchronize();

	// Copy d_in_data back from device to host
	cudaError  cudaStat4 = cudaMemcpy2D(h_out_data_temp,(J/2+1)*sizeof(float2), d_out_data, opitch, (J/2+1)*sizeof(cufftComplex), I, cudaMemcpyDeviceToHost); 
	//cout << cudaGetErrorString(cudaStat4) << endl;

	for (int i = 0; i < I ; i++)
		for (int j = 0; j < J ; j++)
			if ( j < J/2 + 1)
				h_out_data[i*J + j] = h_out_data_temp[i*(J/2+1) + j];
			else if ( i == 0)
				{ 
				h_out_data[i*J + j].x = h_out_data_temp[i*(J/2+1) + (J-j)].x;
				h_out_data[i*J + j].y = -h_out_data_temp[i*(J/2+1) + (J-j)].y;
				}
			else 
				{
				h_out_data[i*J + j].x = h_out_data_temp[(I-i)*(J/2+1) + (J-j)].x;
				h_out_data[i*J + j].y = -h_out_data_temp[(I-i)*(J/2+1) + (J-j)].y;
				}

	cufftDestroy(plan);	
	cudaFree(d_out_data);
	cudaFree(d_in_data);
	
}

void fft_R2C(float2 *h_out_data,float *h_in_data,int I,int J){

	// Output array - host side
	float2 *h_out_data_temp = new float2[I*(J/2+1)] ; 
  
	cufftHandle plan;
	cufftReal *d_in_data;       // Input array - device side
	cufftComplex * d_out_data;  // Output array - device side
	int n[NRANK] = {I, J};

 	
  	cudaMalloc((void**)&d_in_data, sizeof(cufftReal)*I*J);
    cudaMemcpy(d_in_data, h_in_data, sizeof(cufftReal)*I*J, cudaMemcpyHostToDevice);    	  
    cudaMalloc((void**)&d_out_data, sizeof(cufftComplex)*I*(J/2 + 1));
    

     //Performe the fft
	 cufftPlan2d(&plan, NX, NY , CUFFT_R2C);
     cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
     cufftExecR2C(plan, d_in_data, d_out_data);
     cudaThreadSynchronize();
	 cudaError  cudaStat2 = 	cudaMemcpy(h_out_data_temp,d_out_data,  sizeof(cufftComplex)*I*(J/2+1), cudaMemcpyDeviceToHost);
 
	 	for (int i = 0; i < I ; i++)
		for (int j = 0; j < J ; j++)
			if ( j < J/2 + 1)
				h_out_data[i*J + j] = h_out_data_temp[i*(J/2+1) + j];
			else if ( i == 0)
				{ 
				h_out_data[i*J + j].x = h_out_data_temp[i*(J/2+1) + (J-j)].x;
				h_out_data[i*J + j].y = -h_out_data_temp[i*(J/2+1) + (J-j)].y;
				}
			else 
				{
				h_out_data[i*J + j].x = h_out_data_temp[(I-i)*(J/2+1) + (J-j)].x;
				h_out_data[i*J + j].y = -h_out_data_temp[(I-i)*(J/2+1) + (J-j)].y;
				}
	

	 cufftDestroy(plan);
	 cudaFree(d_out_data);


	}