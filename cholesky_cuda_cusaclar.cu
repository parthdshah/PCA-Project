// cholesky decomposition using the library - cuSolver

//**********************************
// cuSolver library must be installed
//**********************************

// link to the documentation of the library - http://docs.nvidia.com/cuda/cusolver/index.html#axzz3V3SakC7i

// author: yathindra kota 
// mail: yatkota@ufl.edu
// last modified: 21st, March, 2015

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudense.h>

#define num_rows 100
#define num_cols 100

#define NUM_THREADS (num_rows*num_cols)
#define BLOCK_WIDTH 1000


void check_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char* argv[])
{ 
	cudsHandle_t cudenseH = NULL;
	cublasHandle_t cublasH = NULL;
    	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  
	int lwork = 0;
	int *devInfo = NULL;
	
	// allocating matrix on the host device
	
	double h_matrix[num_rows*num_cols];
	double h_matrix_ouput[num_rows*num_cols];
	
	//initialize the matrix to a random set of values (0 to 10) of type float 	
	for(int tempi=0; tempi<num_rows;tempi++)
	{
		for(int tempj=0; tempi<num_cols;tempj++)	
		{
			h_matrix[ (tempi*num_rows) + tempj] = (rand()%100)/10; // set the input matrix elements 
		}	
	}		

	// create cudense/cublas handle
    	cusolver_status = cudsCreate(&cudenseH);
    	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    	cublas_status = cublasCreate(&cublasH);
    	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	
	// allocate memory in the GPU and set the output matrix to 0s.. And also copy the input matrix from hot to device
	
	double *d_matrix_input, *d_matrix_output;
    	check_error(cudaMalloc((void **) &d_matrix_input, num_rows * num_cols * sizeof(float)));
	check_error(cudaMalloc((void **) &d_matrix_output, num_rows * num_cols * sizeof(float)));
	check_error(cudaMemset(*d_matrix_output, 0, num_rows * num_cols * sizeof(float)));
	check_error(cudaMemcpy(*d_matrix_input, *h_matrix, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

	//------	
	// alocation of the buffer in the GPU
	// The following function allocates required amount of memory in the GPU and this
	// function is specific to Cholesky function which is called next
	// This function is also from the library cuSolver
		
	cusolverStatus_t cusolverDnDpotrf_bufferSize(cudenseH,
                 NULL, /*cublasFillMode_t uplo, Maybe "CUBLAS_FILL_MODE_LOWER" */
                 num_rows,
                 d_matrix_input,
                 num_rows,
                 &Lwork);
	
	double *d_work = NULL; 	
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    	assert(cudaSuccess == cudaStat1);
	
	timer.start();
	
	// implementation of Cholesky
	cusolverStatus_t cusolverDnDpotrf(cudenseH,
           NULL,/* not sure, needs to be checked*/
           num_rows,
           d_matrix_input,
           num_rows,
           d_work, /*not sure */
           Lwork,
           devInfo );
	
	cudaStat1 = cudaDeviceSynchronize(); // synchronization
    	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    	assert(cudaSuccess == cudaStat1);
	
	timer.stop();	
	//// check if Cholesky is good or not

	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    	assert(cudaSuccess == cudaStat1);
	
	// following can be added if required
    	//printf("after geqrf: info_gpu = %d\n", info_gpu); 
    	assert(0 == info_gpu);

	cudaStat1 = cudaMemcpy(h_matrix_ouput,d_work , sizeof(double)*num_rows*num_cols, cudaMemcpyDeviceToHost); 
	//d_work needs to checked
    	
    	assert(cudaSuccess == cudaStat1);
	
	printf("Time elapsed = %g ms\n", timer.Elapsed());
	
	// print output to file
	FILE *fp;
	fp = fopen("cholesky_output.txt","w");
	fprintf(fp,"output is\n");	
	
	for(tempi=0; tempi<num_rows;tempi++)
	{
		fprintf(fp,"\n");
		for(tempj=0; tempi<num_cols;tempj++)	
		{
			fprintf(fp,"%f", h_matrix_ouput[(tempj * num_rows) + tempi]);
		}	
		
	}

	//de-allocating the memory
	if(d_matrix_input)
		cudaFree(d_matrix_input);
		
	if(d_matrix_output)
		cudaFree(d_matrix_output);
		
	if(d_work)
		cudaFree(d_work);
	
	if (cublasH ) cublasDestroy(cublasH);   
    	if (cudenseH) cudsDestroy(cudenseH); 	

	return 0;	
}
